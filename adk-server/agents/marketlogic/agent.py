"""MarketLogic ADK agent.

Used by the ADK server at /v1/run.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types as genai_types
from loguru import logger

from app.core.config import get_settings
from .orchestrator import (
    build_evidence_request,
    context_matches,
    decide_action,
    resolve_orchestrator_input,
    run_data_agent,
    run_marketlogic_orchestrator,
    run_risk_agent,
    run_strategy_agent,
    run_validation,
    run_valuation_agent,
    session_dict,
    session_risk,
)
from .tools import combine_validation_warnings, exchange_rate_tool, format_scorecard

settings = get_settings()

_TEMP_INPUT_KEY = "temp:orchestrator_input"
_TEMP_EVIDENCE_KEY = "temp:evidence_bundle"
_TEMP_RISK_KEY = "temp:risk"
_TEMP_VALUATION_KEY = "temp:valuation"
_TEMP_STRATEGY_KEY = "temp:strategy"
_TEMP_FOLLOWUP_KEY = "temp:strategy_followup"
_TEMP_ROUTE_KEY = "temp:orchestrator_route"
_PROVIDER_FLAG_KEY = "app:provider_enabled"


def _content_text(content: genai_types.Content | None) -> str:
    if content is None or not content.parts:
        return ""
    return "\n".join(part.text for part in content.parts if getattr(part, "text", None)).strip()


def _user_content(message: str) -> genai_types.Content:
    return genai_types.Content(role="user", parts=[genai_types.Part(text=message)])


def _model_content(message: str) -> genai_types.Content:
    return genai_types.Content(role="model", parts=[genai_types.Part(text=message)])


async def _resolve_stage(*, callback_context: Any) -> genai_types.Content:
    session_state = callback_context.state.to_dict()
    message = _content_text(callback_context.user_content)
    orchestrator_input = resolve_orchestrator_input(message=message, session_state=session_state)
    route = decide_action(orchestrator_input)
    callback_context.state[_TEMP_INPUT_KEY] = orchestrator_input
    callback_context.state[_TEMP_ROUTE_KEY] = route
    return _model_content("resolved")


async def _data_stage(*, callback_context: Any) -> genai_types.Content:
    session_state = callback_context.state.to_dict()
    orchestrator_input = callback_context.state[_TEMP_INPUT_KEY]
    route = callback_context.state[_TEMP_ROUTE_KEY]
    if route.get("action") != "run_workflow":
        callback_context.state[_TEMP_FOLLOWUP_KEY] = False
        return _model_content("data_skipped")

    same_context = context_matches(orchestrator_input, session_state)
    strategy_followup = same_context and orchestrator_input.get("scenario_override") is not None
    callback_context.state[_TEMP_FOLLOWUP_KEY] = strategy_followup

    previous_evidence = session_dict(session_state, "evidence_bundle")
    if strategy_followup and previous_evidence is not None:
        evidence = previous_evidence
    else:
        evidence = await run_data_agent(build_evidence_request(orchestrator_input))

    callback_context.state[_TEMP_EVIDENCE_KEY] = evidence
    return _model_content("data_ready")


async def _risk_stage(*, callback_context: Any) -> genai_types.Content:
    route = callback_context.state[_TEMP_ROUTE_KEY]
    if route.get("action") != "run_workflow":
        return _model_content("risk_skipped")

    session_state = callback_context.state.to_dict()
    evidence = callback_context.state[_TEMP_EVIDENCE_KEY]
    strategy_followup = bool(callback_context.state.get(_TEMP_FOLLOWUP_KEY))
    previous_risk = session_risk(session_state)

    if strategy_followup and previous_risk is not None:
        risk_flags = previous_risk
    else:
        risk_flags = await run_risk_agent(evidence)

    callback_context.state[_TEMP_RISK_KEY] = risk_flags
    return _model_content("risk_ready")


async def _valuation_stage(*, callback_context: Any) -> genai_types.Content:
    route = callback_context.state[_TEMP_ROUTE_KEY]
    if route.get("action") != "run_workflow":
        return _model_content("valuation_skipped")

    session_state = callback_context.state.to_dict()
    evidence = callback_context.state[_TEMP_EVIDENCE_KEY]
    risk_flags = callback_context.state[_TEMP_RISK_KEY]
    strategy_followup = bool(callback_context.state.get(_TEMP_FOLLOWUP_KEY))
    previous_valuation = session_dict(session_state, "valuation")

    if strategy_followup and previous_valuation is not None:
        valuation = previous_valuation
    else:
        valuation = await run_valuation_agent(evidence=evidence, risk_flags=risk_flags)

    callback_context.state[_TEMP_VALUATION_KEY] = valuation
    return _model_content("valuation_ready")


async def _strategy_stage(*, callback_context: Any) -> genai_types.Content:
    route = callback_context.state[_TEMP_ROUTE_KEY]
    if route.get("action") != "run_workflow":
        return _model_content("strategy_skipped")

    orchestrator_input = callback_context.state[_TEMP_INPUT_KEY]
    evidence = callback_context.state[_TEMP_EVIDENCE_KEY]
    risk_flags = callback_context.state[_TEMP_RISK_KEY]
    valuation = callback_context.state[_TEMP_VALUATION_KEY]

    strategy = await run_strategy_agent(orchestrator_input, evidence, valuation, risk_flags)
    callback_context.state[_TEMP_STRATEGY_KEY] = strategy
    return _model_content("strategy_ready")


async def _finalize_stage(*, callback_context: Any) -> genai_types.Content:
    route = callback_context.state[_TEMP_ROUTE_KEY]
    if route.get("action") != "run_workflow":
        text = str(route.get("direct_response") or "Please share more details so I can help.")
        callback_context.state["last_agent_response_type"] = route.get("action")
        return _model_content(text)

    provider_enabled = bool(callback_context.state.get(_PROVIDER_FLAG_KEY, False))
    orchestrator_input = callback_context.state[_TEMP_INPUT_KEY]
    evidence = callback_context.state[_TEMP_EVIDENCE_KEY]
    risk_flags = callback_context.state[_TEMP_RISK_KEY]
    valuation = callback_context.state[_TEMP_VALUATION_KEY]
    strategy = callback_context.state[_TEMP_STRATEGY_KEY]

    exchange = evidence.get("db_evidence", {}).get("exchange_rates", {})
    acquisition_local = exchange_rate_tool(
        amount_usd=valuation["mg_estimate_usd"],
        rate_to_usd=float(exchange.get("rate_to_usd", 1.0)),
    )

    confidence = round((valuation["sufficiency_score"] + evidence["data_sufficiency_score"]) / 2.0, 3)
    validation = run_validation(
        evidence=evidence,
        valuation=valuation,
        confidence=confidence,
        provider_enabled=provider_enabled,
    )
    warnings = combine_validation_warnings(validation)

    scorecard = format_scorecard(
        territory=orchestrator_input["territory"],
        theatrical_projection_usd=valuation["theatrical_projection_usd"],
        vod_projection_usd=valuation["vod_projection_usd"],
        acquisition_price_usd=valuation["mg_estimate_usd"],
        release_mode=strategy["release_mode"],
        release_window_days=strategy["release_window_days"],
        risk_flags=risk_flags,
        citations=evidence["citations"],
        confidence=confidence,
        warnings=warnings,
    )

    callback_context.state["resolved_context"] = {
        "movie": orchestrator_input["movie"],
        "territory": orchestrator_input["territory"],
        "intent": orchestrator_input["intent"],
        "scenario_override": orchestrator_input.get("scenario_override"),
    }
    callback_context.state["evidence_bundle"] = evidence
    callback_context.state["valuation"] = valuation
    callback_context.state["risk"] = risk_flags
    callback_context.state["strategy"] = strategy
    callback_context.state["last_scorecard"] = scorecard
    callback_context.state["recommended_acquisition_local"] = {
        "currency": exchange.get("currency_code", "USD"),
        "amount": acquisition_local,
    }

    return _model_content(json.dumps(scorecard, ensure_ascii=True))


root_agent = SequentialAgent(
    name="MarketLogicOrchestrator",
    description="Workflow orchestrator for film acquisition valuation, risk analysis, and release strategy.",
    sub_agents=[
        Agent(name="ResolveAgent", before_agent_callback=_resolve_stage),
        Agent(name="DataAgent", before_agent_callback=_data_stage),
        Agent(name="RiskAgent", before_agent_callback=_risk_stage),
        Agent(name="ValuationAgent", before_agent_callback=_valuation_stage),
        Agent(name="StrategyAgent", before_agent_callback=_strategy_stage),
        Agent(name="FinalizeAgent", before_agent_callback=_finalize_stage),
    ],
)

_session_service: DatabaseSessionService | None = None
_runner: Runner | None = None
_session_state_cache: dict[str, dict[str, Any]] = {}


def _get_session_service() -> DatabaseSessionService:
    global _session_service
    if _session_service is None:
        logger.info("adk_runner_init app_name={} model={}", settings.app_name, settings.adk_model)
        _session_service = DatabaseSessionService(settings.database_url)
    return _session_service


def _get_runner(session_service: DatabaseSessionService) -> Runner:
    global _runner
    if _runner is None:
        _runner = Runner(app_name=settings.app_name, agent=root_agent, session_service=session_service)
    return _runner


def _load_state(session_id: str, session_obj: Any) -> dict[str, Any]:
    if session_id in _session_state_cache:
        return _session_state_cache[session_id]

    raw_state = getattr(session_obj, "state", None)
    if isinstance(raw_state, dict):
        state = dict(raw_state)
    else:
        state = {}

    _session_state_cache[session_id] = state
    return state


def _persist_state(session_id: str, session_obj: Any, state: dict[str, Any]) -> None:
    _session_state_cache[session_id] = state
    raw_state = getattr(session_obj, "state", None)
    if isinstance(raw_state, dict):
        raw_state.clear()
        raw_state.update(state)
        return
    try:
        setattr(session_obj, "state", state)
    except Exception:
        logger.debug("session_state_assign_skipped session_id={}", session_id)


async def run_agent(message: str, user_id: str, session_id: str | None) -> tuple[str, str]:
    logger.debug(
        "agent_run_start user_id={} session_id={} message_len={}",
        user_id,
        session_id or "new",
        len(message),
    )

    provider_enabled = bool(settings.google_api_key or settings.google_genai_use_vertexai)
    if not provider_enabled:
        logger.warning("agent_run_model_disabled user_id={} session_id={}", user_id, session_id or "new")

    session_service = _get_session_service()
    runner = _get_runner(session_service)

    session = None
    if session_id:
        session = await session_service.get_session(
            app_name=settings.app_name,
            user_id=user_id,
            session_id=session_id,
        )

    if session is None:
        session = await session_service.create_session(
            app_name=settings.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        logger.debug("agent_session_created user_id={} session_id={}", user_id, session.id)
    else:
        logger.debug("agent_session_reused user_id={} session_id={}", user_id, session.id)

    final_text = ""
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=_user_content(message),
            state_delta={_PROVIDER_FLAG_KEY: provider_enabled},
        ):
            if event.is_final_response() and event.content:
                final_text = _content_text(event.content)
    except Exception:
        logger.exception("agent_workflow_failed user_id={} session_id={}", user_id, session.id)

    if not final_text:
        logger.warning("agent_workflow_fallback user_id={} session_id={}", user_id, session.id)
        state = _load_state(session.id, session)
        response_payload, state_delta = await run_marketlogic_orchestrator(
            message=message,
            session_state=state,
            provider_enabled=provider_enabled,
        )
        state.update(state_delta)
        _persist_state(session.id, session, state)
        if isinstance(response_payload, str):
            final_text = response_payload
        else:
            final_text = json.dumps(response_payload, ensure_ascii=True)
    else:
        refreshed = await session_service.get_session(
            app_name=settings.app_name,
            user_id=user_id,
            session_id=session.id,
        )
        if refreshed is not None:
            _load_state(session.id, refreshed)

    logger.info(
        "agent_run_complete user_id={} session_id={} reply_len={}",
        user_id,
        session.id,
        len(final_text),
    )
    return final_text, session.id
