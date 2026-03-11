"""MarketLogic ADK agent.

Used by the ADK server at /v1/run.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types as genai_types
from loguru import logger

from app.core.config import get_settings
from .orchestrator import run_marketlogic_orchestrator

settings = get_settings()

_PROVIDER_FLAG_KEY = "app:provider_enabled"


def _content_text(content: genai_types.Content | None) -> str:
    if content is None or not content.parts:
        return ""
    return "\n".join(part.text for part in content.parts if getattr(part, "text", None)).strip()


def _user_content(message: str) -> genai_types.Content:
    return genai_types.Content(role="user", parts=[genai_types.Part(text=message)])


def _model_content(message: str) -> genai_types.Content:
    return genai_types.Content(role="model", parts=[genai_types.Part(text=message)])


async def _run_stage(*, callback_context: Any) -> genai_types.Content:
    provider_enabled = bool(callback_context.state.get(_PROVIDER_FLAG_KEY, False))
    session_state = callback_context.state.to_dict()
    message = _content_text(callback_context.user_content)

    response_payload, state_delta = await run_marketlogic_orchestrator(
        message=message,
        session_state=session_state,
        provider_enabled=provider_enabled,
    )

    for key, value in state_delta.items():
        callback_context.state[key] = value
    return _model_content(json.dumps(response_payload, ensure_ascii=True))


root_agent = Agent(
    name="MarketLogicOrchestrator",
    model=settings.adk_model,
    description="Single-entry workflow orchestrator for film acquisition valuation, risk analysis, and release strategy.",
    before_agent_callback=_run_stage,
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
