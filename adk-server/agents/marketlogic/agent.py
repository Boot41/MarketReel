"""MarketLogic ADK agent.

Used by the ADK server at /v1/run.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent
from google.adk.sessions import DatabaseSessionService
from loguru import logger

from app.core.config import get_settings
from .orchestrator import run_marketlogic_orchestrator
from .tools import IndexNavigator, IndexRegistry, SufficiencyChecker, TargetedFetcher

settings = get_settings()

root_agent = Agent(
    name="MarketLogicOrchestrator",
    model=settings.adk_model,
    description=(
        "Top-level orchestrator for film acquisition valuation, risk analysis, and release strategy."
    ),
    instruction=(
        "You are MarketLogicOrchestrator. Route each query across data, valuation, risk, and strategy "
        "workflows and return a strict JSON scorecard with citations and confidence signals."
    ),
    tools=[IndexRegistry, IndexNavigator, TargetedFetcher, SufficiencyChecker],
)

_session_service: DatabaseSessionService | None = None
_session_state_cache: dict[str, dict[str, Any]] = {}


def _get_session_service() -> DatabaseSessionService:
    global _session_service
    if _session_service is None:
        logger.info("adk_runner_init app_name={} model={}", settings.app_name, settings.adk_model)
        _session_service = DatabaseSessionService(settings.database_url)
    return _session_service


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

    state = _load_state(session.id, session)
    scorecard, state_delta = await run_marketlogic_orchestrator(
        message=message,
        session_state=state,
        provider_enabled=provider_enabled,
    )
    state.update(state_delta)
    _persist_state(session.id, session, state)

    final_text = json.dumps(scorecard, ensure_ascii=True)
    logger.info(
        "agent_run_complete user_id={} session_id={} reply_len={}",
        user_id,
        session.id,
        len(final_text),
    )
    return final_text, session.id
