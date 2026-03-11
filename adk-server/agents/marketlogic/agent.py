"""MarketLogic ADK agent.

Used by the ADK server at /v1/run.
"""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types
from loguru import logger

from app.core.config import get_settings

settings = get_settings()


def summarize_film(title: str, logline: str) -> dict[str, str]:
    """Provide a short acquisition-focused summary."""
    return {
        "summary": (
            f"{title}: {logline} Focus on global appeal, comparable comps, and a launch plan."
        )
    }


root_agent = Agent(
    name="marketlogic_agent",
    model=settings.adk_model,
    description="Assists film distribution executives with acquisition and release strategy.",
    instruction=(
        "You are MarketLogic AI. Help evaluate indie films for global acquisition and "
        "theatrical release strategy. Ask for missing deal inputs when needed."
    ),
    tools=[summarize_film],
)

_session_service: DatabaseSessionService | None = None
_runner: Runner | None = None


def _get_runner() -> tuple[DatabaseSessionService, Runner]:
    global _session_service, _runner
    if _session_service is None or _runner is None:
        logger.info("adk_runner_init app_name={} model={}", settings.app_name, settings.adk_model)
        _session_service = DatabaseSessionService(settings.database_url)
        _runner = Runner(agent=root_agent, app_name=settings.app_name, session_service=_session_service)
    return _session_service, _runner


def _get_session_service() -> DatabaseSessionService:
    global _session_service
    if _session_service is None:
        _session_service = DatabaseSessionService(settings.database_url)
    return _session_service


async def run_agent(message: str, user_id: str, session_id: str | None) -> tuple[str, str]:
    logger.debug(
        "agent_run_start user_id={} session_id={} message_len={}",
        user_id,
        session_id or "new",
        len(message),
    )
    # If no API key is configured, avoid calling the model runtime and return a safe stub.
    if not settings.google_api_key and not settings.google_genai_use_vertexai:
        logger.warning("agent_run_stub_no_provider user_id={} session_id={}", user_id, session_id or "new")
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
        return (
            "MarketLogic AI is running without a Google API key. "
            "Set GOOGLE_API_KEY to enable model responses.",
            session.id,
        )

    session_service, runner = _get_runner()

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

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=message)],
    )

    final_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text or ""

    logger.info(
        "agent_run_complete user_id={} session_id={} reply_len={}",
        user_id,
        session.id,
        len(final_text),
    )
    return final_text, session.id
