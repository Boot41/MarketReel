"""MarketLogic ADK agent.

Used by the ADK server at /v1/run.
"""

from __future__ import annotations

from pathlib import Path

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.tools.agent_tool import AgentTool
from google.genai import types as genai_types
from loguru import logger

from app.core.config import get_settings
from .config import config
from .sub_agents import (
    data_agent,
    explainability_agent,
    risk_agent,
    strategy_agent,
    valuation_agent,
)

settings = get_settings()

_ORCHESTRATOR_PROMPT = (
    Path(__file__).resolve().parent / "prompts" / "MarketLogicOrchestrator_prompt.txt"
).read_text(encoding="utf-8").strip()


def _content_text(content: genai_types.Content | None) -> str:
    if content is None or not content.parts:
        return ""
    return "\n".join(
        part.text for part in content.parts if getattr(part, "text", None)
    ).strip()


def _user_content(message: str) -> genai_types.Content:
    return genai_types.Content(role="user", parts=[genai_types.Part(text=message)])


root_agent = Agent(
    name="MarketLogicOrchestrator",
    model=config.worker_model,
    description=(
        "Top-level controller for MarketReel film distribution intelligence. "
        "Handles all user messages, routes to specialist sub-agents, and returns "
        "structured acquisition and release strategy recommendations."
    ),
    instruction=_ORCHESTRATOR_PROMPT,
    sub_agents=[data_agent, risk_agent, valuation_agent, strategy_agent, explainability_agent],
)

_session_service: DatabaseSessionService | None = None
_runner: Runner | None = None


def _get_session_service() -> DatabaseSessionService:
    global _session_service
    if _session_service is None:
        logger.info(
            "adk_runner_init app_name={} model={}", settings.app_name, settings.adk_model
        )
        _session_service = DatabaseSessionService(settings.database_url)
    return _session_service


def _get_runner(session_service: DatabaseSessionService) -> Runner:
    global _runner
    if _runner is None:
        _runner = Runner(
            app_name=settings.app_name,
            agent=root_agent,
            session_service=session_service,
        )
    return _runner


async def run_agent(
    message: str, user_id: str, session_id: str | None
) -> tuple[str, str]:
    logger.debug(
        "agent_run_start user_id={} session_id={} message_len={}",
        user_id,
        session_id or "new",
        len(message),
    )

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
        logger.debug(
            "agent_session_created user_id={} session_id={}", user_id, session.id
        )
    else:
        logger.debug(
            "agent_session_reused user_id={} session_id={}", user_id, session.id
        )

    final_text = ""
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=_user_content(message),
        ):
            author = getattr(event, "author", None)
            has_content = bool(event.content and event.content.parts)
            part_types = (
                [
                    "text" if getattr(p, "text", None) else
                    "function_call" if getattr(p, "function_call", None) else
                    "function_response" if getattr(p, "function_response", None) else
                    "other"
                    for p in event.content.parts
                ]
                if has_content else []
            )
            is_final = event.is_final_response()
            logger.debug(
                "adk_event author={} is_final={} part_types={} session_id={}",
                author, is_final, part_types, session.id,
            )

            if is_final and author == root_agent.name and event.content:
                text = _content_text(event.content)
                if text:
                    final_text = text
                else:
                    logger.warning(
                        "adk_final_event_no_text author={} part_types={} session_id={}",
                        author, part_types, session.id,
                    )
            elif is_final and author != root_agent.name and event.content:
                # Sub-agent marked its output as final (agent-transfer pattern).
                # Do not capture — wait for the orchestrator to synthesise and reply.
                logger.debug(
                    "adk_sub_agent_final_skipped author={} session_id={}",
                    author, session.id,
                )
            elif has_content and author == root_agent.name:
                # Fallback: capture last text from root agent even if
                # is_final_response() never fires with text content
                text = _content_text(event.content)
                if text:
                    final_text = text
    except Exception:
        logger.exception(
            "agent_workflow_failed user_id={} session_id={}", user_id, session.id
        )
        raise

    logger.info(
        "agent_run_complete user_id={} session_id={} reply_len={}",
        user_id,
        session.id,
        len(final_text),
    )
    return final_text, session.id
