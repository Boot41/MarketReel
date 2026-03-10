"""MarketLogic ADK agent.

Run via API: POST /v1/run
"""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

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
    model="gemini-2.0-flash",
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
        _session_service = DatabaseSessionService(settings.database_url)
        _runner = Runner(agent=root_agent, app_name=settings.app_name, session_service=_session_service)
    return _session_service, _runner


async def run_agent(message: str, user_id: str, session_id: str | None) -> tuple[str, str]:
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

    return final_text, session.id
