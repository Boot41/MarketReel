from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://localhost/marketreeldb")
os.environ.setdefault("SECRET_KEY", "test")
os.environ.setdefault("ADK_API_KEY", "test")
os.environ.setdefault("ADK_MODEL", "gemini-2.0-flash")
os.environ.setdefault("BACKEND_BASE_URL", "http://localhost:8010")

from agents.marketlogic import agent  # noqa: E402


class _DummySessionService:
    async def get_session(self, *, app_name: str, user_id: str, session_id: str):
        return None

    async def create_session(self, *, app_name: str, user_id: str, session_id: str | None):
        return SimpleNamespace(id=session_id or "sess-1")


class _DummyRunner:
    def __init__(self, events: list[SimpleNamespace]) -> None:
        self._events = events

    async def run_async(self, **kwargs):
        for event in self._events:
            yield event


def _event(*, author: str, text: str, is_final: bool) -> SimpleNamespace:
    content = SimpleNamespace(parts=[SimpleNamespace(text=text)])
    return SimpleNamespace(
        author=author,
        content=content,
        is_final_response=lambda: is_final,
    )


@pytest.mark.asyncio
async def test_run_agent_uses_orchestrator_final_text(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        _event(author="DataAgent", text="sub-agent-final", is_final=True),
        _event(author=agent.root_agent.name, text="orchestrator-final", is_final=True),
    ]
    monkeypatch.setattr(agent, "_get_session_service", lambda: _DummySessionService())
    monkeypatch.setattr(agent, "_get_runner", lambda session_service: _DummyRunner(events))

    reply, session_id = await agent.run_agent(
        message="hello",
        user_id="u1",
        session_id=None,
    )

    assert reply == "orchestrator-final"
    assert session_id == "sess-1"


@pytest.mark.asyncio
async def test_run_agent_ignores_sub_agent_final_when_root_fallback_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        _event(author="ValuationAgent", text="sub-agent-final", is_final=True),
        _event(author=agent.root_agent.name, text="root-non-final-text", is_final=False),
    ]
    monkeypatch.setattr(agent, "_get_session_service", lambda: _DummySessionService())
    monkeypatch.setattr(agent, "_get_runner", lambda session_service: _DummyRunner(events))

    reply, session_id = await agent.run_agent(
        message="hello",
        user_id="u1",
        session_id=None,
    )

    assert reply == "root-non-final-text"
    assert session_id == "sess-1"
