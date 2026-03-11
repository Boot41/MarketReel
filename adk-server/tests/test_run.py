from __future__ import annotations

import sys
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.marketlogic import agent  # noqa: E402
from app import main  # noqa: E402


@pytest.mark.asyncio
async def test_run_endpoint(monkeypatch):
    async def _fake_run(message: str, user_id: str, session_id: str | None):
        return f"Echo: {message}", session_id or "sess-1"

    monkeypatch.setattr(agent, "run_agent", _fake_run)

    monkeypatch.setattr(main.settings, "adk_api_key", "test-key")

    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/run",
            json={"message": "Hello", "user_id": "u1", "session_id": None},
            headers={"X-ADK-API-Key": "test-key"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "Echo: Hello"
    assert data["session_id"] == "sess-1"


@pytest.mark.asyncio
async def test_run_requires_api_key(monkeypatch):
    monkeypatch.setattr(main.settings, "adk_api_key", "test-key")

    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/run",
            json={"message": "Hello", "user_id": "u1", "session_id": None},
        )

    assert response.status_code == 401
