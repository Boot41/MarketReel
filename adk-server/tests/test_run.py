from __future__ import annotations

import sys
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import agent  # noqa: E402
from app.main import app  # noqa: E402


@pytest.mark.asyncio
async def test_run_endpoint(monkeypatch):
    async def _fake_run(message: str, user_id: str, session_id: str | None):
        return f"Echo: {message}", session_id or "sess-1"

    monkeypatch.setattr(agent, "run_agent", _fake_run)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/run",
            json={"message": "Hello", "user_id": "u1", "session_id": None},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "Echo: Hello"
    assert data["session_id"] == "sess-1"
