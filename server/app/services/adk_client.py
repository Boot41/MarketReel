from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.core.config import get_settings


@dataclass(slots=True)
class AdkRunResult:
    reply: str
    session_id: str


async def run_adk(message: str, user_id: str, session_id: str | None) -> AdkRunResult:
    settings = get_settings()
    payload = {"message": message, "user_id": user_id, "session_id": session_id}
    headers = {"X-ADK-API-Key": settings.adk_api_key}
    async with httpx.AsyncClient(base_url=settings.adk_base_url, timeout=30.0) as client:
        response = await client.post("/v1/run", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    return AdkRunResult(reply=data["reply"], session_id=data["session_id"])
