from __future__ import annotations

import secrets

from fastapi import Depends, FastAPI, Header, HTTPException, status
import httpx
from loguru import logger
from pydantic import BaseModel, Field

from agents.marketlogic import agent
from app.core.config import get_settings

app = FastAPI(title="MarketLogic ADK", version="1.0.0")
settings = get_settings()


class RunRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    user_id: str = Field(min_length=1, max_length=200)
    session_id: str | None = None


class RunResponse(BaseModel):
    reply: str
    session_id: str


def verify_api_key(x_adk_api_key: str | None = Header(default=None)) -> None:
    if not x_adk_api_key or not secrets.compare_digest(x_adk_api_key, settings.adk_api_key):
        logger.warning("adk_auth_failed: missing_or_invalid_api_key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
async def health_ready() -> dict[str, str]:
    backend_url = settings.backend_base_url.rstrip("/")
    headers = {"X-Internal-API-Key": settings.internal_api_key or settings.adk_api_key}
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{backend_url}/internal/v1/meta/registry", headers=headers)
        if response.status_code == 200:
            return {"status": "ready"}
        logger.warning("adk_ready_check_failed status_code={}", response.status_code)
        return {"status": "degraded"}
    except Exception:
        logger.warning("adk_ready_check_failed exception=backend_unreachable")
        return {"status": "degraded"}


@app.post("/v1/run", response_model=RunResponse, dependencies=[Depends(verify_api_key)])
async def run(request: RunRequest) -> RunResponse:
    logger.info(
        "adk_run_start user_id={} session_id={} message_len={}",
        request.user_id,
        request.session_id or "new",
        len(request.message),
    )
    try:
        reply, session_id = await agent.run_agent(
            message=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
        )
    except Exception as exc:
        logger.exception(
            "adk_run_failed user_id={} session_id={}",
            request.user_id,
            request.session_id or "new",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ADK run failed",
        ) from exc

    logger.info(
        "adk_run_success user_id={} session_id={} reply_len={}",
        request.user_id,
        session_id,
        len(reply),
    )
    return RunResponse(reply=reply, session_id=session_id)
