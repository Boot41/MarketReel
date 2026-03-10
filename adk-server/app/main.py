from __future__ import annotations

import secrets

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from app import agent
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
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/run", response_model=RunResponse, dependencies=[Depends(verify_api_key)])
async def run(request: RunRequest) -> RunResponse:
    try:
        reply, session_id = await agent.run_agent(
            message=request.message,
            user_id=request.user_id,
            session_id=request.session_id,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ADK run failed",
        ) from exc

    return RunResponse(reply=reply, session_id=session_id)
