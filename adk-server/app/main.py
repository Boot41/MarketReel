from __future__ import annotations

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from app import agent

app = FastAPI(title="MarketLogic ADK", version="1.0.0")


class RunRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    user_id: str = Field(min_length=1, max_length=200)
    session_id: str | None = None


class RunResponse(BaseModel):
    reply: str
    session_id: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/run", response_model=RunResponse)
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
