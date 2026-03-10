from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.agent.agent import run_agent

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentRequest(BaseModel):
    message: str


class AgentResponse(BaseModel):
    reply: str


@router.post("/run", response_model=AgentResponse)
async def agent_run(body: AgentRequest) -> AgentResponse:
    reply = await run_agent(body.message)
    return AgentResponse(reply=reply)
