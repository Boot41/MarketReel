from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import db_session
from app.auth.deps import require_user
from app.db.models import User
from app.services.chats import add_message_and_run, create_chat, get_chat, list_chats

router = APIRouter(prefix="/chats", tags=["chats"])


class ChatCreateRequest(BaseModel):
    title: str = Field(default="New Chat", min_length=1, max_length=200)


class MessageCreateRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatOut(BaseModel):
    id: int
    title: str
    adk_session_id: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ChatDetail(ChatOut):
    messages: list[MessageOut]


@router.post("", response_model=ChatOut)
async def create_chat_route(
    body: ChatCreateRequest,
    db: AsyncSession = Depends(db_session),
    user: User = Depends(require_user),
) -> ChatOut:
    chat = await create_chat(db, user.id, body.title)
    return ChatOut.model_validate(chat)


@router.get("", response_model=list[ChatOut])
async def list_chats_route(
    db: AsyncSession = Depends(db_session),
    user: User = Depends(require_user),
) -> list[ChatOut]:
    chats = await list_chats(db, user.id)
    return [ChatOut.model_validate(chat) for chat in chats]


@router.get("/{chat_id}", response_model=ChatDetail)
async def get_chat_route(
    chat_id: int,
    db: AsyncSession = Depends(db_session),
    user: User = Depends(require_user),
) -> ChatDetail:
    chat = await get_chat(db, user.id, chat_id)
    if chat is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    messages = sorted(chat.messages, key=lambda msg: msg.created_at)
    return ChatDetail(
        **ChatOut.model_validate(chat).model_dump(),
        messages=[MessageOut.model_validate(msg) for msg in messages],
    )


@router.post("/{chat_id}/messages", response_model=list[MessageOut])
async def add_message_route(
    chat_id: int,
    body: MessageCreateRequest,
    db: AsyncSession = Depends(db_session),
    user: User = Depends(require_user),
) -> list[MessageOut]:
    try:
        user_msg, assistant_msg = await add_message_and_run(db, user.id, chat_id, body.content)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found") from exc

    return [
        MessageOut.model_validate(user_msg),
        MessageOut.model_validate(assistant_msg),
    ]
