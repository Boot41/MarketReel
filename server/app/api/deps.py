from __future__ import annotations

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.session import get_db_session


async def db_session(session: AsyncSession = Depends(get_db_session)) -> AsyncSession:
    return session


def settings(s: Settings = Depends(get_settings)) -> Settings:
    return s
