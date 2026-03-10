from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.passwords import verify_password
from app.core.config import get_settings
from app.db.models import User


async def login(db: AsyncSession, username: str, password: str) -> tuple[str | None, User | None]:
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(password, user.password_hash):
        return None, None

    settings = get_settings()
    payload = {
        "sub": str(user.id),
        "exp": datetime.now(UTC) + timedelta(days=7),
    }
    token = jwt.encode(payload, settings.secret_key, algorithm="HS256")
    return token, user
