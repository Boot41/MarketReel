from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.db.models import User


@pytest.mark.asyncio
async def test_login_success(client: AsyncClient, seed_user: User) -> None:
    response = await client.post(
        "/auth/login", json={"username": "testuser", "password": "testpass"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["user"]["username"] == "testuser"


@pytest.mark.asyncio
async def test_login_wrong_password(client: AsyncClient, seed_user: User) -> None:
    response = await client.post("/auth/login", json={"username": "testuser", "password": "wrong"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_login_unknown_user(client: AsyncClient) -> None:
    response = await client.post("/auth/login", json={"username": "nobody", "password": "pass"})
    assert response.status_code == 401
