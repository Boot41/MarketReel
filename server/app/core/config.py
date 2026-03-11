from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

SERVER_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = SERVER_DIR.parent

load_dotenv(ROOT_DIR / ".env")


class Settings(BaseSettings):
    env: Literal["development", "test", "production"] = "development"
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5433/marketreeldb"
    secret_key: str = "app-scaffold-dev-secret"
    google_api_key: str = ""
    google_genai_use_vertexai: bool = False
    adk_base_url: str = "http://localhost:8011"
    adk_api_key: str = ""
    internal_api_key: str = ""
    internal_api_next_key: str = ""

    @property
    def effective_db_url(self) -> str:
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
