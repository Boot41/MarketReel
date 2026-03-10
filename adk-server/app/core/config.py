from __future__ import annotations

from functools import lru_cache

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).resolve().parents[3]

load_dotenv(ROOT_DIR / ".env")


class Settings(BaseSettings):
    env: str = "development"
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5433/marketreeldb"
    google_api_key: str = ""
    google_genai_use_vertexai: bool = False
    adk_api_key: str = ""
    app_name: str = "marketlogic_adk"
    adk_model: str = "gemini-2.5-flash"


@lru_cache
def get_settings() -> Settings:
    return Settings()
