from __future__ import annotations

from dataclasses import dataclass

from app.core.config import get_settings

settings = get_settings()


@dataclass(frozen=True)
class MarketLogicConfiguration:
    """Runtime configuration for hybrid specialist reasoning."""

    worker_model: str = settings.adk_model
    critic_model: str = settings.adk_model
    schema_retry_limit: int = 1


config = MarketLogicConfiguration()
