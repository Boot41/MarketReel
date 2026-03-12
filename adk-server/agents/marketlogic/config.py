from __future__ import annotations

from dataclasses import dataclass

from app.core.config import get_settings

settings = get_settings()


@dataclass(frozen=True)
class MarketLogicConfiguration:
    """Runtime configuration for agent model selection."""

    worker_model: str = settings.adk_model
    critic_model: str = settings.adk_model


config = MarketLogicConfiguration()
