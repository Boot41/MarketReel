from __future__ import annotations

from typing import Any

def _has_keys(payload: dict[str, Any], required_keys: list[str]) -> bool:
    return all(key in payload for key in required_keys)


def validate_valuation_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return _has_keys(
        payload,
        [
            "mg_estimate_usd",
            "confidence_interval_low_usd",
            "confidence_interval_high_usd",
            "theatrical_projection_usd",
            "vod_projection_usd",
            "comparable_films",
            "sufficiency_score",
        ],
    )


def validate_risk_payload(payload: Any) -> bool:
    if not isinstance(payload, list):
        return False
    for item in payload:
        if not isinstance(item, dict):
            return False
        if not _has_keys(
            item,
            ["category", "severity", "scene_ref", "source_ref", "mitigation", "confidence"],
        ):
            return False
    return True


def validate_strategy_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return _has_keys(
        payload,
        ["release_mode", "release_window_days", "marketing_spend_usd", "platform_priority", "roi_scenarios"],
    )
