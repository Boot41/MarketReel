from __future__ import annotations

from ..types import EvidenceBundle, OrchestratorInput, RiskFlag, StrategyResult, ValuationResult


class StrategyAgent:
    """Deterministic release strategy synthesis."""

    @classmethod
    async def run(
        cls,
        orchestrator_input: OrchestratorInput,
        evidence: EvidenceBundle,
        valuation: ValuationResult,
        risk_flags: list[RiskFlag],
    ) -> StrategyResult:
        db = evidence.get("db_evidence", {})
        windows = db.get("theatrical_windows", [])
        release_window = 45
        if windows:
            release_window = max(14, min(90, int(windows[0].get("days", 45))))

        high_risk = any(flag["severity"] == "HIGH" for flag in risk_flags)
        scenario_override = orchestrator_input.get("scenario_override")
        if scenario_override == "streaming_first":
            release_mode = "streaming_first"
        elif scenario_override == "theatrical_first":
            release_mode = "theatrical_first"
        else:
            release_mode = "streaming_first" if high_risk else "theatrical_first"

        theatrical = valuation["theatrical_projection_usd"]
        vod = valuation["vod_projection_usd"]

        if release_mode == "streaming_first":
            theatrical *= 0.55
            vod *= 1.25

        marketing_spend = max(
            250_000.0,
            (theatrical + vod) * (0.12 if release_mode == "theatrical_first" else 0.08),
        )

        roi_theatrical = ((theatrical + vod) - valuation["mg_estimate_usd"] - marketing_spend) / max(
            1.0, valuation["mg_estimate_usd"] + marketing_spend
        )
        roi_streaming = (
            (theatrical * 0.6 + vod * 1.2) - valuation["mg_estimate_usd"] - marketing_spend * 0.8
        ) / max(1.0, valuation["mg_estimate_usd"] + marketing_spend * 0.8)

        return {
            "release_mode": release_mode,
            "release_window_days": int(release_window),
            "marketing_spend_usd": round(marketing_spend, 2),
            "platform_priority": ["theatrical", "premium_vod", "svod"]
            if release_mode == "theatrical_first"
            else ["svod", "premium_vod", "theatrical_limited"],
            "roi_scenarios": {
                "base": round(roi_theatrical, 3),
                "streaming_first": round(roi_streaming, 3),
            },
        }
