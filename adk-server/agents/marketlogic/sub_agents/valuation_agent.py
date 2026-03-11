from __future__ import annotations

from ..tools import mg_calculator_tool
from ..types import EvidenceBundle, RiskFlag, ValuationResult


class ValuationAgent:
    """Deterministic valuation and projection calculations."""

    @staticmethod
    def _estimate_risk_penalty(risk_flags: list[RiskFlag]) -> float:
        if not risk_flags:
            return 0.05
        penalty = 0.0
        for flag in risk_flags:
            severity = flag["severity"]
            if severity == "HIGH":
                penalty += 0.18
            elif severity == "MEDIUM":
                penalty += 0.08
            else:
                penalty += 0.03
        return min(0.6, penalty)

    @classmethod
    async def run(cls, evidence: EvidenceBundle, risk_flags: list[RiskFlag]) -> ValuationResult:
        db = evidence.get("db_evidence", {})
        box_office = db.get("box_office", {})
        actor_signals = db.get("actor_signals", {})
        comparables = db.get("comparable_films", [])
        tool_diagnostics = evidence.get("tool_diagnostics", [])
        has_db_tool_failure = any(
            isinstance(item, dict) and str(item.get("source", "")).startswith("db")
            for item in tool_diagnostics
        )

        comparable_values = [float(item.get("territory_gross_usd", 0.0)) for item in comparables]
        comparable_avg = sum(comparable_values) / len(comparable_values) if comparable_values else 0.0

        mg_estimate_usd = mg_calculator_tool(
            avg_box_office_usd=float(box_office.get("avg_gross_usd", 0.0)),
            avg_qscore=float(actor_signals.get("avg_qscore", 0.0)),
            comparable_avg_gross_usd=float(comparable_avg),
            risk_penalty=cls._estimate_risk_penalty(risk_flags),
            allow_baseline_fallback=not has_db_tool_failure,
        )

        theatrical_projection = max(
            mg_estimate_usd * 2.4,
            float(box_office.get("avg_gross_usd", 0.0)) * 0.75,
        )

        vod = db.get("vod_benchmarks", {})
        vod_projection = max(
            mg_estimate_usd * 0.7,
            float(vod.get("avg_price_max_usd", 0.0)) * 1.1,
        )

        confidence = max(0.25, min(0.95, evidence["data_sufficiency_score"] * 0.9))
        interval_low = mg_estimate_usd * (0.8 - (1.0 - confidence) * 0.15)
        interval_high = mg_estimate_usd * (1.2 + (1.0 - confidence) * 0.2)

        return {
            "mg_estimate_usd": round(mg_estimate_usd, 2),
            "confidence_interval_low_usd": round(interval_low, 2),
            "confidence_interval_high_usd": round(interval_high, 2),
            "theatrical_projection_usd": round(theatrical_projection, 2),
            "vod_projection_usd": round(vod_projection, 2),
            "comparable_films": [str(item.get("title", "")) for item in comparables if item.get("title")],
            "sufficiency_score": round(confidence, 3),
        }
