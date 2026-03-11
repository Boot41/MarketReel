from __future__ import annotations

from typing import Any

from ..config import config
from ..prompt_runtime import VALUATION_AGENT_PROMPT, provider_runtime_enabled, run_prompt_json
from ..types import EvidenceBundle, RiskFlag, ValuationResult
from .reasoner_contracts import validate_valuation_payload
from .valuation_agent import ValuationAgent


class ValuationReasoner:
    """Prompt-contract valuation reasoner with schema-only retries."""

    @staticmethod
    def _adapt_prompt_payload(payload: dict[str, Any]) -> ValuationResult | None:
        mg = payload.get("mg_estimate")
        theatrical = payload.get("theatrical_revenue_projection")
        vod = payload.get("vod_revenue_projection")
        comps = payload.get("comparable_films_used", [])
        if not isinstance(mg, dict) or not isinstance(theatrical, dict) or not isinstance(vod, dict):
            return None

        mg_mid = float(mg.get("currency_usd", mg.get("mid", 0.0)) or 0.0)
        interval_low = float(mg.get("low", 0.0) or 0.0)
        interval_high = float(mg.get("high", 0.0) or 0.0)
        theatrical_mid = float(theatrical.get("mid", 0.0) or 0.0)
        vod_mid = float(vod.get("mid", 0.0) or 0.0)
        confidence = float(mg.get("confidence", 0.0) or 0.0)

        comparable_titles: list[str] = []
        if isinstance(comps, list):
            for item in comps:
                if isinstance(item, dict) and item.get("title"):
                    comparable_titles.append(str(item["title"]))

        result: ValuationResult = {
            "mg_estimate_usd": round(mg_mid, 2),
            "confidence_interval_low_usd": round(interval_low, 2),
            "confidence_interval_high_usd": round(interval_high, 2),
            "theatrical_projection_usd": round(theatrical_mid, 2),
            "vod_projection_usd": round(vod_mid, 2),
            "comparable_films": comparable_titles,
            "sufficiency_score": max(0.0, min(1.0, round(confidence, 3))),
        }
        if validate_valuation_payload(result):
            return result
        return None

    @classmethod
    async def _run_prompt_path(
        cls,
        *,
        evidence: EvidenceBundle,
        risk_flags: list[RiskFlag],
    ) -> ValuationResult | None:
        payload: dict[str, Any] = {
            "movie_id": evidence["movie"],
            "territory": evidence["territory"],
            "evidence_bundle": evidence,
            "risk_flags": risk_flags,
        }
        raw = await run_prompt_json(
            prompt=VALUATION_AGENT_PROMPT,
            input_payload=payload,
            model=config.worker_model,
        )
        if not isinstance(raw, dict):
            return None
        return cls._adapt_prompt_payload(raw)

    @classmethod
    async def run(
        cls,
        *,
        evidence: EvidenceBundle,
        risk_flags: list[RiskFlag],
        provider_enabled: bool,
    ) -> tuple[ValuationResult | None, str | None]:
        max_attempts = max(1, int(config.schema_retry_limit) + 1)
        if provider_runtime_enabled(provider_enabled):
            for _ in range(max_attempts):
                prompt_result = await cls._run_prompt_path(
                    evidence=evidence,
                    risk_flags=risk_flags,
                )
                if prompt_result is not None and validate_valuation_payload(prompt_result):
                    return prompt_result, None
            return None, "valuation_reasoner_schema_invalid"

        for _ in range(max_attempts):
            payload = await ValuationAgent.run(evidence=evidence, risk_flags=risk_flags)
            if validate_valuation_payload(payload):
                return payload, None
        return None, "valuation_reasoner_schema_invalid"
