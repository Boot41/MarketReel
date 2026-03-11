from __future__ import annotations

from typing import Any

from ..config import config
from ..prompt_runtime import RISK_AGENT_PROMPT, provider_runtime_enabled, run_prompt_json
from ..types import EvidenceBundle, RiskFlag
from .risk_agent import RiskAgent
from .reasoner_contracts import validate_risk_payload


class RiskReasoner:
    """Prompt-contract risk reasoner with schema-only retries."""

    @staticmethod
    def _adapt_prompt_flags(payload: dict[str, Any]) -> list[RiskFlag] | None:
        raw_flags = payload.get("risk_flags")
        if not isinstance(raw_flags, list):
            return None
        adapted: list[RiskFlag] = []
        for item in raw_flags:
            if not isinstance(item, dict):
                continue
            category = str(item.get("category", "")).upper()
            severity = str(item.get("severity", "")).upper()
            if category not in {"CENSORSHIP", "CULTURAL_SENSITIVITY", "MARKET"}:
                continue
            if severity not in {"LOW", "MEDIUM", "HIGH"}:
                continue
            adapted.append(
                {
                    "category": category,  # type: ignore[typeddict-item]
                    "severity": severity,  # type: ignore[typeddict-item]
                    "scene_ref": str(item.get("scene_id") or item.get("flag_id") or "unknown"),
                    "source_ref": str(item.get("guideline_reference", "unknown")),
                    "mitigation": str(item.get("mitigation", "")),
                    "confidence": float(item.get("confidence", 0.0) or 0.0),
                }
            )
        if validate_risk_payload(adapted):
            return adapted
        return None

    @classmethod
    async def _run_prompt_path(cls, *, evidence: EvidenceBundle) -> list[RiskFlag] | None:
        input_payload: dict[str, Any] = {
            "movie_id": evidence["movie"],
            "territory": evidence["territory"],
            "evidence_bundle": evidence,
        }
        raw = await run_prompt_json(
            prompt=RISK_AGENT_PROMPT,
            input_payload=input_payload,
            model=config.worker_model,
        )
        if not isinstance(raw, dict):
            return None
        return cls._adapt_prompt_flags(raw)

    @classmethod
    async def run(
        cls,
        *,
        evidence: EvidenceBundle,
        provider_enabled: bool,
    ) -> tuple[list[RiskFlag] | None, str | None]:
        max_attempts = max(1, int(config.schema_retry_limit) + 1)
        if provider_runtime_enabled(provider_enabled):
            for _ in range(max_attempts):
                prompt_result = await cls._run_prompt_path(evidence=evidence)
                if prompt_result is not None and validate_risk_payload(prompt_result):
                    return prompt_result, None
            return None, "risk_reasoner_schema_invalid"

        for _ in range(max_attempts):
            payload = await RiskAgent.run(evidence=evidence)
            if validate_risk_payload(payload):
                return payload, None
        return None, "risk_reasoner_schema_invalid"
