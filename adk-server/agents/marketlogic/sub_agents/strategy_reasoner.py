from __future__ import annotations

from datetime import datetime
from typing import Any

from ..config import config
from ..prompt_runtime import STRATEGY_AGENT_PROMPT, provider_runtime_enabled, run_prompt_json
from ..types import EvidenceBundle, OrchestratorInput, RiskFlag, StrategyResult, ValuationResult
from .reasoner_contracts import validate_strategy_payload
from .strategy_agent import StrategyAgent


class StrategyReasoner:
    """Prompt-contract strategy reasoner with schema-only retries."""

    @staticmethod
    def _months_delta(start: str, end: str) -> int:
        try:
            s = datetime.strptime(start + "-01", "%Y-%m-%d")
            e = datetime.strptime(end + "-01", "%Y-%m-%d")
            delta = (e.year - s.year) * 12 + (e.month - s.month)
            return max(1, delta)
        except Exception:
            return 2

    @classmethod
    def _adapt_prompt_payload(cls, payload: dict[str, Any]) -> StrategyResult | None:
        rec = payload.get("release_recommendation")
        scenarios = payload.get("scenario_comparison", [])
        if not isinstance(rec, dict):
            return None

        mode_raw = str(rec.get("mode", "")).upper()
        if mode_raw in {"STREAMING_FIRST", "VOD_ONLY"}:
            release_mode = "streaming_first"
        else:
            release_mode = "theatrical_first"

        start = str(rec.get("window_start", ""))
        end = str(rec.get("window_end", ""))
        release_window_days = cls._months_delta(start, end) * 30

        marketing = rec.get("marketing_spend_usd", {})
        if isinstance(marketing, dict):
            marketing_spend = float(marketing.get("mid", 0.0) or 0.0)
        else:
            marketing_spend = 0.0

        platforms = rec.get("platform_priority", [])
        platform_priority = [str(item) for item in platforms] if isinstance(platforms, list) else []

        roi_scenarios: dict[str, float] = {}
        if isinstance(scenarios, list):
            for item in scenarios:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("scenario_name", "")).strip()
                if not name:
                    continue
                roi_pct = float(item.get("roi_pct", 0.0) or 0.0)
                roi_scenarios[name] = round(roi_pct / 100.0, 3)

        if not roi_scenarios:
            roi_scenarios = {"base": 0.0}

        result: StrategyResult = {
            "release_mode": release_mode,  # type: ignore[typeddict-item]
            "release_window_days": int(release_window_days),
            "marketing_spend_usd": round(marketing_spend, 2),
            "platform_priority": platform_priority,
            "roi_scenarios": roi_scenarios,
        }
        if validate_strategy_payload(result):
            return result
        return None

    @classmethod
    async def _run_prompt_path(
        cls,
        *,
        orchestrator_input: OrchestratorInput,
        evidence: EvidenceBundle,
        valuation: ValuationResult,
        risk_flags: list[RiskFlag],
    ) -> StrategyResult | None:
        input_payload: dict[str, Any] = {
            "movie_id": evidence["movie"],
            "territory": evidence["territory"],
            "scenario_overrides": orchestrator_input.get("scenario_override"),
            "valuation_output": valuation,
            "risk_output": risk_flags,
            "evidence_bundle": evidence,
        }
        raw = await run_prompt_json(
            prompt=STRATEGY_AGENT_PROMPT,
            input_payload=input_payload,
            model=config.worker_model,
        )
        if not isinstance(raw, dict):
            return None
        return cls._adapt_prompt_payload(raw)

    @classmethod
    async def run(
        cls,
        *,
        orchestrator_input: OrchestratorInput,
        evidence: EvidenceBundle,
        valuation: ValuationResult,
        risk_flags: list[RiskFlag],
        provider_enabled: bool,
    ) -> tuple[StrategyResult | None, str | None]:
        max_attempts = max(1, int(config.schema_retry_limit) + 1)
        if provider_runtime_enabled(provider_enabled):
            for _ in range(max_attempts):
                prompt_result = await cls._run_prompt_path(
                    orchestrator_input=orchestrator_input,
                    evidence=evidence,
                    valuation=valuation,
                    risk_flags=risk_flags,
                )
                if prompt_result is not None and validate_strategy_payload(prompt_result):
                    return prompt_result, None
            return None, "strategy_reasoner_schema_invalid"

        for _ in range(max_attempts):
            payload = await StrategyAgent.run(
                orchestrator_input=orchestrator_input,
                evidence=evidence,
                valuation=valuation,
                risk_flags=risk_flags,
            )
            if validate_strategy_payload(payload):
                return payload, None
        return None, "strategy_reasoner_schema_invalid"
