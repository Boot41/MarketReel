from __future__ import annotations

from ..config import config
from ..types import EvidenceBundle, OrchestratorInput, RiskFlag, StrategyResult, ValuationResult
from .reasoner_contracts import validate_strategy_payload
from .strategy_agent import StrategyAgent


class StrategyReasoner:
    """Prompt-contract strategy reasoner with schema-only retries."""

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
        _ = provider_enabled
        max_attempts = max(1, int(config.schema_retry_limit) + 1)
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
