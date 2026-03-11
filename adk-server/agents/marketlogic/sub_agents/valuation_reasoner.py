from __future__ import annotations

from ..config import config
from ..types import EvidenceBundle, RiskFlag, ValuationResult
from .reasoner_contracts import validate_valuation_payload
from .valuation_agent import ValuationAgent


class ValuationReasoner:
    """Prompt-contract valuation reasoner with schema-only retries."""

    @classmethod
    async def run(
        cls,
        *,
        evidence: EvidenceBundle,
        risk_flags: list[RiskFlag],
        provider_enabled: bool,
    ) -> tuple[ValuationResult | None, str | None]:
        max_attempts = max(1, int(config.schema_retry_limit) + 1)
        for _ in range(max_attempts):
            # Provider-enabled prompt execution is intentionally constrained by deterministic
            # controls; deterministic calculator remains the source of truth.
            payload = await ValuationAgent.run(evidence=evidence, risk_flags=risk_flags)
            if validate_valuation_payload(payload):
                return payload, None
        return None, "valuation_reasoner_schema_invalid"
