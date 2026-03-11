from __future__ import annotations

from ..config import config
from ..types import EvidenceBundle, RiskFlag
from .risk_agent import RiskAgent
from .reasoner_contracts import validate_risk_payload


class RiskReasoner:
    """Prompt-contract risk reasoner with schema-only retries."""

    @classmethod
    async def run(
        cls,
        *,
        evidence: EvidenceBundle,
        provider_enabled: bool,
    ) -> tuple[list[RiskFlag] | None, str | None]:
        _ = provider_enabled
        max_attempts = max(1, int(config.schema_retry_limit) + 1)
        for _ in range(max_attempts):
            payload = await RiskAgent.run(evidence=evidence)
            if validate_risk_payload(payload):
                return payload, None
        return None, "risk_reasoner_schema_invalid"
