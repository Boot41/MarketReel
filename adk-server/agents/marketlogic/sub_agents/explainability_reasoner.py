from __future__ import annotations

from typing import Any


class ExplainabilityReasoner:
    """Deterministic explainability view over prior analytical artifacts."""

    @classmethod
    def run(cls, session_state: dict[str, Any]) -> dict[str, Any]:
        last_scorecard = session_state.get("last_scorecard")
        evidence_bundle = session_state.get("evidence_bundle")
        if not isinstance(last_scorecard, dict) or not isinstance(evidence_bundle, dict):
            return {
                "response_type": "clarification_response",
                "message": "I do not have prior analytical artifacts in this session yet. Run an analysis first.",
            }

        citations = last_scorecard.get("citations", [])
        top_citations = citations[:3] if isinstance(citations, list) else []
        confidence = last_scorecard.get("confidence")
        warnings = last_scorecard.get("warnings", [])

        return {
            "response_type": "conversation_response",
            "message": "Here is the evidence summary from the latest analysis in this session.",
            "explainability": {
                "confidence": confidence,
                "warnings": warnings if isinstance(warnings, list) else [],
                "top_citations": top_citations,
                "data_sufficiency_score": evidence_bundle.get("data_sufficiency_score", 0.0),
            },
        }
