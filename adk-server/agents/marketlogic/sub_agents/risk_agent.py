from __future__ import annotations

from ..types import EvidenceBundle, RiskFlag


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


class RiskAgent:
    """Deterministic risk extraction from retrieved evidence."""

    @staticmethod
    def _risk_from_text(territory: str, text: str) -> tuple[str, str] | None:
        row = _normalize(text)
        territory_norm = _normalize(territory)
        if territory_norm not in row:
            return None
        if "high" in row:
            return "HIGH", "Content likely needs significant edits or alternative distribution mode."
        if "medium" in row:
            return "MEDIUM", "Apply territory-specific edit plan and pre-clear with local legal/distribution."
        if "low" in row:
            return "LOW", "Standard territory compliance process should be sufficient."
        return None

    @classmethod
    async def run(cls, evidence: EvidenceBundle) -> list[RiskFlag]:
        flags: list[RiskFlag] = []
        territory = evidence["territory"]

        for item in evidence["document_evidence"].get("documents", []):
            text = str(item.get("text", ""))
            source = str(item.get("source_path", ""))
            if "censorship" in source:
                detected = cls._risk_from_text(territory, text)
                if detected:
                    severity, mitigation = detected
                    flags.append(
                        {
                            "category": "CENSORSHIP",
                            "severity": severity,
                            "scene_ref": str(item.get("doc_id", "unknown")),
                            "source_ref": source,
                            "mitigation": mitigation,
                            "confidence": 0.8 if severity != "LOW" else 0.65,
                        }
                    )
            if "cultural_sensitivity" in source and _normalize(territory) in _normalize(text):
                flags.append(
                    {
                        "category": "CULTURAL_SENSITIVITY",
                        "severity": "MEDIUM",
                        "scene_ref": str(item.get("doc_id", "unknown")),
                        "source_ref": source,
                        "mitigation": "Localize campaign and trailer cut with culturally aligned messaging.",
                        "confidence": 0.62,
                    }
                )

        if not flags:
            flags.append(
                {
                    "category": "MARKET",
                    "severity": "LOW",
                    "scene_ref": "market_baseline",
                    "source_ref": "derived:insufficient-risk-signal",
                    "mitigation": "Proceed with baseline compliance and territory pre-screening.",
                    "confidence": 0.45,
                }
            )

        return flags
