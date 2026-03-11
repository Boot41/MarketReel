from __future__ import annotations

from typing import Any

from ..tools import IndexNavigator, SufficiencyChecker, TargetedFetcher


class DocumentRetrievalAgent:
    """Deterministic document retrieval agent over indexed corpus."""

    @staticmethod
    def _expand_retrieval_plan(plan: dict[str, Any]) -> dict[str, Any]:
        expanded = dict(plan)
        max_docs = int(plan.get("max_docs", 10))
        max_scenes = int(plan.get("max_scenes", 6))
        expanded["max_docs"] = min(20, max_docs + 6)
        expanded["max_scenes"] = min(12, max_scenes + 4)
        return expanded

    @classmethod
    async def run(
        cls,
        *,
        movie: str,
        territory: str,
        intent: str,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
        plan = IndexNavigator(movie=movie, territory=territory, intent=intent)
        fetched = TargetedFetcher(plan)
        sufficiency = SufficiencyChecker(fetched)

        if sufficiency.get("status") != "PASS":
            expanded_plan = cls._expand_retrieval_plan(plan)
            expanded_fetch = TargetedFetcher(expanded_plan)
            expanded_sufficiency = SufficiencyChecker(expanded_fetch)
            if float(expanded_sufficiency.get("score", 0.0)) >= float(sufficiency.get("score", 0.0)):
                fetched = expanded_fetch
                sufficiency = expanded_sufficiency

        return fetched, sufficiency
