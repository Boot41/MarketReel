from __future__ import annotations

import asyncio
from typing import Any

from ..tools import (
    get_actor_qscore,
    get_box_office_by_genre_territory,
    get_comparable_films,
    get_exchange_rates,
    get_theatrical_window_trends,
    get_vod_price_benchmarks,
    source_citation_tool,
)
from ..types import EvidenceBundle, EvidenceRequest
from .document_retrieval_agent import DocumentRetrievalAgent


class DataAgent:
    """Single gateway for doc and DB evidence retrieval."""

    @classmethod
    async def run(cls, request: EvidenceRequest) -> EvidenceBundle:
        movie = request["movie"]
        territory = request["territory"]

        fetched: dict[str, list[dict[str, Any]]] = {"documents": [], "scenes": []}
        sufficiency: dict[str, Any] = {"status": "EXPAND", "score": 0.0}
        if request["needs_docs"]:
            fetched, sufficiency = await DocumentRetrievalAgent.run(
                movie=movie,
                territory=territory,
                intent=request["intent"],
            )

        all_records = fetched.get("documents", []) + fetched.get("scenes", [])
        citations = source_citation_tool(all_records)

        db_evidence: dict[str, Any] = {}
        if request["needs_db"]:
            box_office, qscore, windows, fx, vod, comparables = await asyncio.gather(
                get_box_office_by_genre_territory(movie, territory),
                get_actor_qscore(movie),
                get_theatrical_window_trends(territory),
                get_exchange_rates(territory),
                get_vod_price_benchmarks(territory),
                get_comparable_films(movie, territory),
            )
            db_evidence = {
                "box_office": box_office,
                "actor_signals": qscore,
                "theatrical_windows": windows,
                "exchange_rates": fx,
                "vod_benchmarks": vod,
                "comparable_films": comparables,
            }

        grouped_documents: dict[str, list[dict[str, Any]]] = {
            "documents": fetched.get("documents", []),
            "scenes": fetched.get("scenes", []),
        }

        return {
            "movie": movie,
            "territory": territory,
            "intent": request["intent"],
            "document_evidence": grouped_documents,
            "db_evidence": db_evidence,
            "citations": citations,
            "data_sufficiency_score": float(sufficiency.get("score", 0.0)),
        }
