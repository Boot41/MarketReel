from __future__ import annotations

import json
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

ROOT_DIR = Path(__file__).resolve().parents[3]
DOCS_ROOT = ROOT_DIR / "adk-server" / "docs"


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


@lru_cache
def _load_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return cast(dict[str, Any], payload)
    return {}


@lru_cache
def _load_jsonl(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                items.append(payload)
    return items


@lru_cache
def page_index_items() -> list[dict[str, Any]]:
    return _load_jsonl(str(DOCS_ROOT / "page_index" / "pages.jsonl"))


@lru_cache
def scene_index_items() -> list[dict[str, Any]]:
    return _load_jsonl(str(DOCS_ROOT / "scripts_indexed" / "scenes.jsonl"))


@lru_cache
def page_manifest() -> dict[str, Any]:
    return _load_json(str(DOCS_ROOT / "page_index" / "manifest.json"))


@lru_cache
def scene_manifest() -> dict[str, Any]:
    return _load_json(str(DOCS_ROOT / "scripts_indexed" / "scene_manifest.json"))


def known_movies() -> list[str]:
    names: set[str] = set()
    for item in page_manifest().get("documents", []):
        movie = item.get("movie")
        if isinstance(movie, str) and movie.strip():
            names.add(movie.strip())
    for item in scene_manifest().get("scripts", []):
        movie = item.get("movie")
        if isinstance(movie, str) and movie.strip():
            names.add(movie.strip())
    return sorted(names)


def known_territories() -> list[str]:
    names: set[str] = set()
    for item in page_manifest().get("documents", []):
        country = item.get("country")
        if isinstance(country, str) and country.strip():
            names.add(country.strip())
    return sorted(names)


def index_registry() -> dict[str, Any]:
    return {
        "page_index_manifest": page_manifest(),
        "scene_manifest": scene_manifest(),
        "known_movies": known_movies(),
        "known_territories": known_territories(),
    }


def build_retrieval_plan(movie: str, territory: str, intent: str) -> dict[str, Any]:
    intent_key = _normalize(intent)
    doc_types = ["synopses", "reviews", "marketing"]
    if intent_key in {"risk", "full_scorecard"}:
        doc_types.extend(
            [
                "cultural_sensitivity",
                "censorship",
                "censorship_guidelines_countries",
            ]
        )
    if intent_key in {"strategy", "full_scorecard"}:
        doc_types.append("scripts")
    return {
        "movie": movie,
        "territory": territory,
        "intent": intent,
        "doc_types": sorted(set(doc_types)),
        "max_docs": 10,
        "max_scenes": 6,
    }


def _movie_match(record: dict[str, Any], movie: str) -> bool:
    movie_norm = _normalize(movie)
    candidates = [
        str(record.get("movie", "")),
        str(record.get("doc_id", "")),
        str(record.get("source_path", "")),
    ]
    return any(movie_norm in _normalize(candidate) for candidate in candidates if candidate)


def _territory_match(record: dict[str, Any], territory: str) -> bool:
    territory_norm = _normalize(territory)
    candidates = [str(record.get("country", "")), str(record.get("text", ""))]
    return any(territory_norm in _normalize(candidate) for candidate in candidates if candidate)


def targeted_fetch(plan: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    movie = str(plan.get("movie", "")).strip()
    territory = str(plan.get("territory", "")).strip()
    wanted_types = {str(item) for item in plan.get("doc_types", [])}
    max_docs = int(plan.get("max_docs", 10))
    max_scenes = int(plan.get("max_scenes", 6))

    docs: list[dict[str, Any]] = []
    for item in page_index_items():
        doc_type = str(item.get("doc_type", ""))
        if doc_type and wanted_types and doc_type not in wanted_types:
            continue
        if movie and not _movie_match(item, movie):
            if doc_type == "censorship_guidelines_countries" and territory and _territory_match(
                item, territory
            ):
                pass
            else:
                continue
        if doc_type == "censorship_guidelines_countries" and territory and not _territory_match(
            item, territory
        ):
            continue
        docs.append(item)
        if len(docs) >= max_docs:
            break

    scenes: list[dict[str, Any]] = []
    for item in scene_index_items():
        if movie and not _movie_match(item, movie):
            continue
        scenes.append(item)
        if len(scenes) >= max_scenes:
            break

    return {"documents": docs, "scenes": scenes}


def sufficiency_check(
    fetched: dict[str, list[dict[str, Any]]],
    min_items: int = 4,
) -> dict[str, Any]:
    documents = fetched.get("documents", [])
    scenes = fetched.get("scenes", [])
    total = len(documents) + len(scenes)
    score = min(1.0, total / float(max(1, min_items * 2)))
    status = "PASS" if total >= min_items else "EXPAND"
    return {"status": status, "score": round(score, 3), "total_items": total}


def _citation_from_record(item: dict[str, Any]) -> dict[str, Any]:
    excerpt = str(item.get("text", "")).strip()
    page = item.get("page")
    return {
        "source_path": str(item.get("source_path", "")),
        "doc_id": str(item.get("doc_id", "")),
        "page": page if isinstance(page, int) else item.get("start_page"),
        "excerpt": excerpt[:220],
    }


def source_citations(items: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for item in items:
        citations.append(_citation_from_record(item))
        if len(citations) >= limit:
            break
    return citations


async def _query_rows(sql: str, db: AsyncSession, params: dict[str, Any]) -> list[dict[str, Any]]:
    result = await db.execute(text(sql), params)
    return [dict(row._mapping) for row in result]


async def get_box_office_by_genre_territory(
    db: AsyncSession, movie: str, territory: str
) -> dict[str, Any]:
    sql = """
    WITH target_film AS (
        SELECT f.id AS film_id
        FROM films f
        WHERE lower(f.title) = lower(:movie)
        LIMIT 1
    ), target_genres AS (
        SELECT fg.genre_id
        FROM film_genres fg
        JOIN target_film tf ON tf.film_id = fg.film_id
    )
    SELECT
        COALESCE(AVG(bo.gross_usd), 0) AS avg_gross_usd,
        COALESCE(SUM(bo.gross_usd), 0) AS total_gross_usd,
        COUNT(*) AS samples
    FROM box_office bo
    JOIN territories t ON t.id = bo.territory_id
    WHERE bo.genre_id IN (SELECT genre_id FROM target_genres)
      AND lower(t.name) = lower(:territory)
    """
    rows = await _query_rows(sql, db=db, params={"movie": movie, "territory": territory})
    if not rows:
        return {"avg_gross_usd": 0.0, "total_gross_usd": 0.0, "samples": 0}
    row = rows[0]
    return {
        "avg_gross_usd": float(row.get("avg_gross_usd") or 0.0),
        "total_gross_usd": float(row.get("total_gross_usd") or 0.0),
        "samples": int(row.get("samples") or 0),
    }


async def get_actor_qscore(db: AsyncSession, movie: str) -> dict[str, Any]:
    sql = """
    SELECT
      COALESCE(AVG(a.q_score), 0) AS avg_qscore,
      COALESCE(SUM(a.social_reach), 0) AS total_social_reach
    FROM films f
    JOIN film_cast fc ON fc.film_id = f.id
    JOIN actors a ON a.id = fc.actor_id
    WHERE lower(f.title) = lower(:movie)
    """
    rows = await _query_rows(sql, db=db, params={"movie": movie})
    if not rows:
        return {"avg_qscore": 0.0, "total_social_reach": 0}
    row = rows[0]
    return {
        "avg_qscore": float(row.get("avg_qscore") or 0.0),
        "total_social_reach": int(row.get("total_social_reach") or 0),
    }


async def get_theatrical_window_trends(db: AsyncSession, territory: str) -> list[dict[str, Any]]:
    sql = """
    SELECT tw.window_type, tw.days
    FROM theatrical_windows tw
    JOIN territories t ON t.id = tw.territory_id
    WHERE lower(t.name) = lower(:territory)
    ORDER BY tw.days ASC
    """
    rows = await _query_rows(sql, db=db, params={"territory": territory})
    return [
        {"window_type": str(item.get("window_type") or ""), "days": int(item.get("days") or 0)}
        for item in rows
    ]


async def get_exchange_rates(db: AsyncSession, territory: str) -> dict[str, Any]:
    sql = """
    SELECT cr.currency_code, cr.rate_to_usd, cr.rate_date
    FROM currency_rates cr
    JOIN territories t ON t.currency_code = cr.currency_code
    WHERE lower(t.name) = lower(:territory)
    ORDER BY cr.rate_date DESC
    LIMIT 1
    """
    rows = await _query_rows(sql, db=db, params={"territory": territory})
    if not rows:
        return {"currency_code": "USD", "rate_to_usd": 1.0, "rate_date": None}
    row = rows[0]
    rate_date = row.get("rate_date")
    rate_date_value = (
        rate_date.isoformat() if isinstance(rate_date, (date, datetime)) else None
    )
    return {
        "currency_code": str(row.get("currency_code") or "USD"),
        "rate_to_usd": float(row.get("rate_to_usd") or 1.0),
        "rate_date": rate_date_value,
    }


async def get_vod_price_benchmarks(db: AsyncSession, territory: str) -> dict[str, Any]:
    sql = """
    SELECT
      COALESCE(AVG(v.price_min_usd), 0) AS avg_price_min_usd,
      COALESCE(AVG(v.price_max_usd), 0) AS avg_price_max_usd
    FROM vod_price_benchmarks v
    JOIN territories t ON t.id = v.territory_id
    WHERE lower(t.name) = lower(:territory)
    """
    rows = await _query_rows(sql, db=db, params={"territory": territory})
    if not rows:
        return {"avg_price_min_usd": 0.0, "avg_price_max_usd": 0.0}
    row = rows[0]
    return {
        "avg_price_min_usd": float(row.get("avg_price_min_usd") or 0.0),
        "avg_price_max_usd": float(row.get("avg_price_max_usd") or 0.0),
    }


async def get_comparable_films(
    db: AsyncSession, movie: str, territory: str, limit: int = 5
) -> list[dict[str, Any]]:
    sql = """
    WITH target_film AS (
        SELECT id AS film_id
        FROM films
        WHERE lower(title) = lower(:movie)
        LIMIT 1
    ), target_genres AS (
        SELECT genre_id
        FROM film_genres
        WHERE film_id = (SELECT film_id FROM target_film)
    )
    SELECT f.title, COALESCE(SUM(bo.gross_usd), 0) AS territory_gross_usd
    FROM box_office bo
    JOIN films f ON f.id = bo.film_id
    JOIN territories t ON t.id = bo.territory_id
    WHERE bo.genre_id IN (SELECT genre_id FROM target_genres)
      AND lower(t.name) = lower(:territory)
      AND lower(f.title) <> lower(:movie)
    GROUP BY f.title
    ORDER BY territory_gross_usd DESC
    LIMIT :limit
    """
    rows = await _query_rows(
        sql,
        db=db,
        params={"movie": movie, "territory": territory, "limit": limit},
    )
    return [
        {
            "title": str(item.get("title") or ""),
            "territory_gross_usd": float(item.get("territory_gross_usd") or 0.0),
        }
        for item in rows
    ]


def docs_search(
    *,
    movie: str,
    territory: str,
    intent: str,
    max_docs: int = 10,
    max_scenes: int = 6,
) -> dict[str, Any]:
    plan = build_retrieval_plan(movie=movie, territory=territory, intent=intent)
    plan["max_docs"] = max_docs
    plan["max_scenes"] = max_scenes
    fetched = targeted_fetch(plan)
    sufficiency = sufficiency_check(fetched)
    records = fetched.get("documents", []) + fetched.get("scenes", [])
    citations = source_citations(records)
    return {
        "movie": movie,
        "territory": territory,
        "intent": intent,
        "documents": fetched.get("documents", []),
        "scenes": fetched.get("scenes", []),
        "citations": citations,
        "sufficiency": sufficiency,
    }


async def evidence_bundle(
    db: AsyncSession,
    *,
    movie: str,
    territory: str,
    intent: str,
    needs_db: bool,
    needs_docs: bool,
) -> dict[str, Any]:
    document_evidence: dict[str, Any] = {"documents": [], "scenes": []}
    citations: list[dict[str, Any]] = []
    data_sufficiency_score = 0.0

    if needs_docs:
        docs_result = docs_search(movie=movie, territory=territory, intent=intent)
        document_evidence = {
            "documents": docs_result["documents"],
            "scenes": docs_result["scenes"],
        }
        citations = list(docs_result["citations"])
        data_sufficiency_score = float(docs_result["sufficiency"]["score"])

    db_evidence: dict[str, Any] = {}
    if needs_db:
        box_office = await get_box_office_by_genre_territory(db, movie=movie, territory=territory)
        actor_signals = await get_actor_qscore(db, movie=movie)
        theatrical_windows = await get_theatrical_window_trends(db, territory=territory)
        exchange_rates = await get_exchange_rates(db, territory=territory)
        vod_benchmarks = await get_vod_price_benchmarks(db, territory=territory)
        comparable_films = await get_comparable_films(db, movie=movie, territory=territory)
        db_evidence = {
            "box_office": box_office,
            "actor_signals": actor_signals,
            "theatrical_windows": theatrical_windows,
            "exchange_rates": exchange_rates,
            "vod_benchmarks": vod_benchmarks,
            "comparable_films": comparable_films,
        }

    return {
        "movie": movie,
        "territory": territory,
        "intent": intent,
        "document_evidence": document_evidence,
        "db_evidence": db_evidence,
        "citations": citations,
        "data_sufficiency_score": data_sufficiency_score,
    }
