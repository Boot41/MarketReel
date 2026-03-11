from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.config import get_settings
from .types import Citation, Scorecard, ValidationReport

settings = get_settings()

_ADK_ROOT = Path(__file__).resolve().parents[2]
_DOCS_ROOT = _ADK_ROOT / "docs"

_engine: AsyncEngine | None = None


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


@lru_cache
def _load_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
def _page_index_items() -> list[dict[str, Any]]:
    return _load_jsonl(str(_DOCS_ROOT / "page_index" / "pages.jsonl"))


@lru_cache
def _scene_index_items() -> list[dict[str, Any]]:
    return _load_jsonl(str(_DOCS_ROOT / "scripts_indexed" / "scenes.jsonl"))


@lru_cache
def _page_manifest() -> dict[str, Any]:
    return _load_json(str(_DOCS_ROOT / "page_index" / "manifest.json"))


@lru_cache
def _scene_manifest() -> dict[str, Any]:
    return _load_json(str(_DOCS_ROOT / "scripts_indexed" / "scene_manifest.json"))


@lru_cache
def _known_movies() -> list[str]:
    names: set[str] = set()
    for item in _page_manifest().get("documents", []):
        movie = item.get("movie")
        if isinstance(movie, str) and movie.strip():
            names.add(movie.strip())
    for item in _scene_manifest().get("scripts", []):
        movie = item.get("movie")
        if isinstance(movie, str) and movie.strip():
            names.add(movie.strip())
    return sorted(names)


@lru_cache
def _known_territories() -> list[str]:
    names: set[str] = set()
    for item in _page_manifest().get("documents", []):
        country = item.get("country")
        if isinstance(country, str) and country.strip():
            names.add(country.strip())
    return sorted(names)


def IndexRegistry() -> dict[str, Any]:
    """Return available docs/scenes metadata for retrieval planning."""
    return {
        "page_index_manifest": _page_manifest(),
        "scene_manifest": _scene_manifest(),
        "known_movies": _known_movies(),
        "known_territories": _known_territories(),
    }


def IndexNavigator(movie: str, territory: str, intent: str) -> dict[str, Any]:
    """Build a deterministic retrieval plan from known indexes."""
    intent_key = _normalize(intent)
    doc_types = ["synopses", "reviews", "marketing"]
    if intent_key in {"risk", "full_scorecard"}:
        doc_types.extend(["cultural_sensitivity", "censorship", "censorship_guidelines_countries"])
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


def TargetedFetcher(plan: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Fetch targeted page/scene chunks based on a retrieval plan."""
    movie = str(plan.get("movie", "")).strip()
    territory = str(plan.get("territory", "")).strip()
    wanted_types = {str(item) for item in plan.get("doc_types", [])}
    max_docs = int(plan.get("max_docs", 10))
    max_scenes = int(plan.get("max_scenes", 6))

    docs: list[dict[str, Any]] = []
    for item in _page_index_items():
        doc_type = str(item.get("doc_type", ""))
        if doc_type and wanted_types and doc_type not in wanted_types:
            continue
        if movie and not _movie_match(item, movie):
            if doc_type == "censorship_guidelines_countries" and territory and _territory_match(item, territory):
                pass
            else:
                continue
        if doc_type == "censorship_guidelines_countries" and territory and not _territory_match(item, territory):
            continue
        docs.append(item)
        if len(docs) >= max_docs:
            break

    scenes: list[dict[str, Any]] = []
    for item in _scene_index_items():
        if movie and not _movie_match(item, movie):
            continue
        scenes.append(item)
        if len(scenes) >= max_scenes:
            break

    return {
        "documents": docs,
        "scenes": scenes,
    }


def SufficiencyChecker(fetched: dict[str, list[dict[str, Any]]], min_items: int = 4) -> dict[str, Any]:
    """Check if retrieval result is sufficient for downstream reasoning."""
    documents = fetched.get("documents", [])
    scenes = fetched.get("scenes", [])
    total = len(documents) + len(scenes)
    score = min(1.0, total / float(max(1, min_items * 2)))
    status = "PASS" if total >= min_items else "EXPAND"
    return {"status": status, "score": round(score, 3), "total_items": total}


def _citation_from_record(item: dict[str, Any]) -> Citation:
    excerpt = str(item.get("text", "")).strip()
    return {
        "source_path": str(item.get("source_path", "")),
        "doc_id": str(item.get("doc_id", "")),
        "page": item.get("page") if isinstance(item.get("page"), int) else item.get("start_page"),
        "excerpt": excerpt[:220],
    }


def source_citation_tool(items: list[dict[str, Any]], limit: int = 12) -> list[Citation]:
    citations: list[Citation] = []
    for item in items:
        citations.append(_citation_from_record(item))
        if len(citations) >= limit:
            break
    return citations


def _get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(settings.database_url, future=True)
    return _engine


async def _query_rows(sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        engine = _get_engine()
        async with engine.connect() as conn:
            result = await conn.execute(text(sql), params)
            return [dict(row._mapping) for row in result]
    except Exception as exc:
        logger.warning("db_tool_query_failed sql_tag={} error={}", sql.split()[0], str(exc))
        return []


async def get_box_office_by_genre_territory(movie: str, territory: str) -> dict[str, Any]:
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
    rows = await _query_rows(sql, {"movie": movie, "territory": territory})
    if not rows:
        return {"avg_gross_usd": 0.0, "total_gross_usd": 0.0, "samples": 0}
    row = rows[0]
    return {
        "avg_gross_usd": float(row.get("avg_gross_usd") or 0.0),
        "total_gross_usd": float(row.get("total_gross_usd") or 0.0),
        "samples": int(row.get("samples") or 0),
    }


async def get_actor_qscore(movie: str) -> dict[str, Any]:
    sql = """
    SELECT
      COALESCE(AVG(a.q_score), 0) AS avg_qscore,
      COALESCE(SUM(a.social_reach), 0) AS total_social_reach
    FROM films f
    JOIN film_cast fc ON fc.film_id = f.id
    JOIN actors a ON a.id = fc.actor_id
    WHERE lower(f.title) = lower(:movie)
    """
    rows = await _query_rows(sql, {"movie": movie})
    if not rows:
        return {"avg_qscore": 0.0, "total_social_reach": 0}
    row = rows[0]
    return {
        "avg_qscore": float(row.get("avg_qscore") or 0.0),
        "total_social_reach": int(row.get("total_social_reach") or 0),
    }


async def get_theatrical_window_trends(territory: str) -> list[dict[str, Any]]:
    sql = """
    SELECT tw.window_type, tw.days
    FROM theatrical_windows tw
    JOIN territories t ON t.id = tw.territory_id
    WHERE lower(t.name) = lower(:territory)
    ORDER BY tw.days ASC
    """
    rows = await _query_rows(sql, {"territory": territory})
    return [
        {
            "window_type": str(item.get("window_type") or ""),
            "days": int(item.get("days") or 0),
        }
        for item in rows
    ]


async def get_exchange_rates(territory: str) -> dict[str, Any]:
    sql = """
    SELECT cr.currency_code, cr.rate_to_usd, cr.rate_date
    FROM currency_rates cr
    JOIN territories t ON t.currency_code = cr.currency_code
    WHERE lower(t.name) = lower(:territory)
    ORDER BY cr.rate_date DESC
    LIMIT 1
    """
    rows = await _query_rows(sql, {"territory": territory})
    if not rows:
        return {"currency_code": "USD", "rate_to_usd": 1.0}
    row = rows[0]
    return {
        "currency_code": str(row.get("currency_code") or "USD"),
        "rate_to_usd": float(row.get("rate_to_usd") or 1.0),
    }


async def get_vod_price_benchmarks(territory: str) -> dict[str, Any]:
    sql = """
    SELECT
      COALESCE(AVG(v.price_min_usd), 0) AS avg_price_min_usd,
      COALESCE(AVG(v.price_max_usd), 0) AS avg_price_max_usd
    FROM vod_price_benchmarks v
    JOIN territories t ON t.id = v.territory_id
    WHERE lower(t.name) = lower(:territory)
    """
    rows = await _query_rows(sql, {"territory": territory})
    if not rows:
        return {"avg_price_min_usd": 0.0, "avg_price_max_usd": 0.0}
    row = rows[0]
    return {
        "avg_price_min_usd": float(row.get("avg_price_min_usd") or 0.0),
        "avg_price_max_usd": float(row.get("avg_price_max_usd") or 0.0),
    }


async def get_comparable_films(movie: str, territory: str, limit: int = 5) -> list[dict[str, Any]]:
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
    rows = await _query_rows(sql, {"movie": movie, "territory": territory, "limit": limit})
    return [
        {
            "title": str(item.get("title") or ""),
            "territory_gross_usd": float(item.get("territory_gross_usd") or 0.0),
        }
        for item in rows
    ]


def mg_calculator_tool(
    avg_box_office_usd: float,
    avg_qscore: float,
    comparable_avg_gross_usd: float,
    risk_penalty: float,
) -> float:
    base = comparable_avg_gross_usd * 0.12 if comparable_avg_gross_usd > 0 else avg_box_office_usd * 0.08
    if base <= 0:
        base = 1_200_000.0
    talent_multiplier = 1.0 + min(0.25, max(0.0, avg_qscore / 400.0))
    sanitized_penalty = min(0.6, max(0.0, risk_penalty))
    mg = base * talent_multiplier * (1.0 - sanitized_penalty)
    return round(max(250_000.0, mg), 2)


def exchange_rate_tool(amount_usd: float, rate_to_usd: float) -> float:
    if rate_to_usd <= 0:
        return round(amount_usd, 2)
    return round(amount_usd / rate_to_usd, 2)


def financial_sanity_check(
    mg_estimate_usd: float,
    theatrical_projection_usd: float,
    vod_projection_usd: float,
) -> bool:
    projected_total = theatrical_projection_usd + vod_projection_usd
    if projected_total <= 0:
        return False
    return mg_estimate_usd <= projected_total * 0.7


def hallucination_check(citations: list[Citation], min_citations: int = 3) -> bool:
    present = [item for item in citations if item.get("source_path")]
    return len(present) >= min_citations


def confidence_threshold_check(confidence: float, threshold: float = 0.55) -> bool:
    return confidence >= threshold


def format_scorecard(
    territory: str,
    theatrical_projection_usd: float,
    vod_projection_usd: float,
    acquisition_price_usd: float,
    release_mode: str,
    release_window_days: int,
    risk_flags: list[dict[str, Any]],
    citations: list[Citation],
    confidence: float,
    warnings: list[str],
) -> Scorecard:
    return {
        "projected_revenue_by_territory": {
            territory: round(theatrical_projection_usd + vod_projection_usd, 2)
        },
        "risk_flags": risk_flags,
        "recommended_acquisition_price": round(acquisition_price_usd, 2),
        "release_timeline": {
            "release_mode": release_mode,
            "theatrical_window_days": release_window_days,
        },
        "citations": citations,
        "confidence": round(confidence, 3),
        "warnings": warnings,
    }


def combine_validation_warnings(report: ValidationReport) -> list[str]:
    warnings = list(report.get("warnings", []))
    if not report.get("financial_sanity_pass", False):
        warnings.append("Financial sanity check failed.")
    if not report.get("hallucination_pass", False):
        warnings.append("Insufficient citations for one or more claims.")
    if not report.get("confidence_threshold_pass", False):
        warnings.append("Overall confidence is below threshold.")
    deduped: list[str] = []
    seen: set[str] = set()
    for warning in warnings:
        if warning in seen:
            continue
        seen.add(warning)
        deduped.append(warning)
    return deduped
