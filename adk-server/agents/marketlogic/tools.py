from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from loguru import logger

from app.core.config import get_settings

settings = get_settings()

_PAGE_INDEX_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "page_index"
_SCRIPTS_INDEX_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "scripts_indexed"


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _internal_api_key() -> str:
    return settings.internal_api_key or settings.adk_api_key


def _backend_headers() -> dict[str, str]:
    return {
        "X-Internal-API-Key": _internal_api_key(),
        "X-Request-ID": str(uuid4()),
    }


def _record_tool_failure(
    *,
    source: str,
    error_type: str,
    endpoint: str,
    status_code: int | None = None,
    message: str | None = None,
) -> None:
    logger.warning(
        "tool_call_failed source={} endpoint={} error_type={} status_code={} message={}",
        source,
        endpoint,
        error_type,
        status_code or 0,
        message or "",
    )


def _classify_status_error(status_code: int) -> str:
    if status_code in {401, 403}:
        return "auth"
    if status_code in {408, 429, 500, 502, 503, 504}:
        return "backend_unavailable"
    return "request_error"


def _classify_exception(exc: Exception) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.ConnectError):
        return "network"
    if isinstance(exc, httpx.NetworkError):
        return "network"
    return "unknown"


def _retry_delays() -> list[float]:
    retries = max(0, int(settings.internal_api_retries))
    return [0.0] + [0.25] * retries


def _should_retry(exc: Exception | None, status_code: int | None) -> bool:
    if exc is not None:
        return True
    return bool(status_code and status_code in {408, 429, 500, 502, 503, 504})


async def _request_json(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> Any:
    base_url = settings.backend_base_url.rstrip("/")
    timeout = float(settings.internal_api_timeout_sec)

    last_exception: Exception | None = None
    for delay in _retry_delays():
        if delay > 0:
            await _sleep(delay)
        try:
            async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
                response = await client.request(
                    method,
                    path,
                    params=params,
                    json=payload,
                    headers=_backend_headers(),
                )
            if response.is_success:
                return response.json()
            if not _should_retry(None, response.status_code):
                _record_tool_failure(
                    source="db",
                    endpoint=path,
                    error_type=_classify_status_error(response.status_code),
                    status_code=response.status_code,
                    message=f"http_{response.status_code}",
                )
                return {}
            last_exception = RuntimeError(f"backend_status_{response.status_code}")
        except Exception as exc:
            if not _should_retry(exc, None):
                _record_tool_failure(
                    source="db",
                    endpoint=path,
                    error_type=_classify_exception(exc),
                    message=str(exc),
                )
                return {}
            last_exception = exc
    if last_exception is not None:
        if isinstance(last_exception, RuntimeError) and "backend_status_" in str(last_exception):
            status_code = int(str(last_exception).split("_")[-1])
            _record_tool_failure(
                source="db",
                endpoint=path,
                error_type=_classify_status_error(status_code),
                status_code=status_code,
                message=str(last_exception),
            )
        else:
            _record_tool_failure(
                source="db",
                endpoint=path,
                error_type=_classify_exception(last_exception),
                message=str(last_exception),
            )
        return {}
    return {}


async def _sleep(seconds: float) -> None:
    if seconds <= 0:
        return
    import asyncio

    await asyncio.sleep(seconds)


def index_registry(movie: str, territory: str) -> dict[str, Any]:
    """Return the document catalog for the given movie and territory.

    Reads local index manifests to show what evidence is available before
    building a retrieval plan. Always call this first.

    Args:
        movie: Film name or identifier (e.g. "deadpool", "interstellar").
        territory: Target distribution territory (e.g. "india", "saudi_arabia").
    """
    movie_slug = _normalize(movie).replace(" ", "_")
    territory_slug = _normalize(territory).replace(" ", "_")
    available_docs: list[dict[str, Any]] = []
    known_movies: list[str] = []
    known_territories: list[str] = []

    try:
        manifest = json.loads((_PAGE_INDEX_PATH / "manifest.json").read_text(encoding="utf-8"))
        for doc in manifest.get("documents", []):
            doc_id = _normalize(str(doc.get("doc_id", ""))).replace(" ", "_")
            doc_type = str(doc.get("doc_type", ""))
            if doc_type == "censorship_guidelines_countries":
                slug = doc_id.replace("censorship_guidelines_", "")
                if slug not in known_territories:
                    known_territories.append(slug)
                if territory_slug and territory_slug in doc_id:
                    available_docs.append(doc)
            else:
                doc_movie = _normalize(str(doc.get("movie", ""))).replace(" ", "_")
                base = doc_id
                for suffix in ("_censorship_guidelines", "_censorship", "_cultural_sensitivity",
                               "_synopsis", "_synopses", "_reviews", "_marketing"):
                    base = base.replace(suffix, "")
                if base and base not in known_movies:
                    known_movies.append(base)
                if movie_slug and (movie_slug in doc_id or movie_slug in doc_movie):
                    available_docs.append(doc)
    except Exception as exc:
        logger.warning("index_registry_page_index_error error={}", exc)

    try:
        scene_manifest = json.loads(
            (_SCRIPTS_INDEX_PATH / "scene_manifest.json").read_text(encoding="utf-8")
        )
        for entry in scene_manifest.get("scripts", []):
            em = _normalize(str(entry.get("movie", ""))).replace(" ", "_")
            if em and em not in known_movies:
                known_movies.append(em)
            doc_id = _normalize(str(entry.get("doc_id", ""))).replace(" ", "_")
            if movie_slug and (movie_slug in em or movie_slug in doc_id):
                available_docs.append({
                    "doc_id": entry.get("doc_id"),
                    "doc_type": "script_scenes",
                    "movie": entry.get("movie"),
                    "scene_count": entry.get("scenes", 0),
                    "source_path": entry.get("source_path", ""),
                })
    except Exception as exc:
        logger.warning("index_registry_scene_manifest_error error={}", exc)

    return {
        "available_docs": available_docs,
        "movie_slug": movie_slug,
        "territory_slug": territory_slug,
        "known_movies": sorted(known_movies),
        "known_territories": sorted(known_territories),
    }


def index_navigator(movie: str, territory: str, retrieval_intent: str) -> dict[str, Any]:
    """Build a targeted retrieval plan for the given movie, territory, and intent.

    Returns a plan with recommended doc_types, max_docs, and max_scenes. Pass
    these fields directly to targeted_fetcher.

    Args:
        movie: Film name or identifier.
        territory: Target distribution territory.
        retrieval_intent: Plain-language retrieval goal — e.g. "censorship risk
            for saudi arabia", "valuation evidence", "full_scorecard",
            "reviews and sentiment".
    """
    intent_key = _normalize(retrieval_intent).replace(" ", "_")
    doc_types: list[str] = ["synopses", "reviews", "marketing"]
    if any(k in intent_key for k in ("risk", "censorship", "full_scorecard", "cultural")):
        doc_types.extend(["cultural_sensitivity", "censorship", "censorship_guidelines_countries"])
    if any(k in intent_key for k in ("valuation", "full_scorecard", "script", "scene")):
        doc_types.append("script_scenes")

    max_docs = 12 if "full_scorecard" in intent_key else 8
    max_scenes = 8 if any(
        k in intent_key for k in ("risk", "censorship", "valuation", "full_scorecard")
    ) else 4

    return {
        "movie": movie,
        "territory": territory,
        "retrieval_intent": retrieval_intent,
        "doc_types": sorted(set(doc_types)),
        "max_docs": max_docs,
        "max_scenes": max_scenes,
    }


def targeted_fetcher(
    movie: str,
    territory: str,
    doc_types: list[str],
    max_docs: int = 10,
    max_scenes: int = 6,
) -> dict[str, Any]:
    """Fetch documents and scenes from the local corpus for a movie and territory.

    Reads docs/page_index/pages.jsonl for document types (synopses, reviews,
    marketing, censorship, cultural_sensitivity) and
    docs/scripts_indexed/scenes.jsonl for script scenes. Returns matched items
    with source references attached.

    Args:
        movie: Film name to retrieve documents for.
        territory: Territory used to filter censorship_guidelines_countries docs.
        doc_types: Document types to fetch. Valid values: "synopses", "reviews",
            "marketing", "censorship", "censorship_guidelines_countries",
            "cultural_sensitivity", "script_scenes".
        max_docs: Maximum page-level documents to return (default 10).
        max_scenes: Maximum script scenes to return (default 6).
    """
    movie_slug = _normalize(movie).replace(" ", "_")
    territory_slug = _normalize(territory).replace(" ", "_")
    type_set = {_normalize(t).replace(" ", "_") for t in (doc_types or [])}

    documents: list[dict[str, Any]] = []
    scenes: list[dict[str, Any]] = []

    page_types = type_set - {"script_scenes"}
    if page_types:
        try:
            with (_PAGE_INDEX_PATH / "pages.jsonl").open(encoding="utf-8") as fh:
                for raw in fh:
                    if len(documents) >= max_docs:
                        break
                    try:
                        item = json.loads(raw)
                    except Exception:
                        continue
                    doc_type = _normalize(str(item.get("doc_type", ""))).replace(" ", "_")
                    if doc_type not in page_types:
                        continue
                    doc_id = _normalize(str(item.get("doc_id", ""))).replace(" ", "_")
                    if doc_type == "censorship_guidelines_countries":
                        if territory_slug and territory_slug not in doc_id:
                            continue
                    else:
                        doc_movie = _normalize(str(item.get("movie", ""))).replace(" ", "_")
                        if movie_slug and movie_slug not in doc_id and movie_slug not in doc_movie:
                            continue
                    documents.append({
                        "doc_id": item.get("doc_id"),
                        "doc_type": item.get("doc_type"),
                        "page": item.get("page"),
                        "content": str(item.get("text", ""))[:2000],
                        "movie": item.get("movie"),
                        "source_reference": str(item.get("source_path", "")),
                    })
        except Exception as exc:
            logger.warning("targeted_fetcher_page_index_error error={}", exc)

    if "script_scenes" in type_set:
        try:
            with (_SCRIPTS_INDEX_PATH / "scenes.jsonl").open(encoding="utf-8") as fh:
                for raw in fh:
                    if len(scenes) >= max_scenes:
                        break
                    try:
                        item = json.loads(raw)
                    except Exception:
                        continue
                    item_movie = _normalize(str(item.get("movie", ""))).replace(" ", "_")
                    doc_id = _normalize(str(item.get("doc_id", ""))).replace(" ", "_")
                    if movie_slug and movie_slug not in item_movie and movie_slug not in doc_id:
                        continue
                    scenes.append({
                        "doc_id": item.get("doc_id"),
                        "scene_title": item.get("scene_title"),
                        "start_page": item.get("start_page"),
                        "end_page": item.get("end_page"),
                        "content": str(item.get("text", ""))[:2000],
                        "movie": item.get("movie"),
                        "source_reference": str(item.get("source_path", "")),
                    })
        except Exception as exc:
            logger.warning("targeted_fetcher_scenes_error error={}", exc)

    return {
        "documents": documents,
        "scenes": scenes,
        "total_documents": len(documents),
        "total_scenes": len(scenes),
    }


def sufficiency_checker(
    total_documents: int,
    total_scenes: int,
    retrieval_intent: str,
) -> dict[str, Any]:
    """Check if retrieved evidence meets the threshold for the retrieval intent.

    Call this after targeted_fetcher using the total_documents and total_scenes
    counts from that result. Returns PASS or EXPAND with guidance on next steps.

    Args:
        total_documents: Number of documents returned by targeted_fetcher.
        total_scenes: Number of script scenes returned by targeted_fetcher.
        retrieval_intent: The same intent string passed to index_navigator.
    """
    intent_key = _normalize(retrieval_intent)
    total = total_documents + total_scenes
    if "risk" in intent_key or "censorship" in intent_key:
        min_items = 5
    elif "full_scorecard" in intent_key:
        min_items = 8
    else:
        min_items = 3

    score = min(1.0, total / float(max(1, min_items * 2)))
    status = "PASS" if total >= min_items else "EXPAND"
    guidance = (
        f"Only {total} items retrieved; {min_items} required. "
        "Call targeted_fetcher again with max_docs increased by 6 and max_scenes by 4."
        if status == "EXPAND"
        else ""
    )
    return {
        "status": status,
        "score": round(score, 3),
        "total_items": total,
        "min_required": min_items,
        "guidance": guidance,
    }


async def get_box_office_by_genre_territory(movie: str, territory: str) -> dict[str, Any]:
    payload = await _request_json(
        "GET",
        "/internal/v1/market/box-office",
        params={"movie": movie, "territory": territory},
    )
    if not isinstance(payload, dict):
        return {"avg_gross_usd": 0.0, "total_gross_usd": 0.0, "samples": 0}
    return {
        "avg_gross_usd": float(payload.get("avg_gross_usd") or 0.0),
        "total_gross_usd": float(payload.get("total_gross_usd") or 0.0),
        "samples": int(payload.get("samples") or 0),
    }


async def get_actor_qscore(movie: str) -> dict[str, Any]:
    payload = await _request_json("GET", "/internal/v1/market/actor-signals", params={"movie": movie})
    if not isinstance(payload, dict):
        return {"avg_qscore": 0.0, "total_social_reach": 0}
    return {
        "avg_qscore": float(payload.get("avg_qscore") or 0.0),
        "total_social_reach": int(payload.get("total_social_reach") or 0),
    }


async def get_theatrical_window_trends(territory: str) -> list[dict[str, Any]]:
    payload = await _request_json(
        "GET",
        "/internal/v1/market/theatrical-windows",
        params={"territory": territory},
    )
    if not isinstance(payload, list):
        return []
    return [
        {
            "window_type": str(item.get("window_type") or ""),
            "days": int(item.get("days") or 0),
        }
        for item in payload
        if isinstance(item, dict)
    ]


async def get_exchange_rates(territory: str) -> dict[str, Any]:
    payload = await _request_json(
        "GET",
        "/internal/v1/market/exchange-rate",
        params={"territory": territory},
    )
    if not isinstance(payload, dict):
        return {"currency_code": "USD", "rate_to_usd": 1.0}
    return {
        "currency_code": str(payload.get("currency_code") or "USD"),
        "rate_to_usd": float(payload.get("rate_to_usd") or 1.0),
        "rate_date": payload.get("rate_date"),
    }


async def get_vod_price_benchmarks(territory: str) -> dict[str, Any]:
    payload = await _request_json(
        "GET",
        "/internal/v1/market/vod-benchmarks",
        params={"territory": territory},
    )
    if not isinstance(payload, dict):
        return {"avg_price_min_usd": 0.0, "avg_price_max_usd": 0.0}
    return {
        "avg_price_min_usd": float(payload.get("avg_price_min_usd") or 0.0),
        "avg_price_max_usd": float(payload.get("avg_price_max_usd") or 0.0),
    }


async def get_comparable_films(movie: str, territory: str, limit: int = 5) -> list[dict[str, Any]]:
    payload = await _request_json(
        "GET",
        "/internal/v1/market/comparables",
        params={"movie": movie, "territory": territory, "limit": limit},
    )
    if not isinstance(payload, list):
        return []
    return [
        {
            "title": str(item.get("title") or ""),
            "territory_gross_usd": float(item.get("territory_gross_usd") or 0.0),
        }
        for item in payload
        if isinstance(item, dict)
    ]


def mg_calculator_tool(
    avg_box_office_usd: float,
    avg_qscore: float,
    comparable_avg_gross_usd: float,
    risk_penalty: float,
    allow_baseline_fallback: bool = True,
) -> float:
    base = comparable_avg_gross_usd * 0.12 if comparable_avg_gross_usd > 0 else avg_box_office_usd * 0.08
    if base <= 0:
        if allow_baseline_fallback:
            base = 1_200_000.0
        else:
            base = 0.0
    talent_multiplier = 1.0 + min(0.25, max(0.0, avg_qscore / 400.0))
    sanitized_penalty = min(0.6, max(0.0, risk_penalty))
    mg = base * talent_multiplier * (1.0 - sanitized_penalty)
    if mg <= 0:
        return 0.0
    return round(max(250_000.0, mg), 2)


def exchange_rate_tool(amount_usd: float, rate_to_usd: float) -> float:
    if rate_to_usd <= 0:
        return round(amount_usd, 2)
    return round(amount_usd / rate_to_usd, 2)
