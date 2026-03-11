from __future__ import annotations

import time
from contextvars import ContextVar
from functools import lru_cache
from typing import Any
from uuid import uuid4

import httpx
from loguru import logger

from app.core.config import get_settings
from .types import Citation, Scorecard, ValidationReport

settings = get_settings()
_tool_diagnostics_var: ContextVar[list[dict[str, Any]]] = ContextVar(
    "tool_diagnostics", default=[]
)


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _internal_api_key() -> str:
    return settings.internal_api_key or settings.adk_api_key


def _backend_headers() -> dict[str, str]:
    return {
        "X-Internal-API-Key": _internal_api_key(),
        "X-Request-ID": str(uuid4()),
    }


def reset_tool_diagnostics() -> None:
    _tool_diagnostics_var.set([])


def get_tool_diagnostics() -> list[dict[str, Any]]:
    diagnostics = _tool_diagnostics_var.get()
    return [dict(item) for item in diagnostics]


def _record_tool_failure(
    *,
    source: str,
    error_type: str,
    endpoint: str,
    status_code: int | None = None,
    message: str | None = None,
) -> None:
    diagnostics = list(_tool_diagnostics_var.get())
    diagnostics.append(
        {
            "source": source,
            "error_type": error_type,
            "endpoint": endpoint,
            "status_code": status_code,
            "message": message or "",
        }
    )
    _tool_diagnostics_var.set(diagnostics)
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


def _request_json_sync(
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
            time.sleep(delay)
        try:
            with httpx.Client(base_url=base_url, timeout=timeout) as client:
                response = client.request(
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
                    source="docs",
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
                    source="docs",
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
                source="docs",
                endpoint=path,
                error_type=_classify_status_error(status_code),
                status_code=status_code,
                message=str(last_exception),
            )
        else:
            _record_tool_failure(
                source="docs",
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


@lru_cache
def IndexRegistry() -> dict[str, Any]:
    payload = _request_json_sync("GET", "/internal/v1/meta/registry")
    if isinstance(payload, dict):
        return payload
    return {
        "page_index_manifest": {"documents": []},
        "scene_manifest": {"scripts": []},
        "known_movies": [],
        "known_territories": [],
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


def TargetedFetcher(plan: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Fetch targeted page/scene chunks via backend internal docs endpoint."""
    payload = _request_json_sync(
        "POST",
        "/internal/v1/docs/search",
        payload={
            "movie": str(plan.get("movie", "")),
            "territory": str(plan.get("territory", "")),
            "intent": str(plan.get("intent", "full_scorecard")),
            "doc_types": [str(item) for item in plan.get("doc_types", [])],
            "max_docs": int(plan.get("max_docs", 10)),
            "max_scenes": int(plan.get("max_scenes", 6)),
        },
    )
    if not isinstance(payload, dict):
        return {"documents": [], "scenes": []}

    documents = payload.get("documents")
    scenes = payload.get("scenes")
    if not isinstance(documents, list) or not isinstance(scenes, list):
        return {"documents": [], "scenes": []}

    return {
        "documents": [item for item in documents if isinstance(item, dict)],
        "scenes": [item for item in scenes if isinstance(item, dict)],
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
    marketing_spend_usd: float,
    platform_priority: list[str],
    roi_scenarios: dict[str, float],
    risk_flags: list[dict[str, Any]],
    citations: list[Citation],
    confidence: float,
    warnings: list[str],
    evidence_basis: str,
    degraded_mode: dict[str, Any],
    response_type: str,
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
        "marketing_spend_usd": round(marketing_spend_usd, 2),
        "platform_priority": platform_priority,
        "roi_scenarios": roi_scenarios,
        "citations": citations,
        "confidence": round(confidence, 3),
        "warnings": warnings,
        "evidence_basis": evidence_basis,
        "degraded_mode": degraded_mode,
        "response_type": response_type,
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
