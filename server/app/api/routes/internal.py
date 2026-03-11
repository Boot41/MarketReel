from __future__ import annotations

import secrets
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import db_session
from app.core.config import Settings, get_settings
from app.services import internal_market_data as market_data

router = APIRouter(prefix="/internal/v1", tags=["internal"])


class BoxOfficeResponse(BaseModel):
    avg_gross_usd: float
    total_gross_usd: float
    samples: int


class ActorSignalsResponse(BaseModel):
    avg_qscore: float
    total_social_reach: int


class WindowItem(BaseModel):
    window_type: str
    days: int


class ExchangeRateResponse(BaseModel):
    currency_code: str
    rate_to_usd: float
    rate_date: str | None = None


class VodBenchmarksResponse(BaseModel):
    avg_price_min_usd: float
    avg_price_max_usd: float


class ComparableFilm(BaseModel):
    title: str
    territory_gross_usd: float


class DocsSearchRequest(BaseModel):
    movie: str = Field(min_length=1, max_length=250)
    territory: str = Field(min_length=1, max_length=120)
    intent: str = Field(min_length=1, max_length=80)
    max_docs: int = Field(default=10, ge=1, le=30)
    max_scenes: int = Field(default=6, ge=1, le=20)


class Citation(BaseModel):
    source_path: str
    doc_id: str
    page: int | None
    excerpt: str


class Sufficiency(BaseModel):
    status: str
    score: float
    total_items: int


class DocsSearchResponse(BaseModel):
    movie: str
    territory: str
    intent: str
    documents: list[dict[str, Any]]
    scenes: list[dict[str, Any]]
    citations: list[Citation]
    sufficiency: Sufficiency


class EvidenceBundleRequest(BaseModel):
    movie: str = Field(min_length=1, max_length=250)
    territory: str = Field(min_length=1, max_length=120)
    intent: str = Field(min_length=1, max_length=80)
    needs_db: bool = True
    needs_docs: bool = True


class EvidenceBundleResponse(BaseModel):
    movie: str
    territory: str
    intent: str
    document_evidence: dict[str, list[dict[str, Any]]]
    db_evidence: dict[str, Any]
    citations: list[Citation]
    data_sufficiency_score: float


class MetaRegistryResponse(BaseModel):
    known_movies: list[str]
    known_territories: list[str]
    page_index_manifest: dict[str, Any]
    scene_manifest: dict[str, Any]


def _request_id(request: Request) -> str:
    incoming = request.headers.get("X-Request-ID", "").strip()
    return incoming or str(uuid4())


def verify_internal_api_key(
    request: Request,
    settings: Settings = Depends(get_settings),
    x_internal_api_key: str | None = Header(default=None),
) -> None:
    current_key = settings.internal_api_key
    next_key = settings.internal_api_next_key
    provided_key = x_internal_api_key or ""
    request_id = _request_id(request)

    if not current_key and not next_key:
        logger.error("internal_auth_misconfigured request_id={}", request_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Internal API key not configured",
        )

    valid_current = bool(current_key) and secrets.compare_digest(provided_key, current_key)
    valid_next = bool(next_key) and secrets.compare_digest(provided_key, next_key)

    if not (valid_current or valid_next):
        logger.warning("internal_auth_failed request_id={}", request_id)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


@router.get(
    "/market/box-office",
    response_model=BoxOfficeResponse,
    dependencies=[Depends(verify_internal_api_key)],
)
async def market_box_office(
    movie: str,
    territory: str,
    db: AsyncSession = Depends(db_session),
) -> BoxOfficeResponse:
    data = await market_data.get_box_office_by_genre_territory(db, movie=movie, territory=territory)
    return BoxOfficeResponse.model_validate(data)


@router.get(
    "/market/actor-signals",
    response_model=ActorSignalsResponse,
    dependencies=[Depends(verify_internal_api_key)],
)
async def market_actor_signals(
    movie: str,
    db: AsyncSession = Depends(db_session),
) -> ActorSignalsResponse:
    data = await market_data.get_actor_qscore(db, movie=movie)
    return ActorSignalsResponse.model_validate(data)


@router.get(
    "/market/theatrical-windows",
    response_model=list[WindowItem],
    dependencies=[Depends(verify_internal_api_key)],
)
async def market_theatrical_windows(
    territory: str,
    db: AsyncSession = Depends(db_session),
) -> list[WindowItem]:
    rows = await market_data.get_theatrical_window_trends(db, territory=territory)
    return [WindowItem.model_validate(item) for item in rows]


@router.get(
    "/market/exchange-rate",
    response_model=ExchangeRateResponse,
    dependencies=[Depends(verify_internal_api_key)],
)
async def market_exchange_rate(
    territory: str,
    db: AsyncSession = Depends(db_session),
) -> ExchangeRateResponse:
    row = await market_data.get_exchange_rates(db, territory=territory)
    return ExchangeRateResponse.model_validate(row)


@router.get(
    "/market/vod-benchmarks",
    response_model=VodBenchmarksResponse,
    dependencies=[Depends(verify_internal_api_key)],
)
async def market_vod_benchmarks(
    territory: str,
    db: AsyncSession = Depends(db_session),
) -> VodBenchmarksResponse:
    row = await market_data.get_vod_price_benchmarks(db, territory=territory)
    return VodBenchmarksResponse.model_validate(row)


@router.get(
    "/market/comparables",
    response_model=list[ComparableFilm],
    dependencies=[Depends(verify_internal_api_key)],
)
async def market_comparables(
    movie: str,
    territory: str,
    limit: int = 5,
    db: AsyncSession = Depends(db_session),
) -> list[ComparableFilm]:
    rows = await market_data.get_comparable_films(db, movie=movie, territory=territory, limit=limit)
    return [ComparableFilm.model_validate(item) for item in rows]


@router.post(
    "/docs/search",
    response_model=DocsSearchResponse,
    dependencies=[Depends(verify_internal_api_key)],
)
async def docs_search(body: DocsSearchRequest) -> DocsSearchResponse:
    result = market_data.docs_search(
        movie=body.movie,
        territory=body.territory,
        intent=body.intent,
        max_docs=body.max_docs,
        max_scenes=body.max_scenes,
    )
    return DocsSearchResponse.model_validate(result)


@router.post(
    "/evidence/bundle",
    response_model=EvidenceBundleResponse,
    dependencies=[Depends(verify_internal_api_key)],
)
async def evidence_bundle(
    body: EvidenceBundleRequest,
    db: AsyncSession = Depends(db_session),
) -> EvidenceBundleResponse:
    result = await market_data.evidence_bundle(
        db,
        movie=body.movie,
        territory=body.territory,
        intent=body.intent,
        needs_db=body.needs_db,
        needs_docs=body.needs_docs,
    )
    return EvidenceBundleResponse.model_validate(result)


@router.get(
    "/meta/registry",
    response_model=MetaRegistryResponse,
    dependencies=[Depends(verify_internal_api_key)],
)
async def meta_registry() -> MetaRegistryResponse:
    payload = market_data.index_registry()
    return MetaRegistryResponse.model_validate(payload)
