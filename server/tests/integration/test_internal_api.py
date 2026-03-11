from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import (
    Actor,
    BoxOffice,
    CurrencyRate,
    Film,
    FilmCast,
    FilmGenre,
    Genre,
    Territory,
    TheatricalWindow,
    VodPriceBenchmark,
)


def _internal_headers() -> dict[str, str]:
    settings = get_settings()
    settings.internal_api_key = "internal-test-key"
    settings.internal_api_next_key = ""
    return {"X-Internal-API-Key": "internal-test-key"}


@pytest.fixture
async def seed_market_data(db: AsyncSession) -> None:
    genre = Genre(name="Sci-Fi")
    territory = Territory(name="India", region_code="APAC", currency_code="INR")
    film = Film(
        title="Interstellar",
        release_year=2014,
        runtime_min=169,
        budget_usd=165_000_000,
        logline="Explorers travel through a wormhole.",
        synopsis_doc_path="synopses/interstellar.md",
        script_doc_path="scripts/interstellar.md",
    )
    comp = Film(
        title="Arrival",
        release_year=2016,
        runtime_min=116,
        budget_usd=47_000_000,
        logline="Linguist faces first contact.",
        synopsis_doc_path="synopses/arrival.md",
        script_doc_path="scripts/arrival.md",
    )
    actor = Actor(name="Matthew McConaughey", q_score=68.5, social_reach=21_000_000)

    db.add_all([genre, territory, film, comp, actor])
    await db.flush()

    db.add(FilmGenre(film_id=film.id, genre_id=genre.id))
    db.add(FilmGenre(film_id=comp.id, genre_id=genre.id))
    db.add(
        FilmCast(
            film_id=film.id,
            actor_id=actor.id,
            billing_order=1,
            role_name="Cooper",
        )
    )
    db.add(
        BoxOffice(
            film_id=film.id,
            territory_id=territory.id,
            genre_id=genre.id,
            gross_local=1_500_000_000,
            gross_usd=18_000_000,
            admissions=2_200_000,
            release_date=date(2014, 11, 7),
        )
    )
    db.add(
        BoxOffice(
            film_id=comp.id,
            territory_id=territory.id,
            genre_id=genre.id,
            gross_local=850_000_000,
            gross_usd=12_000_000,
            admissions=1_300_000,
            release_date=date(2016, 12, 2),
        )
    )
    db.add(TheatricalWindow(territory_id=territory.id, window_type="standard", days=45))
    db.add(CurrencyRate(currency_code="INR", rate_to_usd=83.2, rate_date=date(2026, 3, 1)))
    db.add(
        VodPriceBenchmark(
            territory_id=territory.id,
            license_type="SVOD",
            window_months=6,
            price_min_usd=600_000,
            price_max_usd=1_200_000,
        )
    )
    await db.commit()


@pytest.mark.asyncio
async def test_internal_route_requires_api_key(client) -> None:  # type: ignore[no-untyped-def]
    get_settings().internal_api_key = "internal-test-key"
    response = await client.get("/internal/v1/meta/registry")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_internal_route_rejects_wrong_api_key(client) -> None:  # type: ignore[no-untyped-def]
    get_settings().internal_api_key = "internal-test-key"
    response = await client.get(
        "/internal/v1/meta/registry",
        headers={"X-Internal-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_internal_market_and_bundle_endpoints(client, seed_market_data) -> None:  # type: ignore[no-untyped-def]
    headers = _internal_headers()

    box = await client.get(
        "/internal/v1/market/box-office",
        params={"movie": "Interstellar", "territory": "India"},
        headers=headers,
    )
    assert box.status_code == 200
    assert box.json()["samples"] >= 1

    actor = await client.get(
        "/internal/v1/market/actor-signals",
        params={"movie": "Interstellar"},
        headers=headers,
    )
    assert actor.status_code == 200
    assert actor.json()["total_social_reach"] > 0

    comparables = await client.get(
        "/internal/v1/market/comparables",
        params={"movie": "Interstellar", "territory": "India"},
        headers=headers,
    )
    assert comparables.status_code == 200
    assert len(comparables.json()) >= 1

    bundle = await client.post(
        "/internal/v1/evidence/bundle",
        json={
            "movie": "Interstellar",
            "territory": "India",
            "intent": "full_scorecard",
            "needs_db": True,
            "needs_docs": True,
        },
        headers=headers,
    )
    assert bundle.status_code == 200
    body = bundle.json()
    assert "box_office" in body["db_evidence"]
    assert isinstance(body["document_evidence"]["documents"], list)

