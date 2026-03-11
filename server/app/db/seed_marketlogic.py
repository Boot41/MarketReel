from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import models

SEED_DIR = Path(__file__).with_name("seed_data")


def _load_json(name: str) -> list[dict[str, Any]]:
    path = SEED_DIR / name
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


async def seed_marketlogic(db: AsyncSession) -> dict[str, int]:
    counts: dict[str, int] = {}

    genres = _load_json("genres.json")
    territories = _load_json("territories.json")
    actors = _load_json("actors.json")
    films = _load_json("films.json")
    film_genres = _load_json("film_genres.json")
    film_cast = _load_json("film_cast.json")
    box_office = _load_json("box_office.json")
    theatrical_windows = _load_json("theatrical_windows.json")
    currency_rates = _load_json("currency_rates.json")
    vod_price_benchmarks = _load_json("vod_price_benchmarks.json")

    existing_genres = {
        genre.name: genre
        for genre in (await db.execute(select(models.Genre))).scalars().all()
    }
    created = 0
    for raw in genres:
        name = str(raw.get("name", "")).strip()
        if not name:
            continue
        if name in existing_genres:
            continue
        obj = models.Genre(name=name)
        db.add(obj)
        existing_genres[name] = obj
        created += 1
    counts["genres_created"] = created

    existing_territories = {
        territory.name: territory
        for territory in (await db.execute(select(models.Territory))).scalars().all()
    }
    created = 0
    for raw in territories:
        name = str(raw.get("name", "")).strip()
        region_code = str(raw.get("region_code", "")).strip()
        currency_code = str(raw.get("currency_code", "")).strip()
        if not name:
            continue
        territory = existing_territories.get(name)
        if territory is None:
            territory = models.Territory(
                name=name,
                region_code=region_code,
                currency_code=currency_code,
            )
            db.add(territory)
            existing_territories[name] = territory
            created += 1
        else:
            territory.region_code = region_code
            territory.currency_code = currency_code
    counts["territories_created"] = created

    existing_actors = {
        actor.name: actor
        for actor in (await db.execute(select(models.Actor))).scalars().all()
    }
    created = 0
    for raw in actors:
        name = str(raw.get("name", "")).strip()
        if not name:
            continue
        actor = existing_actors.get(name)
        q_score = float(raw.get("q_score", 0.0))
        social_reach = int(raw.get("social_reach", 0))
        if actor is None:
            actor = models.Actor(name=name, q_score=q_score, social_reach=social_reach)
            db.add(actor)
            existing_actors[name] = actor
            created += 1
        else:
            actor.q_score = q_score
            actor.social_reach = social_reach
    counts["actors_created"] = created

    existing_films = {
        (film.title, film.release_year): film
        for film in (await db.execute(select(models.Film))).scalars().all()
    }
    created = 0
    for raw in films:
        title = str(raw.get("title", "")).strip()
        release_year = int(raw.get("release_year", 0))
        if not title or not release_year:
            continue
        key = (title, release_year)
        film = existing_films.get(key)
        runtime_min = int(raw.get("runtime_min", 0))
        budget_usd = float(raw.get("budget_usd", 0.0))
        logline = str(raw.get("logline", "")).strip()
        synopsis_doc_path = str(raw.get("synopsis_doc_path", "")).strip()
        script_doc_path = str(raw.get("script_doc_path", "")).strip()
        if film is None:
            film = models.Film(
                title=title,
                release_year=release_year,
                runtime_min=runtime_min,
                budget_usd=budget_usd,
                logline=logline,
                synopsis_doc_path=synopsis_doc_path,
                script_doc_path=script_doc_path,
            )
            db.add(film)
            existing_films[key] = film
            created += 1
        else:
            film.runtime_min = runtime_min
            film.budget_usd = budget_usd
            film.logline = logline
            film.synopsis_doc_path = synopsis_doc_path
            film.script_doc_path = script_doc_path
    counts["films_created"] = created

    await db.flush()

    existing_film_genres = {
        (fg.film_id, fg.genre_id)
        for fg in (await db.execute(select(models.FilmGenre))).scalars().all()
    }
    created = 0
    for raw in film_genres:
        title = str(raw.get("film_title", "")).strip()
        release_year = int(raw.get("release_year", 0))
        genre_name = str(raw.get("genre", "")).strip()
        film = existing_films.get((title, release_year))
        genre = existing_genres.get(genre_name)
        if film is None or genre is None:
            continue
        key = (film.id, genre.id)
        if key in existing_film_genres:
            continue
        db.add(models.FilmGenre(film_id=film.id, genre_id=genre.id))
        existing_film_genres.add(key)
        created += 1
    counts["film_genres_created"] = created

    existing_film_cast = {
        (cast.film_id, cast.actor_id)
        for cast in (await db.execute(select(models.FilmCast))).scalars().all()
    }
    created = 0
    for raw in film_cast:
        title = str(raw.get("film_title", "")).strip()
        release_year = int(raw.get("release_year", 0))
        actor_name = str(raw.get("actor_name", "")).strip()
        billing_order = int(raw.get("billing_order", 0))
        role_name = str(raw.get("role_name", "")).strip()
        film = existing_films.get((title, release_year))
        actor = existing_actors.get(actor_name)
        if film is None or actor is None:
            continue
        key = (film.id, actor.id)
        if key in existing_film_cast:
            continue
        db.add(
            models.FilmCast(
                film_id=film.id,
                actor_id=actor.id,
                billing_order=billing_order,
                role_name=role_name,
            )
        )
        existing_film_cast.add(key)
        created += 1
    counts["film_cast_created"] = created

    existing_box_office = {
        (bo.film_id, bo.territory_id, bo.release_date)
        for bo in (await db.execute(select(models.BoxOffice))).scalars().all()
    }
    created = 0
    for raw in box_office:
        title = str(raw.get("film_title", "")).strip()
        release_year = int(raw.get("release_year", 0))
        territory_name = str(raw.get("territory", "")).strip()
        genre_name = str(raw.get("genre", "")).strip()
        release_date = _parse_date(str(raw.get("release_date", "")))
        film = existing_films.get((title, release_year))
        territory = existing_territories.get(territory_name)
        genre = existing_genres.get(genre_name)
        if film is None or territory is None or genre is None:
            continue
        key = (film.id, territory.id, release_date)
        if key in existing_box_office:
            continue
        db.add(
            models.BoxOffice(
                film_id=film.id,
                territory_id=territory.id,
                genre_id=genre.id,
                gross_local=float(raw.get("gross_local", 0.0)),
                gross_usd=float(raw.get("gross_usd", 0.0)),
                admissions=int(raw.get("admissions", 0)),
                release_date=release_date,
            )
        )
        existing_box_office.add(key)
        created += 1
    counts["box_office_created"] = created

    existing_windows = {
        (window.territory_id, window.window_type)
        for window in (await db.execute(select(models.TheatricalWindow))).scalars().all()
    }
    created = 0
    for raw in theatrical_windows:
        territory_name = str(raw.get("territory", "")).strip()
        window_type = str(raw.get("window_type", "")).strip()
        days = int(raw.get("days", 0))
        territory = existing_territories.get(territory_name)
        if territory is None or not window_type:
            continue
        key = (territory.id, window_type)
        if key in existing_windows:
            continue
        db.add(
            models.TheatricalWindow(
                territory_id=territory.id,
                window_type=window_type,
                days=days,
            )
        )
        existing_windows.add(key)
        created += 1
    counts["theatrical_windows_created"] = created

    existing_rates = {
        (rate.currency_code, rate.rate_date)
        for rate in (await db.execute(select(models.CurrencyRate))).scalars().all()
    }
    created = 0
    for raw in currency_rates:
        currency_code = str(raw.get("currency_code", "")).strip()
        rate_date = _parse_date(str(raw.get("rate_date", "")))
        rate_to_usd = float(raw.get("rate_to_usd", 0.0))
        if not currency_code:
            continue
        key = (currency_code, rate_date)
        if key in existing_rates:
            continue
        db.add(
            models.CurrencyRate(
                currency_code=currency_code,
                rate_to_usd=rate_to_usd,
                rate_date=rate_date,
            )
        )
        existing_rates.add(key)
        created += 1
    counts["currency_rates_created"] = created

    existing_vod = {
        (bench.territory_id, bench.license_type, bench.window_months)
        for bench in (await db.execute(select(models.VodPriceBenchmark))).scalars().all()
    }
    created = 0
    for raw in vod_price_benchmarks:
        territory_name = str(raw.get("territory", "")).strip()
        license_type = str(raw.get("license_type", "")).strip()
        window_months = int(raw.get("window_months", 0))
        territory = existing_territories.get(territory_name)
        if territory is None or not license_type:
            continue
        key = (territory.id, license_type, window_months)
        if key in existing_vod:
            continue
        db.add(
            models.VodPriceBenchmark(
                territory_id=territory.id,
                license_type=license_type,
                window_months=window_months,
                price_min_usd=float(raw.get("price_min_usd", 0.0)),
                price_max_usd=float(raw.get("price_max_usd", 0.0)),
            )
        )
        existing_vod.add(key)
        created += 1
    counts["vod_price_benchmarks_created"] = created

    await db.commit()
    return counts
