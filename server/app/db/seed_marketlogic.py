from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import models

SEED_SQL_FILE = Path(__file__).resolve().parents[3] / "seed.txt"
ALLOWED_TERRITORIES = {
    "Australia",
    "China",
    "Germany",
    "India",
    "Japan",
    "Russia",
    "Saudi Arabia",
    "United Arab Emirates",
    "United Kingdom",
    "United States",
}


def _canonical_title(raw: str) -> str:
    mapping = {
        "Avengers: Endgame": "Avengers Endgame",
        "Ford v Ferrari": "Ford V Ferrari",
        "How to Train Your Dragon": "How To Train Your Dragon",
        "The Lion King": "Lion King",
    }
    return mapping.get(raw.strip(), raw.strip())


def _canonical_territory(raw: str) -> str:
    value = raw.strip()
    mapping = {
        "UAE": "United Arab Emirates",
        "Uae": "United Arab Emirates",
        "UAE Middle East": "United Arab Emirates",
        "Uae Middle East": "United Arab Emirates",
        "United States Of America": "United States",
        "UK": "United Kingdom",
    }
    return mapping.get(value, value)


def _slugify(value: str) -> str:
    slug = "_".join(value.lower().replace(":", "").replace("-", " ").split())
    return slug


def _doc_paths(title: str) -> tuple[str, str]:
    repo_root = Path(__file__).resolve().parents[3]
    slug = _slugify(title)

    script_candidates = [
        repo_root / "adk-server" / "docs" / "scripts" / f"{slug}.md",
    ]
    synopsis_candidates = [
        repo_root / "adk-server" / "docs" / "synopses" / "md" / f"{slug}_synopsis.md",
        repo_root
        / "adk-server"
        / "docs"
        / "synopses"
        / "md"
        / f"{slug.replace('_your_', '_')}_synopsis.md",
    ]

    script_rel = f"adk-server/docs/scripts/{slug}.md"
    synopsis_rel = f"adk-server/docs/synopses/md/{slug}_synopsis.md"

    for path in script_candidates:
        if path.exists():
            script_rel = str(path.relative_to(repo_root))
            break
    for path in synopsis_candidates:
        if path.exists():
            synopsis_rel = str(path.relative_to(repo_root))
            break

    return synopsis_rel, script_rel


def _parse_token(token: str) -> Any:
    raw = token.strip()
    if not raw:
        return None
    upper = raw.upper()
    if upper == "NULL":
        return None
    if upper == "TRUE":
        return True
    if upper == "FALSE":
        return False
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1].replace("''", "'")
    try:
        if any(char in raw for char in [".", "e", "E"]):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _split_fields(raw_tuple: str) -> list[Any]:
    fields: list[str] = []
    current: list[str] = []
    in_string = False
    i = 0
    while i < len(raw_tuple):
        ch = raw_tuple[i]
        if ch == "'":
            current.append(ch)
            if in_string and i + 1 < len(raw_tuple) and raw_tuple[i + 1] == "'":
                current.append("'")
                i += 1
            else:
                in_string = not in_string
        elif ch == "," and not in_string:
            fields.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
        i += 1
    if current:
        fields.append("".join(current).strip())
    return [_parse_token(field) for field in fields]


def _extract_rows(values_blob: str) -> list[list[Any]]:
    cleaned_lines = [line for line in values_blob.splitlines() if not line.strip().startswith("--")]
    text = "\n".join(cleaned_lines)

    rows: list[list[Any]] = []
    in_string = False
    depth = 0
    current: list[str] = []

    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "'":
            current.append(ch)
            if in_string and i + 1 < len(text) and text[i + 1] == "'":
                current.append("'")
                i += 1
            else:
                in_string = not in_string
        elif not in_string and ch == "(":
            depth += 1
            if depth == 1:
                current = []
            else:
                current.append(ch)
        elif not in_string and ch == ")":
            depth -= 1
            if depth == 0:
                rows.append(_split_fields("".join(current)))
            else:
                current.append(ch)
        else:
            if depth >= 1:
                current.append(ch)
        i += 1

    return rows


def _parse_seed_sql(seed_text: str) -> dict[str, list[dict[str, Any]]]:
    inserts: dict[str, list[dict[str, Any]]] = {}
    marker = "INSERT INTO "
    cursor = 0

    while True:
        start = seed_text.find(marker, cursor)
        if start == -1:
            break

        paren_start = seed_text.find("(", start)
        values_pos = seed_text.find("VALUES", paren_start)
        paren_end = seed_text.rfind(")", paren_start, values_pos)
        if paren_start == -1 or paren_end == -1 or values_pos == -1:
            break

        table_name = seed_text[start + len(marker) : paren_start].strip().split()[0]
        columns_blob = seed_text[paren_start + 1 : paren_end]
        columns = [col.strip() for col in columns_blob.split(",")]

        stmt_end = seed_text.find(";", values_pos)
        if stmt_end == -1:
            break

        values_blob = seed_text[values_pos + len("VALUES") : stmt_end]
        row_values = _extract_rows(values_blob)
        inserts[table_name] = [
            {columns[idx]: row[idx] for idx in range(min(len(columns), len(row)))}
            for row in row_values
        ]

        cursor = stmt_end + 1

    return inserts


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(str(value).strip())
    except ValueError:
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return default


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    return date.fromisoformat(text)


def _region_for_code(code: str) -> str:
    mapping = {
        "US": "NA",
        "CA": "NA",
        "MX": "NA",
        "GB": "EU",
        "DE": "EU",
        "FR": "EU",
        "IT": "EU",
        "ES": "EU",
        "RU": "EU",
        "CN": "APAC",
        "JP": "APAC",
        "KR": "APAC",
        "IN": "APAC",
        "AU": "APAC",
        "BR": "LATAM",
        "LA": "LATAM",
        "SA": "MENA",
        "AE": "MENA",
    }
    return mapping.get(code.upper(), "GLOBAL")


async def _reset_marketlogic_tables(db: AsyncSession) -> None:
    ordered_models = [
        models.FilmCast,
        models.FilmGenre,
        models.BoxOffice,
        models.TheatricalWindow,
        models.CurrencyRate,
        models.VodPriceBenchmark,
        models.FestivalPerformance,
        models.GenreTerritoryBenchmark,
        models.MarketingPerformance,
        models.CensorshipRiskFlag,
        models.AcquisitionDeal,
        models.StreamingPlatformMarketShare,
        models.TerritoryRiskIndex,
        models.MgBenchmark,
        models.Actor,
        models.Film,
        models.Genre,
        models.Territory,
    ]
    for model in ordered_models:
        await db.execute(delete(model))


async def seed_marketlogic(db: AsyncSession) -> dict[str, int]:
    if not SEED_SQL_FILE.exists():
        return {"error": 1}

    parsed = _parse_seed_sql(SEED_SQL_FILE.read_text(encoding="utf-8"))

    films_seed = parsed.get("films", [])
    box_office_seed = parsed.get("box_office_performance", [])
    actors_seed = parsed.get("actor_metrics", [])
    film_cast_seed = parsed.get("film_cast", [])
    windows_seed = parsed.get("theatrical_window_trends", [])
    fx_seed = parsed.get("currency_exchange_rates", [])
    vod_seed = parsed.get("vod_licensing_benchmarks", [])
    festival_seed = parsed.get("festival_performance", [])
    genre_bench_seed = parsed.get("genre_territory_benchmarks", [])
    marketing_seed = parsed.get("marketing_performance", [])
    censorship_seed = parsed.get("censorship_risk_flags", [])
    deals_seed = parsed.get("acquisition_deals", [])
    stream_seed = parsed.get("streaming_platform_market_share", [])
    risk_seed = parsed.get("territory_risk_index", [])
    mg_seed = parsed.get("mg_benchmarks", [])

    await _reset_marketlogic_tables(db)

    counts: dict[str, int] = {"seed_source": 1}

    genre_names: set[str] = set()
    film_rows: list[dict[str, Any]] = []
    for film in films_seed:
        title = _canonical_title(str(film.get("title") or "").strip())
        if not title:
            continue
        main_genre = str(film.get("genre") or "Unknown").strip() or "Unknown"
        subgenre = str(film.get("subgenre") or "").strip()
        if main_genre:
            genre_names.add(main_genre)
        if subgenre:
            genre_names.add(subgenre)
        synopsis_doc_path, script_doc_path = _doc_paths(title)
        film_rows.append(
            {
                "title": title,
                "release_year": _as_int(film.get("year"), 2000),
                "runtime_min": _as_int(film.get("runtime_minutes"), 100),
                "budget_usd": _as_float(film.get("production_budget_usd"), 0.0),
                "logline": (
                    f"{title} ({main_genre}/{subgenre or 'General'}) "
                    "market profile for global strategy."
                ),
                "synopsis_doc_path": synopsis_doc_path,
                "script_doc_path": script_doc_path,
                "main_genre": main_genre,
                "subgenre": subgenre,
            }
        )

    genre_by_name: dict[str, models.Genre] = {}
    for name in sorted(genre_names):
        genre = models.Genre(name=name)
        db.add(genre)
        genre_by_name[name] = genre
    counts["genres_created"] = len(genre_by_name)

    territory_code_by_name: dict[str, str] = {}
    for row in box_office_seed:
        territory = _canonical_territory(str(row.get("territory") or "").strip())
        code = str(row.get("territory_code") or "").strip().upper()
        if territory and code and territory in ALLOWED_TERRITORIES:
            territory_code_by_name[territory] = code
    for row in (
        windows_seed
        + festival_seed
        + marketing_seed
        + censorship_seed
        + deals_seed
        + risk_seed
        + stream_seed
    ):
        territory = _canonical_territory(str(row.get("territory") or "").strip())
        code = str(row.get("territory_code") or "").strip().upper()
        if (
            territory
            and code
            and territory in ALLOWED_TERRITORIES
            and territory not in territory_code_by_name
        ):
            territory_code_by_name[territory] = code

    currency_by_territory: dict[str, str] = {}
    for row in fx_seed:
        territory = _canonical_territory(str(row.get("territory") or "").strip())
        currency = str(row.get("currency_code") or "USD").strip().upper()
        if territory in ALLOWED_TERRITORIES:
            currency_by_territory[territory] = currency

    all_territories = set(ALLOWED_TERRITORIES)

    territory_by_name: dict[str, models.Territory] = {}
    for territory_name in sorted(all_territories):
        code = territory_code_by_name.get(territory_name, "GL")
        territory = models.Territory(
            name=territory_name,
            region_code=_region_for_code(code),
            currency_code=currency_by_territory.get(territory_name, "USD"),
        )
        db.add(territory)
        territory_by_name[territory_name] = territory
    counts["territories_created"] = len(territory_by_name)

    film_by_seed_id: dict[int, models.Film] = {}
    for idx, row in enumerate(film_rows, start=1):
        film = models.Film(
            title=row["title"],
            release_year=row["release_year"],
            runtime_min=row["runtime_min"],
            budget_usd=row["budget_usd"],
            logline=row["logline"],
            synopsis_doc_path=row["synopsis_doc_path"],
            script_doc_path=row["script_doc_path"],
        )
        db.add(film)
        film_by_seed_id[idx] = film
    counts["films_created"] = len(film_by_seed_id)

    actor_by_seed_id: dict[int, models.Actor] = {}
    for idx, row in enumerate(actors_seed, start=1):
        q_us = _as_float(row.get("q_score_us"), 0.0)
        q_global = _as_float(row.get("q_score_global"), 0.0)
        q_score = (q_us + q_global) / 2.0 if q_us and q_global else max(q_us, q_global)
        social_m = (
            _as_float(row.get("instagram_followers_m"))
            + _as_float(row.get("twitter_followers_m"))
            + _as_float(row.get("tiktok_followers_m"))
            + _as_float(row.get("youtube_subscribers_m"))
        )
        actor = models.Actor(
            name=str(row.get("name") or "Unknown Actor").strip(),
            q_score=q_score,
            social_reach=int(social_m * 1_000_000),
        )
        db.add(actor)
        actor_by_seed_id[idx] = actor
    counts["actors_created"] = len(actor_by_seed_id)

    await db.flush()

    film_genres_created = 0
    for idx, row in enumerate(film_rows, start=1):
        film = film_by_seed_id[idx]
        primary = row["main_genre"]
        secondary = row["subgenre"]
        if primary in genre_by_name:
            db.add(models.FilmGenre(film_id=film.id, genre_id=genre_by_name[primary].id))
            film_genres_created += 1
        if secondary and secondary in genre_by_name and secondary != primary:
            db.add(models.FilmGenre(film_id=film.id, genre_id=genre_by_name[secondary].id))
            film_genres_created += 1
    counts["film_genres_created"] = film_genres_created

    film_cast_created = 0
    for row in film_cast_seed:
        film = film_by_seed_id.get(_as_int(row.get("film_id")))
        actor = actor_by_seed_id.get(_as_int(row.get("actor_id")))
        if film is None or actor is None:
            continue
        db.add(
            models.FilmCast(
                film_id=film.id,
                actor_id=actor.id,
                billing_order=_as_int(row.get("billing_order"), 0),
                role_name=str(row.get("role_type") or "supporting"),
            )
        )
        film_cast_created += 1
    counts["film_cast_created"] = film_cast_created

    box_office_created = 0
    for row in box_office_seed:
        film = film_by_seed_id.get(_as_int(row.get("film_id")))
        territory_name = _canonical_territory(str(row.get("territory") or "").strip())
        territory = territory_by_name.get(territory_name)
        release_date = _as_date(row.get("release_date"))
        if film is None or territory is None or release_date is None:
            continue

        film_source = film_rows[_as_int(row.get("film_id"), 1) - 1]
        genre = genre_by_name.get(film_source["main_genre"])
        if genre is None:
            continue

        gross_usd = _as_float(row.get("total_gross_usd"), 0.0)
        avg_ticket_price = _as_float(row.get("avg_ticket_price_usd"), 0.0)
        admissions = int(gross_usd / avg_ticket_price) if avg_ticket_price > 0 else 0

        db.add(
            models.BoxOffice(
                film_id=film.id,
                territory_id=territory.id,
                genre_id=genre.id,
                gross_local=gross_usd,
                gross_usd=gross_usd,
                admissions=admissions,
                release_date=release_date,
            )
        )
        box_office_created += 1
    counts["box_office_created"] = box_office_created

    latest_window_by_territory: dict[str, dict[str, Any]] = {}
    for row in windows_seed:
        territory_name = _canonical_territory(str(row.get("territory") or "").strip())
        if territory_name not in ALLOWED_TERRITORIES:
            continue
        year = _as_int(row.get("year"), 0)
        previous = latest_window_by_territory.get(territory_name)
        if previous is None or year > _as_int(previous.get("year"), 0):
            latest_window_by_territory[territory_name] = row

    windows_created = 0
    for territory_name, row in latest_window_by_territory.items():
        territory = territory_by_name.get(territory_name)
        if territory is None:
            continue
        window_entries = [
            ("theatrical_to_vod", _as_int(row.get("avg_theatrical_window_days"), 0)),
            ("premium_vod", _as_int(row.get("premium_vod_window_days"), 0)),
            ("theatrical_to_streaming", _as_int(row.get("streaming_window_days"), 0)),
        ]
        for window_type, days in window_entries:
            db.add(
                models.TheatricalWindow(
                    territory_id=territory.id, window_type=window_type, days=days
                )
            )
            windows_created += 1
    counts["theatrical_windows_created"] = windows_created

    rates_created = 0
    for row in fx_seed:
        currency_code = str(row.get("currency_code") or "USD").strip().upper()
        territory_name = _canonical_territory(str(row.get("territory") or "").strip())
        usd_rate = _as_float(row.get("usd_rate"), 1.0)
        rate_date = _as_date(row.get("rate_date"))
        if (
            not currency_code
            or rate_date is None
            or usd_rate <= 0
            or territory_name not in ALLOWED_TERRITORIES
        ):
            continue
        db.add(
            models.CurrencyRate(
                currency_code=currency_code,
                rate_to_usd=round(1.0 / usd_rate, 6),
                rate_date=rate_date,
            )
        )
        rates_created += 1
    counts["currency_rates_created"] = rates_created

    vod_groups: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in vod_seed:
        territory = _canonical_territory(str(row.get("territory") or "").strip())
        license_type = str(row.get("deal_type") or "").strip()
        window_months = _as_int(row.get("license_term_months"), 0)
        fee = _as_float(row.get("license_fee_usd"), 0.0)
        if (
            territory in ALLOWED_TERRITORIES
            and license_type
            and window_months > 0
            and fee > 0
        ):
            vod_groups[(territory, license_type, window_months)].append(fee)

    vod_created = 0
    for (territory_name, license_type, window_months), fees in vod_groups.items():
        territory = territory_by_name.get(territory_name)
        if territory is None:
            continue
        db.add(
            models.VodPriceBenchmark(
                territory_id=territory.id,
                license_type=license_type,
                window_months=window_months,
                price_min_usd=min(fees),
                price_max_usd=max(fees),
            )
        )
        vod_created += 1
    counts["vod_price_benchmarks_created"] = vod_created

    festival_created = 0
    for row in festival_seed:
        film = film_by_seed_id.get(_as_int(row.get("film_id")))
        if film is None:
            continue
        db.add(
            models.FestivalPerformance(
                film_id=film.id,
                festival_name=str(row.get("festival_name") or ""),
                festival_year=_as_int(row.get("festival_year"), 0),
                award_category=str(row.get("award_category") or ""),
                award_result=str(row.get("award_result") or ""),
                audience_score=_as_float(row.get("audience_score"), 0.0),
                critic_score=_as_float(row.get("critic_score"), 0.0),
                buzz_score=_as_int(row.get("buzz_score"), 0),
            )
        )
        festival_created += 1
    counts["festival_performance_created"] = festival_created

    genre_bench_created = 0
    for row in genre_bench_seed:
        territory_name = _canonical_territory(str(row.get("territory") or ""))
        if territory_name not in ALLOWED_TERRITORIES:
            continue
        db.add(
            models.GenreTerritoryBenchmark(
                genre=str(row.get("genre") or ""),
                territory=territory_name,
                territory_code=str(row.get("territory_code") or ""),
                avg_opening_weekend_usd=_as_int(row.get("avg_opening_weekend_usd"), 0),
                avg_total_gross_usd=_as_int(row.get("avg_total_gross_usd"), 0),
                avg_multiplier=_as_float(row.get("avg_multiplier"), 0.0),
                sample_size=_as_int(row.get("sample_size"), 0),
                year_range_start=_as_int(row.get("year_range_start"), 0),
                year_range_end=_as_int(row.get("year_range_end"), 0),
            )
        )
        genre_bench_created += 1
    counts["genre_territory_benchmarks_created"] = genre_bench_created

    marketing_created = 0
    for row in marketing_seed:
        film = film_by_seed_id.get(_as_int(row.get("film_id")))
        territory_name = _canonical_territory(str(row.get("territory") or ""))
        if territory_name not in ALLOWED_TERRITORIES:
            continue
        if film is None:
            continue
        db.add(
            models.MarketingPerformance(
                film_id=film.id,
                territory=territory_name,
                p_and_a_spend_usd=_as_int(row.get("p_and_a_spend_usd"), 0),
                digital_spend_usd=_as_int(row.get("digital_spend_usd"), 0),
                tv_spend_usd=_as_int(row.get("tv_spend_usd"), 0),
                outdoor_spend_usd=_as_int(row.get("outdoor_spend_usd"), 0),
                social_spend_usd=_as_int(row.get("social_spend_usd"), 0),
                revenue_generated_usd=_as_int(row.get("revenue_generated_usd"), 0),
                roi_pct=_as_float(row.get("roi_pct"), 0.0),
                campaign_type=str(row.get("campaign_type") or ""),
            )
        )
        marketing_created += 1
    counts["marketing_performance_created"] = marketing_created

    censorship_created = 0
    for row in censorship_seed:
        film = film_by_seed_id.get(_as_int(row.get("film_id")))
        territory_name = _canonical_territory(str(row.get("territory") or ""))
        if territory_name not in ALLOWED_TERRITORIES:
            continue
        if film is None:
            continue
        db.add(
            models.CensorshipRiskFlag(
                film_id=film.id,
                territory=territory_name,
                territory_code=str(row.get("territory_code") or ""),
                risk_level=str(row.get("risk_level") or ""),
                content_type=str(row.get("content_type") or ""),
                description=str(row.get("description") or ""),
                required_cuts=bool(row.get("required_cuts")),
                rating_assigned=str(row.get("rating_assigned") or ""),
                approved=bool(row.get("approved")),
            )
        )
        censorship_created += 1
    counts["censorship_risk_flags_created"] = censorship_created

    deals_created = 0
    for row in deals_seed:
        film = film_by_seed_id.get(_as_int(row.get("film_id")))
        territory_name = _canonical_territory(str(row.get("territory") or ""))
        deal_date = _as_date(row.get("deal_date"))
        if (
            film is None
            or deal_date is None
            or territory_name not in ALLOWED_TERRITORIES
        ):
            continue
        db.add(
            models.AcquisitionDeal(
                film_id=film.id,
                territory=territory_name,
                deal_type=str(row.get("deal_type") or ""),
                minimum_guarantee_usd=_as_int(row.get("minimum_guarantee_usd"), 0),
                advance_usd=_as_int(row.get("advance_usd"), 0),
                recoupment_threshold_usd=_as_int(row.get("recoupment_threshold_usd"), 0),
                backend_pct=_as_float(row.get("backend_pct"), 0.0),
                acquirer=str(row.get("acquirer") or ""),
                deal_date=deal_date,
                outcome=str(row.get("outcome") or ""),
                actual_revenue_usd=_as_int(row.get("actual_revenue_usd"), 0),
            )
        )
        deals_created += 1
    counts["acquisition_deals_created"] = deals_created

    stream_created = 0
    for row in stream_seed:
        territory_name = _canonical_territory(str(row.get("territory") or ""))
        if territory_name not in ALLOWED_TERRITORIES:
            continue
        db.add(
            models.StreamingPlatformMarketShare(
                platform=str(row.get("platform") or ""),
                territory=territory_name,
                territory_code=str(row.get("territory_code") or ""),
                year=_as_int(row.get("year"), 0),
                subscribers_m=_as_float(row.get("subscribers_m"), 0.0),
                market_share_pct=_as_float(row.get("market_share_pct"), 0.0),
                avg_monthly_revenue_usd=_as_float(row.get("avg_monthly_revenue_usd"), 0.0),
                content_budget_usd=_as_int(row.get("content_budget_usd"), 0),
                film_licensing_budget_usd=_as_int(row.get("film_licensing_budget_usd"), 0),
            )
        )
        stream_created += 1
    counts["streaming_platform_market_share_created"] = stream_created

    risk_created = 0
    for row in risk_seed:
        territory_name = _canonical_territory(str(row.get("territory") or ""))
        if territory_name not in ALLOWED_TERRITORIES:
            continue
        db.add(
            models.TerritoryRiskIndex(
                territory=territory_name,
                territory_code=str(row.get("territory_code") or ""),
                political_risk=_as_float(row.get("political_risk"), 0.0),
                currency_risk=_as_float(row.get("currency_risk"), 0.0),
                censorship_risk=_as_float(row.get("censorship_risk"), 0.0),
                piracy_risk=_as_float(row.get("piracy_risk"), 0.0),
                collection_risk=_as_float(row.get("collection_risk"), 0.0),
                overall_risk=_as_float(row.get("overall_risk"), 0.0),
                market_attractiveness=_as_float(row.get("market_attractiveness"), 0.0),
                year=_as_int(row.get("year"), 0),
                notes=str(row.get("notes") or ""),
            )
        )
        risk_created += 1
    counts["territory_risk_index_created"] = risk_created

    mg_created = 0
    for row in mg_seed:
        territory_name = _canonical_territory(str(row.get("territory") or ""))
        if territory_name not in ALLOWED_TERRITORIES:
            continue
        db.add(
            models.MgBenchmark(
                genre=str(row.get("genre") or ""),
                territory_tier=str(row.get("territory_tier") or ""),
                territory=territory_name,
                budget_range=str(row.get("budget_range") or ""),
                min_mg_usd=_as_int(row.get("min_mg_usd"), 0),
                max_mg_usd=_as_int(row.get("max_mg_usd"), 0),
                typical_mg_usd=_as_int(row.get("typical_mg_usd"), 0),
                notes=str(row.get("notes") or ""),
                year_updated=_as_int(row.get("year_updated"), 0),
            )
        )
        mg_created += 1
    counts["mg_benchmarks_created"] = mg_created

    actual_territories = set(territory_by_name.keys())
    if actual_territories != ALLOWED_TERRITORIES:
        missing = sorted(ALLOWED_TERRITORIES - actual_territories)
        extra = sorted(actual_territories - ALLOWED_TERRITORIES)
        raise ValueError(
            f"Territory validation failed. Missing={missing}, unexpected={extra}"
        )

    await db.commit()
    return counts
