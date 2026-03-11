"""marketlogic data models

Revision ID: 20260311_000002
Revises: 20260310_000001
Create Date: 2026-03-11 00:00:02
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260311_000002"
down_revision = "20260310_000001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "genres",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=100), nullable=False),
    )
    op.create_index("ix_genres_name", "genres", ["name"], unique=True)

    op.create_table(
        "territories",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("region_code", sa.String(length=20), nullable=False),
        sa.Column("currency_code", sa.String(length=10), nullable=False),
    )
    op.create_index("ix_territories_name", "territories", ["name"], unique=True)
    op.create_index("ix_territories_region_code", "territories", ["region_code"], unique=False)
    op.create_index(
        "ix_territories_currency_code",
        "territories",
        ["currency_code"],
        unique=False,
    )

    op.create_table(
        "films",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("release_year", sa.Integer(), nullable=False),
        sa.Column("runtime_min", sa.Integer(), nullable=False),
        sa.Column("budget_usd", sa.Float(), nullable=False),
        sa.Column("logline", sa.String(length=400), nullable=False),
        sa.Column("synopsis_doc_path", sa.String(length=300), nullable=False),
        sa.Column("script_doc_path", sa.String(length=300), nullable=False),
        sa.UniqueConstraint("title", "release_year", name="uq_films_title_year"),
    )
    op.create_index("ix_films_title", "films", ["title"], unique=False)
    op.create_index("ix_films_release_year", "films", ["release_year"], unique=False)

    op.create_table(
        "film_genres",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("film_id", sa.Integer(), nullable=False),
        sa.Column("genre_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["film_id"], ["films.id"]),
        sa.ForeignKeyConstraint(["genre_id"], ["genres.id"]),
        sa.UniqueConstraint("film_id", "genre_id", name="uq_film_genres"),
    )
    op.create_index("ix_film_genres_film_id", "film_genres", ["film_id"], unique=False)
    op.create_index("ix_film_genres_genre_id", "film_genres", ["genre_id"], unique=False)

    op.create_table(
        "actors",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("q_score", sa.Float(), nullable=False),
        sa.Column("social_reach", sa.Integer(), nullable=False),
    )
    op.create_index("ix_actors_name", "actors", ["name"], unique=True)

    op.create_table(
        "film_cast",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("film_id", sa.Integer(), nullable=False),
        sa.Column("actor_id", sa.Integer(), nullable=False),
        sa.Column("billing_order", sa.Integer(), nullable=False),
        sa.Column("role_name", sa.String(length=120), nullable=False),
        sa.ForeignKeyConstraint(["film_id"], ["films.id"]),
        sa.ForeignKeyConstraint(["actor_id"], ["actors.id"]),
        sa.UniqueConstraint("film_id", "actor_id", name="uq_film_cast"),
    )
    op.create_index("ix_film_cast_film_id", "film_cast", ["film_id"], unique=False)
    op.create_index("ix_film_cast_actor_id", "film_cast", ["actor_id"], unique=False)

    op.create_table(
        "box_office",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("film_id", sa.Integer(), nullable=False),
        sa.Column("territory_id", sa.Integer(), nullable=False),
        sa.Column("genre_id", sa.Integer(), nullable=False),
        sa.Column("gross_local", sa.Float(), nullable=False),
        sa.Column("gross_usd", sa.Float(), nullable=False),
        sa.Column("admissions", sa.Integer(), nullable=False),
        sa.Column("release_date", sa.Date(), nullable=False),
        sa.ForeignKeyConstraint(["film_id"], ["films.id"]),
        sa.ForeignKeyConstraint(["territory_id"], ["territories.id"]),
        sa.ForeignKeyConstraint(["genre_id"], ["genres.id"]),
        sa.UniqueConstraint(
            "film_id",
            "territory_id",
            "release_date",
            name="uq_box_office_film_territory_date",
        ),
    )
    op.create_index("ix_box_office_film_id", "box_office", ["film_id"], unique=False)
    op.create_index(
        "ix_box_office_territory_id",
        "box_office",
        ["territory_id"],
        unique=False,
    )
    op.create_index("ix_box_office_genre_id", "box_office", ["genre_id"], unique=False)

    op.create_table(
        "theatrical_windows",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("territory_id", sa.Integer(), nullable=False),
        sa.Column("window_type", sa.String(length=50), nullable=False),
        sa.Column("days", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["territory_id"], ["territories.id"]),
        sa.UniqueConstraint("territory_id", "window_type", name="uq_theatrical_windows"),
    )
    op.create_index(
        "ix_theatrical_windows_territory_id",
        "theatrical_windows",
        ["territory_id"],
        unique=False,
    )

    op.create_table(
        "currency_rates",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("currency_code", sa.String(length=10), nullable=False),
        sa.Column("rate_to_usd", sa.Float(), nullable=False),
        sa.Column("rate_date", sa.Date(), nullable=False),
        sa.UniqueConstraint("currency_code", "rate_date", name="uq_currency_rates"),
    )
    op.create_index(
        "ix_currency_rates_currency_code",
        "currency_rates",
        ["currency_code"],
        unique=False,
    )

    op.create_table(
        "vod_price_benchmarks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("territory_id", sa.Integer(), nullable=False),
        sa.Column("license_type", sa.String(length=50), nullable=False),
        sa.Column("window_months", sa.Integer(), nullable=False),
        sa.Column("price_min_usd", sa.Float(), nullable=False),
        sa.Column("price_max_usd", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["territory_id"], ["territories.id"]),
        sa.UniqueConstraint(
            "territory_id",
            "license_type",
            "window_months",
            name="uq_vod_price_benchmarks",
        ),
    )
    op.create_index(
        "ix_vod_price_benchmarks_territory_id",
        "vod_price_benchmarks",
        ["territory_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_vod_price_benchmarks_territory_id", table_name="vod_price_benchmarks")
    op.drop_table("vod_price_benchmarks")
    op.drop_index("ix_currency_rates_currency_code", table_name="currency_rates")
    op.drop_table("currency_rates")
    op.drop_index("ix_theatrical_windows_territory_id", table_name="theatrical_windows")
    op.drop_table("theatrical_windows")
    op.drop_index("ix_box_office_genre_id", table_name="box_office")
    op.drop_index("ix_box_office_territory_id", table_name="box_office")
    op.drop_index("ix_box_office_film_id", table_name="box_office")
    op.drop_table("box_office")
    op.drop_index("ix_film_cast_actor_id", table_name="film_cast")
    op.drop_index("ix_film_cast_film_id", table_name="film_cast")
    op.drop_table("film_cast")
    op.drop_index("ix_actors_name", table_name="actors")
    op.drop_table("actors")
    op.drop_index("ix_film_genres_genre_id", table_name="film_genres")
    op.drop_index("ix_film_genres_film_id", table_name="film_genres")
    op.drop_table("film_genres")
    op.drop_index("ix_films_release_year", table_name="films")
    op.drop_index("ix_films_title", table_name="films")
    op.drop_table("films")
    op.drop_index("ix_territories_currency_code", table_name="territories")
    op.drop_index("ix_territories_region_code", table_name="territories")
    op.drop_index("ix_territories_name", table_name="territories")
    op.drop_table("territories")
    op.drop_index("ix_genres_name", table_name="genres")
    op.drop_table("genres")
