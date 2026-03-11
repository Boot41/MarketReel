"""add seedtxt extended models

Revision ID: 20260311_000003
Revises: 20260311_000002
Create Date: 2026-03-11 00:00:03
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "20260311_000003"
down_revision = "20260311_000002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "festival_performance",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("film_id", sa.Integer(), nullable=False),
        sa.Column("festival_name", sa.String(length=150), nullable=False),
        sa.Column("festival_year", sa.Integer(), nullable=False),
        sa.Column("award_category", sa.String(length=150), nullable=False),
        sa.Column("award_result", sa.String(length=50), nullable=False),
        sa.Column("audience_score", sa.Float(), nullable=False),
        sa.Column("critic_score", sa.Float(), nullable=False),
        sa.Column("buzz_score", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["film_id"], ["films.id"]),
    )
    op.create_index(
        "ix_festival_performance_film_id", "festival_performance", ["film_id"], unique=False
    )

    op.create_table(
        "genre_territory_benchmarks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("genre", sa.String(length=100), nullable=False),
        sa.Column("territory", sa.String(length=100), nullable=False),
        sa.Column("territory_code", sa.String(length=2), nullable=False),
        sa.Column("avg_opening_weekend_usd", sa.BigInteger(), nullable=False),
        sa.Column("avg_total_gross_usd", sa.BigInteger(), nullable=False),
        sa.Column("avg_multiplier", sa.Float(), nullable=False),
        sa.Column("sample_size", sa.Integer(), nullable=False),
        sa.Column("year_range_start", sa.Integer(), nullable=False),
        sa.Column("year_range_end", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
    )
    op.create_index(
        "ix_genre_territory_benchmarks_genre", "genre_territory_benchmarks", ["genre"], unique=False
    )
    op.create_index(
        "ix_genre_territory_benchmarks_territory",
        "genre_territory_benchmarks",
        ["territory"],
        unique=False,
    )

    op.create_table(
        "marketing_performance",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("film_id", sa.Integer(), nullable=False),
        sa.Column("territory", sa.String(length=100), nullable=False),
        sa.Column("p_and_a_spend_usd", sa.BigInteger(), nullable=False),
        sa.Column("digital_spend_usd", sa.BigInteger(), nullable=False),
        sa.Column("tv_spend_usd", sa.BigInteger(), nullable=False),
        sa.Column("outdoor_spend_usd", sa.BigInteger(), nullable=False),
        sa.Column("social_spend_usd", sa.BigInteger(), nullable=False),
        sa.Column("revenue_generated_usd", sa.BigInteger(), nullable=False),
        sa.Column("roi_pct", sa.Float(), nullable=False),
        sa.Column("campaign_type", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["film_id"], ["films.id"]),
    )
    op.create_index(
        "ix_marketing_performance_film_id", "marketing_performance", ["film_id"], unique=False
    )
    op.create_index(
        "ix_marketing_performance_territory", "marketing_performance", ["territory"], unique=False
    )

    op.create_table(
        "censorship_risk_flags",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("film_id", sa.Integer(), nullable=False),
        sa.Column("territory", sa.String(length=100), nullable=False),
        sa.Column("territory_code", sa.String(length=2), nullable=False),
        sa.Column("risk_level", sa.String(length=20), nullable=False),
        sa.Column("content_type", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("required_cuts", sa.Boolean(), nullable=False),
        sa.Column("rating_assigned", sa.String(length=20), nullable=False),
        sa.Column("approved", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["film_id"], ["films.id"]),
    )
    op.create_index(
        "ix_censorship_risk_flags_film_id", "censorship_risk_flags", ["film_id"], unique=False
    )
    op.create_index(
        "ix_censorship_risk_flags_territory", "censorship_risk_flags", ["territory"], unique=False
    )

    op.create_table(
        "acquisition_deals",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("film_id", sa.Integer(), nullable=False),
        sa.Column("territory", sa.String(length=100), nullable=False),
        sa.Column("deal_type", sa.String(length=50), nullable=False),
        sa.Column("minimum_guarantee_usd", sa.BigInteger(), nullable=False),
        sa.Column("advance_usd", sa.BigInteger(), nullable=False),
        sa.Column("recoupment_threshold_usd", sa.BigInteger(), nullable=False),
        sa.Column("backend_pct", sa.Float(), nullable=False),
        sa.Column("acquirer", sa.String(length=150), nullable=False),
        sa.Column("deal_date", sa.Date(), nullable=False),
        sa.Column("outcome", sa.String(length=50), nullable=False),
        sa.Column("actual_revenue_usd", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["film_id"], ["films.id"]),
    )
    op.create_index("ix_acquisition_deals_film_id", "acquisition_deals", ["film_id"], unique=False)
    op.create_index(
        "ix_acquisition_deals_territory", "acquisition_deals", ["territory"], unique=False
    )

    op.create_table(
        "streaming_platform_market_share",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("platform", sa.String(length=100), nullable=False),
        sa.Column("territory", sa.String(length=100), nullable=False),
        sa.Column("territory_code", sa.String(length=2), nullable=False),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("subscribers_m", sa.Float(), nullable=False),
        sa.Column("market_share_pct", sa.Float(), nullable=False),
        sa.Column("avg_monthly_revenue_usd", sa.Float(), nullable=False),
        sa.Column("content_budget_usd", sa.BigInteger(), nullable=False),
        sa.Column("film_licensing_budget_usd", sa.BigInteger(), nullable=False),
    )
    op.create_index(
        "ix_streaming_platform_market_share_platform",
        "streaming_platform_market_share",
        ["platform"],
        unique=False,
    )
    op.create_index(
        "ix_streaming_platform_market_share_territory",
        "streaming_platform_market_share",
        ["territory"],
        unique=False,
    )
    op.create_index(
        "ix_streaming_platform_market_share_year",
        "streaming_platform_market_share",
        ["year"],
        unique=False,
    )

    op.create_table(
        "territory_risk_index",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("territory", sa.String(length=100), nullable=False),
        sa.Column("territory_code", sa.String(length=2), nullable=False),
        sa.Column("political_risk", sa.Float(), nullable=False),
        sa.Column("currency_risk", sa.Float(), nullable=False),
        sa.Column("censorship_risk", sa.Float(), nullable=False),
        sa.Column("piracy_risk", sa.Float(), nullable=False),
        sa.Column("collection_risk", sa.Float(), nullable=False),
        sa.Column("overall_risk", sa.Float(), nullable=False),
        sa.Column("market_attractiveness", sa.Float(), nullable=False),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False),
    )
    op.create_index(
        "ix_territory_risk_index_territory", "territory_risk_index", ["territory"], unique=False
    )
    op.create_index("ix_territory_risk_index_year", "territory_risk_index", ["year"], unique=False)

    op.create_table(
        "mg_benchmarks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("genre", sa.String(length=100), nullable=False),
        sa.Column("territory_tier", sa.String(length=50), nullable=False),
        sa.Column("territory", sa.String(length=100), nullable=False),
        sa.Column("budget_range", sa.String(length=100), nullable=False),
        sa.Column("min_mg_usd", sa.BigInteger(), nullable=False),
        sa.Column("max_mg_usd", sa.BigInteger(), nullable=False),
        sa.Column("typical_mg_usd", sa.BigInteger(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False),
        sa.Column("year_updated", sa.Integer(), nullable=False),
    )
    op.create_index("ix_mg_benchmarks_genre", "mg_benchmarks", ["genre"], unique=False)
    op.create_index("ix_mg_benchmarks_territory", "mg_benchmarks", ["territory"], unique=False)
    op.create_index(
        "ix_mg_benchmarks_territory_tier", "mg_benchmarks", ["territory_tier"], unique=False
    )
    op.create_index(
        "ix_mg_benchmarks_year_updated", "mg_benchmarks", ["year_updated"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_mg_benchmarks_year_updated", table_name="mg_benchmarks")
    op.drop_index("ix_mg_benchmarks_territory_tier", table_name="mg_benchmarks")
    op.drop_index("ix_mg_benchmarks_territory", table_name="mg_benchmarks")
    op.drop_index("ix_mg_benchmarks_genre", table_name="mg_benchmarks")
    op.drop_table("mg_benchmarks")

    op.drop_index("ix_territory_risk_index_year", table_name="territory_risk_index")
    op.drop_index("ix_territory_risk_index_territory", table_name="territory_risk_index")
    op.drop_table("territory_risk_index")

    op.drop_index(
        "ix_streaming_platform_market_share_year", table_name="streaming_platform_market_share"
    )
    op.drop_index(
        "ix_streaming_platform_market_share_territory", table_name="streaming_platform_market_share"
    )
    op.drop_index(
        "ix_streaming_platform_market_share_platform", table_name="streaming_platform_market_share"
    )
    op.drop_table("streaming_platform_market_share")

    op.drop_index("ix_acquisition_deals_territory", table_name="acquisition_deals")
    op.drop_index("ix_acquisition_deals_film_id", table_name="acquisition_deals")
    op.drop_table("acquisition_deals")

    op.drop_index("ix_censorship_risk_flags_territory", table_name="censorship_risk_flags")
    op.drop_index("ix_censorship_risk_flags_film_id", table_name="censorship_risk_flags")
    op.drop_table("censorship_risk_flags")

    op.drop_index("ix_marketing_performance_territory", table_name="marketing_performance")
    op.drop_index("ix_marketing_performance_film_id", table_name="marketing_performance")
    op.drop_table("marketing_performance")

    op.drop_index(
        "ix_genre_territory_benchmarks_territory", table_name="genre_territory_benchmarks"
    )
    op.drop_index("ix_genre_territory_benchmarks_genre", table_name="genre_territory_benchmarks")
    op.drop_table("genre_territory_benchmarks")

    op.drop_index("ix_festival_performance_film_id", table_name="festival_performance")
    op.drop_table("festival_performance")
