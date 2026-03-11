from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    full_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Chat(Base):
    __tablename__ = "chats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    adk_session_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="chat", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id"), index=True)
    role: Mapped[Literal["user", "assistant", "system"]] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    chat: Mapped[Chat] = relationship(back_populates="messages")


class Genre(Base):
    __tablename__ = "genres"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)


class Territory(Base):
    __tablename__ = "territories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    region_code: Mapped[str] = mapped_column(String(20), index=True)
    currency_code: Mapped[str] = mapped_column(String(10), index=True)


class Film(Base):
    __tablename__ = "films"
    __table_args__ = (UniqueConstraint("title", "release_year", name="uq_films_title_year"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(200), index=True)
    release_year: Mapped[int] = mapped_column(Integer, index=True)
    runtime_min: Mapped[int] = mapped_column(Integer)
    budget_usd: Mapped[float] = mapped_column(Float)
    logline: Mapped[str] = mapped_column(String(400))
    synopsis_doc_path: Mapped[str] = mapped_column(String(300))
    script_doc_path: Mapped[str] = mapped_column(String(300))


class FilmGenre(Base):
    __tablename__ = "film_genres"
    __table_args__ = (UniqueConstraint("film_id", "genre_id", name="uq_film_genres"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    film_id: Mapped[int] = mapped_column(ForeignKey("films.id"), index=True)
    genre_id: Mapped[int] = mapped_column(ForeignKey("genres.id"), index=True)


class Actor(Base):
    __tablename__ = "actors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    q_score: Mapped[float] = mapped_column(Float)
    social_reach: Mapped[int] = mapped_column(Integer)


class FilmCast(Base):
    __tablename__ = "film_cast"
    __table_args__ = (UniqueConstraint("film_id", "actor_id", name="uq_film_cast"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    film_id: Mapped[int] = mapped_column(ForeignKey("films.id"), index=True)
    actor_id: Mapped[int] = mapped_column(ForeignKey("actors.id"), index=True)
    billing_order: Mapped[int] = mapped_column(Integer)
    role_name: Mapped[str] = mapped_column(String(120))


class BoxOffice(Base):
    __tablename__ = "box_office"
    __table_args__ = (
        UniqueConstraint(
            "film_id",
            "territory_id",
            "release_date",
            name="uq_box_office_film_territory_date",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    film_id: Mapped[int] = mapped_column(ForeignKey("films.id"), index=True)
    territory_id: Mapped[int] = mapped_column(ForeignKey("territories.id"), index=True)
    genre_id: Mapped[int] = mapped_column(ForeignKey("genres.id"), index=True)
    gross_local: Mapped[float] = mapped_column(Float)
    gross_usd: Mapped[float] = mapped_column(Float)
    admissions: Mapped[int] = mapped_column(Integer)
    release_date: Mapped[date] = mapped_column(Date)


class TheatricalWindow(Base):
    __tablename__ = "theatrical_windows"
    __table_args__ = (
        UniqueConstraint("territory_id", "window_type", name="uq_theatrical_windows"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    territory_id: Mapped[int] = mapped_column(ForeignKey("territories.id"), index=True)
    window_type: Mapped[str] = mapped_column(String(50))
    days: Mapped[int] = mapped_column(Integer)


class CurrencyRate(Base):
    __tablename__ = "currency_rates"
    __table_args__ = (
        UniqueConstraint("currency_code", "rate_date", name="uq_currency_rates"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    currency_code: Mapped[str] = mapped_column(String(10), index=True)
    rate_to_usd: Mapped[float] = mapped_column(Float)
    rate_date: Mapped[date] = mapped_column(Date)


class VodPriceBenchmark(Base):
    __tablename__ = "vod_price_benchmarks"
    __table_args__ = (
        UniqueConstraint(
            "territory_id",
            "license_type",
            "window_months",
            name="uq_vod_price_benchmarks",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    territory_id: Mapped[int] = mapped_column(ForeignKey("territories.id"), index=True)
    license_type: Mapped[str] = mapped_column(String(50))
    window_months: Mapped[int] = mapped_column(Integer)
    price_min_usd: Mapped[float] = mapped_column(Float)
    price_max_usd: Mapped[float] = mapped_column(Float)
