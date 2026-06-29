"""
SQLAlchemy ORM models - maps to PostgreSQL schema defined in sql/schema.sql.
Uses declarative base with UUID primary keys and TIMESTAMPTZ columns.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


def generate_uuid() -> uuid.UUID:
    """Generate a new UUID object."""
    return uuid.uuid4()


class League(Base):
    """Sports league model (e.g., La Liga, Premier League)."""

    __tablename__ = "leagues"

    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    external_id = Column(String(64), unique=True)
    name = Column(String(128), nullable=False)
    sport = Column(String(64), nullable=False)
    country = Column(String(64))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    matches = relationship("Match", back_populates="league", cascade="all, delete-orphan")


class Team(Base):
    """Sports team model."""

    __tablename__ = "teams"

    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    external_id = Column(String(64), unique=True)
    name = Column(String(128), nullable=False)
    short_name = Column(String(32))
    country = Column(String(64))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    home_matches = relationship(
        "Match", foreign_keys="Match.home_team_id", back_populates="home_team"
    )
    away_matches = relationship(
        "Match", foreign_keys="Match.away_team_id", back_populates="away_team"
    )


class Bookmaker(Base):
    """Bookmaker model (e.g., Pinnacle, Bet365)."""

    __tablename__ = "bookmakers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    external_id = Column(String(64), unique=True)
    name = Column(String(128), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    odds = relationship("OddsHistory", back_populates="bookmaker", cascade="all, delete-orphan")


class Market(Base):
    """Betting market model (e.g., h2h, spreads, totals)."""

    __tablename__ = "markets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    key = Column(String(64), unique=True, nullable=False)
    name = Column(String(128), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    odds = relationship("OddsHistory", back_populates="market", cascade="all, delete-orphan")


class Match(Base):
    """Match/event model with kickoff time and scores."""

    __tablename__ = "matches"

    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    external_id = Column(String(64), unique=True)
    league_id = Column(
        UUID(as_uuid=True), ForeignKey("leagues.id", ondelete="CASCADE"), nullable=False
    )
    home_team_id = Column(
        UUID(as_uuid=True), ForeignKey("teams.id", ondelete="CASCADE"), nullable=False
    )
    away_team_id = Column(
        UUID(as_uuid=True), ForeignKey("teams.id", ondelete="CASCADE"), nullable=False
    )
    kickoff = Column(DateTime(timezone=True), nullable=False)
    status = Column(String(32), default="scheduled")
    home_score = Column(SmallInteger)
    away_score = Column(SmallInteger)
    season = Column(String(16))
    round = Column(String(16))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    league = relationship("League", back_populates="matches")
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    odds = relationship("OddsHistory", back_populates="match", cascade="all, delete-orphan")


class OddsHistory(Base):
    """Historical odds data with bookmaker and market context."""

    __tablename__ = "odds_history"
    __table_args__ = (
        UniqueConstraint(
            "match_id",
            "bookmaker_id",
            "market_id",
            "selection",
            "fetched_at",
            name="uq_odds_snapshot",
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    match_id = Column(
        UUID(as_uuid=True), ForeignKey("matches.id", ondelete="CASCADE"), nullable=False
    )
    bookmaker_id = Column(
        UUID(as_uuid=True), ForeignKey("bookmakers.id", ondelete="CASCADE"), nullable=False
    )
    market_id = Column(
        UUID(as_uuid=True), ForeignKey("markets.id", ondelete="CASCADE"), nullable=False
    )
    selection = Column(String(64), nullable=False)
    odds_decimal = Column(Numeric(8, 4), nullable=False)
    odds_implied = Column(Numeric(6, 4))
    stake_limit = Column(Numeric(12, 2))
    fetched_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    raw_api_data = Column(JSON)

    match = relationship("Match", back_populates="odds")
    bookmaker = relationship("Bookmaker", back_populates="odds")
    market = relationship("Market", back_populates="odds")
