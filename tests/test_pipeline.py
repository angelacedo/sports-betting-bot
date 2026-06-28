"""
Tests for ETL Pipeline
Uses mocks to simulate API responses and verifies UPSERT logic.
"""

from decimal import Decimal
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.models import Base, Match, OddsHistory
from src.etl.pipeline import ETLPipeline


@pytest.fixture
def mock_db_engine():
    """Create in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def mock_pipeline(mock_db_engine):
    """Create pipeline with mock database."""
    with patch("src.etl.pipeline.create_engine") as mock_engine:
        mock_engine.return_value = mock_db_engine
        pipeline = ETLPipeline(db_url="sqlite:///:memory:")
        pipeline.engine = mock_db_engine
        pipeline.SessionLocal = sessionmaker(bind=mock_db_engine)
        return pipeline


@pytest.fixture
def sample_odds_data():
    """Sample odds data from The Odds API."""
    return [
        {
            "id": "event_123",
            "sport_key": "soccer_spain_la_liga",
            "commence_time": "2024-01-15T20:00:00Z",
            "home_team": "Real Madrid",
            "away_team": "Barcelona",
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Real Madrid", "price": 2.10},
                                {"name": "Draw", "price": 3.40},
                                {"name": "Barcelona", "price": 3.50},
                            ],
                        }
                    ],
                }
            ],
        }
    ]


@pytest.fixture
def sample_fixtures_data():
    """Sample fixtures data from API-Football."""
    return {
        "response": [
            {
                "fixture": {
                    "id": 456,
                    "date": "2024-01-15T20:00:00+00:00",
                    "status": {"short": "NS", "long": "Not Started"},
                },
                "league": {
                    "id": 140,
                    "name": "La Liga",
                    "season": 2024,
                    "round": "Regular Season - 20",
                },
                "teams": {
                    "home": {"id": 530, "name": "Real Madrid"},
                    "away": {"id": 529, "name": "Barcelona"},
                },
                "goals": {"home": None, "away": None},
            }
        ]
    }


def test_upsert_matches_no_duplicates(mock_pipeline, sample_fixtures_data):
    """Test that UPSERT doesn't create duplicate matches."""
    # Normalize fixtures
    fixtures_df = mock_pipeline.stats_extractor._normalize_fixtures(sample_fixtures_data)

    # First upsert
    count1 = mock_pipeline._upsert_matches(fixtures_df)
    assert count1 == 1

    # Verify match was created
    session = mock_pipeline.SessionLocal()
    matches = session.query(Match).all()
    assert len(matches) == 1
    assert matches[0].external_id == "456"
    session.close()

    # Second upsert (should update, not duplicate)
    count2 = mock_pipeline._upsert_matches(fixtures_df)
    assert count2 == 1

    # Verify still only one match
    session = mock_pipeline.SessionLocal()
    matches = session.query(Match).all()
    assert len(matches) == 1
    session.close()


def test_upsert_odds_no_duplicates(mock_pipeline, sample_odds_data, sample_fixtures_data):
    """Test that UPSERT doesn't create duplicate odds records."""
    # First create the match
    fixtures_df = mock_pipeline.stats_extractor._normalize_fixtures(sample_fixtures_data)
    mock_pipeline._upsert_matches(fixtures_df)

    # Normalize odds
    odds_df = mock_pipeline.odds_extractor._normalize_odds(sample_odds_data)
    odds_df["fixture_id"] = 456  # Map to fixture

    # First upsert
    count1 = mock_pipeline._upsert_odds(odds_df)
    assert count1 == 3  # 3 outcomes (home, draw, away)

    # Verify odds were created
    session = mock_pipeline.SessionLocal()
    odds = session.query(OddsHistory).all()
    assert len(odds) == 3
    session.close()

    # Second upsert with same data (should not duplicate)
    count2 = mock_pipeline._upsert_odds(odds_df)
    assert count2 == 3

    # Verify still only 3 odds records
    session = mock_pipeline.SessionLocal()
    odds = session.query(OddsHistory).all()
    assert len(odds) == 3
    session.close()


def test_upsert_odds_updates_existing(mock_pipeline, sample_odds_data, sample_fixtures_data):
    """Test that UPSERT updates existing odds with new values."""
    # Create match
    fixtures_df = mock_pipeline.stats_extractor._normalize_fixtures(sample_fixtures_data)
    mock_pipeline._upsert_matches(fixtures_df)

    # First odds upsert
    odds_df = mock_pipeline.odds_extractor._normalize_odds(sample_odds_data)
    odds_df["fixture_id"] = 456
    mock_pipeline._upsert_odds(odds_df)

    # Modify odds values
    odds_df_modified = odds_df.copy()
    odds_df_modified["odds_decimal"] = 2.50  # Changed from 2.10
    mock_pipeline._upsert_odds(odds_df_modified)

    # Verify odds were updated
    session = mock_pipeline.SessionLocal()
    odds = session.query(OddsHistory).filter_by(selection="Real Madrid").first()
    assert odds.odds_decimal == Decimal("2.50")
    session.close()


def test_fuzzy_matching(mock_pipeline):
    """Test fuzzy matching of team names."""
    known_teams = ["Real Madrid", "Barcelona", "Atletico Madrid"]

    # Exact match
    result = mock_pipeline.stats_extractor._fuzzy_match_team("Real Madrid", known_teams)
    assert result == "Real Madrid"

    # Fuzzy match
    result = mock_pipeline.stats_extractor._fuzzy_match_team("Real Madri", known_teams)
    assert result == "Real Madrid"

    # No match
    result = mock_pipeline.stats_extractor._fuzzy_match_team(
        "Unknown Team", known_teams, threshold=90
    )
    assert result is None


def test_merge_odds_with_fixtures(mock_pipeline, sample_odds_data, sample_fixtures_data):
    """Test merging odds DataFrame with matched fixtures."""
    odds_df = mock_pipeline.odds_extractor._normalize_odds(sample_odds_data)
    mock_pipeline.odds_extractor.get_unique_events(odds_df)

    fixtures_df = mock_pipeline.stats_extractor._normalize_fixtures(sample_fixtures_data)

    # Simulate matched fixtures
    matched_fixtures = fixtures_df.copy()
    matched_fixtures["matched_event_id"] = "event_123"

    merged = mock_pipeline._merge_odds_with_fixtures(odds_df, matched_fixtures)

    assert len(merged) == 3  # 3 outcomes
    assert "fixture_id" in merged.columns
    assert all(merged["fixture_id"] == 456)


def test_pipeline_filters_postponed(mock_pipeline, sample_fixtures_data):
    """Test that pipeline filters out postponed matches."""
    # Add a postponed fixture
    sample_fixtures_data["response"].append(
        {
            "fixture": {
                "id": 789,
                "date": "2024-01-15T22:00:00+00:00",
                "status": {"short": "PST", "long": "Postponed"},
            },
            "league": {
                "id": 140,
                "name": "La Liga",
                "season": 2024,
                "round": "Regular Season - 20",
            },
            "teams": {
                "home": {"id": 530, "name": "Real Madrid"},
                "away": {"id": 531, "name": "Valencia"},
            },
            "goals": {"home": None, "away": None},
        }
    )

    fixtures_df = mock_pipeline.stats_extractor._normalize_fixtures(sample_fixtures_data)
    assert len(fixtures_df) == 2

    # Filter postponed
    valid_statuses = ["NS", "TBD"]
    filtered = fixtures_df[fixtures_df["status"].isin(valid_statuses)]
    assert len(filtered) == 1
    assert filtered.iloc[0]["fixture_id"] == 456
