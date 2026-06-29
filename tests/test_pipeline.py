"""
Tests for ETL Pipeline
Uses mocks to simulate API responses and verifies UPSERT logic.
"""

from decimal import Decimal
from unittest.mock import patch

import pandas as pd
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
            "sport_key": "soccer_epl",
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
def sample_events_df():
    """Sample events DataFrame."""
    return pd.DataFrame(
        {
            "event_id": ["event_123"],
            "home_team": ["Real Madrid"],
            "away_team": ["Barcelona"],
            "commence_time": ["2024-01-15T20:00:00Z"],
        }
    )


def test_upsert_matches_no_duplicates(mock_pipeline, sample_events_df):
    """Test that UPSERT doesn't create duplicate matches."""
    # First upsert
    count1 = mock_pipeline._upsert_matches(sample_events_df, "soccer_epl")
    assert count1 == 1

    # Verify match was created
    session = mock_pipeline.SessionLocal()
    matches = session.query(Match).all()
    assert len(matches) == 1
    assert matches[0].external_id == "event_123"
    session.close()

    # Second upsert (should update, not duplicate)
    count2 = mock_pipeline._upsert_matches(sample_events_df, "soccer_epl")
    assert count2 == 1

    # Verify still only one match
    session = mock_pipeline.SessionLocal()
    matches = session.query(Match).all()
    assert len(matches) == 1
    session.close()


def test_upsert_odds_no_duplicates(mock_pipeline, sample_odds_data, sample_events_df):
    """Test that UPSERT doesn't create duplicate odds records."""
    # First create the match
    mock_pipeline._upsert_matches(sample_events_df, "soccer_epl")

    # Normalize odds
    odds_df = mock_pipeline.odds_extractor._normalize_odds(sample_odds_data)

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


def test_upsert_odds_updates_existing(mock_pipeline, sample_odds_data, sample_events_df):
    """Test that UPSERT updates existing odds with new values."""
    # Create match
    mock_pipeline._upsert_matches(sample_events_df, "soccer_epl")

    # First odds upsert
    odds_df = mock_pipeline.odds_extractor._normalize_odds(sample_odds_data)
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


def test_pipeline_extracts_odds(mock_pipeline, sample_odds_data):
    """Test that pipeline correctly extracts odds data."""
    with patch.object(mock_pipeline.odds_extractor, "extract") as mock_extract:
        mock_extract.return_value = mock_pipeline.odds_extractor._normalize_odds(sample_odds_data)

        result = mock_pipeline.run("soccer_epl")

        mock_extract.assert_called_once_with("soccer_epl")
        assert result["matches"] == 1
        assert result["odds"] == 3


def test_empty_odds_aborts_pipeline(mock_pipeline):
    """Test that pipeline aborts when no odds data is found."""
    with patch.object(mock_pipeline.odds_extractor, "extract") as mock_extract:
        mock_extract.return_value = pd.DataFrame()

        result = mock_pipeline.run("soccer_epl")

        assert result == {"matches": 0, "odds": 0}
