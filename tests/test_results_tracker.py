"""
Tests for Results Tracker Module
Validates bet settlement logic, CLV calculation, and analytics generation.
"""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.models import Base, BotBet, League, Market, Match, OddsHistory, Team
from src.execution.results_tracker import ResultsTracker


@pytest.fixture
def mock_db_engine():
    """Create in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def mock_tracker(mock_db_engine):
    """Create ResultsTracker with mock database."""
    tracker = ResultsTracker(db_url="sqlite:///:memory:")
    tracker.engine = mock_db_engine
    tracker.SessionLocal = sessionmaker(bind=mock_db_engine)
    return tracker


@pytest.fixture
def sample_data(mock_tracker):
    """Create sample data for testing."""
    session = mock_tracker.SessionLocal()

    league_id = uuid.uuid4()
    home_team_id = uuid.uuid4()
    away_team_id = uuid.uuid4()
    market_1x2_id = uuid.uuid4()
    market_totals_id = uuid.uuid4()
    match_id = uuid.uuid4()

    league = League(
        id=league_id,
        external_id="epl",
        name="Premier League",
        sport="soccer",
    )
    session.add(league)

    home_team = Team(
        id=home_team_id,
        external_id="arsenal",
        name="Arsenal",
    )
    away_team = Team(
        id=away_team_id,
        external_id="chelsea",
        name="Chelsea",
    )
    session.add_all([home_team, away_team])

    market_1x2 = Market(
        id=market_1x2_id,
        key="h2h",
        name="1X2",
    )
    market_totals = Market(
        id=market_totals_id,
        key="totals",
        name="Over/Under",
    )
    session.add_all([market_1x2, market_totals])

    match = Match(
        id=match_id,
        external_id="arsenal_chelsea_2024",
        league_id=league_id,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        kickoff=datetime.now(UTC) - timedelta(days=1),
        status="finished",
        home_score=2,
        away_score=1,
    )
    session.add(match)

    bet_won = BotBet(
        id=uuid.uuid4(),
        match_id=match_id,
        market_id=market_1x2_id,
        selection="Home",
        odds_decimal=Decimal("2.00"),
        stake=Decimal("100.00"),
        status="pending",
    )
    bet_lost = BotBet(
        id=uuid.uuid4(),
        match_id=match_id,
        market_id=market_1x2_id,
        selection="Away",
        odds_decimal=Decimal("3.50"),
        stake=Decimal("50.00"),
        status="pending",
    )
    bet_over = BotBet(
        id=uuid.uuid4(),
        match_id=match_id,
        market_id=market_totals_id,
        selection="Over 2.5",
        odds_decimal=Decimal("1.80"),
        stake=Decimal("75.00"),
        status="pending",
    )
    session.add_all([bet_won, bet_lost, bet_over])

    closing_odds = OddsHistory(
        id=uuid.uuid4(),
        match_id=match_id,
        bookmaker_id=uuid.uuid4(),
        market_id=market_1x2_id,
        selection="Home",
        odds_decimal=Decimal("1.80"),
        is_closing_line=True,
        fetched_at=datetime.now(UTC),
    )
    session.add(closing_odds)

    session.commit()
    session.close()

    return {
        "match": match,
        "bet_won": bet_won,
        "bet_lost": bet_lost,
        "bet_over": bet_over,
    }


def test_settle_bet_won(mock_tracker, sample_data):
    """Test settling a winning bet."""
    session = mock_tracker.SessionLocal()
    bet = session.query(BotBet).first()

    mock_tracker._settle_bet(session, bet)
    session.commit()

    assert bet.status == "won"
    assert bet.pnl == Decimal("100.00")
    assert bet.settled_at is not None

    session.close()


def test_settle_bet_lost(mock_tracker, sample_data):
    """Test settling a losing bet."""
    session = mock_tracker.SessionLocal()
    bets = session.query(BotBet).all()
    bet = next(b for b in bets if b.selection == "Away")

    mock_tracker._settle_bet(session, bet)
    session.commit()

    assert bet.status == "lost"
    assert bet.pnl == Decimal("-50.00")

    session.close()


def test_settle_bet_over_won(mock_tracker, sample_data):
    """Test settling Over 2.5 bet (total goals = 3, should win)."""
    session = mock_tracker.SessionLocal()
    bets = session.query(BotBet).all()
    bet = next(b for b in bets if "Over" in b.selection)

    mock_tracker._settle_bet(session, bet)
    session.commit()

    assert bet.status == "won"
    assert bet.pnl == Decimal("60.00")

    session.close()


def test_clv_calculation(mock_tracker, sample_data):
    """Test CLV calculation."""
    session = mock_tracker.SessionLocal()
    bet = session.query(BotBet).first()

    mock_tracker._settle_bet(session, bet)
    session.commit()

    assert bet.closing_odds == Decimal("1.80")
    assert bet.clv is not None

    expected_clv = Decimal(str((1 / 1.80) - (1 / 2.00)))
    assert abs(bet.clv - expected_clv) < Decimal("0.0001")

    session.close()


def test_update_pending_bets(mock_tracker, sample_data):
    """Test updating all pending bets."""
    stats = mock_tracker.update_pending_bets()

    assert stats["won"] == 2
    assert stats["lost"] == 1
    assert stats["void"] == 0
    assert stats["error"] == 0


def test_generate_analytics(mock_tracker, sample_data):
    """Test analytics generation."""
    mock_tracker.update_pending_bets()

    analytics = mock_tracker.generate_analytics(days=30)

    assert analytics["total_bets"] == 3
    assert analytics["won_bets"] == 2
    assert analytics["lost_bets"] == 1
    assert analytics["hit_rate"] == pytest.approx(66.67, rel=0.01)
    assert analytics["total_staked"] == 225.00
    assert analytics["total_pnl"] == 110.00


def test_evaluate_1x2(mock_tracker):
    """Test 1X2 evaluation logic."""
    match = MagicMock()
    match.home_score = 2
    match.away_score = 1

    assert mock_tracker._evaluate_1x2(match, "home") == "won"
    assert mock_tracker._evaluate_1x2(match, "away") == "lost"
    assert mock_tracker._evaluate_1x2(match, "draw") == "lost"

    match.home_score = 1
    match.away_score = 1
    assert mock_tracker._evaluate_1x2(match, "draw") == "won"


def test_evaluate_totals(mock_tracker):
    """Test Over/Under evaluation logic."""
    match = MagicMock()
    match.home_score = 2
    match.away_score = 1

    assert mock_tracker._evaluate_totals(match, "Over 2.5") == "won"
    assert mock_tracker._evaluate_totals(match, "Under 2.5") == "lost"
    assert mock_tracker._evaluate_totals(match, "Over 3.5") == "lost"


def test_extract_line(mock_tracker):
    """Test line extraction from selection string."""
    assert mock_tracker._extract_line("Over 2.5", 2.5) == 2.5
    assert mock_tracker._extract_line("Under 3.0", 2.5) == 3.0
    assert mock_tracker._extract_line("Home -1.5", 0.0) == 1.5
    assert mock_tracker._extract_line("No number", 2.5) == 2.5
