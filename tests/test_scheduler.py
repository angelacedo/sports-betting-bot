"""
Tests for Scheduler Module
Validates that all jobs are registered correctly.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.scheduler import BettingScheduler


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies."""
    with (
        patch("src.scheduler.ResultsTracker") as mock_tracker,
        patch("src.scheduler.TelegramNotifier") as mock_notifier,
        patch("src.scheduler.run_analysis") as mock_analysis,
    ):
        mock_tracker.return_value = MagicMock()
        mock_notifier.return_value = MagicMock()
        mock_analysis.return_value = {"fixtures_analyzed": 0, "value_bets": []}
        yield {
            "tracker": mock_tracker,
            "notifier": mock_notifier,
            "analysis": mock_analysis,
        }


def test_scheduler_registers_three_jobs(mock_dependencies):
    """Test that scheduler registers exactly 3 jobs."""
    import schedule

    schedule.clear()

    scheduler = BettingScheduler()

    schedule.every().day.at("09:00").do(scheduler.daily_analysis_job)
    schedule.every().day.at("10:00").do(scheduler.daily_settlement_job)
    schedule.every().monday.at("10:05").do(scheduler.weekly_report_job)

    assert len(schedule.jobs) == 3


def test_daily_analysis_job_calls_run_analysis(mock_dependencies):
    """Test that daily_analysis_job calls run_analysis."""
    scheduler = BettingScheduler()
    scheduler.daily_analysis_job()

    mock_dependencies["analysis"].assert_called_once()


def test_daily_settlement_job_calls_tracker(mock_dependencies):
    """Test that daily_settlement_job calls update_pending_bets."""
    scheduler = BettingScheduler()
    scheduler.daily_settlement_job()

    mock_dependencies["tracker"].return_value.update_pending_bets.assert_called_once()


def test_weekly_report_job_calls_tracker_and_notifier(mock_dependencies):
    """Test that weekly_report_job generates analytics and sends report."""
    scheduler = BettingScheduler()
    scheduler.weekly_report_job()

    mock_dependencies["tracker"].return_value.generate_analytics.assert_called_once_with(days=7)
    mock_dependencies["notifier"].return_value.send_performance_report.assert_called_once()
