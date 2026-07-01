"""
Scheduler Module
Automated job scheduler for daily analysis, bet settlement and performance reporting.
Jobs:
- 09:00 daily: Analyze fixtures and find value bets
- 10:00 daily: Settle pending bets
- 10:05 Monday: Send weekly performance report
"""

import time
from typing import Any

import schedule

from src.execution.results_tracker import ResultsTracker
from src.main import run_analysis
from src.utils.ev_calculator import ValueBet
from src.utils.logger import logger
from src.utils.notifier import TelegramNotifier


class BettingScheduler:
    """Scheduler for automated betting tasks."""

    def __init__(self) -> None:
        """Initialize scheduler with tracker and notifier."""
        self.tracker = ResultsTracker()
        self.notifier = TelegramNotifier()

    def daily_analysis_job(self) -> None:
        """Daily job: analyze fixtures and send value bets to Telegram."""
        logger.info("Running daily analysis job...")
        try:
            result = run_analysis()
            fixtures_analyzed: int = result.get("fixtures_analyzed", 0)
            value_bets: list[ValueBet] = result.get("value_bets", [])

            if value_bets:
                self.notifier.send_value_bets(value_bets, fixtures_analyzed)
                logger.info(f"Sent {len(value_bets)} value bets to Telegram")
            else:
                logger.info("No value bets found today")
                self.notifier.send_message(
                    "📊 Análisis diario completado. No se encontraron value bets hoy."
                )
        except Exception as e:
            logger.error(f"Error in daily analysis: {e}")

    def daily_settlement_job(self) -> None:
        """Daily job: settle pending bets."""
        logger.info("Running daily settlement job...")
        try:
            stats = self.tracker.update_pending_bets()
            logger.info(f"Daily settlement completed: {stats}")
        except Exception as e:
            logger.error(f"Error in daily settlement: {e}")

    def weekly_report_job(self) -> None:
        """Weekly job: generate and send performance report."""
        logger.info("Running weekly report job...")
        try:
            analytics = self.tracker.generate_analytics(days=7)
            self.notifier.send_performance_report(analytics)
            logger.info("Weekly report sent successfully")
        except Exception as e:
            logger.error(f"Error in weekly report: {e}")

    def start(self) -> None:
        """Start the scheduler."""
        logger.info("Starting betting scheduler...")

        schedule.every().day.at("09:00").do(self.daily_analysis_job)
        logger.info("Scheduled daily analysis at 09:00")

        schedule.every().day.at("10:00").do(self.daily_settlement_job)
        logger.info("Scheduled daily settlement at 10:00")

        schedule.every().monday.at("10:05").do(self.weekly_report_job)
        logger.info("Scheduled weekly report on Mondays at 10:05")

        logger.info("Scheduler is running. Press Ctrl+C to stop.")

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")


def start_scheduler() -> None:
    """Start the betting scheduler (entry point)."""
    scheduler = BettingScheduler()
    scheduler.start()


def main() -> None:
    """Run scheduler."""
    start_scheduler()


if __name__ == "__main__":
    main()
