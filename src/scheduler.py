"""
Scheduler Module
Automated job scheduler for bet settlement and performance reporting.
Runs daily settlement and weekly analytics reports.
"""

import time

import schedule

from src.execution.results_tracker import ResultsTracker
from src.utils.logger import logger
from src.utils.notifier import TelegramNotifier


class BettingScheduler:
    """Scheduler for automated betting tasks."""

    def __init__(self) -> None:
        """Initialize scheduler with tracker and notifier."""
        self.tracker = ResultsTracker()
        self.notifier = TelegramNotifier()

    def daily_settlement(self) -> None:
        """Daily job: settle pending bets."""
        logger.info("Running daily settlement job...")
        try:
            stats = self.tracker.update_pending_bets()
            logger.info(f"Daily settlement completed: {stats}")
        except Exception as e:
            logger.error(f"Error in daily settlement: {e}")

    def weekly_report(self) -> None:
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

        schedule.every().day.at("10:00").do(self.daily_settlement)
        logger.info("Scheduled daily settlement at 10:00")

        schedule.every().monday.at("10:05").do(self.weekly_report)
        logger.info("Scheduled weekly report on Mondays at 10:05")

        logger.info("Scheduler is running. Press Ctrl+C to stop.")

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")


def main() -> None:
    """Run scheduler."""
    scheduler = BettingScheduler()
    scheduler.start()


if __name__ == "__main__":
    main()
