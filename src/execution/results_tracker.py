"""
Results Tracker Module
Tracks bet outcomes, calculates CLV (Closing Line Value), and generates performance analytics.
Automatically settles pending bets and computes ROI/Yield metrics.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import and_
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import BotBet, Market, Match, OddsHistory
from src.utils.config import DATABASE_URL
from src.utils.logger import logger


class ResultsTracker:
    """Tracks bet results, calculates CLV, and generates performance analytics."""

    def __init__(self, db_url: str = DATABASE_URL) -> None:
        """Initialize ResultsTracker with database connection."""
        from sqlalchemy import create_engine

        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def update_pending_bets(self) -> dict[str, int]:
        """
        Update all pending bets with match results.
        Evaluates bet outcomes (WON/LOST/VOID), calculates P&L, and CLV.

        Returns:
            Dict with counts of settled bets by status
        """
        session = self.SessionLocal()
        stats = {"won": 0, "lost": 0, "void": 0, "error": 0}

        try:
            pending_bets = (
                session.query(BotBet)
                .join(Match, BotBet.match_id == Match.id)
                .filter(
                    and_(
                        BotBet.status == "pending",
                        Match.kickoff < datetime.now(UTC),
                        Match.status == "finished",
                    )
                )
                .all()
            )

            logger.info(f"Found {len(pending_bets)} pending bets to settle")

            for bet in pending_bets:
                try:
                    self._settle_bet(session, bet)
                    stats[bet.status] += 1
                except Exception as e:
                    logger.error(f"Error settling bet {bet.id}: {e}")
                    stats["error"] += 1
                    session.rollback()

            session.commit()
            logger.info(f"Settled bets: {stats}")

        except Exception as e:
            logger.error(f"Error in update_pending_bets: {e}")
            session.rollback()
        finally:
            session.close()

        return stats

    def _settle_bet(self, session: Session, bet: BotBet) -> None:
        """Settle a single bet based on match result."""
        match = session.query(Match).filter(Match.id == bet.match_id).first()
        if not match:
            raise ValueError(f"Match not found for bet {bet.id}")

        if match.home_score is None or match.away_score is None:
            raise ValueError(f"Match {match.id} has no final score")

        market = session.query(Market).filter(Market.id == bet.market_id).first()
        if not market:
            raise ValueError(f"Market not found for bet {bet.id}")

        market_key = market.key.lower()
        selection = bet.selection.lower()

        if market_key in ["h2h", "1x2", "match_winner"]:
            result = self._evaluate_1x2(match, selection)
        elif market_key in ["totals", "over_under"]:
            result = self._evaluate_totals(match, selection)
        elif market_key in ["spreads", "handicap"]:
            result = self._evaluate_handicap(match, selection, bet.odds_decimal)
        else:
            logger.warning(f"Unknown market type: {market_key}")
            result = "VOID"

        bet.status = result
        bet.settled_at = datetime.now(UTC)

        if result == "won":
            bet.pnl = (bet.odds_decimal - 1) * bet.stake
        elif result == "lost":
            bet.pnl = -bet.stake
        else:
            bet.pnl = Decimal("0")

        self._calculate_clv(session, bet)

        logger.info(
            f"Bet {bet.id}: {result} | P&L: {bet.pnl} | CLV: {bet.clv if bet.clv else 'N/A'}"
        )

    def _evaluate_1x2(self, match: Match, selection: str) -> str:
        """Evaluate 1X2 market outcome."""
        home = match.home_score
        away = match.away_score

        if selection in ["home", "h", "1"]:
            return "won" if home > away else "lost"
        elif selection in ["away", "a", "2"]:
            return "won" if away > home else "lost"
        elif selection in ["draw", "d", "x"]:
            return "won" if home == away else "lost"
        else:
            return "void"

    def _evaluate_totals(self, match: Match, selection: str) -> str:
        """Evaluate Over/Under market outcome."""
        total_goals = match.home_score + match.away_score
        selection_lower = selection.lower()

        if "over" in selection_lower:
            line = self._extract_line(selection, default=2.5)
            return "won" if total_goals > line else "lost"
        elif "under" in selection_lower:
            line = self._extract_line(selection, default=2.5)
            return "won" if total_goals < line else "lost"
        else:
            return "void"

    def _evaluate_handicap(self, match: Match, selection: str, odds: Decimal) -> str:
        """Evaluate handicap/spread market outcome."""
        handicap = self._extract_line(selection, default=0.0)
        home = match.home_score
        away = match.away_score

        if "home" in selection or "h" in selection:
            adjusted = home + handicap
            return "won" if adjusted > away else "lost"
        elif "away" in selection or "a" in selection:
            adjusted = away + handicap
            return "won" if adjusted > home else "lost"
        else:
            return "void"

    def _extract_line(self, selection: str, default: float = 2.5) -> float:
        """Extract numeric line from selection string."""
        import re

        match = re.search(r"(\d+\.?\d*)", selection)
        if match:
            return float(match.group(1))
        return default

    def _calculate_clv(self, session: Session, bet: BotBet) -> None:
        """Calculate Closing Line Value for a bet."""
        closing_odds_record = (
            session.query(OddsHistory)
            .filter(
                and_(
                    OddsHistory.match_id == bet.match_id,
                    OddsHistory.market_id == bet.market_id,
                    OddsHistory.selection.ilike(bet.selection),
                    OddsHistory.is_closing_line.is_(True),
                )
            )
            .order_by(OddsHistory.fetched_at.desc())
            .first()
        )

        if closing_odds_record and closing_odds_record.odds_decimal:
            bet.closing_odds = closing_odds_record.odds_decimal
            odds_taken = float(bet.odds_decimal)
            closing = float(bet.closing_odds)

            if odds_taken > 0 and closing > 0:
                implied_taken = 1 / odds_taken
                implied_closing = 1 / closing
                bet.clv = Decimal(str(implied_closing - implied_taken))

    def generate_analytics(self, days: int = 30) -> dict[str, Any]:
        """
        Generate performance analytics for the specified period.

        Args:
            days: Number of days to analyze (default: 30)

        Returns:
            Dict with performance metrics
        """
        session = self.SessionLocal()

        try:
            cutoff_date = datetime.now(UTC) - timedelta(days=days)

            bets = (
                session.query(BotBet)
                .filter(
                    and_(
                        BotBet.settled_at >= cutoff_date,
                        BotBet.status.in_(["won", "lost"]),
                    )
                )
                .all()
            )

            if not bets:
                logger.info(f"No settled bets found in last {days} days")
                return {
                    "period_days": days,
                    "total_bets": 0,
                    "hit_rate": 0.0,
                    "yield_pct": 0.0,
                    "roi_pct": 0.0,
                    "total_pnl": 0.0,
                    "total_staked": 0.0,
                    "avg_clv": 0.0,
                    "avg_odds": 0.0,
                }

            total_bets = len(bets)
            won_bets = sum(1 for b in bets if b.status == "won")
            hit_rate = (won_bets / total_bets) * 100 if total_bets > 0 else 0.0

            total_staked = sum(float(b.stake) for b in bets)
            total_pnl = sum(float(b.pnl) for b in bets if b.pnl)

            yield_pct = (total_pnl / total_staked) * 100 if total_staked > 0 else 0.0
            roi_pct = yield_pct

            clv_values = [float(b.clv) for b in bets if b.clv is not None]
            avg_clv = sum(clv_values) / len(clv_values) if clv_values else 0.0

            avg_odds = sum(float(b.odds_decimal) for b in bets) / total_bets

            analytics = {
                "period_days": days,
                "total_bets": total_bets,
                "won_bets": won_bets,
                "lost_bets": total_bets - won_bets,
                "hit_rate": round(hit_rate, 2),
                "yield_pct": round(yield_pct, 2),
                "roi_pct": round(roi_pct, 2),
                "total_pnl": round(total_pnl, 2),
                "total_staked": round(total_staked, 2),
                "avg_clv": round(avg_clv, 4),
                "avg_odds": round(avg_odds, 2),
            }

            logger.info(f"Analytics ({days} days): {analytics}")
            return analytics

        except Exception as e:
            logger.error(f"Error generating analytics: {e}")
            return {}
        finally:
            session.close()


def main() -> None:
    """Run results tracker manually."""
    tracker = ResultsTracker()

    logger.info("Updating pending bets...")
    stats = tracker.update_pending_bets()
    logger.info(f"Settlement stats: {stats}")

    logger.info("Generating analytics...")
    analytics = tracker.generate_analytics(days=30)
    logger.info(f"Analytics: {analytics}")


if __name__ == "__main__":
    main()
