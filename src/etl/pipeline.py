"""
ETL Pipeline Orchestrator
Coordinates the full data pipeline: extract -> transform -> load.
Extracts odds data from The Odds API and loads to database.
"""

from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import (
    Bookmaker,
    League,
    Market,
    Match,
    OddsHistory,
    Team,
)
from src.etl.odds_extractor import OddsExtractor
from src.utils.config import DATABASE_URL
from src.utils.logger import logger


class ETLPipeline:
    """Orchestrates the ETL pipeline for sports betting odds data."""

    def __init__(self, db_url: str = DATABASE_URL):
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.odds_extractor = OddsExtractor()

    def run(self, sport_key: str = "soccer_epl") -> dict[str, int]:
        """
        Execute full ETL pipeline.
        Returns dict with counts of inserted/updated records.
        """
        logger.info(f"Starting ETL pipeline for {sport_key}")
        stats = {"matches": 0, "odds": 0}

        try:
            # STEP A: Extract odds data
            logger.info("Step A: Extracting odds from The Odds API")
            odds_df = self.odds_extractor.extract(sport_key)

            if odds_df.empty:
                logger.warning("No odds data found, aborting pipeline")
                return stats

            odds_events_df = self.odds_extractor.get_unique_events(odds_df)

            # STEP B: Load matches to database
            logger.info("Step B: Loading matches to database")
            stats["matches"] = self._upsert_matches(odds_events_df, sport_key)

            # STEP C: Load odds to database
            logger.info("Step C: Loading odds to database")
            stats["odds"] = self._upsert_odds(odds_df)

            logger.info(f"Pipeline completed successfully: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    def _upsert_matches(self, events_df: pd.DataFrame, sport_key: str) -> int:
        """Insert or update matches in database. Returns count of upserted records."""
        session = self.SessionLocal()
        count = 0

        try:
            league = self._get_or_create_league(
                session,
                external_id=sport_key,
                name=sport_key.replace("_", " ").title(),
                sport="soccer",
            )

            for _, row in events_df.iterrows():
                event_id = str(row["event_id"])
                home_team_name = str(row["home_team"])
                away_team_name = str(row["away_team"])
                kickoff_raw = row.get("commence_time")
                kickoff = (
                    pd.to_datetime(kickoff_raw, utc=True) if kickoff_raw else datetime.now(UTC)
                )

                home_team = self._get_or_create_team(
                    session,
                    external_id=home_team_name,
                    name=home_team_name,
                )
                away_team = self._get_or_create_team(
                    session,
                    external_id=away_team_name,
                    name=away_team_name,
                )

                match = session.query(Match).filter_by(external_id=event_id).first()

                if match:
                    match.kickoff = kickoff
                    match.updated_at = datetime.now(UTC)
                else:
                    match = Match(
                        external_id=event_id,
                        league_id=league.id,
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        kickoff=kickoff,
                        status="scheduled",
                    )
                    session.add(match)

                count += 1

            session.commit()
            logger.info(f"Upserted {count} matches")
            return count

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to upsert matches: {e}")
            raise
        finally:
            session.close()

    def _upsert_odds(self, odds_df: pd.DataFrame) -> int:
        """Insert odds records with UPSERT logic. Returns count."""
        session = self.SessionLocal()
        count = 0

        try:
            for _, row in odds_df.iterrows():
                event_id = str(row["event_id"])
                match = session.query(Match).filter_by(external_id=event_id).first()
                if not match:
                    continue

                bookmaker_name = str(row["bookmaker"])
                market_key = str(row["market"])
                selection = str(row["selection"])
                odds_decimal = Decimal(str(row["odds_decimal"]))
                fetched_at = row["fetched_at"]
                raw_data = row.get("raw_api_data")

                bookmaker = self._get_or_create_bookmaker(
                    session,
                    external_id=bookmaker_name,
                    name=bookmaker_name,
                )

                market = self._get_or_create_market(
                    session,
                    key=market_key,
                    name=market_key,
                )

                existing = (
                    session.query(OddsHistory)
                    .filter_by(
                        match_id=match.id,
                        bookmaker_id=bookmaker.id,
                        market_id=market.id,
                        selection=selection,
                        fetched_at=fetched_at,
                    )
                    .first()
                )

                if existing:
                    existing.odds_decimal = odds_decimal
                    existing.raw_api_data = raw_data
                else:
                    implied_prob = Decimal(str(1 / float(odds_decimal))) if odds_decimal else None
                    odds_record = OddsHistory(
                        match_id=match.id,
                        bookmaker_id=bookmaker.id,
                        market_id=market.id,
                        selection=selection,
                        odds_decimal=odds_decimal,
                        odds_implied=implied_prob,
                        fetched_at=fetched_at,
                        raw_api_data=raw_data,
                    )
                    session.add(odds_record)

                count += 1

            session.commit()
            logger.info(f"Upserted {count} odds records")
            return count

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to upsert odds: {e}")
            raise
        finally:
            session.close()

    def _get_or_create_league(
        self,
        session,
        external_id: str,
        name: str,
        sport: str,
        country: str | None = None,
    ) -> League:
        """Get existing league or create new one."""
        league = session.query(League).filter_by(external_id=external_id).first()
        if not league:
            league = League(
                external_id=external_id,
                name=name,
                sport=sport,
                country=country,
            )
            session.add(league)
            session.flush()
        return league

    def _get_or_create_team(
        self,
        session,
        external_id: str,
        name: str,
        short_name: str | None = None,
        country: str | None = None,
    ) -> Team:
        """Get existing team or create new one."""
        team = session.query(Team).filter_by(external_id=external_id).first()
        if not team:
            team = Team(
                external_id=external_id,
                name=name,
                short_name=short_name,
                country=country,
            )
            session.add(team)
            session.flush()
        return team

    def _get_or_create_bookmaker(self, session, external_id: str, name: str) -> Bookmaker:
        """Get existing bookmaker or create new one."""
        bookmaker = session.query(Bookmaker).filter_by(external_id=external_id).first()
        if not bookmaker:
            bookmaker = Bookmaker(external_id=external_id, name=name)
            session.add(bookmaker)
            session.flush()
        return bookmaker

    def _get_or_create_market(
        self, session, key: str, name: str, description: str | None = None
    ) -> Market:
        """Get existing market or create new one."""
        market = session.query(Market).filter_by(key=key).first()
        if not market:
            market = Market(key=key, name=name, description=description)
            session.add(market)
            session.flush()
        return market


def main():
    """Run the ETL pipeline."""
    pipeline = ETLPipeline()
    stats = pipeline.run()
    logger.info(f"Final stats: {stats}")


if __name__ == "__main__":
    main()
