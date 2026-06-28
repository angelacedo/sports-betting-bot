"""
ETL Pipeline Orchestrator
Coordinates the full data pipeline: extract -> transform -> load.
Merges odds and stats data, handles UPSERT logic to avoid duplicates.
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
    MatchStat,
    OddsHistory,
    Team,
)
from src.etl.odds_extractor import OddsExtractor
from src.etl.stats_extractor import StatsExtractor
from src.utils.config import DATABASE_URL
from src.utils.logger import logger


class ETLPipeline:
    """Orchestrates the full ETL pipeline for sports betting data."""

    def __init__(self, db_url: str = DATABASE_URL):
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.odds_extractor = OddsExtractor()
        self.stats_extractor = StatsExtractor()

    def run(self, sport_key: str = "soccer_spain_la_liga", league_id: int = 140) -> dict[str, int]:
        """
        Execute full ETL pipeline.
        Returns dict with counts of inserted/updated records.
        """
        logger.info(f"Starting ETL pipeline for {sport_key}")
        stats = {"matches": 0, "odds": 0, "stats": 0}

        try:
            # STEP A: Extract stats and get valid fixtures
            logger.info("Step A: Extracting fixtures and stats from API-Football")
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            fixtures_df = self.stats_extractor.get_fixtures_by_date(today, league_id=league_id)

            if fixtures_df.empty:
                logger.warning("No fixtures found, aborting pipeline")
                return stats

            # Filter out postponed/cancelled matches
            valid_statuses = ["NS", "TBD"]  # Not Started, To Be Determined
            fixtures_df = fixtures_df[fixtures_df["status"].isin(valid_statuses)]
            logger.info(f"Valid fixtures after filtering: {len(fixtures_df)}")

            if fixtures_df.empty:
                logger.warning("No valid fixtures after filtering, aborting")
                return stats

            # STEP B: Extract odds data
            logger.info("Step B: Extracting odds from The Odds API")
            odds_df = self.odds_extractor.extract(sport_key)

            if odds_df.empty:
                logger.warning("No odds data found, aborting pipeline")
                return stats

            odds_events_df = self.odds_extractor.get_unique_events(odds_df)

            # STEP C: Match fixtures to odds events using fuzzy matching
            logger.info("Step C: Matching fixtures to odds events")
            matched_fixtures = self.stats_extractor.match_fixtures_to_odds(
                fixtures_df, odds_events_df
            )

            if matched_fixtures.empty:
                logger.warning("No matches found between fixtures and odds, aborting")
                return stats

            # STEP D: Transform and merge data
            logger.info("Step D: Transforming and merging data")
            merged_odds = self._merge_odds_with_fixtures(odds_df, matched_fixtures)
            merged_stats = self._extract_match_stats(matched_fixtures)

            # STEP E: Load to database with UPSERT
            logger.info("Step E: Loading data to database")
            stats["matches"] = self._upsert_matches(matched_fixtures)
            stats["odds"] = self._upsert_odds(merged_odds)
            stats["stats"] = self._upsert_stats(merged_stats)

            logger.info(f"Pipeline completed successfully: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    def _merge_odds_with_fixtures(
        self, odds_df: pd.DataFrame, matched_fixtures: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge odds DataFrame with matched fixture IDs."""
        event_to_fixture: dict[str, int] = dict(
            zip(
                matched_fixtures["matched_event_id"].astype(str),
                matched_fixtures["fixture_id"].astype(int),
            )
        )

        odds_df = odds_df.copy()
        odds_df["fixture_id"] = odds_df["event_id"].astype(str).map(event_to_fixture)
        odds_df = odds_df.dropna(subset=["fixture_id"])

        logger.info(f"Merged {len(odds_df)} odds records with fixture IDs")
        return odds_df

    def _extract_match_stats(self, matched_fixtures: pd.DataFrame) -> pd.DataFrame:
        """Extract pre-match stats for matched fixtures."""
        all_stats = []

        for _, fixture in matched_fixtures.iterrows():
            fixture_id = int(fixture["fixture_id"])  # type: ignore[arg-type]
            try:
                stats_df = self.stats_extractor.get_pre_match_stats(fixture_id)
                if not stats_df.empty:
                    all_stats.append(stats_df)
            except Exception as e:
                logger.warning(f"Failed to fetch stats for fixture {fixture_id}: {e}")
                continue

        if not all_stats:
            return pd.DataFrame()

        return pd.concat(all_stats, ignore_index=True)

    def _upsert_matches(self, fixtures_df: pd.DataFrame) -> int:
        """Insert or update matches in database. Returns count of upserted records."""
        session = self.SessionLocal()
        count = 0

        try:
            for _, row in fixtures_df.iterrows():
                league_name = str(row.get("league_name", "Unknown"))
                home_team_name = str(row.get("home_team_name", "Unknown"))
                away_team_name = str(row.get("away_team_name", "Unknown"))
                fixture_id = str(row.get("fixture_id"))
                status = str(row.get("status", "scheduled"))
                date_val = row.get("date")

                league = self._get_or_create_league(
                    session,
                    external_id=str(row.get("league_id")),
                    name=league_name,
                    sport="soccer",
                )

                home_team = self._get_or_create_team(
                    session,
                    external_id=str(row.get("home_team_id")),
                    name=home_team_name,
                )
                away_team = self._get_or_create_team(
                    session,
                    external_id=str(row.get("away_team_id")),
                    name=away_team_name,
                )

                kickoff = pd.to_datetime(date_val, utc=True)

                match = session.query(Match).filter_by(external_id=fixture_id).first()

                if match:
                    match.status = status
                    match.kickoff = kickoff
                    match.updated_at = datetime.now(UTC)
                else:
                    match = Match(
                        external_id=fixture_id,
                        league_id=league.id,
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        kickoff=kickoff,
                        status=status,
                        season=str(row.get("season", "")),
                        round=row.get("round"),
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
                fixture_id_str = str(int(row["fixture_id"]))  # type: ignore[arg-type]
                match = session.query(Match).filter_by(external_id=fixture_id_str).first()
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

    def _upsert_stats(self, stats_df: pd.DataFrame) -> int:
        """Insert match stats with UPSERT logic. Returns count."""
        if stats_df.empty:
            return 0

        session = self.SessionLocal()
        count = 0

        try:
            for _, row in stats_df.iterrows():
                fixture_id_str = str(int(row["fixture_id"]))  # type: ignore[arg-type]
                match = session.query(Match).filter_by(external_id=fixture_id_str).first()
                if not match:
                    continue

                stat_key = str(row["stat_key"])
                period = row.get("period")
                recorded_at = row["recorded_at"]
                stat_value = row["stat_value"]
                raw_data = row.get("raw_api_data")

                existing = (
                    session.query(MatchStat)
                    .filter_by(
                        match_id=match.id,
                        stat_key=stat_key,
                        period=period,
                        recorded_at=recorded_at,
                    )
                    .first()
                )

                if existing:
                    existing.stat_value = stat_value
                    existing.raw_api_data = raw_data
                else:
                    stat_record = MatchStat(
                        match_id=match.id,
                        stat_key=stat_key,
                        period=period,
                        stat_value=stat_value,
                        recorded_at=recorded_at,
                        raw_api_data=raw_data,
                    )
                    session.add(stat_record)

                count += 1

            session.commit()
            logger.info(f"Upserted {count} stat records")
            return count

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to upsert stats: {e}")
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
