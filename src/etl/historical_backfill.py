"""
Historical Backfill Module
Loads historical match data from football-data.co.uk CSVs into PostgreSQL.
CSVs contain: dates, teams, scores, odds (B365, PS, etc).
Uses Polars for fast parsing and UPSERTs into database.
"""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import League, Match, Team
from src.utils.config import DATA_RAW_DIR, DATABASE_URL
from src.utils.logger import logger

LEAGUE_CSV_MAP: dict[str, str] = {
    "Premier League": "premier_league",
    "La Liga": "la_liga",
    "Bundesliga": "bundesliga",
    "Serie A": "serie_a",
    "Ligue 1": "ligue_1",
}


class HistoricalBackfill:
    """
    Loads football-data.co.uk CSVs into PostgreSQL.
    CSVs contain match results and betting odds.
    """

    def __init__(self, db_url: str = DATABASE_URL) -> None:
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.data_dir = DATA_RAW_DIR / "historical"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _parse_csv(self, csv_path: Path) -> pl.DataFrame:
        """Parse CSV using Polars."""
        try:
            df = pl.read_csv(csv_path, ignore_errors=True, null_values=["", "NA", "-"])
            logger.info(f"Parsed {len(df)} rows from {csv_path.name}")
            return df
        except Exception as e:
            logger.error(f"Failed to parse {csv_path}: {e}")
            return pl.DataFrame()

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date from CSV (DD/MM/YY or DD/MM/YYYY)."""
        for fmt in ("%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(date_str, fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
        logger.warning(f"Cannot parse date: {date_str}")
        return datetime.now(UTC)

    def _get_or_create_league(self, session, league_name: str) -> League:
        league = session.query(League).filter_by(name=league_name).first()
        if not league:
            league = League(
                external_id=league_name.lower().replace(" ", "_"),
                name=league_name,
                sport="soccer",
            )
            session.add(league)
            session.flush()
        return league

    def _get_or_create_team(self, session, team_name: str) -> Team:
        team = session.query(Team).filter_by(name=team_name).first()
        if not team:
            team = Team(
                external_id=team_name.lower().replace(" ", "_"),
                name=team_name,
            )
            session.add(team)
            session.flush()
        return team

    def _upsert_matches(self, df: pl.DataFrame, league_name: str, season: str) -> int:
        """UPSERT matches into PostgreSQL."""
        if df.is_empty():
            return 0

        session = self.SessionLocal()
        count = 0

        try:
            league = self._get_or_create_league(session, league_name)

            for row in df.iter_rows(named=True):
                home_name = row.get("HomeTeam") or row.get("Home")
                away_name = row.get("AwayTeam") or row.get("Away")
                date_raw = row.get("Date")

                if not home_name or not away_name or not date_raw:
                    continue

                parsed_date = self._parse_date(date_raw)
                home_score = row.get("FTHG") or row.get("HG")
                away_score = row.get("FTAG") or row.get("AG")

                home_team = self._get_or_create_team(session, home_name)
                away_team = self._get_or_create_team(session, away_name)

                date_str = parsed_date.strftime("%Y-%m-%d")
                external_id = f"{date_str}_{home_name}_{away_name}"

                existing = session.query(Match).filter_by(external_id=external_id).first()

                if existing:
                    existing.home_score = home_score
                    existing.away_score = away_score
                    existing.updated_at = datetime.now(UTC)
                else:
                    match = Match(
                        external_id=external_id,
                        league_id=league.id,
                        home_team_id=home_team.id,
                        away_team_id=away_team.id,
                        kickoff=parsed_date,
                        status="finished",
                        home_score=home_score,
                        away_score=away_score,
                        season=season,
                    )
                    session.add(match)

                count += 1

            session.commit()
            logger.info(f"Upserted {count} matches for {league_name} {season}")
            return count

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to upsert {league_name} {season}: {e}")
            return 0
        finally:
            session.close()

    def run(
        self,
        leagues: list[str] | None = None,
        seasons: list[str] | None = None,
    ) -> dict[str, int]:
        """
        Run full backfill: load CSVs and UPSERT to database.
        """
        if leagues is None:
            leagues = list(LEAGUE_CSV_MAP.keys())
        if seasons is None:
            seasons = ["2023-2024", "2022-2023", "2021-2022"]

        logger.info(f"Starting backfill: leagues={leagues}, seasons={seasons}")
        stats: dict[str, int] = {}

        for league_name in leagues:
            league_count = 0
            csv_prefix = LEAGUE_CSV_MAP.get(league_name, league_name.lower().replace(" ", "_"))

            for season in seasons:
                csv_filename = f"{csv_prefix}_{season}.csv"
                csv_path = self.data_dir / csv_filename

                if not csv_path.exists():
                    logger.warning(f"CSV not found: {csv_path}")
                    continue

                df = self._parse_csv(csv_path)
                if df.is_empty():
                    continue

                count = self._upsert_matches(df, league_name, season)
                league_count += count

            stats[league_name] = league_count
            logger.info(f"{league_name}: {league_count} matches loaded")

        total = sum(stats.values())
        logger.info(f"Backfill complete. Total: {total} matches")
        return stats


def main() -> None:
    """Run historical backfill."""
    backfill = HistoricalBackfill()
    stats = backfill.run()
    logger.info(f"Loaded matches: {stats}")


if __name__ == "__main__":
    main()
