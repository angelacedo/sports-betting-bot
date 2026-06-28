"""
Historical Backfill Module
Downloads and loads historical match data from football-data.co.uk into PostgreSQL.
Uses Polars for fast CSV parsing and fuzzy matching for team name alignment.
"""

import re
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from thefuzz import fuzz, process

from src.database.models import League, Match, Team
from src.utils.config import DATA_RAW_DIR, DATABASE_URL
from src.utils.logger import logger


class HistoricalBackfill:
    """
    Downloads historical match data from football-data.co.uk and loads into PostgreSQL.
    Covers multiple leagues and seasons with fuzzy team name matching.
    """

    # League codes mapping for football-data.co.uk URLs
    LEAGUE_CODES = {
        "La Liga": "sp",
        "Premier League": "epl",
        "Bundesliga": "d1",
        "Serie A": "i1",
        "Ligue 1": "f1",
    }

    # Season URL patterns
    SEASON_PATTERNS = {
        "2023-2024": "2324",
        "2022-2023": "2223",
        "2021-2022": "2122",
        "2020-2021": "2021",
        "2019-2020": "1920",
    }

    def __init__(self, db_url: str = DATABASE_URL):
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.data_dir = DATA_RAW_DIR / "historical"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Team name mapping for fuzzy matching
        self.known_teams: dict[str, list[str]] = {}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True,
    )
    def _download_csv(self, url: str, output_path: Path) -> bool:
        """Download CSV file with retry logic."""
        try:
            logger.info(f"Downloading {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            raise

    def _get_csv_url(self, league_name: str, season: str) -> str:
        """Generate CSV URL for a specific league and season."""
        league_code = self.LEAGUE_CODES.get(league_name)
        if not league_code:
            raise ValueError(f"Unknown league: {league_name}")

        season_code = self.SEASON_PATTERNS.get(season)
        if not season_code:
            raise ValueError(f"Unknown season: {season}")

        # football-data.co.uk URL pattern
        base_url = "https://www.football-data.co.uk/mmz4281"
        return f"{base_url}/{season_code}/{league_code}.csv"

    def _parse_csv(self, csv_path: Path) -> pl.DataFrame:
        """Parse CSV using Polars for speed."""
        try:
            df = pl.read_csv(
                csv_path,
                ignore_errors=True,
                null_values=["", "NA", "-"],
            )
            logger.info(f"Parsed {len(df)} rows from {csv_path.name}")
            return df

        except Exception as e:
            logger.error(f"Failed to parse {csv_path}: {e}")
            return pl.DataFrame()

    def _fuzzy_match_team(
        self, team_name: str, known_teams: list[str], threshold: int = 80
    ) -> str | None:
        """Fuzzy match team name against known list."""
        if not known_teams:
            return None

        if team_name in known_teams:
            return team_name

        match = process.extractOne(team_name, known_teams, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= threshold:
            return match[0]

        return None

    def _normalize_team_name(self, team_name: str) -> str:
        """Normalize team name for better matching."""
        # Remove common suffixes/prefixes
        team_name = re.sub(r"\s+(FC|CF|SC|AC|AS|FC)$", "", team_name, flags=re.IGNORECASE)
        team_name = re.sub(r"^(FC|CF|SC|AC|AS)\s+", "", team_name, flags=re.IGNORECASE)
        return team_name.strip()

    def _parse_date(self, date_str: str, time_str: str | None = None) -> datetime:
        """Parse date and time from CSV."""
        try:
            # football-data.co.uk uses DD/MM/YY format
            date_formats = ["%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d"]

            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Cannot parse date: {date_str}")

            # Add time if available
            if time_str:
                try:
                    time_parts = time_str.split(":")
                    parsed_date = parsed_date.replace(
                        hour=int(time_parts[0]),
                        minute=int(time_parts[1]) if len(time_parts) > 1 else 0,
                    )
                except (ValueError, IndexError):
                    pass

            # Set timezone to UTC
            return parsed_date.replace(tzinfo=UTC)

        except Exception as e:
            logger.warning(f"Failed to parse date '{date_str}' time '{time_str}': {e}")
            return datetime.now(UTC)

    def _get_or_create_league(self, session, league_name: str) -> League:
        """Get or create league record."""
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
        """Get or create team record."""
        normalized_name = self._normalize_team_name(team_name)
        team = session.query(Team).filter_by(name=normalized_name).first()

        if not team:
            team = Team(
                external_id=normalized_name.lower().replace(" ", "_"),
                name=normalized_name,
            )
            session.add(team)
            session.flush()

        return team

    def _load_csv_to_db(self, csv_path: Path, league_name: str, season: str) -> int:
        """Load single CSV file into database."""
        df = self._parse_csv(csv_path)
        if df.is_empty():
            return 0

        session = self.SessionLocal()
        count = 0

        try:
            # Get or create league
            league = self._get_or_create_league(session, league_name)

            # Process each row
            for row in df.iter_rows(named=True):
                try:
                    # Extract match data
                    home_team_name = row.get("HomeTeam") or row.get("Home")
                    away_team_name = row.get("AwayTeam") or row.get("Away")
                    date_str = row.get("Date") or row.get("Div")
                    time_str = row.get("Time")

                    if not home_team_name or not away_team_name or not date_str:
                        continue

                    # Get or create teams
                    home_team = self._get_or_create_team(session, home_team_name)
                    away_team = self._get_or_create_team(session, away_team_name)

                    # Parse date
                    kickoff = self._parse_date(date_str, time_str)

                    # Extract scores
                    home_score = row.get("FTHG") or row.get("HG")
                    away_score = row.get("FTAG") or row.get("AG")

                    # Check for existing match (UPSERT logic)
                    external_id = f"{date_str}_{home_team_name}_{away_team_name}"
                    existing_match = session.query(Match).filter_by(external_id=external_id).first()

                    if existing_match:
                        # Update existing
                        existing_match.home_score = home_score
                        existing_match.away_score = away_score
                        existing_match.updated_at = datetime.now(UTC)
                    else:
                        # Create new
                        match = Match(
                            external_id=external_id,
                            league_id=league.id,
                            home_team_id=home_team.id,
                            away_team_id=away_team.id,
                            kickoff=kickoff,
                            status="finished",
                            home_score=home_score,
                            away_score=away_score,
                            season=season,
                        )
                        session.add(match)

                    count += 1

                except Exception as e:
                    logger.warning(f"Failed to process row: {e}")
                    continue

            session.commit()
            logger.info(f"Loaded {count} matches from {csv_path.name}")
            return count

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to load {csv_path.name}: {e}")
            return 0

        finally:
            session.close()

    def run(self, seasons: list[str] | None = None) -> dict[str, int]:
        """
        Run full historical backfill for specified seasons and leagues.
        Returns dict with counts of loaded matches per league.
        """
        if seasons is None:
            seasons = ["2023-2024", "2022-2023", "2021-2022"]

        logger.info(f"Starting historical backfill for seasons: {seasons}")
        stats: dict[str, int] = {}

        for league_name in self.LEAGUE_CODES.keys():
            league_count = 0
            logger.info(f"Processing {league_name}")

            for season in seasons:
                try:
                    # Generate URL and download CSV
                    url = self._get_csv_url(league_name, season)
                    csv_filename = f"{league_name.lower().replace(' ', '_')}_{season}.csv"
                    csv_path = self.data_dir / csv_filename

                    # Download if not exists
                    if not csv_path.exists():
                        self._download_csv(url, csv_path)

                    # Load to database
                    count = self._load_csv_to_db(csv_path, league_name, season)
                    league_count += count

                except Exception as e:
                    logger.error(f"Failed to process {league_name} {season}: {e}")
                    continue

            stats[league_name] = league_count
            logger.info(f"{league_name}: loaded {league_count} matches")

        total = sum(stats.values())
        logger.info(f"Historical backfill complete. Total matches: {total}")
        return stats


def main():
    """Run historical backfill."""
    backfill = HistoricalBackfill()
    stats = backfill.run()
    print(f"Loaded matches: {stats}")


if __name__ == "__main__":
    main()
