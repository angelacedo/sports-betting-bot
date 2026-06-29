"""
Football Data API Extractor
Extracts historical match data from football-data.org (free tier).
Provides match results, odds, and statistics for backfilling.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.utils.config import DATA_RAW_DIR
from src.utils.logger import logger

API_BASE_URL = "https://api.football-data.org/v4"
CACHE_DIR = DATA_RAW_DIR / "football_data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LEAGUE_CODES = {
    "Premier League": "PL",
    "La Liga": "PD",
    "Bundesliga": "BL1",
    "Serie A": "SA",
    "Ligue 1": "FL1",
}


class FootballDataExtractor:
    """Extracts historical data from football-data.org API."""

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key
        self.headers = {"X-Auth-Token": api_key} if api_key else {}

    def _get_cache_path(self, endpoint: str, params: dict) -> Path:
        """Generate cache file path."""
        key = f"{endpoint}_{'_'.join(f'{k}={v}' for k, v in sorted(params.items()))}"
        safe_key = key.replace("/", "_").replace("?", "_")[:100]
        return CACHE_DIR / f"{safe_key}.json"

    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """Check if cache exists and is fresh."""
        if not cache_path.exists():
            return False
        file_age_hours = (datetime.now(UTC).timestamp() - cache_path.stat().st_mtime) / 3600
        return file_age_hours < max_age_hours

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True,
    )
    def _fetch(self, endpoint: str, params: dict) -> dict[str, Any]:
        """Fetch data from API with retry logic."""
        url = f"{API_BASE_URL}/{endpoint}"
        logger.info(f"Fetching {url} with params {params}")

        response = requests.get(url, headers=self.headers, params=params, timeout=30)
        response.raise_for_status()

        remaining = response.headers.get("x-requests-available-minute")
        if remaining:
            logger.info(f"Requests remaining: {remaining}")

        return response.json()

    def get_matches(
        self, league: str, season: int, date_from: str | None = None, date_to: str | None = None
    ) -> pl.DataFrame:
        """
        Get matches for a league/season.
        Returns DataFrame with match data including scores and odds.
        """
        league_code = LEAGUE_CODES.get(league)
        if not league_code:
            logger.error(f"Unknown league: {league}")
            return pl.DataFrame()

        params: dict[str, Any] = {"season": season}
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to

        cache_path = self._get_cache_path(f"competitions/{league_code}/matches", params)

        if self._is_cache_valid(cache_path):
            logger.info(f"Using cached data: {cache_path}")
            with open(cache_path) as f:
                data = json.load(f)
        else:
            try:
                data = self._fetch(f"competitions/{league_code}/matches", params)
                with open(cache_path, "w") as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Cached to {cache_path}")
            except Exception as e:
                logger.error(f"Failed to fetch matches: {e}")
                return pl.DataFrame()

        return self._normalize_matches(data, league, season)

    def _normalize_matches(self, data: dict, league: str, season: int) -> pl.DataFrame:
        """Normalize API response to DataFrame."""
        matches = data.get("matches", [])
        if not matches:
            return pl.DataFrame()

        records = []
        for match in matches:
            score = match.get("score", {})
            full_time = score.get("fullTime", {})

            record = {
                "match_id": match.get("id"),
                "league": league,
                "season": season,
                "date": match.get("utcDate", ""),
                "home_team": match.get("homeTeam", {}).get("name", ""),
                "away_team": match.get("awayTeam", {}).get("name", ""),
                "home_goals": full_time.get("home"),
                "away_goals": full_time.get("away"),
                "status": match.get("status", ""),
            }
            records.append(record)

        df = pl.DataFrame(records)
        logger.info(f"Normalized {len(df)} matches")
        return df

    def get_standings(self, league: str, season: int) -> pl.DataFrame:
        """Get league standings."""
        league_code = LEAGUE_CODES.get(league)
        if not league_code:
            return pl.DataFrame()

        params = {"season": season}
        cache_path = self._get_cache_path(f"competitions/{league_code}/standings", params)

        if self._is_cache_valid(cache_path):
            with open(cache_path) as f:
                data = json.load(f)
        else:
            try:
                data = self._fetch(f"competitions/{league_code}/standings", params)
                with open(cache_path, "w") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to fetch standings: {e}")
                return pl.DataFrame()

        return self._normalize_standings(data, league, season)

    def _normalize_standings(self, data: dict, league: str, season: int) -> pl.DataFrame:
        """Normalize standings data."""
        standings = data.get("standings", [])
        if not standings:
            return pl.DataFrame()

        total_table = next((s for s in standings if s.get("type") == "TOTAL"), None)
        if not total_table:
            return pl.DataFrame()

        records = []
        for entry in total_table.get("table", []):
            record = {
                "league": league,
                "season": season,
                "position": entry.get("position"),
                "team": entry.get("team", {}).get("name", ""),
                "played": entry.get("playedGames"),
                "won": entry.get("won"),
                "draw": entry.get("draw"),
                "lost": entry.get("lost"),
                "goals_for": entry.get("goalsFor"),
                "goals_against": entry.get("goalsAgainst"),
                "goal_difference": entry.get("goalDifference"),
                "points": entry.get("points"),
            }
            records.append(record)

        return pl.DataFrame(records)


def main() -> None:
    """Test extractor."""
    extractor = FootballDataExtractor()
    df = extractor.get_matches("Premier League", 2024)
    logger.info(f"Extracted {len(df)} matches")
    if not df.is_empty():
        logger.info(df.head(10))


if __name__ == "__main__":
    main()
