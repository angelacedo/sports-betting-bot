"""
API-Football Extractor
Extracts match statistics, context, and team data from API-Football (v3.football.api-sports.io).
Implements fuzzy matching for team names and respects rate limits.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from thefuzz import fuzz, process

from src.utils.config import (
    API_FOOTBALL_BASE_URL,
    API_FOOTBALL_KEY,
    CACHE_TTL_HOURS,
    DATA_RAW_DIR,
)
from src.utils.logger import logger


class StatsExtractor:
    """Extracts stats and context data from API-Football with caching."""

    def __init__(self, api_key: str = API_FOOTBALL_KEY):
        self.api_key = api_key
        self.base_url = API_FOOTBALL_BASE_URL
        self.cache_dir = DATA_RAW_DIR / "stats"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {
            "x-apisports-key": self.api_key,
        }
        self._request_count = 0  # Track daily API usage

    def _get_cache_path(self, endpoint: str, params: dict) -> Path:
        """Generate deterministic cache path based on endpoint and params."""
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        today = datetime.now(UTC).strftime("%Y%m%d")
        safe_name = f"{endpoint}_{param_str}_{today}".replace("/", "_")
        return self.cache_dir / f"{safe_name}.json"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache exists and is within TTL."""
        if not cache_path.exists():
            return False
        file_age_hours = (
            datetime.now(UTC).timestamp() - cache_path.stat().st_mtime
        ) / 3600
        return file_age_hours < CACHE_TTL_HOURS

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True,
    )
    def _make_request(self, endpoint: str, params: dict) -> dict:
        """Make API request with rate limit tracking and retry logic."""
        url = f"{self.base_url}/{endpoint}"

        logger.debug(f"API-Football request: {endpoint} {params}")
        response = requests.get(url, headers=self.headers, params=params, timeout=30)
        response.raise_for_status()

        self._request_count += 1
        remaining = response.headers.get("x-ratelimit-requests-remaining")
        if remaining:
            logger.info(f"API-Football - requests remaining today: {remaining}")

        data = response.json()

        # Check for API errors
        if data.get("errors"):
            logger.error(f"API-Football error: {data['errors']}")
            raise ValueError(f"API error: {data['errors']}")

        return data

    def _fuzzy_match_team(
        self, team_name: str, known_teams: list[str], threshold: int = 80
    ) -> str | None:
        """
        Fuzzy match team name against known list.
        Returns best match if similarity >= threshold, else None.
        """
        if not known_teams:
            return None

        # Exact match first
        if team_name in known_teams:
            return team_name

        # Fuzzy match
        match = process.extractOne(team_name, known_teams, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= threshold:
            logger.debug(f"Fuzzy match: '{team_name}' -> '{match[0]}' (score: {match[1]})")
            return match[0]

        logger.warning(f"No fuzzy match found for team: {team_name}")
        return None

    def get_fixtures_by_date(
        self,
        date: str,
        league_id: int = 140,  # La Liga default
        season: int | None = None,
    ) -> pd.DataFrame:
        """
        Fetch fixtures for a specific date and league.
        Returns DataFrame with match info and status.
        """
        if season is None:
            season = datetime.now().year

        params = {"date": date, "league": league_id, "season": season}
        cache_path = self._get_cache_path("fixtures", params)

        if self._is_cache_valid(cache_path):
            logger.info(f"Using cached fixtures: {cache_path}")
            with open(cache_path) as f:
                data = json.load(f)
        else:
            data = self._make_request("fixtures", params)
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)

        return self._normalize_fixtures(data)

    def _normalize_fixtures(self, data: dict) -> pd.DataFrame:
        """Normalize fixtures API response to DataFrame."""
        response = data.get("response", [])

        if not response:
            logger.warning("No fixtures returned")
            return pd.DataFrame()

        records = []
        for fixture_data in response:
            fixture = fixture_data.get("fixture", {})
            league = fixture_data.get("league", {})
            teams = fixture_data.get("teams", {})
            goals = fixture_data.get("goals", {})

            records.append(
                {
                    "fixture_id": fixture.get("id"),
                    "external_id": str(fixture.get("id")),
                    "date": fixture.get("date"),
                    "status": fixture.get("status", {}).get("short"),
                    "status_long": fixture.get("status", {}).get("long"),
                    "league_id": league.get("id"),
                    "league_name": league.get("name"),
                    "season": league.get("season"),
                    "round": league.get("round"),
                    "home_team_id": teams.get("home", {}).get("id"),
                    "home_team_name": teams.get("home", {}).get("name"),
                    "away_team_id": teams.get("away", {}).get("id"),
                    "away_team_name": teams.get("away", {}).get("name"),
                    "home_score": goals.get("home"),
                    "away_score": goals.get("away"),
                    "raw_api_data": fixture_data,
                }
            )

        df = pd.DataFrame(records)
        logger.info(f"Extracted {len(df)} fixtures")
        return df

    def get_pre_match_stats(self, fixture_id: int) -> pd.DataFrame:
        """
        Fetch pre-match statistics (predictions, standings, etc.) for a fixture.
        Returns DataFrame with various statistical indicators.
        """
        params = {"fixture": fixture_id}
        cache_path = self._get_cache_path("predictions", params)

        if self._is_cache_valid(cache_path):
            logger.info(f"Using cached predictions: {cache_path}")
            with open(cache_path) as f:
                data = json.load(f)
        else:
            data = self._make_request("predictions", params)
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)

        return self._normalize_predictions(fixture_id, data)

    def _normalize_predictions(self, fixture_id: int, data: dict) -> pd.DataFrame:
        """Normalize predictions API response."""
        response = data.get("response", {})

        if not response:
            return pd.DataFrame()

        records = []
        recorded_at = datetime.now(UTC)

        # Extract various prediction metrics
        predictions = response.get("predictions", {})
        if predictions:
            records.append(
                {
                    "fixture_id": fixture_id,
                    "stat_key": "predictions",
                    "stat_value": predictions,
                    "period": "pre_match",
                    "recorded_at": recorded_at,
                    "raw_api_data": response,
                }
            )

        # League standings context
        standings = response.get("league", {})
        if standings:
            records.append(
                {
                    "fixture_id": fixture_id,
                    "stat_key": "standings_context",
                    "stat_value": standings,
                    "period": "pre_match",
                    "recorded_at": recorded_at,
                    "raw_api_data": response,
                }
            )

        # Teams form
        teams = response.get("teams", {})
        if teams:
            records.append(
                {
                    "fixture_id": fixture_id,
                    "stat_key": "teams_form",
                    "stat_value": teams,
                    "period": "pre_match",
                    "recorded_at": recorded_at,
                    "raw_api_data": response,
                }
            )

        return pd.DataFrame(records)

    def match_fixtures_to_odds(
        self,
        fixtures_df: pd.DataFrame,
        odds_events_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Match fixtures from API-Football to odds events using fuzzy team name matching.
        Returns merged DataFrame with fixture_id mapped to event_id.
        """
        if fixtures_df.empty or odds_events_df.empty:
            logger.warning("Cannot match: one of the DataFrames is empty")
            return pd.DataFrame()

        # Get unique team names from odds
        odds_home_teams = odds_events_df["home_team"].unique().tolist()
        odds_away_teams = odds_events_df["away_team"].unique().tolist()

        matches = []
        for _, fixture in fixtures_df.iterrows():
            home_name = str(fixture["home_team_name"])
            away_name = str(fixture["away_team_name"])

            # Fuzzy match home team
            matched_home = self._fuzzy_match_team(home_name, odds_home_teams)
            matched_away = self._fuzzy_match_team(away_name, odds_away_teams)

            if matched_home and matched_away:
                # Find corresponding odds event
                odds_match = odds_events_df[
                    (odds_events_df["home_team"] == matched_home)
                    & (odds_events_df["away_team"] == matched_away)
                ]

                if not odds_match.empty:
                    match_record = fixture.to_dict()
                    match_record["matched_event_id"] = odds_match.iloc[0]["event_id"]
                    match_record["matched_home_team"] = matched_home
                    match_record["matched_away_team"] = matched_away
                    matches.append(match_record)
                    logger.debug(
                        f"Matched: {home_name} vs {away_name} -> {matched_home} vs {matched_away}"
                    )

        result_df = pd.DataFrame(matches)
        logger.info(f"Successfully matched {len(result_df)} fixtures to odds events")
        return result_df


def main():
    """Test extraction."""
    extractor = StatsExtractor()
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    fixtures = extractor.get_fixtures_by_date(today, league_id=140)
    print(f"Extracted {len(fixtures)} fixtures")
    if not fixtures.empty:
        print(fixtures.head())


if __name__ == "__main__":
    main()
