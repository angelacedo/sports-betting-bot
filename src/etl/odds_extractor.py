"""
The Odds API Extractor
Extracts real-time odds data from the-odds-api.com.
Filters by sport/league and returns structured odds with bookmaker classification.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.utils.config import (
    CACHE_TTL_HOURS,
    DATA_RAW_DIR,
    SHARP_BOOKMAKERS,
    THE_ODDS_API_BASE_URL,
    THE_ODDS_API_KEY,
)
from src.utils.logger import logger


class OddsExtractor:
    """Extracts odds data from The Odds API with caching and retry logic."""

    def __init__(self, api_key: str = THE_ODDS_API_KEY):
        self.api_key = api_key
        self.base_url = THE_ODDS_API_BASE_URL
        self.cache_dir = DATA_RAW_DIR / "odds"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, sport_key: str) -> Path:
        """Generate cache file path based on sport and date."""
        today = datetime.now(UTC).strftime("%Y%m%d")
        return self.cache_dir / f"{sport_key}_{today}.json"

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
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True,
    )
    def _fetch_odds(self, sport_key: str, regions: str = "eu,uk") -> list[dict[str, Any]]:
        """Fetch odds from API with retry logic."""
        url = f"{self.base_url}/sports/{sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": "h2h,spreads,totals",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }

        logger.info(f"Fetching odds from The Odds API: sport={sport_key}")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        # Check remaining requests
        remaining = response.headers.get("x-requests-remaining")
        if remaining:
            logger.info(f"The Odds API - requests remaining: {remaining}")

        return response.json()

    def extract(self, sport_key: str = "soccer_spain_la_liga") -> pd.DataFrame:
        """
        Extract odds data for a specific sport/league.
        Returns DataFrame with normalized odds data.
        """
        cache_path = self._get_cache_path(sport_key)

        # Check cache first
        if self._is_cache_valid(cache_path):
            logger.info(f"Using cached odds data: {cache_path}")
            with open(cache_path) as f:
                raw_data = json.load(f)
        else:
            # Fetch from API
            raw_data = self._fetch_odds(sport_key)
            # Save to cache
            with open(cache_path, "w") as f:
                json.dump(raw_data, f, indent=2)
            logger.info(f"Cached odds data to: {cache_path}")

        if not raw_data:
            logger.warning(f"No odds data returned for {sport_key}")
            return pd.DataFrame()

        # Normalize to DataFrame
        return self._normalize_odds(raw_data)

    def _normalize_odds(self, raw_data: list[dict]) -> pd.DataFrame:
        """
        Normalize raw API response to structured DataFrame.
        Extracts match info, bookmakers, and odds for each market.
        """
        records = []
        fetched_at = datetime.now(UTC)

        for event in raw_data:
            event_id = event.get("id")
            sport_key = event.get("sport_key")
            commence_time = event.get("commence_time")
            home_team = event.get("home_team")
            away_team = event.get("away_team")

            for bookmaker in event.get("bookmakers", []):
                bookmaker_name = bookmaker.get("key")
                is_sharp = bookmaker_name.lower() in SHARP_BOOKMAKERS

                for market in bookmaker.get("markets", []):
                    market_key = market.get("key")

                    for outcome in market.get("outcomes", []):
                        selection = outcome.get("name")
                        odds = outcome.get("price")
                        point = outcome.get("point")  # For spreads/totals

                        records.append(
                            {
                                "event_id": event_id,
                                "sport_key": sport_key,
                                "commence_time": commence_time,
                                "home_team": home_team,
                                "away_team": away_team,
                                "bookmaker": bookmaker_name,
                                "is_sharp": is_sharp,
                                "market": market_key,
                                "selection": selection,
                                "odds_decimal": odds,
                                "point": point,
                                "fetched_at": fetched_at,
                                "raw_api_data": {
                                    "event": event,
                                    "bookmaker": bookmaker,
                                    "market": market,
                                    "outcome": outcome,
                                },
                            }
                        )

        df = pd.DataFrame(records)
        logger.info(f"Extracted {len(df)} odds records")
        return df

    def get_unique_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract unique events from odds DataFrame."""
        if df.empty:
            return pd.DataFrame()

        events = df[
            ["event_id", "sport_key", "commence_time", "home_team", "away_team"]
        ].drop_duplicates()
        logger.info(f"Found {len(events)} unique events")
        return events


def main():
    """Test extraction."""
    extractor = OddsExtractor()
    df = extractor.extract("soccer_spain_la_liga")
    print(f"Extracted {len(df)} records")
    if not df.empty:
        print(df.head())


if __name__ == "__main__":
    main()
