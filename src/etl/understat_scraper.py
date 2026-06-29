"""
Understat xG Scraper
Scrapes expected goals (xG) data from understat.com using aiohttp + BeautifulSoup.
Extracts match-level xG from JavaScript variables embedded in HTML pages.
"""

import asyncio
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import polars as pl
from bs4 import BeautifulSoup

from src.utils.config import DATA_RAW_DIR
from src.utils.logger import logger

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

LEAGUE_URL_MAP: dict[str, str] = {
    "Premier League": "EPL",
    "La Liga": "La_liga",
    "Bundesliga": "Bundesliga",
    "Serie A": "Serie_A",
    "Ligue 1": "Ligue_1",
}

REQUEST_DELAY_MIN: float = 2.0
REQUEST_DELAY_MAX: float = 5.0


class UnderstatScraper:
    """Async scraper for understat.com xG data with ethical rate limiting."""

    def __init__(self) -> None:
        self.cache_dir = DATA_RAW_DIR / "understat"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {"User-Agent": USER_AGENT}

    def _get_cache_path(self, league: str, season: int) -> Path:
        return self.cache_dir / f"{league}_{season}.json"

    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        if not cache_path.exists():
            return False
        file_age_hours = (datetime.now(UTC).timestamp() - cache_path.stat().st_mtime) / 3600
        return file_age_hours < max_age_hours

    def _parse_js_variable(self, html: str, var_name: str) -> list[dict[str, Any]]:
        """Extract JSON data from JavaScript variable assignments in HTML."""
        pattern = rf"{var_name}\s*=\s*JSON\.parse\(\'(.*?)\'\)"
        match = re.search(pattern, html, re.DOTALL)
        if not match:
            logger.warning(f"Variable '{var_name}' not found in HTML")
            return []

        encoded = match.group(1)
        decoded = encoded.encode().decode("unicode_escape")
        return json.loads(decoded)

    async def _fetch_page(self, session: aiohttp.ClientSession, url: str) -> str:
        """Fetch a single page with rate limiting."""
        await asyncio.sleep(REQUEST_DELAY_MIN)
        async with session.get(
            url, headers=self.headers, timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            resp.raise_for_status()
            return await resp.text()

    async def _scrape_league_season(self, league: str, season: int) -> list[dict[str, Any]]:
        """Scrape all matches for a league/season from understat."""
        league_code = LEAGUE_URL_MAP.get(league)
        if not league_code:
            logger.error(f"Unknown league: {league}")
            return []

        url = f"https://understat.com/league/{league_code}/{season}"
        cache_path = self._get_cache_path(league, season)

        if self._is_cache_valid(cache_path):
            logger.info(f"Using cached understat data: {cache_path}")
            with open(cache_path) as f:
                return json.load(f)

        logger.info(f"Scraping understat: {url}")
        async with aiohttp.ClientSession() as session:
            html = await self._fetch_page(session, url)

        soup = BeautifulSoup(html, "html.parser")
        scripts = soup.find_all("script")

        dates_data: list[dict[str, Any]] = []
        for script in scripts:
            text = script.string or ""
            if "datesData" in text:
                dates_data = self._parse_js_variable(text, "datesData")
                break

        if not dates_data:
            logger.warning(f"No datesData found for {league} {season}")
            return []

        matches: list[dict[str, Any]] = []
        for day in dates_data:
            for match in day.get("matches", []):
                if match.get("isResult", False) or match.get("isLive", False):
                    matches.append(self._normalize_match(match, league, season))

        with open(cache_path, "w") as f:
            json.dump(matches, f, indent=2)
        logger.info(f"Cached {len(matches)} matches to {cache_path}")

        return matches

    def _normalize_match(self, match: dict[str, Any], league: str, season: int) -> dict[str, Any]:
        """Normalize a single match record from understat."""
        home = match.get("h", {})
        away = match.get("a", {})
        return {
            "date": match.get("datetime", {}).get("date", ""),
            "league": league,
            "season": season,
            "home_team": home.get("title", ""),
            "away_team": away.get("title", ""),
            "home_goals": int(home.get("goals", 0)),
            "away_goals": int(away.get("goals", 0)),
            "home_xg": float(home.get("xG", 0.0)),
            "away_xg": float(away.get("xG", 0.0)),
            "match_id": int(match.get("id", 0)),
        }

    async def scrape_all(
        self, leagues: list[str] | None = None, seasons: list[int] | None = None
    ) -> pl.DataFrame:
        """Scrape xG data for multiple leagues and seasons."""
        if leagues is None:
            leagues = list(LEAGUE_URL_MAP.keys())
        if seasons is None:
            seasons = [2024, 2023, 2022, 2021]

        all_matches: list[dict[str, Any]] = []

        for league in leagues:
            for season in seasons:
                try:
                    matches = await self._scrape_league_season(league, season)
                    all_matches.extend(matches)
                    logger.info(f"{league} {season}: {len(matches)} matches scraped")
                except Exception as e:
                    logger.error(f"Failed to scrape {league} {season}: {e}")
                    continue

        if not all_matches:
            logger.warning("No matches scraped")
            return pl.DataFrame()

        df = pl.DataFrame(all_matches)
        logger.info(f"Total xG records scraped: {len(df)}")
        return df

    def scrape(
        self, leagues: list[str] | None = None, seasons: list[int] | None = None
    ) -> pl.DataFrame:
        """Synchronous wrapper for scrape_all."""
        return asyncio.run(self.scrape_all(leagues, seasons))


def main() -> None:
    """Test scraper."""
    scraper = UnderstatScraper()
    df = scraper.scrape(leagues=["Premier League"], seasons=[2024])
    logger.info(f"Scraped {len(df)} matches")
    if not df.is_empty():
        logger.info(df.head(10))


if __name__ == "__main__":
    main()
