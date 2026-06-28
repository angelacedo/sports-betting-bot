"""
Configuration module - loads environment variables from .env file.
Uses python-dotenv for secure credential management.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")


# Database
DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "postgresql://bot:bot_secret@localhost:5432/sportsbot"
)

# API Keys
THE_ODDS_API_KEY: str = os.getenv("THE_ODDS_API_KEY", "")
API_FOOTBALL_KEY: str = os.getenv("API_FOOTBALL_KEY", "")

# Paths
PROJECT_ROOT: Path = _project_root
DATA_RAW_DIR: Path = _project_root / "data" / "raw"
DATA_PROCESSED_DIR: Path = _project_root / "data" / "processed"
LOGS_DIR: Path = _project_root / "logs"

# Ensure directories exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# API Configuration
THE_ODDS_API_BASE_URL: str = "https://api.the-odds-api.com/v4"
API_FOOTBALL_BASE_URL: str = "https://v3.football.api-sports.io"

# Rate limiting & caching
CACHE_TTL_HOURS: int = 6
API_FOOTBALL_DAILY_LIMIT: int = 100

# Sharp bookmakers (low margin, high limit) - used for true probability estimation
SHARP_BOOKMAKERS: set[str] = {
    "pinnacle",
    "sbobet",
    "betfair",
    "circa",
    "bookmaker",
}
