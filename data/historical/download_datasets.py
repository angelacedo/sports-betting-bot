"""
Historical Dataset Downloader
Downloads CSV files from football-data.co.uk for multiple leagues and seasons.
Uses retry logic and progress bars for robust downloading.
"""

import sys
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import DATA_RAW_DIR
from src.utils.logger import logger


# League codes mapping
LEAGUE_CODES = {
    "La Liga": "sp",
    "Premier League": "e0",
    "Bundesliga": "d1",
    "Serie A": "i1",
    "Ligue 1": "f1",
}

# Season codes
SEASONS = {
    "2023-2024": "2324",
    "2022-2023": "2223",
    "2021-2022": "2122",
}


def get_csv_url(league_code: str, season_code: str) -> str:
    """Generate CSV URL for a specific league and season."""
    return f"https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    reraise=True,
)
def download_file(url: str, output_path: Path) -> bool:
    """Download file with retry logic."""
    try:
        logger.info(f"Downloading {url}")
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        # Get total size if available
        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as f:
            if total_size:
                with tqdm(
                    total=total_size, unit="B", unit_scale=True, desc=output_path.name
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        logger.info(f"Saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        raise


def download_all_datasets(
    leagues: dict[str, str] | None = None,
    seasons: dict[str, str] | None = None,
) -> list[Path]:
    """
    Download all historical CSV datasets.
    Returns list of downloaded file paths.
    """
    if leagues is None:
        leagues = LEAGUE_CODES
    if seasons is None:
        seasons = SEASONS

    output_dir = DATA_RAW_DIR / "historical"
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files = []
    total_downloads = len(leagues) * len(seasons)

    logger.info(f"Starting download of {total_downloads} datasets")

    with tqdm(total=total_downloads, desc="Overall progress") as pbar:
        for league_name, league_code in leagues.items():
            for season_name, season_code in seasons.items():
                url = get_csv_url(league_code, season_code)
                filename = f"{league_name.lower().replace(' ', '_')}_{season_name}.csv"
                output_path = output_dir / filename

                try:
                    if output_path.exists():
                        logger.info(f"Skipping {filename} (already exists)")
                    else:
                        download_file(url, output_path)

                    downloaded_files.append(output_path)

                except Exception as e:
                    logger.error(f"Failed to download {league_name} {season_name}: {e}")

                pbar.update(1)

    logger.info(f"Downloaded {len(downloaded_files)}/{total_downloads} datasets")
    return downloaded_files


def main():
    """Download all historical datasets."""
    logger.info("Starting historical dataset download")
    files = download_all_datasets()

    print(f"\nDownloaded {len(files)} files:")
    for f in files:
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
