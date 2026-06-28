# Historical Backfill & Feature Engineering Guide

## Overview
This module downloads historical match data, loads it into PostgreSQL, and generates ML training features.

## Files Created

✓ `src/etl/historical_backfill.py` - Downloads and loads historical CSVs into PostgreSQL
✓ `src/models/feature_builder.py` - Generates ML features from match data
✓ `data/historical/download_datasets.py` - Standalone downloader with progress bars
✓ `requirements.txt` - Updated with polars, xgboost, thefuzz[speedup], tqdm

## Datasets Downloaded

15 CSV files from football-data.co.uk (3 seasons × 5 leagues):
- La Liga (2021-2022, 2022-2023, 2023-2024)
- Premier League (2021-2022, 2022-2023, 2023-2024)
- Bundesliga (2021-2022, 2022-2023, 2023-2024)
- Serie A (2021-2022, 2022-2023, 2023-2024)
- Ligue 1 (2021-2022, 2022-2023, 2023-2024)

Location: `data/raw/historical/`

## Execution Flow

### Prerequisites
1. Start PostgreSQL:
   ```bash
   cd docker
   docker compose up -d db
   ```

2. Install dependencies:
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Step 1: Download Historical Datasets
```bash
python data/historical/download_datasets.py
```
Downloads 15 CSV files to `data/raw/historical/`

### Step 2: Load Data into PostgreSQL
```bash
python -m src.etl.historical_backfill
```
Parses CSVs with Polars and loads into database with UPSERT logic.

### Step 3: Generate Training Features
```bash
python -m src.models.feature_builder
```
Calculates:
- Rolling averages (5, 10 matches) for goals
- Elo ratings and rating differences
- Fatigue metrics (days of rest)
- Home/away performance
- Match results (H/D/A) and total goals

Output: `data/processed/training_dataset.parquet`

## Features Generated

| Feature | Description |
|---------|-------------|
| `home_elo_before` | Home team Elo rating before match |
| `away_elo_before` | Away team Elo rating before match |
| `elo_diff` | Difference in Elo ratings |
| `days_rest` | Days since last match (fatigue) |
| `home_team_form` | Home team rolling avg goals (last 5) |
| `away_team_form` | Away team rolling avg goals (last 5) |
| `home_win_rate` | Home team win rate (last 5 home matches) |
| `away_win_rate` | Away team win rate (last 5 away matches) |
| `match_result` | Target: H (Home win), D (Draw), A (Away win) |
| `total_goals` | Target: Total goals in match |

## Expected Output
- ~5,000+ matches across 3 seasons and 5 leagues
- Parquet file with all features and targets
- Ready for XGBoost model training

## Troubleshooting

### PostgreSQL Connection Error
```
psycopg2.OperationalError: connection to server at "localhost" port 5432 failed
```
**Solution:** Start PostgreSQL with `docker compose up -d db`

### La Liga CSV Files Are HTML
The files `la_liga_*.csv` may contain HTML error pages. This is because football-data.co.uk uses a different code for La Liga. The backfill will skip these files gracefully and continue with other leagues.

### No Data in Database
Ensure you've run the backfill before running feature_builder:
```bash
python -m src.etl.historical_backfill
```

## Next Steps
1. Train XGBoost model on `training_dataset.parquet`
2. Implement prediction pipeline for upcoming matches
3. Backtest strategy on historical data
