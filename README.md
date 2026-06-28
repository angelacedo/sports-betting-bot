# Sports Betting Bot

Data-driven sports betting prediction system using Machine Learning.

> **Disclaimer:** This project is for **educational and data analysis purposes only**. It is not intended to provide financial advice or guarantee profits. Sports betting involves significant risk of financial loss.

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Database | PostgreSQL 15 |
| ORM | SQLAlchemy 2.0 |
| ML | scikit-learn, XGBoost (planned) |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions (Ruff + pytest) |
| Deployment | Hostinger VPS |

## Project Structure

```
src/
├── database/      # SQLAlchemy connections and models
├── etl/           # Data extraction (odds APIs, stats)
├── models/        # ML models (XGBoost, Poisson, etc.)
├── execution/     # Bet execution logic (Betfair API, etc.)
└── utils/         # Helpers (Kelly criterion, EV calculations)
```

## Quick Start

### 1. Clone and configure

```bash
git clone git@github.com:angelacedo/sports-betting-bot.git
cd sports-betting-bot
cp .env.example .env
# Edit .env with your API keys and database credentials
```

### 2. Run with Docker Compose

```bash
cd docker
docker compose up --build
```

This starts:
- **PostgreSQL 15** on port 5432 with the schema auto-loaded
- **Bot** container that verifies DB connectivity

### 3. Local development

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### 4. Run the ETL Pipeline

The ETL pipeline extracts data from both APIs, merges them, and loads to PostgreSQL:

```bash
# Set your API keys in .env first
python -m src.etl.pipeline
```

**Pipeline Steps:**
1. **Extract fixtures** from API-Football (match context, stats, status)
2. **Extract odds** from The Odds API (bookmakers, markets, prices)
3. **Fuzzy match** team names between both APIs
4. **Filter** postponed/cancelled matches
5. **UPSERT** to PostgreSQL (avoids duplicates on repeated runs)
6. **Cache** raw API responses in `/data/raw/` (respects rate limits)

**Caching:** API responses are cached locally for 6 hours to avoid hitting rate limits. Cache files are stored in `data/raw/odds/` and `data/raw/stats/`.

### 5. Run Tests

```bash
pytest tests/ -v
```

## Environment Variables

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `THE_ODDS_API_KEY` | The Odds API key |
| `API_FOOTBALL_KEY` | API-Football key |
| `TELEGRAM_BOT_TOKEN` | Telegram bot for alerts |
| `BETFAIR_APP_KEY` | Betfair exchange API key |

## Responsible Gambling

This project promotes responsible data analysis. If you or someone you know has a gambling problem, please seek help from professional organizations. Never bet more than you can afford to lose.

## License

Private. All rights reserved.
