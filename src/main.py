import logging
import os
import sys

from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("bot")


def main() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL not set. Check your .env file.")
        sys.exit(1)

    engine = create_engine(database_url)

    with engine.connect() as conn:
        result = conn.execute(text("SELECT version();"))
        version = result.scalar()
        logger.info("Connected to database: %s", version)

    logger.info("Sports Betting Bot initialized successfully.")


if __name__ == "__main__":
    main()
