"""
Logging configuration - centralized logging to file and console.
Logs are stored in /logs/etl.log with rotation.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

from src.utils.config import LOGS_DIR


def setup_logger(name: str = "etl", level: int = logging.INFO) -> logging.Logger:
    """Configure logger with file and console handlers."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler with rotation (10MB max, keep 5 backups)
    file_handler = RotatingFileHandler(
        LOGS_DIR / "etl.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Default logger instance
logger = setup_logger()
