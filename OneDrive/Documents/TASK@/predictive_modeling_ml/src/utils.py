"""Small shared helpers. Keep this file boring on purpose."""

import logging

from src.config import LOG_FORMAT


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a consistent format across the project."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger