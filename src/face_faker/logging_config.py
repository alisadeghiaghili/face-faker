"""Logging helpers for library and CLI use."""

from __future__ import annotations

import logging
from typing import Optional

LOGGER_NAME = "face_faker"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a namespaced logger.

    Args:
        name: Optional child logger name under ``face_faker``.

    Returns:
        Configured library logger instance.

    Example:
        >>> get_logger("test").name
        'face_faker.test'
    """
    if not name:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_cli_logging(verbose: bool = False) -> None:
    """Configure stderr logging for CLI entry points.

    Args:
        verbose: When true, emit DEBUG records; otherwise INFO.

    Example:
        >>> configure_cli_logging(False)
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )
