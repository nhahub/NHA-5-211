"""Structured logging setup for SMART IDS v3."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys


def get_logger(
    name: str,
    log_file: Path | None = None,
    level: str | None = None,
) -> logging.Logger:
    """Create or retrieve a configured logger with consistent formatting."""
    logger = logging.getLogger(name)
    resolved_level = (level or os.getenv("SMART_IDS_LOG_LEVEL", "INFO")).upper()
    logger.setLevel(getattr(logging, resolved_level, logging.INFO))

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        abs_target = str(log_file.resolve())
        existing_files = {
            str(Path(h.baseFilename).resolve()) for h in logger.handlers if isinstance(h, logging.FileHandler)
        }
        if abs_target not in existing_files:
            for h in list(logger.handlers):
                if isinstance(h, logging.FileHandler):
                    logger.removeHandler(h)
                    h.close()
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(file_handler)

    logger.propagate = False
    return logger
