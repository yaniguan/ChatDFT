"""
Structured Logging for ChatDFT
===============================

JSON-compatible structured logging for reproducibility and debugging.
All science modules should use ``get_logger(__name__)`` instead of ``print()``.

Usage
-----
    from science.core.logging import get_logger
    logger = get_logger(__name__)

    logger.info("Built graph", extra={"n_nodes": 36, "n_edges": 102})
    logger.warning("Voronoi fallback triggered", extra={"reason": "QhullError"})
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict


class _StructuredFormatter(logging.Formatter):
    """JSON-lines formatter for structured log output."""

    def format(self, record: logging.LogRecord) -> str:
        entry: Dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "module": record.module,
            "msg": record.getMessage(),
        }
        # Merge extra fields (skip standard LogRecord attributes)
        _SKIP = {
            "name", "msg", "args", "created", "relativeCreated", "exc_info",
            "exc_text", "stack_info", "lineno", "funcName", "pathname",
            "filename", "module", "thread", "threadName", "processName",
            "process", "message", "levelname", "levelno", "msecs",
            "taskName",
        }
        for k, v in record.__dict__.items():
            if k not in _SKIP and not k.startswith("_"):
                entry[k] = v
        if record.exc_info and record.exc_info[1]:
            entry["error"] = str(record.exc_info[1])
            entry["error_type"] = type(record.exc_info[1]).__name__
        return json.dumps(entry, default=str)


class _HumanFormatter(logging.Formatter):
    """Human-readable formatter for interactive use."""

    def format(self, record: logging.LogRecord) -> str:
        base = f"[{record.levelname:7s}] {record.module}: {record.getMessage()}"
        # Append extra context
        _SKIP = {
            "name", "msg", "args", "created", "relativeCreated", "exc_info",
            "exc_text", "stack_info", "lineno", "funcName", "pathname",
            "filename", "module", "thread", "threadName", "processName",
            "process", "message", "levelname", "levelno", "msecs",
            "taskName",
        }
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in _SKIP and not k.startswith("_")
        }
        if extras:
            ctx = " ".join(f"{k}={v}" for k, v in extras.items())
            base += f"  ({ctx})"
        return base


_LOG_MODE = "human"  # or "json"


def set_log_mode(mode: str) -> None:
    """Switch between 'human' (default) and 'json' log formats."""
    global _LOG_MODE
    _LOG_MODE = mode


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a ChatDFT structured logger.

    Parameters
    ----------
    name : str
        Module name (use ``__name__``).
    level : int
        Logging level (default: INFO).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(f"chatdft.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        if _LOG_MODE == "json":
            handler.setFormatter(_StructuredFormatter())
        else:
            handler.setFormatter(_HumanFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
