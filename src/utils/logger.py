"""
Structured clinical reasoning logger built on structlog.

Provides subsystem-aware, JSON-compatible structured logging with
consistent field conventions across all pipeline stages.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def _configure_structlog() -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=logging.DEBUG,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=False)
            if not sys.stderr.isatty()
            else structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


_configured = False


def get_logger(name: str, **initial_context: Any) -> structlog.BoundLogger:
    """
    Return a subsystem-bound structured logger.

    Parameters
    ----------
    name:
        Typically __name__ of the calling module.
    **initial_context:
        Key-value pairs bound to every log call from this logger instance.
        Useful for tagging logs with subsystem, stage, or run_id.
    """
    global _configured
    if not _configured:
        _configure_structlog()
        _configured = True

    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger


def get_clinical_logger(
    subsystem: str,
    stage: int | None = None,
    run_id: str | None = None,
) -> structlog.BoundLogger:
    """
    Convenience wrapper that creates a logger pre-bound with clinical pipeline
    context fields (subsystem, stage, run_id).
    """
    ctx: dict[str, Any] = {"subsystem": subsystem}
    if stage is not None:
        ctx["stage"] = stage
    if run_id is not None:
        ctx["run_id"] = run_id
    return get_logger(subsystem, **ctx)
