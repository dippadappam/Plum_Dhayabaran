"""Application logging setup.

Standard-library logging only (no third-party deps): a single app-level logger
(``claims``) writing a consistent, structured ``key=value`` line format. Setup
is idempotent so the repeated ``app.api`` reloads the test suite performs never
stack duplicate handlers.

This module is observability layered on top of the engine. No log statement
participates in — or may change — any decision, amount, trace, or control flow.
"""

import logging
import sys

LOGGER_NAME = "claims"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the app-level ``claims`` logger.

    Idempotent: a second call (e.g. after ``importlib.reload(app.api)``) does
    not add a second handler, so log lines never double. The logger stands
    alone (``propagate = False``) so it does not also bubble to the root logger
    that pytest/uvicorn configure, which would duplicate every line.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def get_logger(name: str) -> logging.Logger:
    """A child logger under the app logger (e.g. ``get_logger("api")`` ->
    ``claims.api``). Child records propagate to the configured ``claims``
    handler."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
