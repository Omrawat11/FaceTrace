"""Structured logging configuration with secret filtering."""

import logging
import re
import sys

_SENSITIVE_PATTERNS = [
    re.compile(r"0x[a-fA-F0-9]{64}"),  # 32-byte Ethereum private keys
    re.compile(r"[a-fA-F0-9]{64}"),    # Raw 64-char hex tokens
]


class SecretFilter(logging.Filter):
    """Filter that masks sensitive private keys or tokens from log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            msg = record.msg
            # Check for patterns that match private keys
            for pattern in _SENSITIVE_PATTERNS:
                msg = pattern.sub("[REDACTED_SECRET]", msg)
            record.msg = msg
        return True


def setup_logger(name: str = "facetrace", level: str = "INFO") -> logging.Logger:
    """Configure and return a standardized logger with secret filtering."""
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level.upper())
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.addFilter(SecretFilter())
        logger.addHandler(handler)

    return logger
