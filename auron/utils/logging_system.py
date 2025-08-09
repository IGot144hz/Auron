"""
Custom logging setup used across the Auron assistant.

This module configures Python's logging module with sensible defaults.  It
supports colourised output via Rich when available and falls back to a
minimal ANSI coloured formatter otherwise.  The log level and colour
settings are controlled via environment variables.
"""
import logging
import os
import sys

# ANSI colour codes used only when Rich is unavailable and stdout is a TTY
_COLOR = {
    "DEBUG": "\033[37m",  # white
    "INFO": "\033[36m",  # cyan
    "WARNING": "\033[33m",  # yellow
    "ERROR": "\033[31m",  # red
    "CRITICAL": "\033[41m",  # red background
    "RESET": "\033[0m",
}


class _ColoredFormatter(logging.Formatter):
    """Basic ANSI‑coloured formatter used as a fallback if Rich is not available."""

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOR.get(record.levelname, "")
        reset = _COLOR["RESET"]
        message = super().format(record)
        return f"{colour}{message}{reset}"


def setup_log_system(name: str, *, level: str | None = None) -> logging.Logger:
    """
    Create (or return) a configured logger.

    - Honours LOG_LEVEL env var (default INFO) unless a ``level`` is explicitly passed.
    - Uses RichHandler when available and stdout is a TTY (and NO_COLOR is not set).
    - Avoids duplicate handlers if called multiple times for the same logger.
    """
    level_str = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, level_str, logging.INFO)

    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(log_level)
        return logger

    no_colour = os.getenv("NO_COLOR") is not None
    is_tty = sys.stdout.isatty()

    handler: logging.Handler
    if not no_colour and is_tty:
        try:
            from rich.logging import RichHandler  # type: ignore

            handler = RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_level=True,
                show_path=False,
                markup=False,
            )
            # RichHandler does its own formatting of time/level
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
        except Exception:
            # Fallback to plain StreamHandler with ANSI colours
            handler = logging.StreamHandler(sys.stdout)
            formatter = _ColoredFormatter(
                "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
            )
            handler.setFormatter(formatter)
    else:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)

    logger.addHandler(handler)
    # Configure the logger's level from the environment or explicit argument
    logger.setLevel(log_level)
    # Propagate log records to the root logger so that global handlers (e.g. web log)
    # also receive them.  Without propagation the WebLogHandler attached to the root
    # logger would never see messages from module‑specific loggers.  Duplicate
    # console output is avoided because only the module logger has a Stream/Rich handler.
    logger.propagate = True
    # Ensure the root logger's level is not more restrictive than this logger's level.
    # If the root logger level is higher (e.g. WARNING) it would filter out INFO
    # messages emitted before the root logger configuration.  By lowering it here,
    # we guarantee that early INFO logs are not lost.
    root_logger = logging.getLogger()
    if root_logger.level > log_level:
        root_logger.setLevel(log_level)
    return logger


# Convenience alias for users who type faster
get_logger = setup_log_system