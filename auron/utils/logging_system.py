# utils/logging_system.py
import logging
import os
import sys

# ANSI color codes used only when Rich is unavailable and stdout is a TTY
_COLOR = {
    "DEBUG": "\033[37m",  # white
    "INFO": "\033[36m",  # cyan
    "WARNING": "\033[33m",  # yellow
    "ERROR": "\033[31m",  # red
    "CRITICAL": "\033[41m",  # red background
    "RESET": "\033[0m",
}


class _ColoredFormatter(logging.Formatter):
    """Basic ANSI-colored formatter used as a fallback if Rich is not available."""

    def format(self, record: logging.LogRecord) -> str:
        color = _COLOR.get(record.levelname, "")
        reset = _COLOR["RESET"]
        message = super().format(record)
        return f"{color}{message}{reset}"


def setup_log_system(name: str, *, level: str | None = None) -> logging.Logger:
    """
    Create (or return) a configured logger.

    - Honors LOG_LEVEL env var (default INFO) unless a `level` is explicitly passed.
    - Uses RichHandler when available and stdout is a TTY (and NO_COLOR is not set).
    - Avoids duplicate handlers if called multiple times for the same logger.
    """
    level_str = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, level_str, logging.INFO)

    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(log_level)
        return logger

    no_color = os.getenv("NO_COLOR") is not None
    is_tty = sys.stdout.isatty()

    handler: logging.Handler
    if not no_color and is_tty:
        try:
            from rich.logging import RichHandler  # type: ignore

            handler = RichHandler(  # pretty console logs
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
            # Fallback to plain StreamHandler with ANSI colors
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
    logger.setLevel(log_level)
    logger.propagate = False
    return logger


# Convenience alias for users who type faster
get_logger = setup_log_system
