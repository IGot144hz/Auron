import logging
import os
import sys

# Define ANSI color codes for log levels (used if Rich is not available)
COLOR_CODES = {
    "DEBUG": "\033[37m",  # white
    "INFO": "\033[36m",  # cyan
    "WARNING": "\033[33m",  # yellow
    "ERROR": "\033[31m",  # red
    "CRITICAL": "\033[41m",  # red background for critical
    "RESET": "\033[0m",  # reset to default color
}


class ColoredFormatter(logging.Formatter):
    """
    Custom logging Formatter that injects ANSI color codes based on log level.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Pick color for this level, default to no color if level name not in map
        color = COLOR_CODES.get(record.levelname, "")
        reset = COLOR_CODES["RESET"]
        message = super().format(record)
        # Return colored message
        return f"{color}{message}{reset}"


def setup_log_sytem(name: str) -> logging.Logger:
    """
    Set up and return a logger with the given name, configured for console output.

    - Uses a colored console formatter for readability.
    - Respects the LOG_LEVEL environment variable (defaults to INFO).
    - If the Rich library is installed, uses RichHandler for enhanced output.
    - Ensures only one handler is added per logger (subsequent calls return the same logger).

    **Parameters:**
      - name (str): The name for the logger (typically __name__ of the module).

    **Returns:**
      - logging.Logger: Configured logger instance.
    """
    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    # Default to INFO if unrecognized level string
    level = getattr(logging, level_str, logging.INFO)

    logger = logging.getLogger(name)
    if logger.handlers:
        # Logger is already configured, return it to avoid adding extra handlers
        return logger

    # Determine if we should output color (disable color if NO_COLOR env or not a TTY)
    no_color = os.getenv("NO_COLOR")
    use_color = False if no_color is not None or not sys.stdout.isatty() else True

    try:
        # Try to use Rich for nice formatted logging if available
        from rich.logging import RichHandler

        handler = RichHandler(rich_tracebacks=True)
        # (RichHandler by default handles its own formatting and color)
    except ImportError:
        # Rich not installed, use standard StreamHandler with optional colors
        handler = logging.StreamHandler(sys.stdout)
        if use_color:
            formatter = ColoredFormatter(
                "[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S"
            )
        else:
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S"
            )
        handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False  # prevent double logging if root logger has handlers

    return logger
