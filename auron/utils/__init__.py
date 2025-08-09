"""Utility functions for Auron.

The ``logging_system`` module provides a configurable logging setup that
honours environment variables and falls back to simple coloured output when
Rich is unavailable.
"""

from .logging_system import setup_log_system, get_logger  # noqa: F401

__all__ = ["setup_log_system", "get_logger"]