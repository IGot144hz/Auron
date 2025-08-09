"""
Web UI package for the Auron assistant.

This package provides a simple Flask application exposing a local
web interface to interact with the assistant.  Users can send
commands, view conversation history, toggle subsystems (voice, TTS,
Discord) and inspect logs via the browser.

Use `python -m auron.web.flask_app` or run with `--web` flag from
``main.py`` to launch the web UI.
"""

from __future__ import annotations

__all__ = []  # Nothing exported at package level