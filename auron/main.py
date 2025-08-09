"""
Entry point for the Auron assistant.

Depending on command‑line flags, this module launches the assistant either
with a graphical user interface or as a headless process that listens for
voice commands.  The assistant controller coordinates voice recognition,
command routing, LLM integration, TTS output and Discord connectivity.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from .assistant_controller import AssistantController

# GUIApp (Tkinter) is no longer used in this version.  The assistant now
# defaults to a web‑based interface served by Flask.
GUIApp = None  # type: ignore


logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Launch the Auron assistant with either the web interface or headless mode."""
    parser = argparse.ArgumentParser(description="Run the Auron assistant")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without any user interface (CLI only)",
    )
    args = parser.parse_args(argv)

    if args.headless:
        # Headless mode: just start voice recognition and wait
        controller = AssistantController()
        controller.start_voice_recognition()
        logger.info("Running headless. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, shutting down…")
        finally:
            controller.stop_voice_recognition()
            controller.stop_discord()
            logger.info("Assistant shutdown.")
    else:
        # Default: start the web interface
        from .web.flask_app import run_app  # type: ignore
        logger.info("Starting web UI on http://127.0.0.1:5000 …")
        try:
            run_app()
        finally:
            logger.info("Web UI terminated. Shutting down subsystems…")


if __name__ == "__main__":
    main(sys.argv[1:])