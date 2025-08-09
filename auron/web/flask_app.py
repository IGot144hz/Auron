"""
Flask application providing a web interface for the Auron assistant.

This module defines a simple REST API and serves a single page web app
allowing the user to interact with the assistant from their browser.
It exposes endpoints to send messages, toggle voice recognition,
text‑to‑speech and Discord integration, fetch the current status and
retrieve recent logs.  The page is built with Bootstrap 5 for a clean
layout and uses JavaScript ``fetch()`` calls to communicate with the
backend.
"""
from __future__ import annotations

import logging
import threading
from typing import List, Dict, Any

from flask import Flask, jsonify, render_template, request, Response, make_response

from ..assistant_controller import AssistantController

app = Flask(__name__, template_folder="templates")

# Suppress default HTTP request logging from Werkzeug to reduce noise in
# the log view.  The assistant itself logs important events separately.
import logging as _flask_logging
_flask_logging.getLogger("werkzeug").setLevel(_flask_logging.WARNING)

# Initialise the assistant controller once.  Start voice recognition
# automatically so voice commands can be recognised even when using the
# web interface.
controller = AssistantController()
controller.start_voice_recognition()

# In‑memory conversation history and log buffer.  In a real application,
# these would likely be stored externally or in a database.
CHAT_HISTORY: List[Dict[str, str]] = []
LOG_BUFFER: List[str] = []
LOG_BUFFER_MAX = 200  # store up to 200 lines


class WebLogHandler(logging.Handler):
    """Custom logging handler that appends log messages to LOG_BUFFER."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        LOG_BUFFER.append(msg)
        # Trim old logs
        if len(LOG_BUFFER) > LOG_BUFFER_MAX:
            del LOG_BUFFER[: len(LOG_BUFFER) - LOG_BUFFER_MAX]


# Attach our handler to the root logger
_web_log_handler = WebLogHandler()
_web_log_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S"))
root_logger = logging.getLogger()
# Remove any pre‑existing handlers on the root logger to avoid duplicate output
for _h in list(root_logger.handlers):
    root_logger.removeHandler(_h)
# Attach only the web log handler to the root logger
root_logger.addHandler(_web_log_handler)
# Set the root logger level to INFO so only informational messages and above
# propagate to the web UI.  DEBUG logs can still be enabled via LOG_LEVEL
# environment variable for individual loggers without overwhelming the web.
root_logger.setLevel(logging.INFO)


@app.route("/")
def index() -> str:
    """Serve the main web page."""
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def api_status() -> Any:
    """Return the current state of the subsystems and lengths of history/logs."""
    return jsonify(
        voice_enabled=controller.voice_enabled,
        tts_enabled=controller.tts_enabled,
        discord_enabled=controller.discord_bridge is not None,
        chat_length=len(CHAT_HISTORY),
        log_length=len(LOG_BUFFER),
    )


@app.route("/api/clear_chat", methods=["POST"])
def api_clear_chat() -> Any:
    """Clear the conversation history."""
    CHAT_HISTORY.clear()
    return jsonify(success=True)


@app.route("/api/clear_logs", methods=["POST"])
def api_clear_logs() -> Any:
    """Clear the log buffer."""
    LOG_BUFFER.clear()
    return jsonify(success=True)


@app.route("/api/shutdown", methods=["POST"])
def api_shutdown() -> Any:
    """Shut down the Flask development server."""
    # Immediately respond to the client before shutting down the assistant.
    def shutdown_system() -> None:
        try:
            # Stop subsystems gracefully
            controller.stop_voice_recognition()
            controller.stop_discord()
            # Fully stop the voice engine to release audio resources
            try:
                controller.engine.stop()
            except Exception:
                pass
        finally:
            # Shut down the Flask server
            func = request.environ.get('werkzeug.server.shutdown')
            if func:
                func()
            # Exit the entire process
            import os
            os._exit(0)
    threading.Thread(target=shutdown_system, daemon=True).start()
    return jsonify(message="Assistant shutting down…")


@app.route("/api/message", methods=["POST"])
def api_message() -> Any:
    """Receive a user message and return the assistant's reply."""
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify(error="Empty message"), 400
    # Append user message to history
    CHAT_HISTORY.append({"role": "User", "text": text})
    # Process via assistant on a separate thread to avoid blocking the server
    reply_holder: Dict[str, Any] = {}

    def _process() -> None:
        response = controller.handle_command(text)
        reply_holder["response"] = response or ""
        # Append assistant reply to history
        CHAT_HISTORY.append({"role": "Assistant", "text": reply_holder["response"]})

    thread = threading.Thread(target=_process, daemon=True)
    thread.start()
    thread.join()  # Wait synchronously for response to keep API simple
    return jsonify(reply=reply_holder.get("response", ""))


@app.route("/api/voice/toggle", methods=["POST"])
def api_toggle_voice() -> Any:
    """Toggle voice recognition on/off."""
    if controller.voice_enabled:
        controller.stop_voice_recognition()
    else:
        controller.start_voice_recognition()
    return jsonify(voice_enabled=controller.voice_enabled)


@app.route("/api/tts/toggle", methods=["POST"])
def api_toggle_tts() -> Any:
    """Toggle text‑to‑speech on/off."""
    controller.tts_enabled = not controller.tts_enabled
    return jsonify(tts_enabled=controller.tts_enabled)

# Restart the TTS engine on demand
@app.route("/api/tts/restart", methods=["POST"])
def api_restart_tts() -> Any:
    success = controller.restart_tts()
    return jsonify(success=success)

# Restart the language model on demand
@app.route("/api/llm/restart", methods=["POST"])
def api_restart_llm() -> Any:
    success = controller.restart_llm()
    return jsonify(success=success)


@app.route("/api/discord/toggle", methods=["POST"])
def api_toggle_discord() -> Any:
    """Toggle the Discord bridge on/off."""
    if controller.discord_bridge is None:
        controller.start_discord()
    else:
        controller.stop_discord()
    return jsonify(discord_enabled=controller.discord_bridge is not None)


@app.route("/api/chat", methods=["GET"])
def api_chat() -> Any:
    """Return the full conversation history."""
    return jsonify(history=CHAT_HISTORY)


@app.route("/api/logs", methods=["GET"])
def api_logs() -> Any:
    """Return the recent log lines."""
    return jsonify(logs=LOG_BUFFER)

# New endpoint: download logs as a plain text file.
@app.route("/api/logs/download", methods=["GET"])
def api_logs_download() -> Any:
    """Return the recent log lines as a downloadable text file."""
    # Join log lines with newlines.  Use LF by default.
    text = "\n".join(LOG_BUFFER)
    # Use make_response to create a proper Flask Response object
    resp = make_response(text)
    resp.headers.set("Content-Type", "text/plain")
    resp.headers.set(
        "Content-Disposition", "attachment; filename=auron_logs.txt"
    )
    return resp


def run_app(host: str = "127.0.0.1", port: int = 5000, debug: bool = False, *, auto_open: bool = True) -> None:
    """
    Run the Flask development server.  Intended to be called from main.

    Parameters
    ----------
    host, port: specify where the server should listen.
    debug: whether to enable Flask debugging (reloader disabled).
    auto_open: if True, open the default web browser to the application URL.
    """
    # Optionally open the browser after a short delay so the server is ready
    if auto_open:
        import webbrowser
        def _open() -> None:
            url = f"http://{host}:{port}"
            try:
                webbrowser.open(url)
            except Exception:
                pass
        # Use a timer so the call does not block the server startup
        threading.Timer(1.0, _open).start()
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    # Run with default parameters when executed directly
    run_app()