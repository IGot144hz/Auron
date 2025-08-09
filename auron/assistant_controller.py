"""
Central controller for the Auron assistant.

This module orchestrates voice recognition, command routing, language model
integration, text‑to‑speech, Discord connectivity and the optional GUI.  The
``AssistantController`` class exposes methods to start/stop voice listening and
Discord, register new internal commands and process arbitrary user inputs.
"""
from __future__ import annotations

import os
import threading
import webbrowser
from typing import Optional, Callable

try:
    # Use python‑dotenv if available to load .env files into the environment
    from dotenv import load_dotenv  # type: ignore[assignment]
except Exception:
    # Define a no‑op fallback when dotenv is not installed
    def load_dotenv(*args, **kwargs) -> None:  # type: ignore[override]
        return

from .utils.logging_system import setup_log_system
from .voice_recognition.voicekey_engine import VoiceKeyEngine
from .voice_recognition.stt_engine import STTEngine
from .commands import CommandRouter
from .llm import OllamaClient
from .tts import TTSPlayer

import logging

logger = setup_log_system("assistant_controller")


class AssistantController:
    """Coordinates all subsystems of the Auron assistant."""

    def __init__(self) -> None:
        # Load environment variables from .env if present
        load_dotenv()

        # Flags controlling subsystem activation
        self.voice_enabled: bool = False
        self.tts_enabled: bool = True
        self.discord_bridge: Optional["DiscordBridge"] = None
        self._discord_thread: Optional[threading.Thread] = None

        # Initialise subsystems
        # Speech‑to‑text engine (faster‑whisper)
        stt_model_size = os.getenv("STT_MODEL_SIZE", "medium")
        stt_device = os.getenv("STT_DEVICE", "auto")
        stt_compute_type = os.getenv("STT_COMPUTE_TYPE") or None
        self.stt = STTEngine(model_size=stt_model_size, device=stt_device, compute_type=stt_compute_type)

        # Wake word engine; the callback will be triggered on a separate thread
        self.engine = VoiceKeyEngine(self._on_wake, cooldown_seconds=1.0)

        # Concurrency lock to avoid overlapping voice processing
        self._busy = threading.Lock()

        # Command router and registration of built‑in commands
        self.router = CommandRouter()
        self._register_default_commands()

        # Language model client
        self.llm = OllamaClient()

        # Text‑to‑speech engine
        tts_voice = os.getenv("TTS_VOICE") or None
        self.tts = TTSPlayer(voice_name=tts_voice)

        # System prompt for LLM requests.  If not set in .env, fall back to a
        # sensible default instructing the model to be helpful, honest and
        # always answer in English (for compatibility with the TTS engine).
        default_prompt = (
            "You are Auron, an AI assistant that helps with everyday tasks. "
            "Always respond in English, be clear and concise, and never hallucinate. "
            "If you do not know the answer, say 'I don't know.'"
        )
        self.system_prompt: str = os.getenv("SYSTEM_PROMPT", default_prompt).strip()

    # ------------------------------------------------------------------
    # Voice recognition lifecycle
    # ------------------------------------------------------------------
    def start_voice_recognition(self) -> None:
        """Start listening for the wake word."""
        if not self.voice_enabled:
            try:
                # If the underlying engine has been fully stopped (stream deleted),
                # reinitialize it to ensure we have a working stream.  The
                # VoiceKeyEngine.start() call will be a no‑op if the stream
                # already exists and is inactive.  Using pause/resume instead
                # of stop prevents destruction of internal resources when
                # toggling voice recognition.
                if getattr(self.engine, "stream", None) is None:
                    # Recreate the engine with the same callback
                    self.engine = VoiceKeyEngine(self._on_wake, cooldown_seconds=1.0)
                self.engine.start()
                self.voice_enabled = True
                logger.info("Voice recognition enabled.")
            except Exception as e:
                logger.error(f"Failed to start voice recognition: {e}", exc_info=True)

    def stop_voice_recognition(self) -> None:
        """Stop listening for the wake word."""
        if self.voice_enabled:
            try:
                # Pause instead of fully stopping the engine.  Stopping destroys
                # the Porcupine stream and prevents restarting it later.  Pausing
                # will stop listening but keep resources intact so that a
                # subsequent call to start_voice_recognition() can resume.
                if getattr(self.engine, "pause", None):
                    self.engine.pause()
                else:
                    # Fallback to stop() if pause is unavailable
                    self.engine.stop()
            finally:
                self.voice_enabled = False
                logger.info("Voice recognition disabled.")

    # ------------------------------------------------------------------
    # Discord integration
    # ------------------------------------------------------------------
    def start_discord(self) -> None:
        """Start the Discord bridge if configured with a token."""
        if self.discord_bridge is not None:
            logger.debug("Discord bridge already running.")
            return
        token = os.getenv("DISCORD_TOKEN")
        if not token:
            logger.warning("Cannot start Discord bridge: DISCORD_TOKEN is not set.")
            return
        # Import here to avoid circular import at module level
        from .discord_bot.discord_bridge import DiscordBridge
        self.discord_bridge = DiscordBridge(assistant=self, token=token, speak=self.tts_enabled)
        # Run the bot in its own thread since run() blocks
        def _run() -> None:
            self.discord_bridge.run_bot()
        self._discord_thread = threading.Thread(target=_run, name="DiscordThread", daemon=True)
        self._discord_thread.start()
        logger.info("Discord bridge started.")

    def stop_discord(self) -> None:
        """Stop the Discord bridge if it is running."""
        if self.discord_bridge is None:
            return
        try:
            # discord.Client.close() stops the event loop gracefully
            self.discord_bridge.close()
        except Exception as e:
            logger.error(f"Failed to stop Discord bridge: {e}", exc_info=True)
        self.discord_bridge = None
        # The thread will exit once the loop is closed
        self._discord_thread = None
        logger.info("Discord bridge stopped.")

    # ------------------------------------------------------------------
    # Command processing
    # ------------------------------------------------------------------
    def _register_default_commands(self) -> None:
        """Register a set of example internal commands for demonstration."""

        def create_folders_handler(text: str) -> str:
            """Create a simple folder structure in the user's home directory."""
            base_dir = os.path.expanduser("~/auron_folders")
            subfolders = ["Documents", "Music", "Videos"]
            try:
                for name in subfolders:
                    os.makedirs(os.path.join(base_dir, name), exist_ok=True)
                logger.info(f"Created folder structure under {base_dir}")
                return f"Created folder structure under {base_dir}"  # speakable
            except Exception as e:
                logger.error(f"Failed to create folders: {e}", exc_info=True)
                return "An error occurred while creating folders."

        def open_youtube_handler(text: str) -> str:
            """Open YouTube in the default web browser."""
            try:
                webbrowser.open("https://www.youtube.com")
                logger.info("Opening YouTube in the default browser.")
                return "Opening YouTube."
            except Exception as e:
                logger.error(f"Failed to open YouTube: {e}", exc_info=True)
                return "An error occurred while opening YouTube."

        def play_spotify_handler(text: str) -> str:
            """Stub for playing a Spotify song (not implemented)."""
            logger.info("Received request to play a Spotify song (stub).")
            # Here you would integrate the Spotify API to play a specific song
            return "Playing Spotify song (not implemented yet)."

        # Register patterns with their handlers.  Use word boundaries to avoid
        # accidental matches.
        self.router.add_internal(r"\berstelle\s+ordnerstruktur\b", create_folders_handler)
        self.router.add_internal(r"\böffne\s+youtube\b", open_youtube_handler)
        self.router.add_internal(r"\bspiele\s+spotify\S*\s+song\b", play_spotify_handler)

        # Add a command to list available built‑in functions.  This responds to
        # queries such as "sage mir alle deine funktionen" or "zähle mir deine
        # funktionen" and returns a summary of the assistant's internal
        # capabilities.  The pattern is deliberately broad to catch common
        # phrasings.
        def list_functions_handler(_: str) -> str:
            return (
                "Ich kann aktuell einige interne Befehle ausführen: "
                "Ordnerstrukturen erstellen (\"erstelle ordnerstruktur\"), "
                "YouTube im Browser öffnen (\"öffne youtube\") und einen Spotify‑Song abspielen (\"spiele spotify … song\")."
            )
        self.router.add_internal(
            r"\b(?:sage|zähl(?:e)?)\s+mir\s+(?:alle\s+)?deine\s+(?:funktionen|fähigkeiten)\b",
            list_functions_handler,
        )
        # Also respond to "was kannst du" questions.
        self.router.add_internal(r"\bwas\s+kannst\s+du\b", list_functions_handler)

    def handle_command(self, text: str) -> Optional[str]:
        """
        Process a user command and return the assistant's response.

        This method routes the command through the ``CommandRouter``.  If an
        internal command matches, its handler is executed.  Otherwise the
        command is forwarded to the language model with the system prompt
        prepended.  The response is returned and, if TTS is enabled, also
        spoken.
        """
        logger.info(f"User command: {text}")
        target, payload = self.router.route(text)
        if target == "internal":
            response = payload or ""
            if response and self.tts_enabled:
                # Speak on a background thread to avoid blocking command handling.
                threading.Thread(target=self.tts.speak, args=(response,), daemon=True).start()
            return response
        # Otherwise send to LLM
        # Compose the full prompt with the system prefix
        full_prompt = f"{self.system_prompt}\n\nUser: {text}\nAssistant:"
        response = self.llm.generate(full_prompt)
        # Log the response for debugging
        logger.info(f"Assistant response: {response}")
        if response and self.tts_enabled:
            # Speak asynchronously to avoid blocking the caller (e.g. API request)
            threading.Thread(target=self.tts.speak, args=(response,), daemon=True).start()
        return response

    # ------------------------------------------------------------------
    # Restartable subsystems
    # ------------------------------------------------------------------
    def restart_tts(self) -> bool:
        """Reinitialise the TTS engine. Returns True if successful."""
        try:
            tts_voice = os.getenv("TTS_VOICE") or None
            self.tts = TTSPlayer(voice_name=tts_voice)
            logger.info("TTS engine restarted.")
            return True
        except Exception as e:
            logger.error(f"Failed to restart TTS engine: {e}", exc_info=True)
            return False

    def restart_llm(self) -> bool:
        """Reinitialise the language model client. Returns True if successful."""
        try:
            self.llm = OllamaClient()
            logger.info("LLM client restarted.")
            return True
        except Exception as e:
            logger.error(f"Failed to restart LLM client: {e}", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Wake word callback
    # ------------------------------------------------------------------
    def _on_wake(self) -> None:
        """
        Callback executed when the wake word is detected.

        This method offloads the recording and transcription pipeline to a
        worker thread to avoid blocking the audio callback.
        """
        if not self._busy.acquire(blocking=False):
            logger.debug("Wakeword detected but a command is already being processed. Ignoring.")
            return
        threading.Thread(target=self._process_wake_event, daemon=True).start()

    def _process_wake_event(self) -> None:
        """
        Record, transcribe and handle the user's speech after the wake word.
        """
        try:
            logger.info("Wake word recognized. Preparing to record command…")
            # Pause the wakeword engine to free the microphone
            self.engine.pause()
            # Record until silence
            try:
                audio = self.stt.record_until_silence()
                logger.info("Voice command recording finished.")
            except Exception as e:
                logger.error(f"Error during voice recording: {e}", exc_info=True)
                audio = None
        finally:
            # Resume listening as soon as possible
            try:
                self.engine.start()
            except Exception as e:
                logger.error(f"Failed to resume wakeword engine: {e}", exc_info=True)
        # Process the recorded audio
        if audio is None or (hasattr(audio, "size") and audio.size == 0):
            logger.warning("No audio captured for transcription.")
            self._busy.release()
            return
        try:
            # Transcribe speech to text
            text = self.stt.transcribe(audio, language="de")
            if text:
                # Dispatch to router/LLM
                response = self.handle_command(text)
                # Optionally integrate with GUI/Discord: the GUI will pick up logs
        except Exception as e:
            logger.error(f"Speech transcription failed: {e}", exc_info=True)
        finally:
            self._busy.release()