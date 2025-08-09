"""Text‑to‑speech (TTS) support for Auron.

This package exposes the :class:`~auron.tts.tts_engine.TTSPlayer` class
which wraps the Chatterbox TTS model for local speech synthesis.  It
supports optional zero‑shot voice cloning via a short reference audio
clip defined by the ``TTS_AUDIO_PROMPT_PATH`` environment variable.
The underlying model is loaded lazily on the first call to
``speak`` and generation/playback is performed synchronously
through the ``sounddevice`` library.
"""

from .tts_engine import TTSPlayer  # noqa: F401

__all__ = ["TTSPlayer"]