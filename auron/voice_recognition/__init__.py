"""Voice recognition components for Auron.

This package exposes the ``VoiceKeyEngine`` and ``STTEngine`` classes which
implement wake‑word detection and speech‑to‑text transcription respectively.
"""

from .voicekey_engine import VoiceKeyEngine  # noqa: F401
from .stt_engine import STTEngine  # noqa: F401

__all__ = ["VoiceKeyEngine", "STTEngine"]