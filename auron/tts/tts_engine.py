"""
Text‑to‑speech (TTS) engine wrapper using the Chatterbox library.

This implementation wraps the open‑source Chatterbox TTS model to
generate natural sounding speech from arbitrary text.  Chatterbox
supports zero‑shot voice cloning via a short reference audio clip.  To
configure the voice, set the environment variable ``TTS_AUDIO_PROMPT_PATH``
to the path of a WAV file containing a few seconds of your desired
voice.  The device used for generation can be selected via
``TTS_DEVICE`` (``cuda``, ``mps``, ``cpu`` or ``auto`` to detect the
best available).

The ``TTSPlayer`` class loads the Chatterbox model on first use and
serialises generation/playback using a thread lock.  Generated audio
is streamed to the system audio output via ``sounddevice``.  If
Chatterbox or its dependencies are not installed, the player logs
errors and disables speech.
"""
from __future__ import annotations

import logging
from ..utils.logging_system import setup_log_system
import os
import threading
from typing import Optional

import numpy as np

# Importing heavy dependencies lazily to avoid slowing down module import.
try:
    import torch  # type: ignore[import]
    import torchaudio  # type: ignore[import]
    from chatterbox.tts import ChatterboxTTS  # type: ignore[import]
    import sounddevice as sd  # type: ignore[import]
except Exception as _exc:
    # Failure will be handled in TTSPlayer initialisation
    torch = None  # type: ignore[assignment]
    torchaudio = None  # type: ignore[assignment]
    ChatterboxTTS = None  # type: ignore[assignment]
    sd = None  # type: ignore[assignment]
    _import_error = _exc
else:
    _import_error = None  # type: ignore[assignment]

# Use the custom logging setup so that TTS messages have their own handler and
# propagate correctly to the web interface.  This ensures informational
# messages about model loading are emitted even before the web app configures
# the root logger.
logger = setup_log_system("tts_engine")


class TTSPlayer:
    """
    Wrapper around the Chatterbox TTS model for speech synthesis.

    Parameters
    ----------
    voice_name: Optional[str]
        This parameter is ignored for Chatterbox but kept for
        backwards‑compatibility with previous implementations.  To
        configure the voice, set ``TTS_AUDIO_PROMPT_PATH`` in the
        environment to point to a WAV file with your voice sample.
    """

    def __init__(self, voice_name: Optional[str] = None) -> None:
        # Determine whether dependencies loaded correctly
        if _import_error is not None:
            logger.error(
                "Chatterbox TTS dependencies are missing: %s", _import_error
            )
            self.model = None
        else:
            # Determine device: auto → choose CUDA or MPS if available
            device_env = os.getenv("TTS_DEVICE", "auto").lower()
            device: str
            if device_env == "auto":
                if torch and torch.cuda.is_available():
                    device = "cuda"
                elif torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            else:
                device = device_env
            try:
                # Load the TTS model
                self.model = ChatterboxTTS.from_pretrained(device=device)
                logger.info(f"Chatterbox TTS model loaded on {device}.")
            except Exception as e:
                logger.error(f"Failed to load Chatterbox TTS model: {e}", exc_info=True)
                self.model = None
            # Retrieve custom audio prompt if provided
            audio_prompt_path = os.getenv("TTS_AUDIO_PROMPT_PATH")
            if audio_prompt_path and os.path.isfile(audio_prompt_path):
                self.audio_prompt_path: Optional[str] = audio_prompt_path
                logger.info(f"Using custom TTS audio prompt: {audio_prompt_path}")
            else:
                self.audio_prompt_path = None
        # Thread lock to serialise generation/playback
        self._lock = threading.Lock()

    def speak(self, text: str) -> None:
        """Generate speech from text and play it through the default audio output.

        If the TTS model is unavailable or ``text`` is empty, this method
        returns immediately.  Any exceptions during generation or playback
        are logged and do not propagate.
        """
        if not text:
            return
        if self.model is None:
            logger.debug("TTS model not available; skipping speech.")
            return
        with self._lock:
            try:
                # Generate speech; returns a torch tensor with shape [channels, samples]
                wav = self.model.generate(
                    text, audio_prompt_path=self.audio_prompt_path  # type: ignore[arg-type]
                )
                # Convert to numpy on CPU
                if hasattr(wav, "cpu"):
                    wav_np = wav.cpu().numpy()
                else:
                    wav_np = np.array(wav, dtype=np.float32)
                # Flatten to mono if multiple channels
                if wav_np.ndim > 1:
                    wav_np = wav_np[0]
                # Ensure float32
                audio = wav_np.astype(np.float32)
                # Play the audio synchronously
                sd.stop()
                sd.play(audio, self.model.sr)  # type: ignore[assignment]
                sd.wait()
            except Exception as e:
                logger.error(f"Error during TTS generation/playback: {e}", exc_info=True)