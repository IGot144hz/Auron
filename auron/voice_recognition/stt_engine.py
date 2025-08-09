"""
Speech‑to‑text engine using Faster Whisper and WebRTC VAD.

This module records audio from the microphone until a period of silence is
detected, then transcribes it using the Faster Whisper model.  It feeds
float32 numpy audio directly to the model and avoids temporary files.
"""
from __future__ import annotations

import warnings
import time
from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

from ..utils.logging_system import setup_log_system

logger = setup_log_system("stt_engine")


@dataclass
class STTConfig:
    sample_rate: int = 16_000
    channels: int = 1
    frame_ms: int = 30  # VAD supports 10, 20, or 30 ms
    vad_aggressiveness: int = 2  # 0..3
    max_record_seconds: int = 20
    min_silence_time: float = 1.2  # seconds of continuous silence to stop
    pre_speech_padding_ms: int = 300  # keep a bit before first detected speech


class STTEngine:
    """
    Records microphone audio until a period of silence and transcribes with
    Faster Whisper.

    - Uses WebRTC VAD to determine end‑of‑speech.
    - Feeds float32 numpy audio directly to Faster Whisper (no temp WAV files).
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",  # 'auto' | 'cpu' | 'cuda'
        compute_type: str | None = None,  # None => smart fallback
        cfg: STTConfig | None = None,
    ) -> None:
        # Suppress noisy warnings from dependencies
        warnings.filterwarnings("ignore", category=UserWarning)
        self.cfg = cfg or STTConfig()
        self._frame_samples = int(self.cfg.sample_rate * self.cfg.frame_ms / 1000)
        self._pre_pad_frames = max(1, int(self.cfg.pre_speech_padding_ms / self.cfg.frame_ms))

        # Smart compute_type fallback to avoid CPU float16 errors when 'auto' picks CPU
        if compute_type is not None:
            preferred: Iterable[str] = (compute_type,)
        elif device == "cpu":
            preferred = ("int8", "int16", "float32")
        else:
            preferred = ("float16", "int8", "int16", "float32")

        last_err: Exception | None = None
        for ct in preferred:
            try:
                self.model = WhisperModel(model_size, device=device, compute_type=ct)
                logger.debug(
                    f"Loaded Whisper model='{model_size}' (device={device}, compute_type={ct})."
                )
                break
            except Exception as e:  # try next compute type
                last_err = e
                logger.warning(f"Failed loading compute_type={ct}, trying next… ({e})")
        else:
            logger.error("Could not initialize Whisper model with any compute_type.")
            if last_err:
                raise last_err
            raise RuntimeError("Whisper model initialization failed with unknown error.")

    # ------------------- Recording -------------------
    def _vad_is_speech(self, frame_int16: np.ndarray, vad: webrtcvad.Vad) -> bool:
        """Return True if the frame contains speech. Expects 1‑D int16 mono samples of length ``_frame_samples``."""
        assert frame_int16.ndim == 1 and frame_int16.dtype == np.int16
        return vad.is_speech(frame_int16.tobytes(), self.cfg.sample_rate)

    def record_until_silence(self) -> np.ndarray:
        """
        Record from the default microphone until VAD registers ``min_silence_time`` after any speech.

        Returns a 1‑D int16 numpy array (mono, ``cfg.sample_rate``).
        """
        vad = webrtcvad.Vad(self.cfg.vad_aggressiveness)
        frame_len = self._frame_samples

        # Ring buffer to keep some audio before first speech (for non‑clipped start)
        pad_buffer: list[np.ndarray] = []
        audio_frames: list[np.ndarray] = []
        have_detected_speech = False
        silence_started_at: float | None = None
        start_time = time.time()

        def on_audio(indata, frames, time_info, status) -> None:
            nonlocal have_detected_speech, silence_started_at
            if status:
                if status.input_overflow:
                    logger.warning("Recording input overflow: some audio frames were lost.")
                if status.input_underflow:
                    logger.warning("Recording input underflow: no audio data available.")
                if not (status.input_overflow or status.input_underflow):
                    logger.warning(f"Recording input status flag: {status}")

            # indata shape: (frames, channels) with dtype=int16
            mono = indata[:, 0].copy()  # channels=1 in our stream config

            # Keep a small pre‑speech buffer
            if not have_detected_speech:
                pad_buffer.append(mono)
                if len(pad_buffer) > self._pre_pad_frames:
                    pad_buffer.pop(0)

            is_speech = self._vad_is_speech(mono, vad)
            if is_speech:
                if not have_detected_speech:
                    # flush pre‑speech padding into main buffer
                    audio_frames.extend(pad_buffer)
                    pad_buffer.clear()
                have_detected_speech = True
                silence_started_at = None
                audio_frames.append(mono)
            else:
                if have_detected_speech:
                    audio_frames.append(mono)
                    if silence_started_at is None:
                        silence_started_at = time.time()

        with sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=self.cfg.channels,
            dtype="int16",
            blocksize=frame_len,
            callback=on_audio,
        ):
            logger.info("Voice recording started (waiting for silence or timeout)…")
            while True:
                now = time.time()
                if now - start_time > self.cfg.max_record_seconds:
                    logger.info("Maximum recording duration reached, stopping.")
                    break
                if have_detected_speech and silence_started_at is not None:
                    if now - silence_started_at >= self.cfg.min_silence_time:
                        logger.debug("Silence detected, stopping recording.")
                        break
                time.sleep(0.02)

        if not audio_frames:
            logger.debug("No audio captured.")
            return np.array([], dtype=np.int16)

        audio = np.concatenate(audio_frames, axis=0).astype(np.int16)
        logger.debug(
            f"Captured {len(audio)} samples (~{len(audio)/self.cfg.sample_rate:.2f}s)."
        )
        return audio

    # ------------------- Transcription -------------------
    def transcribe(
        self,
        audio_int16: np.ndarray,
        *,
        language: str | None = "de",
        beam_size: int = 5,
    ) -> str:
        """
        Transcribe an int16 mono signal.  Returns the concatenated text.
        """
        if audio_int16.size == 0:
            return ""

        audio_f32 = (audio_int16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
        try:
            segments, info = self.model.transcribe(
                audio_f32,
                beam_size=beam_size,
                language=language,
                # You can enable the following if you want extra robustness (slower):
                # vad_filter=True,
                # vad_parameters={"min_silence_duration_ms": int(self.cfg.min_silence_time * 1000)},
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            logger.debug(f"Transcription result: '{text}'")
            return text
        except Exception as e:
            logger.error(f"Error during transcription: {e}", exc_info=True)
            raise

    def record_and_transcribe(
        self, *, language: str | None = "de", beam_size: int = 5
    ) -> str:
        """Helper to record and transcribe in one call."""
        audio = self.record_until_silence()
        return self.transcribe(audio, language=language, beam_size=beam_size)