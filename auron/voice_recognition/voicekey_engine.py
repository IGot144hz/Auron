# voicekey_engine.py
import os
import threading
import time
from collections.abc import Callable, Sequence

import numpy as np
import pvporcupine
import sounddevice as sd
from dotenv import load_dotenv

from utils.logging_system import setup_log_system

# Load env before logger (so LOG_LEVEL etc. are available)
load_dotenv()
logger = setup_log_system("voicekey_engine")


def _resolve_device(device: int | str | None) -> int | None:
    """Resolve a sounddevice input by index or fuzzy name (case-insensitive). Returns index or None."""
    if device is None or isinstance(device, int):
        return device
    name_lc = device.lower()
    for idx, info in enumerate(sd.query_devices()):
        if (
            info.get("max_input_channels", 0) > 0
            and name_lc in str(info.get("name", "")).lower()
        ):
            return idx
    logger.warning(f"Input device '{device}' not found, using default.")
    return None


class VoiceKeyEngine:
    """
    Always-listening wake word detector using Picovoice Porcupine.

    - Triggers `callback` on detection (executed on a daemon thread).
    - Includes a simple cooldown to avoid multi-trigger storms while the callback runs.
    """

    def __init__(
        self,
        callback: Callable[[], None],
        *,
        keyword_paths: Sequence[str] | None = None,  # if None, reads WAKEWORD_PATH
        sensitivities: Sequence[float] | None = None,  # per keyword, else uniform
        model_path: str | None = None,  # if None, reads PORC_MODEL
        access_key: str | None = None,  # if None, reads ACCESS_KEY
        device: int | str | None = None,
        cooldown_seconds: float = 1.0,
        input_latency_ms: int | None = None,  # overrides env INPUT_LATENCY
    ) -> None:
        access_key = access_key or os.getenv("ACCESS_KEY")
        model_path = model_path or os.getenv("PORC_MODEL")
        kp = list(keyword_paths) if keyword_paths else [os.getenv("WAKEWORD_PATH", "")]
        if not access_key:
            logger.critical("Missing Picovoice AccessKey (ACCESS_KEY).")
            raise ValueError("Missing ACCESS_KEY")
        if not all(kp) or not kp[0]:
            logger.critical("Missing Porcupine keyword path(s) (WAKEWORD_PATH).")
            raise ValueError("Missing WAKEWORD_PATH")
        if not model_path:
            logger.critical("Missing Porcupine base model path (PORC_MODEL).")
            raise ValueError("Missing PORC_MODEL")

        sens = (
            list(sensitivities)
            if sensitivities is not None
            else [float(os.getenv("PORC_SENSITIVITY", "0.6"))] * len(kp)
        )

        try:
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=kp,
                model_path=model_path,
                sensitivities=sens,
            )
            logger.debug("Porcupine initialized.")
        except pvporcupine.PorcupineError as e:
            logger.error(f"Failed to initialize Porcupine: {e}")
            raise

        self._callback = callback
        self._cooldown = max(0.0, cooldown_seconds)
        self._last_trigger: float = 0.0

        latency_ms = (
            input_latency_ms
            if input_latency_ms is not None
            else int(os.getenv("INPUT_LATENCY", "0") or 0)
        )
        latency = (latency_ms / 1000.0) if latency_ms and latency_ms > 0 else None

        sd_device = _resolve_device(device)
        try:
            self.stream = sd.RawInputStream(
                samplerate=self.porcupine.sample_rate,
                blocksize=self.porcupine.frame_length,
                dtype="int16",
                channels=1,
                callback=self._on_audio,
                device=sd_device,
                latency=latency,
            )
            logger.debug(
                "Audio input ready (device=%s, %d Hz, frame=%d).",
                str(sd_device),
                self.porcupine.sample_rate,
                self.porcupine.frame_length,
            )
        except Exception as e:
            self.porcupine.delete()
            logger.error(f"Failed to open audio input stream: {e}")
            raise

    # -------------- lifecycle --------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def start(self) -> None:
        if self.stream and not self.stream.active:
            self.stream.start()
            logger.debug("VoiceKeyEngine started (listening).")

    def pause(self) -> None:
        if self.stream and self.stream.active:
            self.stream.stop()
            logger.debug("VoiceKeyEngine paused.")

    def stop(self) -> None:
        try:
            if self.stream:
                if self.stream.active:
                    self.stream.stop()
                self.stream.close()
        finally:
            try:
                self.porcupine.delete()
            finally:
                self.stream = None
                self.porcupine = None
                logger.debug("VoiceKeyEngine stopped and resources released.")

    @property
    def is_listening(self) -> bool:
        return bool(self.stream and self.stream.active)

    # -------------- audio callback --------------
    def _on_audio(self, indata, frames, time_info, status):
        try:
            if status:
                if status.input_overflow:
                    logger.warning("Audio input overflow detected.")
                if status.input_underflow:
                    logger.warning("Audio input underflow detected.")
                if not (status.input_overflow or status.input_underflow):
                    logger.warning(f"Audio input status flag: {status}")

            pcm = np.frombuffer(indata, dtype=np.int16)
            result = self.porcupine.process(pcm)
            if result >= 0:
                now = time.time()
                if now - self._last_trigger < self._cooldown:
                    return  # debounce
                self._last_trigger = now

                def _run():
                    try:
                        self._callback()
                    except Exception as e:
                        logger.error(
                            f"Exception in wakeword callback: {e}", exc_info=True
                        )

                threading.Thread(target=_run, daemon=True).start()
        except Exception as e:
            logger.error(f"Error in wakeword audio callback: {e}", exc_info=True)
