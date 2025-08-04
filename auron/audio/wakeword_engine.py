import os
import sys
import numpy as np
import pvporcupine
import sounddevice as sd
from dotenv import load_dotenv
from core.logger import setup_logger

# Load environment variables from .env (if present) *before* setting up logger,
# so that LOG_LEVEL and other configs are read.
load_dotenv()

# Initialize logger for this module
logger = setup_logger("Wakeword_Engine")


class WakewordEngine:
    """
    An always-listening wake word detection engine using Picovoice Porcupine.

    This engine opens an audio input stream and continuously listens for the configured
    wake word. When the wake word is detected, a callback function is invoked.

    **Environment Variables:**
      - `ACCESS_KEY`: Picovoice AccessKey for Porcupine (required).
      - `WAKEWORD_PATH`: Filesystem path to the Porcupine wake word model (.ppn file).
      - `PORC_MODEL`: Path to the Porcupine base model file (e.g., .pv file for language model).
      - `LOG_LEVEL` (optional): Log verbosity level (DEBUG, INFO, etc.).
      - `NO_COLOR` (optional): If set, disable colored logging output.

    **Parameters:**
      - callback (callable): A function to call when the wake word is detected.
          It should accept no arguments (or handle no arguments) and return quickly.
          **Note:** This callback is called from the audio thread – avoid long operations inside it.
      - sensitivity (float): Detection sensitivity for the wake word (0.0 to 1.0).
          Higher values reduce misses but increase false alarms. Default is 0.6.
      - device (int or str, optional): Audio input device (index or name). If not provided, uses the system default input.

    **Raises:**
      - ValueError: If required environment variables are missing.
      - pvporcupine.PorcupineError: If Porcupine initialization fails (e.g., invalid AccessKey or model paths).
    """

    def __init__(self, callback, sensitivity: float = 0.6, device=None):
        # Ensure required environment variables are present
        access_key = os.getenv("ACCESS_KEY")
        wakeword_path = os.getenv("WAKEWORD_PATH")
        model_path = os.getenv("PORC_MODEL")
        if not access_key:
            raise ValueError(
                "ACCESS_KEY is missing. Set it in environment or .env file."
            )
        if not wakeword_path:
            raise ValueError(
                "WAKEWORD_PATH is missing. Set it in environment or .env file."
            )
        if not model_path:
            raise ValueError(
                "PORC_MODEL is missing. Set it in environment or .env file."
            )

        # Initialize Porcupine wake word engine
        try:
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=[wakeword_path],
                model_path=model_path,
                sensitivities=[sensitivity],
            )
        except pvporcupine.PorcupineError as e:
            # Log the error and re-raise to prevent running without a functional engine
            logger.error("Failed to initialize Porcupine engine: %s", e)
            raise

        self.callback = callback  # Function to call when wake word is detected

        # Optional: allow custom latency setting via env (to help avoid dropouts if needed)
        latency_ms = os.getenv("INPUT_LATENCY")
        latency = float(latency_ms) / 1000.0 if latency_ms else None

        # Set up the input audio stream for the microphone
        self.stream = sd.RawInputStream(
            samplerate=self.porcupine.sample_rate,
            blocksize=self.porcupine.frame_length,  # frames per buffer matches Porcupine requirement
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
            device=device,
            latency=latency,
        )
        logger.debug(
            "Audio stream initialized (device=%s, samplerate=%d, blocksize=%d)",
            str(device),
            self.porcupine.sample_rate,
            self.porcupine.frame_length,
        )

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Internal method: Sounddevice audio callback. Processes incoming audio frames.
        This is called in a separate thread by the sound device engine.
        """
        try:
            # Check for microphone errors or buffer issues
            if status:
                if status.input_overflow:
                    logger.warning(
                        "Audio input overflow detected (missed audio frames)."
                    )
                if status.input_underflow:
                    logger.warning(
                        "Audio input underflow detected (no data available)."
                    )
                # Log any status flags (if not specifically handled above)
                if not (status.input_overflow or status.input_underflow):
                    logger.warning("Audio input status: %s", status)

            # Convert raw bytes to NumPy array of int16
            pcm = np.frombuffer(indata, dtype=np.int16)
            # Process the audio frame with Porcupine
            result = self.porcupine.process(pcm)
            if result >= 0:
                # Invoke the callback. Note: callback should be non-blocking.
                self.callback()
        except Exception as e:
            # Catch all to prevent the callback from stopping the stream on error
            logger.error("Error in audio callback: %s", e)
            # Optionally, we could stop the stream on critical errors:
            # import sounddevice; if isinstance(e, sounddevice.PortAudioError): self.stop()

    def start(self):
        """Start listening for the wake word (opens the audio stream)."""
        self.stream.start()
        logger.info("WakewordEngine started, listening for wake word...")

    def stop(self):
        """Stop listening and clean up resources (closes audio stream and Porcupine engine)."""
        try:
            # Stop and close the audio stream
            if self.stream:
                self.stream.stop()
                self.stream.close()
        finally:
            # Always delete the Porcupine engine to release resources (avoid memory leaks).
            if hasattr(self, "porcupine") and self.porcupine is not None:
                self.porcupine.delete()
                self.porcupine = None
        logger.info("WakewordEngine stopped.")
