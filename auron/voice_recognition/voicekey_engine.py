import os
import numpy as np
import pvporcupine
import sounddevice as sd
from dotenv import load_dotenv
from utils.log_system import setup_log_sytem
import threading  # for handling callback in separate thread

# Load environment variables from .env (if present) before setting up the logger
load_dotenv()

# Initialize logger for this module
logger = setup_log_sytem("VoiceKeyEngine")


class VoiceKeyEngine:
    """
    Always-listening wake word detection engine using Picovoice Porcupine.

    Continuously listens on an audio input stream for a configured wake word and
    triggers a callback when the wake word is detected.

    **Environment Variables:**
      - `ACCESS_KEY`: Picovoice AccessKey for Porcupine (required).
      - `WAKEWORD_PATH`: Filesystem path to the Porcupine wake word model (.ppn file) (required).
      - `PORC_MODEL`: Path to the Porcupine base model file (.pv file, language model) (required).
      - `INPUT_LATENCY` (optional): Desired input latency in milliseconds (to avoid dropouts).
      - `LOG_LEVEL` (optional): Logging verbosity (DEBUG, INFO, etc.).
      - `NO_COLOR` (optional): If set, disable colored logging output.

    **Parameters:**
      - callback (callable): Function to call when the wake word is detected. The function
          should not perform long-running tasks or blocking calls (it will be executed on a separate thread).
      - sensitivity (float): Detection sensitivity (0.0 to 1.0, higher values = more sensitive). Default 0.6.
      - device (int or str, optional): Audio input device (index or name). If not provided, uses the default input.

    **Raises:**
      - ValueError: If required environment variables are missing.
      - pvporcupine.PorcupineError: If Porcupine initialization fails (e.g., invalid AccessKey or model paths).
    """

    def __init__(self, callback, sensitivity: float = 0.6, device=None):
        # Ensure required environment variables are set
        access_key = os.getenv("ACCESS_KEY")
        wakeword_path = os.getenv("WAKEWORD_PATH")
        model_path = os.getenv("PORC_MODEL")
        if not access_key:
            logger.critical(
                "ACCESS_KEY is missing. Set it in environment or .env file."
            )
            raise ValueError("Missing Picovoice AccessKey (ACCESS_KEY).")
        if not wakeword_path:
            logger.critical(
                "WAKEWORD_PATH is missing. Set it in environment or .env file."
            )
            raise ValueError("Missing Porcupine wakeword model path (WAKEWORD_PATH).")
        if not model_path:
            logger.critical(
                "PORC_MODEL is missing. Set it in environment or .env file."
            )
            raise ValueError("Missing Porcupine base model path (PORC_MODEL).")

        # Initialize Porcupine wake word detector
        try:
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=[wakeword_path],
                model_path=model_path,
                sensitivities=[sensitivity],
            )
            logger.debug("Porcupine wakeword engine initialized.")
        except pvporcupine.PorcupineError as e:
            logger.error("Failed to initialize Porcupine engine: %s", e)
            raise

        self.callback = callback  # function to call when wake word is detected
        # Optional: allow custom input latency via env to avoid audio dropouts
        latency_ms = os.getenv("INPUT_LATENCY")
        latency = float(latency_ms) / 1000.0 if latency_ms else None

        # Set up the input audio stream for microphone in always-listening mode
        try:
            self.stream = sd.RawInputStream(
                samplerate=self.porcupine.sample_rate,
                blocksize=self.porcupine.frame_length,
                dtype="int16",
                channels=1,
                callback=self._audio_callback,
                device=device,
                latency=latency,
            )
            logger.debug(
                "Audio input stream initialized (device=%s, samplerate=%d Hz, blocksize=%d frames).",
                str(device),
                self.porcupine.sample_rate,
                self.porcupine.frame_length,
            )
        except Exception as e:
            logger.error("Failed to open audio input stream: %s", e)
            # Clean up Porcupine if the stream failed to open
            self.porcupine.delete()
            self.porcupine = None
            raise

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Internal callback for the audio input stream. Processes incoming audio frames and detects the wake word.
        This runs on a separate audio thread and should not perform blocking operations.
        """
        try:
            # Handle any audio status flags (overflows, underflows)
            if status:
                if status.input_overflow:
                    logger.warning(
                        "Audio input overflow detected - missed some audio frames."
                    )
                if status.input_underflow:
                    logger.warning(
                        "Audio input underflow detected - no audio data available."
                    )
                # Log any other status flags if present
                if not (status.input_overflow or status.input_underflow):
                    logger.warning("Audio input status flag: %s", status)

            # Convert raw audio bytes to NumPy int16 array for processing
            pcm = np.frombuffer(indata, dtype=np.int16)
            result = self.porcupine.process(pcm)
            if result >= 0:
                # Wake word was detected. Invoke the callback in a new thread to avoid blocking this callback:contentReference[oaicite:2]{index=2}.
                def _run_callback():
                    try:
                        self.callback()
                    except Exception as e:
                        logger.error(
                            "Exception in wakeword callback: %s", e, exc_info=True
                        )

                threading.Thread(target=_run_callback, daemon=True).start()
        except Exception as e:
            # Log any exception to prevent the stream from stopping.
            logger.error("Error in wakeword audio callback: %s", e, exc_info=True)

    def start(self):
        """Start listening for the wake word (opens or resumes the audio stream)."""
        if self.stream:
            if not self.stream.active:
                self.stream.start()
                logger.debug("WakewordEngine started and listening for the wake word.")
            else:
                logger.debug(
                    "WakewordEngine start called, but the stream is already active."
                )

    def pause(self):
        """Pause listening for the wake word (temporarily stops the audio stream)."""
        if self.stream and self.stream.active:
            self.stream.stop()
            logger.debug("WakewordEngine paused (audio stream stopped).")
        else:
            logger.debug("WakewordEngine pause called, but the stream was not active.")

    def stop(self):
        """Completely stop the engine and release audio and detection resources."""
        try:
            if self.stream:
                if self.stream.active:
                    self.stream.stop()
                self.stream.close()
                self.stream = None
        except Exception as e:
            logger.warning(
                "Exception while stopping audio stream: %s", e, exc_info=True
            )
        finally:
            if hasattr(self, "porcupine") and self.porcupine is not None:
                self.porcupine.delete()
                self.porcupine = None
        logger.debug("WakewordEngine stopped and resources released.")
