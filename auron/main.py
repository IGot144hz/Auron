# main.py
import time
import threading

from utils.logging_system import setup_log_system
from voice_recognition.voicekey_engine import VoiceKeyEngine
from voice_recognition.stt_engine import STTEngine

logger = setup_log_system("main")


class AssistantApp:
    """Owns the wakeword engine and STT engine; coordinates record → transcribe."""

    def __init__(self) -> None:
        # Adjust model_size/device as you like (STTEngine has smart compute_type fallbacks)
        self.stt = STTEngine(model_size="medium", device="cuda")
        # Cooldown prevents rapid re-triggers while we process the previous one
        self.engine = VoiceKeyEngine(self.on_wake, cooldown_seconds=1.0)

        # Ensures we don't start overlapping record/transcribe cycles
        self._busy = threading.Lock()

    # ------------- Wakeword callback -------------

    def on_wake(self) -> None:
        """Called by VoiceKeyEngine on detection. Offload to worker thread immediately."""
        if not self._busy.acquire(blocking=False):
            logger.debug(
                "Wakeword detected but a command is already being processed. Ignoring."
            )
            return
        threading.Thread(target=self._process_wake_event, daemon=True).start()

    # ------------- Processing pipeline -------------

    def _process_wake_event(self) -> None:
        """
        1) Pause wakeword engine (free the mic)
        2) Record until silence
        3) Resume wakeword engine ASAP
        4) Transcribe in the same worker thread (non-blocking for the audio callback)
        """
        try:
            logger.info("Wake word recognized. Preparing to record command…")
            # 1) Pause detection to free the microphone
            self.engine.pause()
            logger.debug("Wakeword engine paused for voice command recording.")

            # 2) Record user's speech until silence or timeout
            try:
                audio = self.stt.record_until_silence()
                logger.info("Voice command recording finished.")
            except Exception as e:
                logger.error(f"Error during voice recording: {e}", exc_info=True)
                audio = None
        finally:
            # 3) Resume listening as soon as possible
            try:
                self.engine.start()
                logger.debug("Wakeword engine resumed.")
            except Exception as e:
                logger.error(f"Failed to resume wakeword engine: {e}", exc_info=True)

        # 4) Transcribe (if we captured any audio)
        if audio is None or (hasattr(audio, "size") and audio.size == 0):
            logger.warning("No audio captured for transcription.")
            self._busy.release()
            return

        try:
            text = self.stt.transcribe(audio, language="de")
            logger.info(f"User command: {text}")
            # TODO: Route the command (execute actions, TTS feedback, GUI update, etc.)
        except Exception as e:
            logger.error(f"Speech transcription failed: {e}", exc_info=True)
        finally:
            self._busy.release()

    # ------------- App lifecycle -------------

    def run(self) -> None:
        """Start wakeword listening and keep the process alive."""
        self.engine.start()
        logger.info("System is now listening for the wake word…")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.debug("Shutting down (KeyboardInterrupt received)…")
        finally:
            self.engine.stop()
            logger.info("Application terminated.")


if __name__ == "__main__":
    app = AssistantApp()
    app.run()
