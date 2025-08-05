import time
import threading
from utils.log_system import setup_log_sytem
from voice_recognition.voicekey_engine import VoiceKeyEngine
from voice_recognition.stt_engine import sttEngine


logger = setup_log_sytem("Main")

# Initialize the speech-to-text system (using a medium model for better accuracy; adjust as needed)
stt = sttEngine(model_size="medium")


def handle_command(audio_data):
    """
    Transcribe the recorded audio and log the recognized command.
    This function can be extended to perform actions (e.g., execute the command, trigger TTS feedback, update GUI).
    """
    try:
        command_text = stt.transcribe(audio_data)
        logger.info(f"User command: {command_text}")
        # TODO: Implement command handling (e.g., execute actions or respond via TTS/GUI).
    except Exception as e:
        logger.error(f"Speech transcription failed: {e}")


def process_wake_event():
    """
    Handle the workflow after the wake word is detected:
      1. Pause the wakeword engine to avoid interference.
      2. Record the user's voice command until silence.
      3. Resume the wakeword engine to listen for the next wake word.
      4. Transcribe the recorded command in the background.
    """
    # 1. Pause wakeword detection to free the microphone for recording:contentReference[oaicite:5]{index=5}
    engine.pause()
    logger.debug("Wakeword engine paused for voice command recording.")
    audio_data = None
    try:
        # 2. Record the user's speech until silence is detected or timeout
        audio_data = stt.record_until_silence()
        logger.info("Voice command recording finished.")
    except Exception as e:
        logger.error(f"Error during voice recording: {e}")
    finally:
        # 3. Resume wakeword engine ASAP to start listening for the next wake word
        try:
            engine.start()
            logger.debug("Wakeword engine resumed.")
        except Exception as e:
            logger.error(f"Failed to resume wakeword engine: {e}")
    # 4. If we have recorded audio, transcribe it in a separate thread to avoid blocking
    if audio_data is not None:
        threading.Thread(target=handle_command, args=(audio_data,), daemon=True).start()


def on_wake():
    """Callback function invoked by WakewordEngine when the wake word is detected."""
    logger.info("Wake word recognized. Preparing to record command...")
    # Offload processing to a new thread so this callback returns immediately:contentReference[oaicite:6]{index=6}
    threading.Thread(target=process_wake_event, daemon=True).start()


# Instantiate the wakeword engine with the on_wake callback
try:
    engine = VoiceKeyEngine(on_wake)
except Exception as e:
    logger.critical(f"Failed to initialize WakewordEngine: {e}")
    raise

# Start the wakeword listening loop
engine.start()
logger.info("System is now listening for the wake word...")

# Main loop to keep the program running; can also handle other tasks or heartbeats if needed
try:
    while True:
        time.sleep(0.1)  # sleep to reduce CPU usage in the idle loop
except KeyboardInterrupt:
    logger.debug("Shutting down (KeyboardInterrupt received)...")
finally:
    # Ensure resources are cleaned up on exit
    engine.stop()
    logger.info("Application terminated.")
