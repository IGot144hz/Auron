import warnings
import sounddevice as sd
import numpy as np
import queue
import time
import tempfile
import os
from faster_whisper import WhisperModel
import scipy.io.wavfile
from utils.log_system import setup_log_sytem

warnings.filterwarnings("ignore", category=UserWarning)

import webrtcvad  # noqa: E402

logger = setup_log_sytem("SpeechToText")

# Configuration constants
SAMPLE_RATE = 16000  # Audio sample rate in Hz
CHANNELS = 1  # Mono audio
FRAME_DURATION_MS = 30  # Frame size for VAD in milliseconds (10, 20, or 30 ms frames)
VAD_AGGRESSIVENESS = 2  # VAD mode (0 = very sensitive, 3 = very strict)
MAX_RECORD_SECONDS = 10  # Safety limit for maximum recording duration


class sttEngine:
    """
    Speech-to-Text engine that records microphone audio until silence is detected, then transcribes it using Whisper.

    Uses WebRTC VAD (Voice Activity Detection) to determine when the user stops speaking.

    **Parameters:**
      - model_size (str): Which Whisper model to load (e.g., 'small', 'medium'). Defaults to 'small'.
      - device (str): Device for running the model ('cpu', 'cuda', or 'auto'). Defaults to 'auto'.

    **Raises:**
      - Exception: If loading the Whisper model fails.
    """

    def __init__(self, model_size="small", device="auto"):
        try:
            # Load Whisper model (with int8 quantization for efficiency on CPU, if used)
            self.model = WhisperModel(model_size, device=device, compute_type="int8")
            logger.debug(f"Whisper model '{model_size}' loaded (device: {device}).")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            raise

    def record_until_silence(
        self, max_duration=MAX_RECORD_SECONDS, min_silence_time=1.5
    ):
        """
        Record audio from the microphone until a period of silence is detected or until max_duration is reached.

        Returns:
            numpy.ndarray: Recorded audio samples as a 1-D NumPy array of int16.
        """
        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        frame_size = int(
            SAMPLE_RATE * FRAME_DURATION_MS / 1000
        )  # number of samples per VAD frame
        audio_queue = queue.Queue()
        audio_frames = []

        # Callback to capture audio into the queue
        def audio_callback(indata, frames, time_info, status):
            if status:
                # Log buffer over/underflow conditions if any
                if status.input_overflow:
                    logger.warning(
                        "Recording input overflow: some audio frames were lost."
                    )
                if status.input_underflow:
                    logger.warning(
                        "Recording input underflow: no audio data available."
                    )
                if not (status.input_overflow or status.input_underflow):
                    logger.warning(f"Recording input status flag: {status}")
            # Append the audio chunk to the queue for processing
            audio_queue.put(indata.copy())

        # Open an input stream for recording
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=frame_size,
            callback=audio_callback,
        ):
            logger.info("Voice recording started (waiting for silence or timeout)...")
            start_time = time.time()
            silence_start_time = None

            while True:
                # If recording time exceeds maximum, stop recording
                if time.time() - start_time > max_duration:
                    logger.info(
                        "Maximum recording duration reached, stopping recording."
                    )
                    break
                try:
                    data = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    # No data available yet, continue waiting
                    continue

                audio_frames.append(data)
                # Check if the current audio frame contains speech
                is_speech = vad.is_speech(data.tobytes(), SAMPLE_RATE)
                if is_speech:
                    silence_start_time = None  # reset silence timer on speech
                else:
                    if silence_start_time is None:
                        # Start counting silence duration
                        silence_start_time = time.time()
                    elif time.time() - silence_start_time >= min_silence_time:
                        # Sufficient silence detected, stop recording
                        logger.debug("Silence detected, stopping recording.")
                        break

            logger.debug("Voice recording ended.")

        # Combine all recorded frames into a single numpy array
        if audio_frames:
            recorded_audio = np.concatenate(audio_frames, axis=0).flatten()
        else:
            recorded_audio = np.array([], dtype=np.int16)
        return recorded_audio

    def transcribe(self, audio_data: np.ndarray):
        """
        Transcribe the given audio data to text using the Whisper model.

        Parameters:
            audio_data (np.ndarray): 1-D array of int16 audio samples (16 kHz).

        Returns:
            str: The transcribed text.
        """
        logger.debug("Starting transcription...")
        try:
            # Write the audio data to a temporary WAV file (16 kHz, mono)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                scipy.io.wavfile.write(tmp_wav.name, SAMPLE_RATE, audio_data)
                wav_path = tmp_wav.name
            # Use the Whisper model to transcribe the audio file (assuming German language for accuracy)
            segments, info = self.model.transcribe(wav_path, beam_size=5, language="de")
            # Concatenate text from all segments
            result_text = " ".join(segment.text.strip() for segment in segments).strip()
            logger.debug(f"Transcription result: '{result_text}'")
            # Remove the temporary file after transcription
            os.remove(wav_path)
            logger.debug("Transcription completed.")
            return result_text
        except Exception as e:
            logger.error(f"Error during transcription: {e}", exc_info=True)
            raise

    def record_and_transcribe(self):
        """
        Convenience method to record from microphone until silence and then return the transcribed text.
        """
        audio = self.record_until_silence()
        return self.transcribe(audio)
