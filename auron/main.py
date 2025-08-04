import time
from audio.wakeword_engine import WakewordEngine
from core.logger import setup_logger


def on_wake():
    logger.info("WakeWord recognized...")


logger = setup_logger("Main")
engine = WakewordEngine(on_wake)

try:
    engine.start()
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    engine.stop()
