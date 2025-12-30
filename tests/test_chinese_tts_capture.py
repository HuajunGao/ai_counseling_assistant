import sys
import os
import time
import queue
import logging
import shutil

# Try importing sounddevice and verify system dependencies
try:
    import sounddevice as sd
except OSError as e:
    print("\n\033[91m[ERROR] Could not import 'sounddevice'.\033[0m")
    print("This is likely due to missing PortAudio library in WSL/Linux.")
    print("Please run: \033[93msudo apt-get update && sudo apt-get install -y libportaudio2\033[0m\n")
    sys.exit(1)

try:
    import soundfile as sf
except OSError as e:
    print("\n\033[91m[ERROR] Could not import 'soundfile'.\033[0m")
    print("Please run: \033[93msudo apt-get install -y libsndfile1\033[0m\n")
    sys.exit(1)

import numpy as np
from gtts import gTTS

# Add project root to path to allow imports from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Verify ffmpeg
if shutil.which("ffmpeg") is None:
    print("\n\033[91m[ERROR] 'ffmpeg' is not found.\033[0m")
    print("It is required for audio conversion.")
    print("Please run: \033[93msudo apt-get install -y ffmpeg\033[0m\n")
    sys.exit(1)


from core.audio_capture import AudioCapture
from core.transcriber import Transcriber

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_chinese_capture():
    """
    Tests if the system can capture and transcribe Chinese audio played via speakers.
    """

    # Check audio devices
    logger.info("Checking audio devices...")
    try:
        devices = sd.query_devices()
        logger.info(f"Available audio devices:\n{devices}")
        if len(devices) == 0:
            logger.error("No audio devices found! WSL might need configuration.")
            logger.error("Try installing PulseAudio and configuring WSL to use it, or verify WSLg status.")
            return
    except Exception as e:
        logger.error(f"Error querying devices: {e}")
        return

    # 1. Generate Chinese Audio
    text_to_speak = "你好。这是一个测试。我们正在测试语音识别系统。"
    filename = "test_chinese.wav"
    mp3_filename = "test_chinese.mp3"

    logger.info(f"Generating TTS for: {text_to_speak}")
    tts = gTTS(text=text_to_speak, lang="zh")
    tts.save(mp3_filename)

    # Convert mp3 to wav
    logger.info("Converting MP3 to WAV...")
    ret = os.system(f"ffmpeg -y -v error -i {mp3_filename} -ar 16000 -ac 1 {filename}")
    if ret != 0:
        logger.error("ffmpeg conversion failed.")
        sys.exit(1)

    # 2. Setup Audio Capture and Transcriber
    audio_queue = queue.Queue()
    transcribe_queue = queue.Queue()

    capturer = AudioCapture(audio_queue)
    capturer.start()

    transcriber = Transcriber(audio_queue, transcribe_queue)
    transcriber.start()

    time.sleep(1)  # Warmup

    # 3. Play Audio
    logger.info(f"Playing audio {filename}...")
    try:
        data, fs = sf.read(filename)
        sd.play(data, fs)
        sd.wait()  # Wait until playback is finished
        logger.info("Playback finished.")
    except Exception as e:
        logger.error(f"Error playing audio: {e}")
        logger.error("If 'PortAudio not initialized' or similar, check WSL audio settings.")

    # Allow some time for processing
    time.sleep(5)

    # 4. Stop and Check Results
    logger.info("Stopping capture...")
    capturer.stop()
    transcriber.stop()

    logger.info("Checking transcription results...")

    found_keywords = False
    captured_texts = []

    while not transcribe_queue.empty():
        item = transcribe_queue.get()
        if item["type"] == "partial" or item["type"] == "final":
            text = item.get("text", "")
            captured_texts.append(text)
            logger.info(f"Captured: {text}")
            if "测试" in text or "你好" in text or "识别" in text:
                found_keywords = True

    if found_keywords:
        logger.info("Test PASSED: Detected Chinese keywords.")
    else:
        logger.error(f"Test FAILED: Chinese keywords not found. Captured: {captured_texts}")

    # Cleanup
    if os.path.exists(mp3_filename):
        os.remove(mp3_filename)
    if os.path.exists(filename):
        os.remove(filename)


if __name__ == "__main__":
    test_chinese_capture()
