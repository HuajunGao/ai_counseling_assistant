import sys
import os
import time
import queue
import logging
import sounddevice as sd
import soundfile as sf
import numpy as np
from gtts import gTTS

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.audio_capture import AudioCapture
from core.transcriber import Transcriber

# Fix for MKL/OpenMP conflict (just in case)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_listener():
    print("--- Background Listener (Stable) ---", flush=True)
    
    # 31 = Stereo Mix (requires 48000Hz usually)
    INPUT_DEVICE = 31
    NATIVE_RATE = 48000
    
    print(f"Device: {INPUT_DEVICE} @ {NATIVE_RATE}Hz", flush=True)

    audio_queue = queue.Queue()
    transcribe_queue = queue.Queue()
    
    print("Initializing AudioCapture...", flush=True)
    capturer = AudioCapture(audio_queue, native_sample_rate=NATIVE_RATE)
    capturer.start(device_id=INPUT_DEVICE)
    
    print("Initializing Transcriber...", flush=True)
    transcriber = Transcriber(audio_queue, transcribe_queue)
    transcriber.start()
    
    print("\nLISTENING... Play audio on your Realtek Speakers now!", flush=True)
    print("(Press Ctrl+C to stop)", flush=True)
    
    start_time = time.time()
    
    try:
        while True:
            try:
                item = transcribe_queue.get(timeout=0.5)
                if item['type'] == 'partial' or item['type'] == 'final':
                    text = item.get('text', '')
                    timestamp = time.time() - start_time
                    logging.info(f"[+{timestamp:.1f}s] {text}")
                    # Print clearly
                    if item['type'] == 'final' or (item['type'] == 'partial' and len(text) > 5):
                         print(f"Captured: {text}", flush=True)
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
        capturer.stop()
        transcriber.stop()

if __name__ == "__main__":
    run_listener()
