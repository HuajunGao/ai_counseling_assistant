import sys
import os
import time
import queue
import logging

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.audio_capture import AudioCapture
from core.transcriber import Transcriber

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def background_listener():
    print("--- Background Listener (Simulated System Service) ---", flush=True)
    
    # HARDCODED CONFIGURATION BASED ON DIAGNOSTICS
    # Stereo Mix (Input) - Captures Desktop Audio
    INPUT_DEVICE = 31 
    NATIVE_RATE = 48000 # Stereo Mix requires 48k
    
    print(f"Initializing Audio Capture on Device {INPUT_DEVICE} @ {NATIVE_RATE}Hz...", flush=True)
    import numpy
    print(f"NumPy Version: {numpy.__version__}", flush=True)
    
    audio_queue = queue.Queue()
    transcribe_queue = queue.Queue()
    
    # Initialize with native rate logic we just added
    print("Creating AudioCapture object...", flush=True)
    capturer = AudioCapture(audio_queue=audio_queue, native_sample_rate=NATIVE_RATE)
    
    print(f"Starting AudioCapture on device {INPUT_DEVICE}...", flush=True)
    try:
        capturer.start(device_id=INPUT_DEVICE)
        print("AudioCapture started.", flush=True)
    except Exception as e:
        print(f"AudioCapture failed to start: {e}", flush=True)
        raise

    print("Creating Transcriber...", flush=True)
    transcriber = Transcriber(audio_queue, transcribe_queue)
    
    print("Starting Transcriber thread... (SKIPPING FOR DEBUG)", flush=True)
    # transcriber.start() 
    
    print("Listening... (Audio Capture ONLY - 10s test)", flush=True)
    start_time = time.time()
    
    try:
        while time.time() - start_time < 10:
            time.sleep(1)
            remaining = 10 - (time.time() - start_time)
            print(f"Recording... {remaining:.0f}s left", flush=True)
            try:
                item = transcribe_queue.get(timeout=0.5)
                if item['type'] == 'partial' or item['type'] == 'final':
                    text = item.get('text', '')
                    timestamp = time.time() - start_time
                    # Simulating a log file output or backend event
                    print(f"[+ {timestamp:.2f}s] TRANSCRIPTION: {text}", flush=True)
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        print("\nStopping listener...", flush=True)
        capturer.stop()
        transcriber.stop()
        print("Done.")

if __name__ == "__main__":
    try:
        background_listener()
    except Exception as e:
        logger.exception("CRITICAL ERROR IN LISTENER:")
        # Also print to stdout just in case
        print(f"CRITICAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
