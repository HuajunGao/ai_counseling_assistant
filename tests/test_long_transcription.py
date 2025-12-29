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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_long_capture():
    print("--- Audio Device Diagnostics ---", flush=True)
    print(sd.query_devices(), flush=True)
    print(f"Default Devices: {sd.default.device}", flush=True)
    print("--------------------------------", flush=True)

    # --- DEVICE CONFIGURATION ---
    # Change these IDs based on the list above if you can't hear/capture audio.
    # WASAPI devices (Better stability/latency than MME)
    INPUT_DEVICE = 27 
    OUTPUT_DEVICE = 25
    # ----------------------------

    # Long Chinese text (~30 seconds speaking time)
    text_to_speak = (
        "你好。这是一个关于语音识别速度的测试。"
        "我们将播放一段较长的音频，看看系统能否实时转录。"
        "人工智能正在改变我们的生活，从自动驾驶到智能客服，"
        "语音识别技术是人机交互的重要接口。"
        "希望这个测试能够展示出良好的准确率和响应速度。"
        "如果你看到了这段文字，说明测试正在进行中。"
    )
    
    filename = "test_long.wav"
    mp3_filename = "test_long.mp3"
    
    print("Generating long TTS audio (this may take a moment)...", flush=True)
    tts = gTTS(text=text_to_speak, lang='zh')
    tts.save(mp3_filename)
    
    print("Converting MP3 to WAV...", flush=True)
    ret = os.system(f"ffmpeg -y -v error -i {mp3_filename} -ar 16000 -ac 1 {filename}")
    if ret != 0:
        logger.error("ffmpeg conversion failed.")
        sys.exit(1)

    audio_queue = queue.Queue()
    transcribe_queue = queue.Queue()
    
    capturer = AudioCapture(audio_queue)
    capturer.start(device_id=INPUT_DEVICE)
    
    transcriber = Transcriber(audio_queue, transcribe_queue)
    transcriber.start()
    
    time.sleep(2) # Warmup
    
    print(f"Playing audio ({filename})... Watch the timestamps!", flush=True)
    start_time = time.time()
    
    data, fs = sf.read(filename)
    # Ensure float32 for sounddevice/PortAudio compatibility and prevent crashes
    data = data.astype(np.float32)

    try:
        sd.play(data, fs, device=OUTPUT_DEVICE)
    except Exception as e:
        print(f"ERROR playing audio: {e}", flush=True)
        # Continue to see if capture works even if playback fails (if measuring via virtual cable)
    
    playback_duration = len(data) / fs
    print(f"Audio duration: {playback_duration:.2f}s", flush=True)
    
    # Monitor results while playing
    playing = True
    chunks_received = 0
    
    while playing or not transcribe_queue.empty():
        # Check if playback is done
        current_time = time.time()
        if (current_time - start_time) > (playback_duration + 2) and playing:
             playing = False
             print("\nPlayback finished (estimated). Waiting for final results...", flush=True)
             capturer.stop()
        
        try:
            item = transcribe_queue.get(timeout=0.1)
            if item['type'] == 'partial' or item['type'] == 'final':
                text = item.get('text', '')
                latency = time.time() - start_time
                print(f"[T+{latency:.2f}s] Captured: {text}", flush=True)
                chunks_received += 1
        except queue.Empty:
            if not playing and (time.time() - start_time) > (playback_duration + 10):
                # Timeout after playback
                break
            continue

    transcriber.stop()
    logger.info("Test finished.")
    
    # Cleanup
    if os.path.exists(mp3_filename):
       os.remove(mp3_filename)
    if os.path.exists(filename):
       os.remove(filename)

if __name__ == "__main__":
    test_long_capture()
