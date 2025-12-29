import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import time
import requests

def foreground_player():
    print("--- Foreground Player (Simulating Browser/Video) ---", flush=True)
    
    # HARDCODED CONFIGURATION
    # Realtek Speakers (Output) - Paired with Stereo Mix
    OUTPUT_DEVICE = 25
    
    filename = "test_long.wav"
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Run test_long_transcription.py once to generate it.", flush=True)
        return

    print(f"Loading {filename}...", flush=True)
    data, fs = sf.read(filename)
    
    # Resample to 48000 if needed (simple linear interpolation for test)
    # The device defaults to 48k, so we should feed it 48k to avoid errors.
    target_fs = 48000
    if fs != target_fs:
        print(f"Resampling from {fs}Hz to {target_fs}Hz...", flush=True)
        number_of_samples = int(round(len(data) * float(target_fs) / fs))
        # Basic interpolation using numpy (good enough for dev test)
        data = np.interp(
            np.linspace(0.0, 1.0, number_of_samples),
            np.linspace(0.0, 1.0, len(data)),
            data
        )
        fs = target_fs
        
    data = data.astype(np.float32)
    
    print(f"Playing audio on Device {OUTPUT_DEVICE}...", flush=True)
    try:
        sd.play(data, fs, device=OUTPUT_DEVICE)
        print("Playback started. Switch to the Listener window to see results!", flush=True)
        sd.wait()
        print("Playback finished.")
    except Exception as e:
        print(f"Playback error: {e}", flush=True)

if __name__ == "__main__":
    foreground_player()
