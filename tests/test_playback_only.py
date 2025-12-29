import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import time

def test_playback():
    print("--- Audio Device Diagnostics ---", flush=True)
    print(sd.query_devices(), flush=True)
    print(f"Default Devices: {sd.default.device}", flush=True)
    print("--------------------------------", flush=True)

    # --- CONFIGURATION ---
    # Set this to the ID of the device you want to test (e.g., 33 or 50)
    # If None, it uses the system default.
    OUTPUT_DEVICE = None 
    # ---------------------

    filename = "test_long.wav"
    
    # Fallback to tone if file doesn't exist
    if not os.path.exists(filename):
        print(f"File {filename} not found. Generating a 440Hz sine wave...", flush=True)
        fs = 44100
        duration = 3.0
        t = np.linspace(0, duration, int(fs * duration), False)
        data = 0.5 * np.sin(2 * np.pi * 440 * t)
    else:
        print(f"Loading {filename}...", flush=True)
        data, fs = sf.read(filename)
        
    # Ensure float32
    data = data.astype(np.float32)

    print(f"Playing on device ID: {OUTPUT_DEVICE} (Default if None)", flush=True)
    print("Listen carefully...", flush=True)
    
    try:
        sd.play(data, fs, device=OUTPUT_DEVICE)
        sd.wait() # Wait for playback to finish
        print("Playback command finished.", flush=True)
    except Exception as e:
        print(f"Error during playback: {e}", flush=True)

if __name__ == "__main__":
    test_playback()
