import sounddevice as sd
import numpy as np
import time

def check_duplex():
    print("--- Duplex Check ---")
    print(f"Default Devices: {sd.default.device}")
    
    in_dev = sd.default.device[0]
    out_dev = sd.default.device[1]
    
    print(f"Testing simultaneous Input (Device {in_dev}) and Output (Device {out_dev})...")
    
    try:
        # Generate short tone
        fs = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(fs * duration), False)
        tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        
        # Start Input Stream
        print("1. Starting Input Stream...", flush=True)
        def callback(indata, frames, time, status):
            pass
            
        with sd.InputStream(device=in_dev, channels=1, samplerate=fs, callback=callback):
            print("   Input Stream running.", flush=True)
            time.sleep(0.5)
            
            # Try Playback
            print("2. Attempting Playback while Input is active...", flush=True)
            sd.play(tone, fs, device=out_dev, blocking=True)
            print("   Playback finished successfully!", flush=True)
            
        print("\nSUCCESS: Full-duplex seems to work with these devices.")
        
    except Exception as e:
        print(f"\nFAILURE: Could not run duplex: {e}")
        print("This likely means the Audio Driver (MME) cannot do both at once.")
        print("Recommendation: Switch to WASAPI devices.")

if __name__ == "__main__":
    check_duplex()
