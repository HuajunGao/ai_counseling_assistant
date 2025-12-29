import numpy as np
import sounddevice as sd
import sys
import time

print("Starting sanity check...")
try:
    fs = 44100
    t = np.linspace(0, 1, fs, False)
    tone = 0.5*np.sin(2*np.pi*440*t)
    print("Playing 440Hz tone...", flush=True)
    sd.play(tone, fs)
    sd.wait()
    print("Sanity check passed: Audio played.", flush=True)
except Exception as e:
    print(f"Sanity check failed: {e}", file=sys.stderr, flush=True)
    sys.exit(1)
