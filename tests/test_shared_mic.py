"""
Test if two processes can simultaneously read from the same microphone.
This simulates: Zoom using mic + Script using the same mic.

Usage:
  python tests/test_shared_mic.py --mic 0
"""

import argparse
import threading
import time
import warnings
import numpy as np

# Suppress soundcard warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*discontinuity.*")

try:
    import soundcard as sc
except ImportError:
    print("soundcard not installed. Run: pip install soundcard")
    exit(1)


def capture_audio(name: str, mic, duration: float, sample_rate: int = 16000, results: dict = None):
    """Simulate a process capturing from the microphone."""
    print(f"[{name}] Starting capture on: {mic.name}")

    success = False
    try:
        with mic.recorder(samplerate=sample_rate, channels=1) as recorder:
            start_time = time.time()
            count = 0
            while (time.time() - start_time) < duration:
                chunk = recorder.record(numframes=int(sample_rate * 0.1))
                if chunk.ndim == 2:
                    chunk = np.mean(chunk, axis=1)
                rms = float(np.sqrt(np.mean(chunk**2)))

                # Only print every few iterations to reduce noise
                count += 1
                if count % 3 == 0:
                    bar = "#" * int(min(rms * 100, 20))
                    print(f"[{name}] RMS: {rms:.4f} [{bar:<20}]")
            success = True
    except Exception as e:
        print(f"[{name}] ERROR: {e}")

    if results is not None:
        results[name] = success
    print(f"[{name}] Done. {'✅' if success else '❌'}")


def main():
    parser = argparse.ArgumentParser(description="Test shared microphone access.")
    parser.add_argument("--list", "-l", action="store_true", help="List microphones")
    parser.add_argument("--mic", type=int, default=0, help="Microphone device index")
    parser.add_argument("--duration", "-t", type=float, default=5.0, help="Duration (seconds)")
    args = parser.parse_args()

    mics = sc.all_microphones(include_loopback=False)

    if args.list:
        print("\n=== Microphones ===")
        for idx, m in enumerate(mics):
            print(f"  {idx}: {m.name}")
        return

    if args.mic >= len(mics):
        print(f"Error: Mic index {args.mic} out of range (0-{len(mics) - 1})")
        return

    mic = mics[args.mic]

    print(f"\n=== Testing Shared Mic Access ===")
    print(f"Microphone: [{args.mic}] {mic.name}")
    print(f"Duration: {args.duration}s")
    print()
    print("Starting TWO 'processes' reading from the SAME microphone...")
    print("-" * 50)

    results = {}

    # Start two "processes" (threads) reading from the same mic
    t1 = threading.Thread(target=capture_audio, args=("PROCESS_1", mic, args.duration, 16000, results))
    t2 = threading.Thread(target=capture_audio, args=("PROCESS_2", mic, args.duration, 16000, results))

    t1.start()
    time.sleep(0.3)  # Slight delay
    t2.start()

    t1.join()
    t2.join()

    print("-" * 50)

    if results.get("PROCESS_1") and results.get("PROCESS_2"):
        print("\n✅ SUCCESS: Both processes captured audio from the same microphone!")
        print("You CAN use the mic with Zoom/Teams AND this script simultaneously.")
    else:
        print("\n❌ FAILED: One or both processes could not capture audio.")
        print("The microphone may be in exclusive mode.")
        print("Fix: Windows Settings → Sound → Microphone → Advanced → Uncheck 'Exclusive Mode'")


if __name__ == "__main__":
    main()
