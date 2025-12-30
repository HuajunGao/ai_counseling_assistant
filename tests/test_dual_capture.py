"""
Test dual-stream audio capture: Microphone + Loopback simultaneously.
This simulates capturing both your voice and the other person's voice in a call.

Usage:
  python tests/test_dual_capture.py --mic 0 --loopback 5
"""

import argparse
import threading
import time
import numpy as np

try:
    import soundcard as sc
except ImportError:
    print("soundcard not installed. Run: pip install soundcard")
    exit(1)


def list_devices():
    """List available devices."""
    print("\n=== Output Devices (for Loopback) ===")
    for idx, spk in enumerate(sc.all_speakers()):
        print(f"  {idx}: {spk.name}")

    print("\n=== Input Devices (Microphones) ===")
    mics = sc.all_microphones(include_loopback=False)
    for idx, mic in enumerate(mics):
        print(f"  {idx}: {mic.name}")
    print()


class AudioStream:
    """Captures audio from a device and reports RMS."""

    def __init__(self, name: str, recorder, sample_rate: int = 16000):
        self.name = name
        self.recorder = recorder
        self.sample_rate = sample_rate
        self.running = False
        self.current_rms = 0.0
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._capture, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _capture(self):
        with self.recorder:
            while self.running:
                try:
                    chunk = self.recorder.record(numframes=int(self.sample_rate * 0.1))
                    if chunk.ndim == 2:
                        chunk = np.mean(chunk, axis=1)
                    self.current_rms = float(np.sqrt(np.mean(chunk**2)))
                except Exception as e:
                    print(f"[{self.name}] Error: {e}")
                    break


def main():
    parser = argparse.ArgumentParser(description="Test dual audio capture.")
    parser.add_argument("--list", "-l", action="store_true", help="List devices and exit")
    parser.add_argument("--mic", type=int, help="Microphone device index")
    parser.add_argument("--loopback", type=int, help="Loopback (speaker) device index")
    parser.add_argument("--duration", "-t", type=float, default=10.0, help="Test duration (seconds)")
    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    if args.mic is None or args.loopback is None:
        print("Error: Both --mic and --loopback are required.")
        print("Run with --list to see available devices.")
        return

    # Get devices
    speakers = sc.all_speakers()
    mics = sc.all_microphones(include_loopback=False)
    loopback_mics = sc.all_microphones(include_loopback=True)
    loopback_mics = [m for m in loopback_mics if getattr(m, "isloopback", False)]

    if args.mic >= len(mics):
        print(f"Error: Mic index {args.mic} out of range (0-{len(mics) - 1})")
        return
    if args.loopback >= len(speakers):
        print(f"Error: Loopback index {args.loopback} out of range (0-{len(speakers) - 1})")
        return

    mic = mics[args.mic]
    speaker = speakers[args.loopback]

    # Find matching loopback mic for the speaker
    loopback_mic = None
    for lm in loopback_mics:
        if lm.name == speaker.name or speaker.name.lower() in lm.name.lower():
            loopback_mic = lm
            break
    if loopback_mic is None and loopback_mics:
        loopback_mic = loopback_mics[0]

    if loopback_mic is None:
        print("Error: No loopback microphone found for the speaker.")
        return

    print(f"\n=== Starting Dual Capture Test ({args.duration}s) ===")
    print(f"  Mic:      [{args.mic}] {mic.name}")
    print(f"  Loopback: [{args.loopback}] {loopback_mic.name}")
    print()

    sample_rate = 16000

    # Create streams
    mic_stream = AudioStream("MIC", mic.recorder(samplerate=sample_rate, channels=1))
    loopback_stream = AudioStream("LOOPBACK", loopback_mic.recorder(samplerate=sample_rate, channels=1))

    # Start both
    mic_stream.start()
    loopback_stream.start()

    print("Recording... Speak into the mic and play audio to test both streams.\n")

    start_time = time.time()
    try:
        while (time.time() - start_time) < args.duration:
            mic_bar = "#" * int(min(mic_stream.current_rms * 100, 30))
            loop_bar = "#" * int(min(loopback_stream.current_rms * 100, 30))

            print(
                f"\r  MIC:      [{mic_bar:<30}] {mic_stream.current_rms:.4f}  |  "
                f"LOOPBACK: [{loop_bar:<30}] {loopback_stream.current_rms:.4f}",
                end="",
                flush=True,
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    mic_stream.stop()
    loopback_stream.stop()

    print("\n\n=== Test Complete ===")
    print("If both meters showed activity, dual capture works! ✅")


if __name__ == "__main__":
    main()
