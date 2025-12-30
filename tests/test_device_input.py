"""
CLI tool to list audio devices and test real-time input capture.
Usage:
  python tests/test_device_input.py                  # List all devices
  python tests/test_device_input.py --device 25      # Test device 25
  python tests/test_device_input.py --device 25 --loopback  # Test WASAPI loopback
"""

import argparse
import sys
import time
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice not installed. Run: pip install sounddevice")
    sys.exit(1)


def list_devices():
    """Print all audio devices with their capabilities."""
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    print("\n=== Audio Devices ===")
    print(f"{'ID':<4} {'Name':<45} {'API':<15} {'In':<3} {'Out':<3} {'Rate'}")
    print("-" * 85)

    for idx, dev in enumerate(devices):
        api_name = hostapis[dev["hostapi"]]["name"]
        in_ch = dev["max_input_channels"]
        out_ch = dev["max_output_channels"]
        rate = int(dev["default_samplerate"])

        # Highlight WASAPI devices (required for loopback)
        marker = " *" if "WASAPI" in api_name else ""
        print(f"{idx:<4} {dev['name'][:44]:<45} {api_name[:14]:<15} {in_ch:<3} {out_ch:<3} {rate}{marker}")

    print("\n* = WASAPI device (supports loopback capture)")
    print(f"\nDefault Input:  {sd.default.device[0]}")
    print(f"Default Output: {sd.default.device[1]}")


def test_device(device_id: int, duration: float = 5.0, loopback: bool = False):
    """Test audio capture on the specified device."""
    devices = sd.query_devices()
    if device_id < 0 or device_id >= len(devices):
        print(f"Error: Device ID {device_id} not found.")
        return

    dev = devices[device_id]
    print(f"\n=== Testing Device {device_id}: {dev['name']} ===")
    print(f"Channels: {dev['max_input_channels']} in / {dev['max_output_channels']} out")
    print(f"Sample Rate: {int(dev['default_samplerate'])} Hz")

    sample_rate = int(dev["default_samplerate"])
    channels = 1

    # For loopback, we open an OUTPUT device as input
    if loopback:
        if dev["max_output_channels"] <= 0:
            print("Error: Loopback requires an OUTPUT device.")
            return
        if not hasattr(sd, "WasapiSettings"):
            print("Error: WASAPI loopback not available on this system.")
            return
        channels = min(2, dev["max_output_channels"])
        print(f"[LOOPBACK MODE] Capturing system audio from output device...")
    else:
        if dev["max_input_channels"] <= 0:
            print("Error: This is not an input device. Use --loopback for output devices.")
            return
        channels = 1

    print(f"\nListening for {duration} seconds... Speak or play audio now!")
    print("-" * 40)

    peak_rms = 0.0
    samples_received = 0

    def callback(indata, frames, time_info, status):
        nonlocal peak_rms, samples_received
        if status:
            print(f"  [Status: {status}]")

        # Convert to mono if stereo
        data = indata.copy()
        if data.ndim == 2 and data.shape[1] > 1:
            data = np.mean(data, axis=1)

        rms = np.sqrt(np.mean(data**2))
        samples_received += frames

        if rms > peak_rms:
            peak_rms = rms

        # Visual bar
        bar_len = int(min(rms * 500, 50))
        bar = "#" * bar_len
        print(f"\r  RMS: {rms:.4f} [{bar:<50}]", end="", flush=True)

    try:
        extra_settings = None
        if loopback:
            extra_settings = sd.WasapiSettings(loopback=True)

        with sd.InputStream(
            device=device_id,
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            callback=callback,
            extra_settings=extra_settings,
        ):
            time.sleep(duration)
    except Exception as e:
        print(f"\nError: {e}")
        return

    print("\n" + "-" * 40)
    print(f"Samples received: {samples_received}")
    print(f"Peak RMS: {peak_rms:.4f}")

    if peak_rms > 0.01:
        print("\n✅ SUCCESS: Audio signal detected!")
    elif peak_rms > 0.001:
        print("\n⚠️  WEAK: Some audio detected, but very quiet.")
    else:
        print("\n❌ SILENCE: No audio detected. Check device or source.")


def main():
    parser = argparse.ArgumentParser(description="Test audio device input capture.")
    parser.add_argument("--device", "-d", type=int, help="Device ID to test")
    parser.add_argument("--loopback", "-l", action="store_true", help="Use WASAPI loopback (for output devices)")
    parser.add_argument("--duration", "-t", type=float, default=5.0, help="Test duration in seconds")
    args = parser.parse_args()

    if args.device is not None:
        test_device(args.device, args.duration, args.loopback)
    else:
        list_devices()
        print("\nTo test a device, run:")
        print("  python tests/test_device_input.py --device <ID>")
        print("  python tests/test_device_input.py --device <ID> --loopback  # For speakers")


if __name__ == "__main__":
    main()
