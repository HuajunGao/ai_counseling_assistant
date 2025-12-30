import argparse
import sys
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

try:
    import soundcard as sc
except Exception:
    sc = None


def list_devices(backend: str) -> None:
    if backend == "soundcard":
        if sc is None:
            print("soundcard is not installed. Install it to list devices.")
            return
        speakers = sc.all_speakers()
        microphones = sc.all_microphones(include_loopback=True)
        print("Output devices (soundcard):")
        for idx, spk in enumerate(speakers):
            print(f"  {idx:>3} | {spk.name}")
        print("\nLoopback devices (soundcard):")
        for idx, mic in enumerate([m for m in microphones if getattr(m, "isloopback", False)]):
            print(f"  {idx:>3} | {mic.name}")
        return

    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    output_devices = []
    for idx, dev in enumerate(devices):
        if dev["max_output_channels"] <= 0:
            continue
        hostapi_name = hostapis[dev["hostapi"]]["name"]
        output_devices.append(
            f"{idx:>3} | {dev['name']} | {hostapi_name} | "
            f"out {dev['max_output_channels']} | {dev['default_samplerate']} Hz"
        )
    print("Output devices (sounddevice):")
    for line in output_devices:
        print(f"  {line}")


def resolve_backend(requested: str) -> str:
    if requested == "auto":
        return "soundcard" if sc is not None else "sounddevice"
    return requested


def resolve_output_device(device_spec: str | int, backend: str):
    if backend == "soundcard":
        if sc is None:
            raise ValueError("soundcard is not installed.")
        speakers = sc.all_speakers()
        if isinstance(device_spec, int) or str(device_spec).isdigit():
            device_id = int(device_spec)
            if device_id < 0 or device_id >= len(speakers):
                raise ValueError(f"Device id {device_id} is out of range.")
            return device_id, speakers[device_id]
        needle = str(device_spec).strip().lower()
        matches = [(idx, spk) for idx, spk in enumerate(speakers) if needle in spk.name.lower()]
        if not matches:
            raise ValueError(f"No output device matched name: {device_spec}")
        if len(matches) > 1:
            print("Multiple output devices matched. Please be more specific or use --device-id.")
            for idx, spk in matches:
                print(f"  {idx:>3} | {spk.name}")
            raise ValueError("Ambiguous device name.")
        return matches[0]

    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    if isinstance(device_spec, int) or str(device_spec).isdigit():
        device_id = int(device_spec)
        if device_id < 0 or device_id >= len(devices):
            raise ValueError(f"Device id {device_id} is out of range.")
        dev = devices[device_id]
        if dev["max_output_channels"] <= 0:
            raise ValueError(f"Device id {device_id} is not an output device.")
        return device_id, dev
    needle = str(device_spec).strip().lower()
    matches = []
    for idx, dev in enumerate(devices):
        if dev["max_output_channels"] <= 0:
            continue
        if needle in dev["name"].lower():
            matches.append((idx, dev))
    if not matches:
        raise ValueError(f"No output device matched name: {device_spec}")
    if len(matches) > 1:
        print("Multiple output devices matched. Please be more specific or use --device-id.")
        for idx, dev in matches:
            hostapi_name = hostapis[dev["hostapi"]]["name"]
            print(f"  {idx:>3} | {dev['name']} | {hostapi_name}")
        raise ValueError("Ambiguous device name.")
    return matches[0]


def resolve_loopback_microphone(speaker) -> "sc._Microphone":
    microphones = sc.all_microphones(include_loopback=True)
    loopbacks = [m for m in microphones if getattr(m, "isloopback", False)]
    if not loopbacks:
        raise ValueError("No loopback microphones found via soundcard.")
    for mic in loopbacks:
        if mic.name == speaker.name:
            return mic
    needle = speaker.name.lower()
    for mic in loopbacks:
        if needle in mic.name.lower():
            return mic
    return loopbacks[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a short loopback audio clip for benchmark purposes.")
    parser.add_argument("--output", default="benchmark.wav", help="Output WAV file path")
    parser.add_argument("--duration", type=float, default=15.0, help="Duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate")
    parser.add_argument("--channels", type=int, default=1, help="Number of channels")
    parser.add_argument("--device", help="Output device name substring or numeric id")
    parser.add_argument("--device-id", type=int, help="Output device id")
    parser.add_argument("--device-name", help="Output device name substring")
    parser.add_argument("--list-devices", action="store_true", help="List devices and exit")
    parser.add_argument(
        "--backend", default="auto", choices=["auto", "soundcard", "sounddevice"], help="Capture backend"
    )
    args = parser.parse_args()

    backend = resolve_backend(args.backend)

    if args.list_devices:
        list_devices(backend)
        return 0

    if args.duration <= 0:
        print("Duration must be positive.")
        return 2

    device_spec = args.device_id or args.device_name or args.device
    if not device_spec:
        print("Error: --device (or --device-id/--device-name) is required", file=sys.stderr)
        return 2

    try:
        device_id, dev = resolve_output_device(device_spec, backend)
    except ValueError as exc:
        print(str(exc))
        return 2

    print(f"Recording {args.duration:.1f}s loopback to {args.output} ...")
    time.sleep(0.2)

    if backend == "soundcard":
        if sc is None:
            print("soundcard is not installed. Install it or use --backend sounddevice.")
            return 2
        loopback_mic = resolve_loopback_microphone(dev)
        recorder = loopback_mic.recorder(samplerate=args.sample_rate, channels=args.channels)
        with recorder:
            block = recorder.record(numframes=int(args.duration * args.sample_rate))
        if block.ndim == 2 and block.shape[1] > 1:
            block = np.mean(block, axis=1)
        sf.write(args.output, block, args.sample_rate)
    else:
        try:
            wasapi_settings = sd.WasapiSettings(loopback=True, auto_convert=True)
        except Exception:
            print("sounddevice WASAPI loopback is not available. Use --backend soundcard.")
            return 2
        audio = sd.rec(
            int(args.duration * args.sample_rate),
            samplerate=args.sample_rate,
            channels=args.channels,
            device=device_id,
            dtype="float32",
            extra_settings=wasapi_settings,
        )
        sd.wait()
        sf.write(args.output, audio, args.sample_rate)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
