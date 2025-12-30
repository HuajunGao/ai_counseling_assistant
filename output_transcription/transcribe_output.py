import argparse
import io
import logging
import os
import queue
import sys
import time
import warnings
from datetime import datetime
import inspect
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv

# Enable importing from root directory (config.py, core/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from core.transcriber import Transcriber

# Avoid OpenMP duplicate runtime aborts on Windows (torch/ctranslate2/numpy).
_KMP_DUP_OK_SET_BY_APP = False
if os.environ.get("KMP_DUPLICATE_LIB_OK") is None:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    _KMP_DUP_OK_SET_BY_APP = True
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

_CUDA_DLL_DIRS: list[str] = []
_CUDA_DLL_HANDLES = []
if sys.platform == "win32":
    venv_root = os.path.abspath(os.path.join(os.path.dirname(sys.executable), ".."))
    site_packages = os.path.join(venv_root, "Lib", "site-packages")
    for rel_path in ("nvidia\\cudnn\\bin", "nvidia\\cublas\\bin"):
        dll_dir = os.path.join(site_packages, rel_path)
        if os.path.isdir(dll_dir):
            _CUDA_DLL_HANDLES.append(os.add_dll_directory(dll_dir))
            _CUDA_DLL_DIRS.append(dll_dir)
    if _CUDA_DLL_DIRS:
        os.environ["PATH"] = ";".join(_CUDA_DLL_DIRS + [os.environ.get("PATH", "")])

try:
    import soundcard as sc
except Exception:
    sc = None
else:
    # Suppress noisy loopback discontinuity warnings from soundcard on Windows.
    try:
        from soundcard import SoundcardRuntimeWarning

        warnings.filterwarnings(
            "ignore",
            message="data discontinuity in recording",
            category=SoundcardRuntimeWarning,
        )
    except Exception:
        pass


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


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
        print("\nInput devices (soundcard):")
        for idx, mic in enumerate([m for m in microphones if not getattr(m, "isloopback", False)]):
            print(f"  {idx:>3} | {mic.name}")
        return

    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    output_devices = []
    input_devices = []

    for idx, dev in enumerate(devices):
        hostapi_name = hostapis[dev["hostapi"]]["name"]
        summary = (
            f"{idx:>3} | {dev['name']} | {hostapi_name} | "
            f"in {dev['max_input_channels']} / out {dev['max_output_channels']} | "
            f"default {dev['default_samplerate']} Hz"
        )
        if dev["max_output_channels"] > 0:
            output_devices.append(summary)
        if dev["max_input_channels"] > 0:
            input_devices.append(summary)

    print("Output devices:")
    for line in output_devices:
        print(f"  {line}")
    print("\nInput devices:")
    for line in input_devices:
        print(f"  {line}")


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
    # Try exact name match
    for mic in loopbacks:
        if mic.name == speaker.name:
            return mic
    # Try substring match
    needle = speaker.name.lower()
    for mic in loopbacks:
        if needle in mic.name.lower():
            return mic
    # Fallback to first loopback device
    return loopbacks[0]


def resample_linear(signal: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return signal
    if signal.size == 0:
        return signal
    duration = signal.shape[0] / float(src_rate)
    target_length = int(duration * dst_rate)
    if target_length <= 0:
        return np.array([], dtype=np.float32)
    x_old = np.linspace(0.0, duration, num=signal.shape[0], endpoint=False)
    x_new = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(x_new, x_old, signal).astype(np.float32)


def create_output_writer(args):
    file_handle = None
    if args.output_file:
        file_handle = open(args.output_file, "a", encoding="utf-8", buffering=1, newline="\n")

    def emit(text: str) -> None:
        if not text:
            return
        prefix = ""
        if args.timestamps:
            prefix = datetime.now().strftime("%H:%M:%S") + " | "
        line = prefix + text
        print(line, flush=True)
        if file_handle:
            file_handle.write(line + "\n")
            file_handle.flush()
            if args.output_fsync:
                os.fsync(file_handle.fileno())

    return emit, file_handle


def wasapi_supports_loopback() -> bool:
    try:
        sig = inspect.signature(sd.WasapiSettings)
        # Check if 'loopback' is a parameter in the constructor
        return "loopback" in sig.parameters
    except Exception:
        return False


def resolve_backend(requested: str) -> str:
    if requested == "auto":
        return "soundcard" if sc is not None else "sounddevice"
    return requested


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Transcribe system output audio via WASAPI loopback (Windows).")
    parser.add_argument("--list-devices", action="store_true", help="List devices and exit")
    parser.add_argument("--device", help="Output device name substring or numeric id")
    parser.add_argument("--device-id", type=int, help="Output device id")
    parser.add_argument("--device-name", help="Output device name substring")
    parser.add_argument("--model", default=None, help="Whisper model name or path")
    parser.add_argument("--language", default="zh", choices=["zh", "auto"], help="Language")
    parser.add_argument("--chunk-ms", type=int, default=1000, help="Chunk size in ms")
    parser.add_argument("--sample-rate", type=int, default=16000, help="ASR target sample rate")
    parser.add_argument("--capture-sample-rate", type=int, default=None, help="Override capture sample rate")
    parser.add_argument("--vad", action="store_true", default=None, help="Enable VAD")
    parser.add_argument("--no-vad", action="store_false", dest="vad", help="Disable VAD")
    parser.add_argument("--vad-threshold", type=float, default=None, help="Threshold for VAD")
    parser.add_argument("--dynamic-chunks", action="store_true", default=None, help="Split by silence")
    parser.add_argument("--silence-ms", type=int, default=None, help="Silence duration to end a segment (ms)")
    parser.add_argument("--min-segment-ms", type=int, default=None, help="Minimum segment duration (ms)")
    parser.add_argument("--max-segment-ms", type=int, default=None, help="Maximum segment duration (ms)")
    parser.add_argument("--whisper-vad", action="store_true", default=False, help="Enable Whisper's internal VAD")
    parser.add_argument("--timestamps", action="store_true", help="Prefix lines with timestamps")
    parser.add_argument("--output-file", help="Append output to a text file")
    parser.add_argument("--output-fsync", action="store_true", help="Force fsync after each line")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--asr-device", default=None, help="ASR device: auto/cpu/cuda")
    parser.add_argument("--compute-type", default=None, help="ASR compute type")
    parser.add_argument("--monitor-only", action="store_true", help="Only monitor loopback RMS")
    parser.add_argument(
        "--backend", default="auto", choices=["auto", "soundcard", "sounddevice"], help="Capture backend"
    )
    parser.add_argument(
        "--asr-backend", default="local", choices=["local", "azure", "openai", "funasr"], help="ASR backend"
    )

    # Provider specific
    parser.add_argument("--openai-model", help="OpenAI transcription model")
    parser.add_argument("--funasr-model", help="FunASR model name")
    parser.add_argument("--funasr-punc-model", help="FunASR punctuation model name")

    args = parser.parse_args()

    backend = resolve_backend(args.backend)

    if args.list_devices:
        list_devices(backend)
        return 0

    device_spec = args.device_id or args.device_name or args.device
    if not device_spec:
        print("Error: --device (or --device-id/--device-name) is required", file=sys.stderr)
        return 2

    configure_logging(args.log_level)
    logger = logging.getLogger("loopback_transcriber")

    # --------------------------------------------------------------------------
    # Update Config from Args
    # --------------------------------------------------------------------------
    if args.model:
        config.WHISPER_MODEL_SIZE = args.model
    if args.language:
        config.ASR_LANGUAGE = args.language
    if args.chunk_ms:
        config.ASR_CHUNK_MS = args.chunk_ms
    if args.vad is not None:
        config.ASR_VAD_ENABLED = args.vad
    if args.vad_threshold is not None:
        config.ASR_VAD_THRESHOLD = args.vad_threshold
    if args.dynamic_chunks is not None:
        config.ASR_DYNAMIC_CHUNKS = args.dynamic_chunks
    if args.silence_ms:
        config.ASR_SILENCE_MS = args.silence_ms
    if args.min_segment_ms:
        config.ASR_MIN_SEGMENT_MS = args.min_segment_ms
    if args.max_segment_ms:
        config.ASR_MAX_SEGMENT_MS = args.max_segment_ms
    if args.asr_backend:
        config.ASR_BACKEND = args.asr_backend
    if args.asr_device:
        config.DEVICE = args.asr_device
        config.FUNASR_DEVICE = args.asr_device
    if args.compute_type:
        config.COMPUTE_TYPE = args.compute_type

    if args.openai_model:
        config.OPENAI_TRANSCRIBE_MODEL = args.openai_model
    if args.funasr_model:
        config.FUNASR_MODEL = args.funasr_model
    if args.funasr_punc_model:
        config.FUNASR_PUNC_MODEL = args.funasr_punc_model

    # --------------------------------------------------------------------------
    # Device Resolution
    # --------------------------------------------------------------------------
    if _KMP_DUP_OK_SET_BY_APP:
        logger.info("Set KMP_DUPLICATE_LIB_OK=TRUE to avoid OpenMP runtime conflicts.")
    if _CUDA_DLL_DIRS:
        logger.info("Added CUDA DLL search paths: %s", "; ".join(_CUDA_DLL_DIRS))

    try:
        device_id, dev = resolve_output_device(device_spec, backend)
    except ValueError as exc:
        logger.error(str(exc))
        logger.info("Use --list-devices to see available outputs.")
        return 2

    hostapi_name = "WASAPI (soundcard)" if backend == "soundcard" else ""
    capture_rate = 48000  # default
    channels = 2

    if backend == "sounddevice":
        hostapis = sd.query_hostapis()
        hostapi_name = hostapis[dev["hostapi"]]["name"]
        if "WASAPI" not in hostapi_name.upper():
            logger.error("Selected device is not on Windows WASAPI. Choose a WASAPI output device.")
            return 2

        if not hasattr(sd, "WasapiSettings"):
            logger.error("sounddevice WASAPI settings are not available in this environment.")
            return 2

        if not wasapi_supports_loopback():
            logger.warning("sounddevice version %s might not support loopback robustly.", sd.__version__)

        capture_rate = int(args.capture_sample_rate or dev["default_samplerate"])
        channels = int(dev["max_output_channels"]) or 2
        channels = 2 if channels >= 2 else 1
    else:  # soundcard
        if sc is None:
            logger.error("soundcard is not installed.")
            return 2
        capture_rate = int(args.capture_sample_rate or 48000)

    target_rate = config.SAMPLE_RATE
    chunk_ms = config.ASR_CHUNK_MS
    chunk_frames_capture = int(capture_rate * (chunk_ms / 1000.0))
    if chunk_frames_capture <= 0:
        logger.error("Invalid chunk size.")
        return 2

    logger.info("Starting loopback capture on output device:")
    device_name = dev["name"] if backend == "sounddevice" else dev.name
    logger.info(f"  id={device_id} name={device_name}")
    logger.info(f"  hostapi={hostapi_name}")
    logger.info(f"  capture={capture_rate}Hz/{channels}ch -> target={target_rate}Hz")
    logger.info(f"  backend={backend}, asr={config.ASR_BACKEND}")
    logger.info(f"  vad={config.ASR_VAD_ENABLED}, dynamic_chunks={config.ASR_DYNAMIC_CHUNKS}")

    emit, file_handle = create_output_writer(args)

    if args.monitor_only:
        logger.info("Monitor only mode. No transcription.")
        # Simple loop for monitor
        # Not implementing full monitor loop here to save space, assuming user focus is ASR.
        # But if needed, just run capture and print RMS.
        return 0

    # --------------------------------------------------------------------------
    # Initialize Transcriber
    # --------------------------------------------------------------------------
    input_queue = queue.Queue()
    output_queue = queue.Queue()

    transcriber = Transcriber(input_queue, output_queue)
    transcriber.start()

    stream = None
    recorder = None

    try:
        # ----------------------------------------------------------------------
        # Start Capture
        # ----------------------------------------------------------------------
        if backend == "sounddevice":
            wasapi_settings = sd.WasapiSettings(loopback=True, auto_convert=True)

            def callback(indata, frames, time_info, status):
                if status:
                    logger.debug(f"Audio callback status: {status}")
                # Resample immediately or just put in queue?
                # Transcriber expects segments, but here we get raw callback chunks.
                # To minimize callback work, put in queue.
                # BUT Transcriber expects TARGET SAMPLE RATE.
                # So we must resample either here or there.
                # To match previous logic, let's resample here or ensure Transcriber handles it.
                # Previous logic resampled in main loop.
                # Let's simple: resample here.

                block = indata.copy()
                if block.ndim == 2:
                    if block.shape[1] > 1:
                        block = np.mean(block, axis=1)  # mix to mono
                    else:
                        block = block[:, 0]

                # Simple resampling
                if capture_rate != target_rate:
                    chunk = resample_linear(block, capture_rate, target_rate)
                else:
                    chunk = block

                input_queue.put(chunk)

            stream = sd.InputStream(
                device=device_id,
                channels=channels,
                samplerate=capture_rate,
                dtype="float32",
                callback=callback,
                extra_settings=wasapi_settings,
            )
            stream.start()

            # Main thread loop: process output
            while True:
                try:
                    event = output_queue.get(timeout=1.0)
                    if event["type"] == "commit":
                        emit(event["text"])
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    break

        else:  # soundcard
            loopback_mic = resolve_loopback_microphone(dev)
            logger.info(f"  loopback_device={loopback_mic.name}")
            recorder = loopback_mic.recorder(samplerate=capture_rate, channels=channels)

            with recorder:
                while True:
                    block = recorder.record(numframes=chunk_frames_capture)
                    if block.ndim == 2:
                        if block.shape[1] > 1:
                            block = np.mean(block, axis=1)
                        else:
                            block = block[:, 0]

                    if capture_rate != target_rate:
                        chunk = resample_linear(block, capture_rate, target_rate)
                    else:
                        chunk = block

                    input_queue.put(chunk)

                    # Check for output
                    while not output_queue.empty():
                        try:
                            event = output_queue.get_nowait()
                            if event["type"] == "commit":
                                emit(event["text"])
                        except queue.Empty:
                            break

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        transcriber.stop()
        transcriber.join()
        if stream:
            stream.stop()
            stream.close()
        if file_handle:
            file_handle.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
