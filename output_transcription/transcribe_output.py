import argparse
import logging
import os
import queue
import sys
import time
from datetime import datetime
import inspect

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

try:
    import soundcard as sc
except Exception:
    sc = None


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


def rms_energy(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal ** 2)))

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

def load_local_model(model_name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model_name, device=device, compute_type=compute_type)

def init_azure_recognizer(args, emit):
    try:
        import azure.cognitiveservices.speech as speechsdk
    except Exception as exc:
        raise RuntimeError(f"azure.cognitiveservices.speech not available: {exc}") from exc

    key = args.speech_key or os.getenv("SPEECH_KEY")
    region = args.speech_region or os.getenv("SPEECH_REGION")
    endpoint = args.speech_endpoint or os.getenv("SPEECH_ENDPOINT")

    if not key:
        raise RuntimeError("Missing SPEECH_KEY (set --speech-key or env var SPEECH_KEY).")
    if not region and not endpoint:
        raise RuntimeError("Missing SPEECH_REGION or SPEECH_ENDPOINT.")

    if endpoint:
        speech_config = speechsdk.SpeechConfig(subscription=key, endpoint=endpoint)
    else:
        speech_config = speechsdk.SpeechConfig(subscription=key, region=region)

    if args.language == "zh":
        speech_config.speech_recognition_language = "zh-CN"

    stream_format = speechsdk.audio.AudioStreamFormat(
        samples_per_second=16000,
        bits_per_sample=16,
        channels=1,
    )
    push_stream = speechsdk.audio.PushAudioInputStream(stream_format)
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

    auto_lang = None
    if args.language == "auto":
        auto_lang = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=["zh-CN", "en-US"]
        )

    if auto_lang:
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
            auto_detect_source_language_config=auto_lang,
        )
    else:
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )

    def on_recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            emit(evt.result.text)

    def on_canceled(evt):
        logging.getLogger("loopback_transcriber").warning(
            "Azure canceled: %s", evt.reason
        )

    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    recognizer.start_continuous_recognition()

    return recognizer, push_stream

def wasapi_supports_loopback() -> bool:
    try:
        sig = inspect.signature(sd.WasapiSettings)
        return "loopback" in sig.parameters
    except Exception:
        return False

def resolve_backend(requested: str) -> str:
    if requested == "auto":
        return "soundcard" if sc is not None else "sounddevice"
    return requested


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Transcribe system output audio via WASAPI loopback (Windows)."
    )
    parser.add_argument("--list-devices", action="store_true", help="List devices and exit")
    parser.add_argument("--device", help="Output device name substring or numeric id")
    parser.add_argument("--device-id", type=int, help="Output device id")
    parser.add_argument("--device-name", help="Output device name substring")
    parser.add_argument("--model", default="small", help="Whisper model name or path")
    parser.add_argument("--language", default="zh", choices=["zh", "auto"], help="Language")
    parser.add_argument("--chunk-ms", type=int, default=1000, help="Chunk size in ms")
    parser.add_argument("--sample-rate", type=int, default=16000, help="ASR target sample rate")
    parser.add_argument("--capture-sample-rate", type=int, default=None, help="Override capture sample rate")
    parser.add_argument("--vad", action="store_true", default=True, help="Enable energy VAD")
    parser.add_argument("--no-vad", action="store_false", dest="vad", help="Disable energy VAD")
    parser.add_argument("--whisper-vad", action="store_true", default=False, help="Enable Whisper's internal VAD")
    parser.add_argument("--timestamps", action="store_true", help="Prefix lines with timestamps")
    parser.add_argument("--output-file", help="Append output to a text file")
    parser.add_argument("--output-fsync", action="store_true", help="Force fsync after each line")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--rms-log-interval", type=float, default=2.0, help="Seconds between RMS debug logs")
    parser.add_argument("--asr-device", default="auto", help="ASR device: auto/cpu/cuda")
    parser.add_argument("--compute-type", default="auto", help="ASR compute type: auto/int8/float16/etc")
    parser.add_argument("--monitor-only", action="store_true", help="Only monitor loopback RMS without ASR")
    parser.add_argument("--backend", default="auto", choices=["auto", "soundcard", "sounddevice"], help="Capture backend")
    parser.add_argument("--asr-backend", default="local", choices=["local", "azure"], help="ASR backend")
    parser.add_argument("--speech-key", help="Azure Speech key (or SPEECH_KEY env)")
    parser.add_argument("--speech-region", help="Azure Speech region (or SPEECH_REGION env)")
    parser.add_argument("--speech-endpoint", help="Azure Speech endpoint (or SPEECH_ENDPOINT env)")

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

    try:
        device_id, dev = resolve_output_device(device_spec, backend)
    except ValueError as exc:
        logger.error(str(exc))
        logger.info("Use --list-devices to see available outputs.")
        return 2

    hostapi_name = "WASAPI (soundcard)" if backend == "soundcard" else ""
    if backend == "sounddevice":
        hostapis = sd.query_hostapis()
        hostapi_name = hostapis[dev["hostapi"]]["name"]
        if "WASAPI" not in hostapi_name.upper():
            logger.error(
                "Selected device is not on Windows WASAPI. Choose a WASAPI output device."
            )
            return 2

        if not hasattr(sd, "WasapiSettings"):
            logger.error("sounddevice WASAPI settings are not available in this environment.")
            return 2

        if not wasapi_supports_loopback():
            logger.error(
                "sounddevice %s does not expose WASAPI loopback. Use --backend soundcard.",
                sd.__version__,
            )
            return 2

        capture_rate = int(args.capture_sample_rate or dev["default_samplerate"])
    else:
        if sc is None:
            logger.error("soundcard is not installed. Install it or use --backend sounddevice.")
            return 2
        capture_rate = int(args.capture_sample_rate or 48000)
    target_rate = int(args.sample_rate)
    chunk_ms = max(100, int(args.chunk_ms))
    chunk_frames_capture = int(capture_rate * (chunk_ms / 1000.0))
    if chunk_frames_capture <= 0:
        logger.error("Invalid chunk size.")
        return 2

    if backend == "sounddevice":
        channels = int(dev["max_output_channels"]) or 2
        channels = 2 if channels >= 2 else 1
    else:
        channels = 2

    logger.info("Starting loopback capture on output device:")
    device_name = dev["name"] if backend == "sounddevice" else dev.name
    logger.info(f"  id={device_id} name={device_name}")
    logger.info(f"  hostapi={hostapi_name}")
    logger.info(f"  capture_rate={capture_rate} Hz, channels={channels}")
    logger.info(f"  target_rate={target_rate} Hz, chunk_ms={chunk_ms}")
    logger.info(f"  model={args.model}, language={args.language}")
    logger.info(f"  vad={args.vad}, whisper_vad={args.whisper_vad}, timestamps={args.timestamps}")
    logger.info(f"  backend={backend}")
    logger.info(f"  asr_backend={args.asr_backend}")
    if args.output_file:
        logger.info(f"  output_file={args.output_file}")

    emit, file_handle = create_output_writer(args)
    model = None
    azure_recognizer = None
    azure_stream = None
    if not args.monitor_only:
        if args.asr_backend == "local":
            try:
                model = load_local_model(args.model, device=args.asr_device, compute_type=args.compute_type)
                logger.info("Whisper model loaded.")
            except Exception as exc:
                logger.error(f"Failed to load model: {exc}")
                return 2
        elif args.asr_backend == "azure":
            try:
                azure_recognizer, azure_stream = init_azure_recognizer(args, emit)
                logger.info("Azure Speech recognizer started.")
            except Exception as exc:
                logger.error(f"Failed to start Azure Speech: {exc}")
                return 2

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    stream = None
    if backend == "sounddevice":
        wasapi_settings = sd.WasapiSettings(loopback=True, auto_convert=True)

        def callback(indata, frames, time_info, status):
            if status:
                logger.debug(f"Audio callback status: {status}")
            audio_queue.put(indata.copy())

        try:
            stream = sd.InputStream(
                device=device_id,
                channels=channels,
                samplerate=capture_rate,
                dtype="float32",
                callback=callback,
                extra_settings=wasapi_settings,
            )
            stream.start()
        except Exception as exc:
            logger.error(f"Failed to start loopback stream: {exc}")
            return 2

    if args.monitor_only:
        logger.info("Loopback capture started (monitor-only). Press Ctrl+C to stop.")
    else:
        logger.info("Loopback capture started. Press Ctrl+C to stop.")

    buffer = np.array([], dtype=np.float32)
    last_rms_log = 0.0
    try:
        if backend == "soundcard":
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
                    chunk = resample_linear(block, capture_rate, target_rate)
                    chunk_rms = rms_energy(chunk)
                    now = time.time()
                    if now - last_rms_log >= args.rms_log_interval:
                        logger.debug(f"Chunk RMS: {chunk_rms:.6f}")
                        last_rms_log = now

                    if args.monitor_only:
                        continue

                    if args.vad and chunk_rms < 0.01:
                        continue

                    if args.asr_backend == "azure":
                        pcm = np.clip(chunk, -1.0, 1.0)
                        pcm = (pcm * 32767.0).astype(np.int16)
                        azure_stream.write(pcm.tobytes())
                        continue

                    segments, info = model.transcribe(
                        chunk,
                        language=None if args.language == "auto" else args.language,
                        vad_filter=args.whisper_vad,
                        beam_size=5,
                    )
                    text = " ".join(segment.text.strip() for segment in segments).strip()
                    emit(text)
        else:
            while True:
                try:
                    block = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if block.ndim == 2:
                    if block.shape[1] > 1:
                        block = np.mean(block, axis=1)
                    else:
                        block = block[:, 0]
                buffer = np.concatenate((buffer, block))

                while buffer.shape[0] >= chunk_frames_capture:
                    chunk = buffer[:chunk_frames_capture]
                    buffer = buffer[chunk_frames_capture:]

                    chunk = resample_linear(chunk, capture_rate, target_rate)

                    chunk_rms = rms_energy(chunk)
                    now = time.time()
                    if now - last_rms_log >= args.rms_log_interval:
                        logger.debug(f"Chunk RMS: {chunk_rms:.6f}")
                        last_rms_log = now

                    if args.monitor_only:
                        continue

                    if args.vad and chunk_rms < 0.01:
                        continue

                    if args.asr_backend == "azure":
                        pcm = np.clip(chunk, -1.0, 1.0)
                        pcm = (pcm * 32767.0).astype(np.int16)
                        azure_stream.write(pcm.tobytes())
                        continue

                    segments, info = model.transcribe(
                        chunk,
                        language=None if args.language == "auto" else args.language,
                        vad_filter=args.whisper_vad,
                        beam_size=5,
                    )
                    text = " ".join(segment.text.strip() for segment in segments).strip()
                    emit(text)

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        try:
            if stream:
                stream.stop()
                stream.close()
        except Exception:
            pass
        try:
            if azure_recognizer:
                azure_recognizer.stop_continuous_recognition()
        except Exception:
            pass
        try:
            if azure_stream:
                azure_stream.close()
        except Exception:
            pass
        if file_handle:
            file_handle.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
