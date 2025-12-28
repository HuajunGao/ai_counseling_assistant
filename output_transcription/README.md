# System Output (Speaker) Loopback Transcription Test

This folder contains a focused, standalone script to verify **system output loopback** transcription on Windows 11.
It **does not** use a microphone. Audio is captured from the selected **output device** via **WASAPI loopback**.

## Install

From the repo root:

```bash
pip install -r requirements.txt
```

Note: `sounddevice` does not expose WASAPI loopback in its latest release. This script uses the `soundcard`
backend by default if installed.

## List devices

```bash
python output_transcription/transcribe_output.py --list-devices
```

Pick an **output device** that shows `Windows WASAPI` in the list.

## Run

By device name (substring match):

```bash
python output_transcription/transcribe_output.py --device "Headphones (Realtek(R) Audio)"
```

By device id:

```bash
python output_transcription/transcribe_output.py --device-id 21
```

If you want to force the backend:

```bash
python output_transcription/transcribe_output.py --backend soundcard --list-devices
python output_transcription/transcribe_output.py --backend soundcard --device-id 0
```

## Options

- `--model` Whisper model name or path (default: `small`)
- `--language` `zh` or `auto`
- `--chunk-ms` chunk size in ms (default: 1000)
- `--sample-rate` ASR target sample rate (default: 16000)
- `--capture-sample-rate` override device capture sample rate
- `--vad` / `--no-vad` enable or disable simple energy VAD
- `--timestamps` prefix output with `HH:MM:SS`
- `--output-file` append output to a text file
- `--backend` `auto` (default), `soundcard`, or `sounddevice`
- `--asr-backend` `local` (default) or `azure`
- `--speech-key` Azure Speech key (or `SPEECH_KEY` env var)
- `--speech-region` Azure Speech region (or `SPEECH_REGION` env var)
- `--speech-endpoint` Azure Speech endpoint (or `SPEECH_ENDPOINT` env var)

## Notes

- This script uses **WASAPI loopback** to capture **system output**, not the microphone.
- If you disable or unplug the microphone, loopback capture should still work.
- For validation, play a known Chinese audio clip (YouTube or local file) and confirm transcription.
- Azure backend uses `.env` (see repo root) or CLI flags for credentials.

## Troubleshooting

- **No WASAPI device**: ensure you select a device labeled `Windows WASAPI`, or use `--backend soundcard`.
- **Device busy / cannot start**: close apps that have exclusive control of the device.
- **No transcription**: try `--no-vad` or increase `--chunk-ms` to 2000.
- **High latency**: reduce `--chunk-ms` (500 is a good starting point).
