# Plan: Run Chinese TTS → Audio Capture → Transcription Test on Windows

## Prerequisites (system level)
- **Python 3.12+** (install from https://www.python.org/downloads/windows/ and add to PATH)
- **ffmpeg** (download static binaries from https://ffmpeg.org/download.html#build-windows, extract, and add the `bin` folder to the system PATH, or install via Chocolatey: `choco install ffmpeg`).
- **PortAudio** is bundled with the Windows `sounddevice` wheel, so no extra install is needed.

## Steps
1. **Copy/clone the repository** to a Windows directory, e.g. `C:\Users\kgao\ai_counseling_assistant`.
2. **Create a virtual environment**:
   ```powershell
   cd C:\Users\kgao\ai_counseling_assistant
   python -m venv .venv
   .\.venv\Scripts\activate   # PowerShell
   # or .\.venv\Scripts\activate.bat for cmd
   ```
3. **Install Python dependencies**:
   ```powershell
   pip install -r requirements.txt
   # Force reinstall sounddevice and soundfile to ensure Windows wheels are used
   pip install --force-reinstall sounddevice soundfile
   ```
4. **Verify the audio stack**:
   ```powershell
   python -c "import sounddevice as sd; print('PortAudio version:', sd.get_portaudio_version())"
   ```
   You should see a version number (e.g., 19.6.0). If this succeeds, the library is correctly linked.
5. **Optional sanity‑check playback** (plays a 1‑second 440 Hz tone):
   ```powershell
   python - <<'PY'
   import numpy as np, sounddevice as sd
   fs = 44100
   t = np.linspace(0, 1, fs, False)
   tone = 0.5*np.sin(2*np.pi*440*t)
   sd.play(tone, fs)
   sd.wait()
   PY
   ```
6. **Run the Chinese TTS capture test**:
   ```powershell
   python tests\test_chinese_tts_capture.py
   ```
   The script will:
   - Generate Chinese speech with `gTTS` → `test_chinese.mp3`.
   - Convert MP3 → WAV using `ffmpeg`.
   - Start `AudioCapture` (records from the default microphone).
   - Play the WAV file.
   - Transcribe the captured audio with Whisper (language set to Chinese).
   - Verify that the transcription contains keywords like `你好`, `测试`, `识别`.
   Expected successful output includes lines such as:
   ```
   INFO: Generating TTS for: 你好。这是一个测试。我们正在测试语音识别系统。
   INFO: Converting MP3 to WAV...
   INFO: Playing audio test_chinese.wav...
   INFO: Captured: 你好。这是一个测试。我们正在测试语音识别系统。
   INFO: Test PASSED: Detected Chinese keywords.
   ```
7. **Cleanup** (optional):
   ```powershell
   del test_chinese.mp3
   del test_chinese.wav
   ```

## Troubleshooting (Windows)
- **ImportError: sounddevice** – ensure the virtual environment is activated and reinstall with `pip install --force-reinstall sounddevice`.
- **ffmpeg not found** – verify `ffmpeg -version` works; if not, add its `bin` directory to PATH and restart the terminal.
- **No sound heard** – check Windows Sound Settings → Output device is correct and not muted.
- **Transcription empty** – ensure the Whisper model downloaded correctly (first run may take a moment) and that the microphone is picking up the playback.
- **PortAudio errors** – rare on Windows; reinstall `sounddevice` wheel.

## Summary
Following the above steps on a native Windows environment avoids the WSL PortAudio issues and lets you quickly validate that Chinese speech can be generated, played back, captured, and correctly transcribed.
