# AI Counseling Assistant

A local AI-powered counseling copilot that transcribes audio (microphone or system output) and provides real-time suggestions using LLMs (Ollama, OpenAI).

## Features

- **Real-time Transcription**: Supports multiple backends (FasterWhisper, FunASR, OpenAI, Azure).
- **Two Modes**:
    1.  **Counseling Copilot (Main App)**: Captures microphone input for live sessions.
    2.  **System Audio Transcriber**: Captures system output (e.g., meetings, videos) via WASAPI loopback.
- **AI Suggestions**: Generates counseling tips based on the transcript.

## Setup

1.  **Prerequisites**:
    - Python 3.10+
    - [ffmpeg](https://ffmpeg.org/download.html) installed and added to PATH.
    - [Ollama](https://ollama.com/) (optional, for local LLM).

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For GPU support, ensure you have CUDA installed and the appropriate `torch` version.*

3.  **Configuration**:
    Create a `.env` file in the root directory (copied from `.env.example` if available) or set environment variables:
    ```env
    # ASR Selection: local (FasterWhisper), funasr, openai, azure
    ASR_BACKEND=local
    ASR_LANGUAGE=zh
    
    # Model Config
    WHISPER_MODEL_SIZE=small
    DEVICE=cuda
    COMPUTE_TYPE=float16
    
    # LLM Config
    LLM_PROVIDER=ollama
    OLLAMA_MODEL=llama3
    ```

## Usage

### 1. Main Application (Microphone Input)
Use this for counseling sessions where you are speaking into a microphone.

```bash
python main.py
```
- Opens a web interface at `http://localhost:8080`.
- Select your **Microphone** device from the dropdown.

### 2. System Audio Transcription (Loopback)
Use this to transcribe audio playing from your computer (e.g., a Zoom call, video, or the "Foreground Player" simulation).

**Integration Note**: This uses a specialized script with WASAPI Loopback support.

1.  **List Devices**:
    ```bash
    python output_transcription/transcribe_output.py --list-devices
    ```
    *Look for your Speaker/Headphones in the Output devices list (e.g., Device 25).*

2.  **Start Capture**:
    Use the ID of your output device to capture its audio.
    ```bash
    python output_transcription/transcribe_output.py --device-id <DEVICE_ID>
    ```
    *Example: If your Speakerphone is Device 25, use `--device-id 25`. This will capture what you hear.*

## Troubleshooting

- **No Audio**: Check if the correct device ID is selected. WASAPI Loopback requires an active output stream (play some audio to test).
- **Crash on Start**: Ensure `ffmpeg` is on your PATH. If using CUDA, verify Nvidia drivers.
- **MKL Conflict**: If you see a silent crash, set `KMP_DUPLICATE_LIB_OK=TRUE` in your environment.
