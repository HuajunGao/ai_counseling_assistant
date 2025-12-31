# AI Counseling Assistant

A localized AI-powered counseling copilot designed to assist professional counselors. It features real-time dual-stream transcription, live AI-driven supervision, and comprehensive session management including visitor profiles and historical session browsing.

## ✨ Core Features

- **🎬 Real-time Dual-Stream Transcription**: Simulataneously captures input from the Counselor (Microphone) and the Client (System Audio via WASAPI loopback).
- **🤖 AI Live Supervision**: Provides precise, concise professional suggestions based on real-time conversation context. Supports manual queries to the AI.
- **📝 Chronological Dialogue Storage**: Saves session records in a natural dialogue format (sorted by time) instead of separate tracks.
- **📜 Visitor Profiles & History**:
    - **Automated Profiling**: AI automatically generates a one-sentence descriptive summary for each visitor.
    - **History Browser**: Filter past records by Visitor ID, view AI-generated session summaries, and expand full conversational histories.
- **🚀 Multi-Backend ASR Support**: Flexible switching between FunASR (optimized for Mandarin), OpenAI Whisper, Azure Speech, etc.

## 🛠️ Prerequisites

1.  **System Requirements**:
    - Python 3.10+
    - [ffmpeg](https://ffmpeg.org/download.html) installed and added to your system PATH.
    - Microsoft Windows (required for WASAPI loopback support).

2.  **Installation**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For GPU acceleration, ensure CUDA is installed and configured in `config.py`.*

3.  **Environment Setup**:
    Create a `.env` file in the root directory or modify `config.py` directly:
    ```env
    OPENAI_API_KEY=your_key
    OPENAI_BASE_URL=https://api.openai.com/v1
    ASR_BACKEND=funasr  # Options: funasr, openai, azure, local
    ```

## 🚀 Quick Start

```bash
# Start the Streamlit application
uv run streamlit run ui/streamlit_app.py
```

The application will be available at `http://localhost:8501`.

## 📖 Usage Guide

### 1. 📝 Real-time Session
- Select your **Microphone** and **Speaker** devices in the **Settings** tab.
- Click **Start Recording** to begin the session.
- AI feedback will appear on the right panel automatically or upon manual request.

### 2. 💾 Saving Sessions
- Input a **Visitor ID** (defaults to current timestamp).
- Click **Save Session**. The system will generate an AI summary and visitor description, then store the record in the `sessions/` directory.

### 3. 📜 Session History
- Switch to the **History** tab.
- Search or select a Visitor ID to view their AI profile.
- Select a specific session date to see the summary and expand the full dialogue history.

## 💻 Tech Stack

- **Frontend**: Streamlit
- **ASR**: FunASR / OpenAI Whisper / Azure Speech
- **LLM**: GPT-4o / Claude / Ollama
- **Audio Utility**: SoundCard (WASAPI), PyAudio
- **Concurrency**: Threading + Asyncio

## ❓ Troubleshooting

- **MKL Conflict**: If the app crashes silently, set the environment variable `KMP_DUPLICATE_LIB_OK=TRUE`.
- **No audio captured?**: Ensure you've selected correct active devices and the speaker has actual sound output.
