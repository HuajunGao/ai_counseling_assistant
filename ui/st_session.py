"""
Streamlit session state management for audio capture and transcription.
"""

import threading
import queue
import time
import sys
from typing import Optional
import streamlit as st

# Windows COM initialization for Streamlit
if sys.platform == "win32":
    try:
        import pythoncom

        pythoncom.CoInitialize()
    except ImportError:
        pass

from core.dual_capture import DualStreamCapture, list_devices
from core.transcriber import Transcriber
from core.llm_engine import SuggestionEngine


def init_session_state():
    """Initialize all session state variables."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.is_recording = False
        st.session_state.capture = None
        st.session_state.mic_transcriber = None
        st.session_state.loopback_transcriber = None
        st.session_state.my_transcript = []  # List of transcribed text from mic
        st.session_state.other_transcript = []  # List of transcribed text from loopback
        st.session_state.ai_suggestions = []  # List of AI suggestions
        st.session_state.mic_rms = 0.0
        st.session_state.loopback_rms = 0.0
        st.session_state.last_suggestion_time = 0
        st.session_state.last_transcript_len = 0  # Track total messages to avoid duplicate suggestions
        st.session_state.suggestion_engine = SuggestionEngine()
        # Output queues for transcription results
        st.session_state.mic_output_queue = queue.Queue()
        st.session_state.loopback_output_queue = queue.Queue()
        # Cache devices at init to avoid COM issues on rerun
        st.session_state.devices_cache = None


def get_devices():
    """Get available audio devices (cached to avoid COM issues)."""
    if st.session_state.get("devices_cache") is not None:
        return st.session_state.devices_cache

    # Initialize COM for this thread on Windows
    if sys.platform == "win32":
        try:
            import pythoncom

            pythoncom.CoInitialize()
        except:
            pass

    devices = list_devices()
    st.session_state.devices_cache = devices
    return devices


def start_recording(mic_idx: int, loopback_idx: int):
    """Start dual-stream audio capture and transcription."""
    if st.session_state.is_recording:
        return

    # Create capture
    st.session_state.capture = DualStreamCapture(mic_idx, loopback_idx)
    st.session_state.capture.start()

    # Build separate ASR configs for mic and loopback
    import config as base_config
    
    mic_asr_backend = st.session_state.get("mic_asr_backend", base_config.ASR_BACKEND)
    loopback_asr_backend = st.session_state.get("loopback_asr_backend", base_config.ASR_BACKEND)
    
    # Create config-like objects for each transcriber
    class ASRConfig:
        pass
    
    mic_config = ASRConfig()
    for attr in dir(base_config):
        if not attr.startswith("_"):
            setattr(mic_config, attr, getattr(base_config, attr))
    mic_config.ASR_BACKEND = mic_asr_backend

    loopback_config = ASRConfig()
    for attr in dir(base_config):
        if not attr.startswith("_"):
            setattr(loopback_config, attr, getattr(base_config, attr))
    loopback_config.ASR_BACKEND = loopback_asr_backend

    # Create transcribers with separate configs
    st.session_state.mic_transcriber = Transcriber(
        st.session_state.capture.mic_queue, st.session_state.mic_output_queue, config=mic_config
    )
    st.session_state.mic_transcriber.start()

    st.session_state.loopback_transcriber = Transcriber(
        st.session_state.capture.loopback_queue, st.session_state.loopback_output_queue, config=loopback_config
    )
    st.session_state.loopback_transcriber.start()

    st.session_state.is_recording = True


def stop_recording():
    """Stop audio capture and transcription."""
    if not st.session_state.is_recording:
        return

    if st.session_state.capture:
        st.session_state.capture.stop()
    if st.session_state.mic_transcriber:
        st.session_state.mic_transcriber.stop()
    if st.session_state.loopback_transcriber:
        st.session_state.loopback_transcriber.stop()

    st.session_state.is_recording = False


def update_levels():
    """Update audio level values."""
    if st.session_state.capture:
        levels = st.session_state.capture.get_levels()
        st.session_state.mic_rms = levels["mic_rms"]
        st.session_state.loopback_rms = levels["loopback_rms"]


def process_transcripts():
    """Process any pending transcription results."""
    import time as time_module

    def format_time_with_latency(item):
        """Format time string with latency if available."""
        time_str = time_module.strftime("%H:%M:%S")
        latency = item.get("latency", 0)
        if latency > 0:
            return f"{time_str}[{latency:.1f}s]"
        return time_str

    # Process mic transcripts
    while not st.session_state.mic_output_queue.empty():
        try:
            item = st.session_state.mic_output_queue.get_nowait()
            if item.get("text"):
                st.session_state.my_transcript.append({"time": format_time_with_latency(item), "text": item["text"]})
        except:
            break

    # Process loopback transcripts
    while not st.session_state.loopback_output_queue.empty():
        try:
            item = st.session_state.loopback_output_queue.get_nowait()
            if item.get("text"):
                st.session_state.other_transcript.append(
                    {"time": format_time_with_latency(item), "text": item["text"]}
                )
        except:
            break


def generate_ai_suggestion(interval_seconds: int = 30, user_question: str = ""):
    """Generate AI suggestion if enough time has passed or if user asked a question."""
    current_time = time.time()
    
    # If user asked a question, always generate (skip interval check)
    has_question = bool(user_question and user_question.strip())
    if not has_question and current_time - st.session_state.last_suggestion_time < interval_seconds:
        return

    # Get transcripts (speaker = 倾诉者 = other, listener = 倾听者 = my)
    speaker_transcript = st.session_state.other_transcript or []
    listener_transcript = st.session_state.my_transcript or []

    # Check if there is new content since last suggestion (skip if user asked question)
    current_len = len(listener_transcript) + len(speaker_transcript)
    if not has_question and current_len == st.session_state.get("last_transcript_len", 0):
        return

    if not speaker_transcript and not listener_transcript and not has_question:
        return

    try:
        suggestion = st.session_state.suggestion_engine.generate_suggestions(
            speaker_transcript=speaker_transcript,
            listener_transcript=listener_transcript,
            user_question=user_question
        )
        if suggestion:
            st.session_state.ai_suggestions.append({"time": time.strftime("%H:%M:%S"), "text": suggestion})
            st.session_state.last_suggestion_time = current_time
            st.session_state.last_transcript_len = current_len
    except Exception as e:
        pass  # Silently fail if LLM not configured


def clear_session():
    """Clear all transcripts and suggestions."""
    st.session_state.my_transcript = []
    st.session_state.other_transcript = []
    st.session_state.ai_suggestions = []
    st.session_state.last_transcript_len = 0

    # Also clear pending queues
    if "mic_output_queue" in st.session_state:
        while not st.session_state.mic_output_queue.empty():
            try:
                st.session_state.mic_output_queue.get_nowait()
            except:
                break

    if "loopback_output_queue" in st.session_state:
        while not st.session_state.loopback_output_queue.empty():
            try:
                st.session_state.loopback_output_queue.get_nowait()
            except:
                break
