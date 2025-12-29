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
    if 'initialized' not in st.session_state:
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
        st.session_state.suggestion_engine = SuggestionEngine()
        # Output queues for transcription results
        st.session_state.mic_output_queue = queue.Queue()
        st.session_state.loopback_output_queue = queue.Queue()
        # Cache devices at init to avoid COM issues on rerun
        st.session_state.devices_cache = None


def get_devices():
    """Get available audio devices (cached to avoid COM issues)."""
    if st.session_state.get('devices_cache') is not None:
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
    
    # Create transcribers
    st.session_state.mic_transcriber = Transcriber(
        st.session_state.capture.mic_queue,
        st.session_state.mic_output_queue
    )
    st.session_state.mic_transcriber.start()
    
    st.session_state.loopback_transcriber = Transcriber(
        st.session_state.capture.loopback_queue,
        st.session_state.loopback_output_queue
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
        st.session_state.mic_rms = levels['mic_rms']
        st.session_state.loopback_rms = levels['loopback_rms']


def process_transcripts():
    """Process any pending transcription results."""
    import time as time_module
    
    # Process mic transcripts
    while not st.session_state.mic_output_queue.empty():
        try:
            item = st.session_state.mic_output_queue.get_nowait()
            if item.get('text'):
                st.session_state.my_transcript.append({
                    'time': time_module.strftime('%H:%M:%S'),
                    'text': item['text']
                })
        except:
            break
    
    # Process loopback transcripts
    while not st.session_state.loopback_output_queue.empty():
        try:
            item = st.session_state.loopback_output_queue.get_nowait()
            if item.get('text'):
                st.session_state.other_transcript.append({
                    'time': time_module.strftime('%H:%M:%S'),
                    'text': item['text']
                })
        except:
            break


def generate_ai_suggestion(interval_seconds: int = 30):
    """Generate AI suggestion if enough time has passed."""
    current_time = time.time()
    if current_time - st.session_state.last_suggestion_time < interval_seconds:
        return
    
    # Build context from recent conversation (handle dict format)
    my_recent = st.session_state.my_transcript[-5:] if st.session_state.my_transcript else []
    other_recent = st.session_state.other_transcript[-5:] if st.session_state.other_transcript else []
    
    # Extract text from dict items
    my_texts = [item['text'] if isinstance(item, dict) else item for item in my_recent]
    other_texts = [item['text'] if isinstance(item, dict) else item for item in other_recent]
    
    if not my_texts and not other_texts:
        return
    
    context = "对方: " + " ".join(other_texts) + "\n我: " + " ".join(my_texts)
    
    try:
        suggestion = st.session_state.suggestion_engine.generate_suggestions(context)
        if suggestion:
            st.session_state.ai_suggestions.append({
                'time': time.strftime('%H:%M:%S'),
                'text': suggestion
            })
            st.session_state.last_suggestion_time = current_time
    except Exception as e:
        pass  # Silently fail if LLM not configured


def clear_session():
    """Clear all transcripts and suggestions."""
    st.session_state.my_transcript = []
    st.session_state.other_transcript = []
    st.session_state.ai_suggestions = []
    
    # Also clear pending queues
    if 'mic_output_queue' in st.session_state:
        while not st.session_state.mic_output_queue.empty():
            try:
                st.session_state.mic_output_queue.get_nowait()
            except:
                break
                
    if 'loopback_output_queue' in st.session_state:
        while not st.session_state.loopback_output_queue.empty():
            try:
                st.session_state.loopback_output_queue.get_nowait()
            except:
                break
