"""
AI Counseling Copilot - Streamlit Application
Main entry point for the Streamlit UI.
"""
# DLL path setup for Windows - MUST be before any torch imports
import os
import sys

if os.environ.get("KMP_DUPLICATE_LIB_OK") is None:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

_CUDA_DLL_DIRS = []
if sys.platform == "win32":
    # Initialize COM for Windows audio APIs (required by soundcard library)
    try:
        import pythoncom
        pythoncom.CoInitialize()
    except ImportError:
        pass
    
    venv_root = os.path.abspath(os.path.join(os.path.dirname(sys.executable), ".."))
    site_packages = os.path.join(venv_root, "Lib", "site-packages")
    current_path = os.environ.get("PATH", "")
    
    for rel_path in ("nvidia\\cudnn\\bin", "nvidia\\cublas\\bin"):
        dll_dir = os.path.join(site_packages, rel_path)
        if os.path.isdir(dll_dir):
            # Only add DLL directory if not already in PATH
            if dll_dir not in current_path:
                os.add_dll_directory(dll_dir)
                _CUDA_DLL_DIRS.append(dll_dir)
    
    # Only update PATH if we have new dirs to add (avoid repeated appends)
    if _CUDA_DLL_DIRS:
        os.environ["PATH"] = ";".join(_CUDA_DLL_DIRS) + ";" + current_path

import streamlit as st
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.st_session import (
    init_session_state,
    get_devices,
    start_recording,
    stop_recording,
    update_levels,
    process_transcripts,
    generate_ai_suggestion,
    clear_session
)
from ui.st_components import (
    device_selectors,
    level_meters,
    control_buttons,
    transcript_panel,
    ai_suggestions_panel,
    status_indicator
)

# Page config
st.set_page_config(
    page_title="AI Counseling Copilot",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for compact layout that fills viewport
st.markdown("""
<style>
    .main > div {
        padding-top: 0.5rem;
    }
    .stProgress > div > div > div {
        background-color: #1f77b4;
    }
    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        max-width: 100%;
    }
    /* Hide default header/footer if any */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
init_session_state()

# Header - compact but visible
st.markdown("## 🎙️ AI Counseling Copilot")

# Device selection and controls
devices = get_devices()
mic_idx, speaker_idx = device_selectors(devices)

# Level meters and status
col_levels, col_status = st.columns([2, 1])
with col_levels:
    level_meters(st.session_state.mic_rms, st.session_state.loopback_rms)
with col_status:
    status_indicator(st.session_state.is_recording)

# Control buttons
start_clicked, stop_clicked, clear_clicked = control_buttons(st.session_state.is_recording)

if start_clicked:
    start_recording(mic_idx, speaker_idx)
    st.rerun()

if stop_clicked:
    stop_recording()
    st.rerun()

if clear_clicked:
    clear_session()
    st.rerun()


# Main content - 3 columns
col_left, col_center, col_right = st.columns([3, 4, 3])

with col_left:
    transcript_panel("我", "🧑", st.session_state.my_transcript, "blue")

with col_center:
    ai_suggestions_panel(st.session_state.ai_suggestions)

with col_right:
    transcript_panel("对方", "👤", st.session_state.other_transcript, "green")

# Auto-refresh logic when recording
if st.session_state.is_recording:
    # Update levels and process transcripts
    update_levels()
    process_transcripts()
    
    # Generate AI suggestion periodically
    generate_ai_suggestion(interval_seconds=30)
    
    # Auto-refresh every 1 second
    time.sleep(1)
    st.rerun()
