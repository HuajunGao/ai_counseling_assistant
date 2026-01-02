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
    clear_session,
    save_session,
    get_existing_visitor_ids,
    get_all_visitor_info,
    get_sessions_list,
    load_specific_session,
)
from core.session_storage import generate_default_visitor_id
from ui.st_components import (
    device_selectors,
    level_meters,
    control_buttons,
    ai_settings_panel,
    transcript_panel,
    ai_suggestions_panel,
    status_indicator,
    visitor_id_input,
)
import config

# Page config
st.set_page_config(page_title="AI Counseling Copilot", page_icon="🎙️", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for compact layout that fills viewport
st.markdown(
    """
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
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 6px 16px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
init_session_state()

# Header - compact but visible
st.markdown("## 🎙️ AI Counseling Copilot")

# Create tabs: Main, History and Config
tab_main, tab_history, tab_config = st.tabs(["📝 对话", "📜 历史记录", "⚙️ 设置"])

# ===== CONFIG TAB =====
with tab_config:
    st.markdown("### 设备设置")
    devices = get_devices()
    mic_idx, speaker_idx = device_selectors(devices, config.DEFAULT_MIC_NAME, config.DEFAULT_SPEAKER_NAME)

    st.markdown("### AI 设置")
    ai_model, ai_interval, ai_context_len, mic_asr, loopback_asr = ai_settings_panel(
        config.OPENAI_MODELS, config.ASR_BACKENDS, config.OPENAI_MODEL
    )

    # Update suggestion engine with selected model
    if st.session_state.suggestion_engine:
        st.session_state.suggestion_engine.set_model(ai_model)
        st.session_state.suggestion_engine.set_context_length(ai_context_len)

    # Store ASR settings in session state for use when starting recording
    backend_map = {"funasr": "funasr", "openai": "openai", "azure": "azure"}
    st.session_state.mic_asr_backend = backend_map.get(mic_asr, mic_asr)
    st.session_state.loopback_asr_backend = backend_map.get(loopback_asr, loopback_asr)

    # Store device selection in session state
    st.session_state.selected_mic_idx = mic_idx
    st.session_state.selected_speaker_idx = speaker_idx
    st.session_state.selected_ai_interval = ai_interval

# ===== HISTORY TAB =====
with tab_history:
    st.markdown("### 🔍 浏览过往记录")
    from ui.st_components import history_viewer
    from core.session_storage import get_visitor_profile, save_visitor_profile
    
    visitor_info = get_all_visitor_info()
    history_viewer(
        visitor_info=visitor_info,
        get_sessions_func=get_sessions_list,
        load_session_func=load_specific_session,
        get_profile_func=get_visitor_profile,
        save_profile_func=save_visitor_profile
    )

# ===== MAIN TAB =====
with tab_main:
    # Level meters
    level_meters(st.session_state.mic_rms, st.session_state.loopback_rms)

    # Control buttons
    start_clicked, stop_clicked, clear_clicked = control_buttons(st.session_state.is_recording)

    # Visitor ID and Save button
    default_visitor_id = st.session_state.get("current_visitor_id", generate_default_visitor_id())
    existing_ids = get_existing_visitor_ids()
    visitor_id, save_clicked, private_notes = visitor_id_input(default_visitor_id, existing_ids)

    # Store current visitor ID in session state
    st.session_state.current_visitor_id = visitor_id

    # Get device selection from session state (set in config tab)
    mic_idx = st.session_state.get("selected_mic_idx", 0)
    speaker_idx = st.session_state.get("selected_speaker_idx", 0)
    ai_interval = st.session_state.get("selected_ai_interval", 15)

    if start_clicked:
        start_recording(mic_idx, speaker_idx)
        st.rerun()

    if stop_clicked:
        stop_recording()
        st.rerun()

    if clear_clicked:
        # If currently recording, stop it first before clearing
        if st.session_state.is_recording:
            stop_recording()
        clear_session()
        st.rerun()

    # Handle save button click
    if save_clicked:
        with st.spinner("正在保存会话并生成总结..."):
            success, message, filepath = save_session(visitor_id, private_notes)
        if success:
            st.success(message)
        else:
            st.error(message)

    # Main content - 3 columns
    col_left, col_center, col_right = st.columns([3, 4, 3])

    with col_left:
        transcript_panel("倾听者 (我)", "🧑", st.session_state.my_transcript, "blue")

    with col_center:
        user_question = ai_suggestions_panel(st.session_state.ai_suggestions)

    with col_right:
        transcript_panel("倾诉者 (对方)", "👤", st.session_state.other_transcript, "green")

    # Handle user question - if entered, generate immediately
    if user_question and user_question.strip():
        generate_ai_suggestion(interval_seconds=0, user_question=user_question)
        st.rerun()

# Auto-refresh logic when recording
if st.session_state.is_recording:
    # Update levels and process transcripts
    update_levels()
    process_transcripts()

    # Generate AI suggestion periodically with user-selected interval
    generate_ai_suggestion(interval_seconds=ai_interval)

    # Auto-refresh every 1 second
    time.sleep(1)
    st.rerun()
