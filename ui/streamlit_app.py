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
    recording_mode_selector,
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

    st.markdown("### 录制设置")
    # Recording mode selector (disabled during recording)
    current_mode = st.session_state.get("recording_mode", "dual")
    if st.session_state.is_recording:
        mode_names = {'dual': '双轨录制', 'mic_only': '仅麦克风', 'speaker_only': '仅扬声器'}
        st.info(f"⏺️ 当前录制模式: {mode_names[current_mode]}")
        st.caption("⚠️ 停止录制后可更改模式")
    else:
        recording_mode = recording_mode_selector(current_mode)
        st.session_state.recording_mode = recording_mode

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
    # Get recording mode
    recording_mode = st.session_state.get("recording_mode", "dual")
    mic_enabled = recording_mode in ("dual", "mic_only")
    speaker_enabled = recording_mode in ("dual", "speaker_only")
    
    # Compact control row at top
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([2, 2, 2, 1])
    
    with col_ctrl1:
        start_clicked, stop_clicked, clear_clicked = control_buttons(st.session_state.is_recording)
    
    with col_ctrl2:
        # Level meters (compact)
        level_meters(st.session_state.mic_rms, st.session_state.loopback_rms, mic_enabled, speaker_enabled)
    
    with col_ctrl3:
        # Visitor ID input (compact)
        default_visitor_id = st.session_state.get("current_visitor_id", generate_default_visitor_id())
        existing_ids = get_existing_visitor_ids()
        visitor_id = st.text_input("🏷️ 来访 ID", value=default_visitor_id, key="visitor_id_compact", label_visibility="collapsed")
        st.session_state.current_visitor_id = visitor_id
    
    with col_ctrl4:
        # Save button
        save_clicked = st.button("💾", use_container_width=True, help="保存会话")
    
    # Private notes (collapsible)
    with st.expander("📝 私密笔记", expanded=False):
        private_notes = st.text_area(
            "咨询师私密笔记",
            placeholder="记录您的感悟、督导重点或下次会话目标...",
            key="private_notes_compact",
            height=80,
            label_visibility="collapsed"
        )
    
    st.divider()
    
    # === MAIN CONTENT: AI Suggestions (Full Width) ===
    st.markdown("### 💡 AI 建议与提问")
    user_question, ask_button = ai_suggestions_panel(st.session_state.ai_suggestions)
    
    st.divider()
    
    # === TRANSCRIPTS: Collapsible at bottom ===
    with st.expander("📝 对话记录", expanded=False):
        col_left, col_right = st.columns(2)
        
        with col_left:
            transcript_panel("倾听者 (我)", "🧑", st.session_state.my_transcript, "blue", enabled=mic_enabled)
        
        with col_right:
            transcript_panel("倾诉者 (对方)", "👤", st.session_state.other_transcript, "green", enabled=speaker_enabled)
    
    # Get device selection from session state (set in config tab)
    mic_idx = st.session_state.get("selected_mic_idx", 0)
    speaker_idx = st.session_state.get("selected_speaker_idx", 0)
    ai_interval = st.session_state.get("selected_ai_interval", 30)

    if start_clicked:
        start_recording(mic_idx, speaker_idx, mode=recording_mode)
        st.rerun()

    if stop_clicked:
        stop_recording()
        st.rerun()

    if clear_clicked:
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

    # Handle user question - if button clicked or Enter pressed
    # Debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"DEBUG: ask_button={ask_button}, user_question='{user_question}'")
    
    # Button click always triggers if there's text (and prevents Enter from also triggering)
    if ask_button and user_question and user_question.strip():
        logger.info(f"DEBUG: Button clicked with question: '{user_question}'")
        generate_ai_suggestion(interval_seconds=0, user_question=user_question)
        st.session_state.last_processed_question = user_question  # Mark as processed
        st.rerun()
    # Enter key: only trigger if it's a new question AND button wasn't clicked
    elif not ask_button and user_question and user_question.strip():
        last_question = st.session_state.get("last_processed_question", "")
        logger.info(f"DEBUG: Enter key - current: '{user_question}', last: '{last_question}'")
        if user_question != last_question:
            logger.info(f"DEBUG: Processing new question via Enter")
            generate_ai_suggestion(interval_seconds=0, user_question=user_question)
            st.session_state.last_processed_question = user_question
            st.rerun()
        else:
            logger.info(f"DEBUG: Skipping duplicate question")

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
