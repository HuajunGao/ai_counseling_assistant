"""
Reusable Streamlit UI components.
"""
import streamlit as st


def device_selectors(devices: dict) -> tuple:
    """
    Render device selection dropdowns.
    Returns (mic_idx, speaker_idx).
    """
    col1, col2 = st.columns(2)
    
    with col1:
        mic_options = {m['id']: f"🎤 {m['name'][:35]}" for m in devices['mics']}
        mic_idx = st.selectbox(
            "麦克风 (我)",
            options=list(mic_options.keys()),
            format_func=lambda x: mic_options[x],
            key="mic_select"
        )
    
    with col2:
        speaker_options = {s['id']: f"🔊 {s['name'][:35]}" for s in devices['speakers']}
        speaker_idx = st.selectbox(
            "扬声器 (对方)",
            options=list(speaker_options.keys()),
            format_func=lambda x: speaker_options[x],
            key="speaker_select"
        )
    
    return mic_idx, speaker_idx


def level_meters(mic_rms: float, loopback_rms: float):
    """Render audio level meters."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.caption("🎤 Mic Level")
        st.progress(min(mic_rms * 10, 1.0))
    
    with col2:
        st.caption("🔊 Speaker Level")
        st.progress(min(loopback_rms * 10, 1.0))


def control_buttons(is_recording: bool) -> tuple:
    """
    Render start/stop buttons.
    Returns (start_clicked, stop_clicked, clear_clicked).
    """
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        start_clicked = st.button(
            "▶️ 开始录制",
            type="primary",
            disabled=is_recording,
            use_container_width=True
        )
    
    with col2:
        stop_clicked = st.button(
            "⏹️ 停止",
            type="secondary",
            disabled=not is_recording,
            use_container_width=True
        )
    
    with col3:
        clear_clicked = st.button(
            "🗑️ 清空",
            type="secondary",
            use_container_width=True
        )
    
    return start_clicked, stop_clicked, clear_clicked


def transcript_panel(title: str, emoji: str, transcripts: list, color: str = "blue"):
    """Render a transcript panel with scrollable content."""
    st.markdown(f"### {emoji} {title}")
    
    container = st.container(height=450)
    with container:
        if transcripts:
            # Show newest first (reversed order)
            for item in reversed(transcripts):
                if isinstance(item, dict):
                    time_str = item.get('time', '')
                    text = item.get('text', '')
                    st.markdown(f"<div style='padding: 5px; margin: 3px 0; background: rgba(0,0,0,0.05); border-radius: 5px;'><span style='color: #666; font-size: 0.8em;'>{time_str}</span> {text}</div>", unsafe_allow_html=True)
                else:
                    # Legacy format (plain string)
                    st.markdown(f"<div style='padding: 5px; margin: 3px 0; background: rgba(0,0,0,0.05); border-radius: 5px;'>{item}</div>", unsafe_allow_html=True)
        else:
            st.caption("等待转录...")


def ai_suggestions_panel(suggestions: list):
    """Render AI suggestions panel."""
    st.markdown("### 💡 AI 建议")
    
    container = st.container(height=450)
    with container:
        if suggestions:
            for item in reversed(suggestions[-5:]):  # Show last 5, newest first
                with st.expander(f"🕐 {item['time']}", expanded=True):
                    st.markdown(item['text'])
        else:
            st.info("AI 将根据对话内容定期提供建议...")


def status_indicator(is_recording: bool):
    """Show recording status."""
    if is_recording:
        st.success("🔴 正在录制...")
    else:
        st.info("⏸️ 已停止")
