"""
Reusable Streamlit UI components.
"""
import streamlit as st


def device_selectors(devices: dict, default_name: str = "") -> tuple:
    """
    Render device selection dropdowns.
    Returns (mic_idx, speaker_idx).
    """
    col1, col2 = st.columns(2)
    
    # Find default indices based on device name
    default_mic = 0
    default_speaker = 0
    if default_name:
        for m in devices['mics']:
            if default_name.lower() in m['name'].lower():
                default_mic = m['id']
                break
        for s in devices['speakers']:
            if default_name.lower() in s['name'].lower():
                default_speaker = s['id']
                break
    
    with col1:
        mic_options = {m['id']: f"🎤 {m['name'][:35]}" for m in devices['mics']}
        mic_ids = list(mic_options.keys())
        mic_idx = st.selectbox(
            "麦克风 (我)",
            options=mic_ids,
            index=mic_ids.index(default_mic) if default_mic in mic_ids else 0,
            format_func=lambda x: mic_options[x],
            key="mic_select"
        )
    
    with col2:
        speaker_options = {s['id']: f"🔊 {s['name'][:35]}" for s in devices['speakers']}
        speaker_ids = list(speaker_options.keys())
        speaker_idx = st.selectbox(
            "扬声器 (对方)",
            options=speaker_ids,
            index=speaker_ids.index(default_speaker) if default_speaker in speaker_ids else 0,
            format_func=lambda x: speaker_options[x],
            key="speaker_select"
        )
    
    return mic_idx, speaker_idx


def level_meters(mic_rms: float, loopback_rms: float):
    """Render audio level meters."""
    col1, col2 = st.columns(2)
    
    # Convert RMS to percentage (0-100)
    mic_pct = min(int(mic_rms * 1000), 100)
    speaker_pct = min(int(loopback_rms * 1000), 100)
    
    with col1:
        st.markdown(f"""
        <div style='margin-bottom: 5px;'>
            <span style='font-size: 0.8em;'>🎤 Mic</span>
            <div style='background: #333; border-radius: 4px; height: 12px; width: 100%;'>
                <div style='background: linear-gradient(90deg, #22c55e, #86efac); width: {mic_pct}%; height: 100%; border-radius: 4px; transition: width 0.1s;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='margin-bottom: 5px;'>
            <span style='font-size: 0.8em;'>🔊 Speaker</span>
            <div style='background: #333; border-radius: 4px; height: 12px; width: 100%;'>
                <div style='background: linear-gradient(90deg, #3b82f6, #93c5fd); width: {speaker_pct}%; height: 100%; border-radius: 4px; transition: width 0.1s;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def control_buttons(is_recording: bool) -> tuple:
    """
    Render start/stop buttons.
    Returns (start_clicked, stop_clicked, clear_clicked).
    """
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        label = "🔴 正在录制..." if is_recording else "▶️ 开始录制"
        start_clicked = st.button(
            label,
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


def ai_settings_panel(openai_models: list, whisper_models: list, asr_backends: list, current_whisper: str, current_ai_model: str):
    """Render AI and transcription settings."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        ai_model = st.selectbox(
            "🤖 AI Model",
            options=openai_models,
            index=openai_models.index(current_ai_model) if current_ai_model in openai_models else 0,
            key="ai_model_select"
        )
    
    with col2:
        interval = st.slider(
            "⏱️ 间隔(秒)",
            min_value=10,
            max_value=120,
            value=15,
            step=5,
            key="ai_interval"
        )
    
    with col3:
        context_len = st.slider(
            "📝 上下文(行)",
            min_value=3,
            max_value=20,
            value=15,
            step=1,
            key="ai_context_len"
        )
    
    with col4:
        asr_backend = st.selectbox(
            "🎯 转录后端",
            options=asr_backends,
            key="asr_backend_select"
        )
    
    with col5:
        whisper_model = st.selectbox(
            "🎤 Whisper",
            options=whisper_models,
            index=whisper_models.index(current_whisper) if current_whisper in whisper_models else 4,
            key="whisper_model_select",
            disabled=(asr_backend != "whisper")
        )
    
    return ai_model, interval, context_len, asr_backend, whisper_model


def transcript_panel(title: str, emoji: str, transcripts: list, color: str = "blue"):
    """Render a transcript panel with scrollable content."""
    st.markdown(f"**{emoji} {title}**")
    
    container = st.container(height=400)
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
    st.markdown("**💡 AI 建议**")
    
    container = st.container(height=400)
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
