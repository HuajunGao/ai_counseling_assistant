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
        for m in devices["mics"]:
            if default_name.lower() in m["name"].lower():
                default_mic = m["id"]
                break
        for s in devices["speakers"]:
            if default_name.lower() in s["name"].lower():
                default_speaker = s["id"]
                break

    with col1:
        mic_options = {m["id"]: f"🎤 {m['name'][:35]}" for m in devices["mics"]}
        mic_ids = list(mic_options.keys())
        mic_idx = st.selectbox(
            "麦克风 (我)",
            options=mic_ids,
            index=mic_ids.index(default_mic) if default_mic in mic_ids else 0,
            format_func=lambda x: mic_options[x],
            key="mic_select",
        )

    with col2:
        speaker_options = {s["id"]: f"🔊 {s['name'][:35]}" for s in devices["speakers"]}
        speaker_ids = list(speaker_options.keys())
        speaker_idx = st.selectbox(
            "扬声器 (对方)",
            options=speaker_ids,
            index=speaker_ids.index(default_speaker) if default_speaker in speaker_ids else 0,
            format_func=lambda x: speaker_options[x],
            key="speaker_select",
        )

    return mic_idx, speaker_idx


def level_meters(mic_rms: float, loopback_rms: float):
    """Render audio level meters."""
    col1, col2 = st.columns(2)

    # Convert RMS to percentage (0-100)
    mic_pct = min(int(mic_rms * 1000), 100)
    speaker_pct = min(int(loopback_rms * 1000), 100)

    with col1:
        st.markdown(
            f"""
        <div style='margin-bottom: 5px;'>
            <span style='font-size: 0.8em;'>🎤 Mic</span>
            <div style='background: #333; border-radius: 4px; height: 12px; width: 100%;'>
                <div style='background: linear-gradient(90deg, #22c55e, #86efac); width: {mic_pct}%; height: 100%; border-radius: 4px; transition: width 0.1s;'></div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div style='margin-bottom: 5px;'>
            <span style='font-size: 0.8em;'>🔊 Speaker</span>
            <div style='background: #333; border-radius: 4px; height: 12px; width: 100%;'>
                <div style='background: linear-gradient(90deg, #3b82f6, #93c5fd); width: {speaker_pct}%; height: 100%; border-radius: 4px; transition: width 0.1s;'></div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def control_buttons(is_recording: bool) -> tuple:
    """
    Render start/stop buttons.
    Returns (start_clicked, stop_clicked, clear_clicked).
    """
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        label = "🔴 正在录制..." if is_recording else "▶️ 开始录制"
        start_clicked = st.button(label, type="primary", disabled=is_recording, use_container_width=True)

    with col2:
        stop_clicked = st.button("⏹️ 停止", type="secondary", disabled=not is_recording, use_container_width=True)

    with col3:
        clear_clicked = st.button("🗑️ 清空", type="secondary", use_container_width=True)

    return start_clicked, stop_clicked, clear_clicked


def ai_settings_panel(openai_models: list, asr_backends: list, current_ai_model: str):
    """Render AI and transcription settings with dual ASR config."""
    # Row 1: AI settings
    col1, col2, col3 = st.columns(3)

    with col1:
        ai_model = st.selectbox(
            "🤖 AI Model",
            options=openai_models,
            index=openai_models.index(current_ai_model) if current_ai_model in openai_models else 0,
            key="ai_model_select",
        )

    with col2:
        interval = st.slider("⏱️ 间隔(秒)", min_value=10, max_value=120, value=15, step=5, key="ai_interval")

    with col3:
        context_len = st.slider("📝 上下文(行)", min_value=3, max_value=20, value=15, step=1, key="ai_context_len")

    # Row 2: ASR settings (dual config)
    col4, col5 = st.columns(2)

    with col4:
        mic_asr = st.selectbox("🎤 我的 ASR", options=asr_backends, key="mic_asr_select")

    with col5:
        loopback_asr = st.selectbox("🔊 对方 ASR", options=asr_backends, key="loopback_asr_select")

    return ai_model, interval, context_len, mic_asr, loopback_asr


def transcript_panel(title: str, emoji: str, transcripts: list, color: str = "blue"):
    """Render a transcript panel with scrollable content."""
    st.markdown(f"**{emoji} {title}**")

    container = st.container(height=400)
    with container:
        if transcripts:
            # Show newest first (reversed order)
            for item in reversed(transcripts):
                if isinstance(item, dict):
                    time_str = item.get("time", "")
                    text = item.get("text", "")
                    st.markdown(
                        f"<div style='padding: 5px; margin: 3px 0; background: rgba(0,0,0,0.05); border-radius: 5px;'><span style='color: #666; font-size: 0.8em;'>{time_str}</span> {text}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    # Legacy format (plain string)
                    st.markdown(
                        f"<div style='padding: 5px; margin: 3px 0; background: rgba(0,0,0,0.05); border-radius: 5px;'>{item}</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("等待转录...")


def ai_suggestions_panel(suggestions: list):
    """Render AI suggestions panel with question input."""
    st.markdown("**💡 AI 建议**")

    # Question input for the counselor
    user_question = st.text_input(
        "💬 向 AI 提问", placeholder="输入问题后按回车发送...", key="ai_question_input", label_visibility="collapsed"
    )

    container = st.container(height=350)
    with container:
        if suggestions:
            for item in reversed(suggestions[-5:]):  # Show last 5, newest first
                with st.expander(f"🕐 {item['time']}", expanded=True):
                    st.markdown(item["text"])
        else:
            st.info("AI 将根据对话内容定期提供建议，或输入问题直接询问...")

    return user_question


def status_indicator(is_recording: bool):
    """Show recording status."""
    if is_recording:
        st.success("🔴 正在录制...")
    else:
        st.info("⏸️ 已停止")


def visitor_id_input(default_id: str, existing_ids: list) -> tuple:
    """
    Render visitor ID input with save button.
    Returns (visitor_id, save_clicked).
    """
    col1, col2 = st.columns([3, 1])

    with col1:
        # Text input for visitor ID with autocomplete hint
        help_text = f"已有来访者: {', '.join(existing_ids[-5:])}" if existing_ids else "输入来访者ID"
        visitor_id = st.text_input(
            "🏷️ 来访 ID",
            value=default_id,
            placeholder="例如: 20251230 或 client_001",
            help=help_text,
            key="visitor_id_input",
        )

    with col2:
        # Add some vertical spacing to align with input
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        save_clicked = st.button("💾 保存会话", type="primary", use_container_width=True)

    return visitor_id, save_clicked


def history_viewer(visitor_info: list, get_sessions_func, load_session_func):
    """
    Render the history browser.
    
    Args:
        visitor_info: List of {"id": str, "description": str}
        get_sessions_func: Function(visitor_id) -> list of filenames
        load_session_func: Function(visitor_id, filename) -> session_dict
    """
    if not visitor_info:
        st.info("暂无历史记录。")
        return

    # 1. Visitor Selection
    v_ids = [v["id"] for v in visitor_info]
    selected_v_id = st.selectbox(
        "选择来访者", 
        options=v_ids,
        format_func=lambda x: f"{x} - {next(v['description'] for v in visitor_info if v['id'] == x)}"
    )
    
    # 2. Session List for selected visitor
    if selected_v_id:
        sessions = get_sessions_func(selected_v_id)
        if not sessions:
            st.warning("该来访者暂无保存的会话。")
            return
            
        # Reverse to show newest first
        selected_session_file = st.selectbox(
            "选择会话日期",
            options=list(reversed(sessions)),
            format_func=lambda x: x.replace(".json", "")
        )
        
        if selected_session_file:
            session_data = load_session_func(selected_v_id, selected_session_file)
            
            # 3. Session Details
            st.divider()
            
            # Summary Section
            st.subheader("💡 会话提要")
            st.info(session_data.get("summary", "无提要"))
            
            # Dialogue Details
            with st.expander("📝 详细对话历史", expanded=False):
                conversation = session_data.get("conversation", {})
                dialogue = conversation.get("dialogue")
                
                if dialogue:
                    # New chronological format
                    for msg in dialogue:
                        role = msg.get("role", "未知")
                        time_str = msg.get("time", "")
                        text = msg.get("text", "")
                        
                        align = "left" if role == "倾诉者" else "right"
                        bg_color = "#f0fdf4" if role == "倾诉者" else "#eff6ff"
                        label_color = "#166534" if role == "倾诉者" else "#1e40af"
                        
                        st.markdown(
                            f"""
                            <div style='display: flex; flex-direction: column; align-items: {"flex-start" if align=="left" else "flex-end"}; margin: 10px 0;'>
                                <div style='font-size: 0.8em; color: {label_color}; margin-bottom: 2px;'>
                                    {role} [{time_str}]
                                </div>
                                <div style='background: {bg_color}; padding: 10px; border-radius: 10px; max-width: 80%; box-shadow: 0 1px 2px rgba(0,0,0,0.1);'>
                                    {text}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    # Fallback to legacy separate columns
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**倾听者**")
                        for msg in conversation.get("listener", []):
                            st.caption(f"[{msg.get('time')}] {msg.get('text')}")
                    with col2:
                        st.markdown("**倾诉者**")
                        for msg in conversation.get("speaker", []):
                            st.caption(f"[{msg.get('time')}] {msg.get('text')}")
