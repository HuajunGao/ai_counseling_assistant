"""
Reusable Streamlit UI components.
"""

import streamlit as st


def device_selectors(devices: dict, default_mic_name: str = "", default_speaker_name: str = "") -> tuple:
    """
    Render device selection dropdowns.
    Returns (mic_idx, speaker_idx).
    """
    col1, col2 = st.columns(2)

    # Find default indices based on device names
    default_mic = 0
    default_speaker = 0
    if default_mic_name:
        for m in devices["mics"]:
            if default_mic_name.lower() in m["name"].lower():
                default_mic = m["id"]
                break
    if default_speaker_name:
        for s in devices["speakers"]:
            if default_speaker_name.lower() in s["name"].lower():
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
                        f"<div style='padding: 5px; margin: 3px 0; background: rgba(255,255,255,0.05); border: 1px solid rgba(128,128,128,0.2); border-radius: 5px;'><span style='color: gray; font-size: 0.8em;'>{time_str}</span> {text}</div>",
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

    # Add Private Notes text area
    private_notes = st.text_area(
        "📝 咨询师私密笔记 (仅保存在本地，AI 不可见)",
        placeholder="在这里记录您的感悟、督导重点或下次会话目标...",
        key="private_notes_input",
        height=100
    )

    with col2:
        # Add some vertical spacing to align with input
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        save_clicked = st.button("💾 保存会话", type="primary", use_container_width=True)

    return visitor_id, save_clicked, private_notes


def history_viewer(visitor_info: list, get_sessions_func, load_session_func, get_profile_func, save_profile_func=None):
    """
    Render the history browser.
    
    Args:
        visitor_info: List of {"id": str, "description": str}
        get_sessions_func: Function(visitor_id) -> list of filenames
        load_session_func: Function(visitor_id, filename) -> session_dict
        get_profile_func: Function(visitor_id) -> visitor profile dict
        save_profile_func: Function(visitor_id, profile_data) -> None
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
    
    # 2. Display Visitor Personal Info Card (stays visible)
    if selected_v_id:
        profile = get_profile_func(selected_v_id)
        
        # Initialize edit state in session_state if not present
        edit_key = f"edit_mode_{selected_v_id}"
        if edit_key not in st.session_state:
            st.session_state[edit_key] = False
            
        # Personal Info Card Header
        col_title, col_edit = st.columns([5, 1])
        with col_title:
            st.markdown("### 👤 来访者档案")
        with col_edit:
            if not st.session_state[edit_key]:
                if st.button("📝 编辑", key=f"btn_edit_{selected_v_id}"):
                    st.session_state[edit_key] = True
                    st.rerun()
        
        personal_info = profile.get("personal_info", {})
        session_count = profile.get("session_count", 0)
        
        if st.session_state[edit_key]:
            # Edit Mode
            with st.form(key=f"edit_form_{selected_v_id}"):
                st.markdown("**正在编辑档案内容**")
                
                # Basic Description
                description = st.text_area("档案概括", value=profile.get("description", ""), height=100)
                
                col1, col2 = st.columns(2)
                with col1:
                    age = st.text_input("年龄", value=personal_info.get("age") or "")
                with col2:
                    gender = st.selectbox("性别", options=["未录入", "男", "女", "其他"], 
                                         index=["未录入", "男", "女", "其他"].index(personal_info.get("gender") if personal_info.get("gender") in ["男", "女", "其他"] else "未录入"))
                
                occupation = st.text_input("职业", value=personal_info.get("occupation") or "")
                background = st.text_area("背景信息 / 累计历史", value=personal_info.get("background") or "", height=200)
                
                col_save, col_cancel = st.columns([1, 1])
                with col_save:
                    if st.form_submit_button("✅ 保存修改", type="primary", use_container_width=True):
                        # Prepare data
                        save_data = {
                            "description": description,
                            "personal_info": {
                                "age": age if age else None,
                                "gender": gender if gender != "未录入" else None,
                                "occupation": occupation if occupation else None,
                                "background": background
                            }
                        }
                        # Call save function
                        if save_profile_func:
                            save_profile_func(selected_v_id, save_data)
                        else:
                            # Fallback
                            from core.session_storage import save_visitor_profile
                            save_visitor_profile(selected_v_id, save_data)
                            
                        st.session_state[edit_key] = False
                        st.success("档案已更新")
                        st.rerun()
                
                with col_cancel:
                    if st.form_submit_button("❌ 取消", use_container_width=True):
                        st.session_state[edit_key] = False
                        st.rerun()
        else:
            # View Mode
            # Create info display with columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("累计会话", f"{session_count} 次")
            
            with col2:
                age_val = personal_info.get("age") or "未录入"
                st.metric("年龄", age_val)
            
            with col3:
                gender_val = personal_info.get("gender") or "未录入"
                st.metric("性别", gender_val)
            
            # Occupation and Background in expandable section
            with st.expander("📋 详细信息", expanded=True):
                st.markdown(f"**档案概括**: {profile.get('description', '无')}")
                st.divider()
                
                occ_val = personal_info.get("occupation")
                st.markdown(f"**职业**: {occ_val or '未录入'}")
                
                bg_val = personal_info.get("background")
                st.markdown(f"**背景信息 / 累计历史**:")
                if bg_val:
                    st.markdown(bg_val)
                else:
                    st.markdown("*暂无背景信息*")
        
        st.divider()
        
        # 3. Session List for selected visitor
        sessions = get_sessions_func(selected_v_id)
        if not sessions:
            st.warning("该来访者暂无保存的会话。")
            return
            
        # Session selection
        st.markdown("### 📅 会话记录")
        selected_session_file = st.selectbox(
            "选择会话日期",
            options=list(reversed(sessions)),
            format_func=lambda x: x.replace(".json", ""),
            label_visibility="collapsed"
        )
        
        if selected_session_file:
            session_data = load_session_func(selected_v_id, selected_session_file)
            
            # Summary Section
            st.markdown("#### 💡 会话提要")
            st.info(session_data.get("summary", "无提要"))
            
            # Private Notes Section
            private_notes = session_data.get("private_notes")
            if private_notes:
                st.text_area("📝 私密笔记", value=private_notes, height=150, disabled=True)
            
            # Dialogue Details
            with st.expander("📝 详细对话历史", expanded=False):
                conversation = session_data.get("conversation", {})
                
                # Support new structure (original/corrected) and legacy (dialogue)
                if "original" in conversation:
                    # New structure
                    original_dialogue = conversation.get("original", [])
                    corrected_dialogue = conversation.get("corrected", [])
                    
                    # Add Smart Correction toggle
                    use_corrected = st.checkbox("🛠️ 智能纠错模式", value=True, key=f"correction_toggle_{selected_v_id}_{selected_session_file}")
                    
                    dialogue = corrected_dialogue if use_corrected else original_dialogue
                else:
                    # Legacy structure - fallback to dialogue field
                    dialogue = conversation.get("dialogue")
                
                if dialogue:
                    # Chronological format
                    for msg in dialogue:
                        role = msg.get("role", "未知")
                        time_str = msg.get("time", "")
                        text = msg.get("text", "")
                        
                        # Show merge indicator if present
                        merged_from = msg.get("merged_from")
                        merge_indicator = " 🔗" if merged_from else ""
                        
                        align = "left" if role == "倾诉者" else "right"
                        bg_color = "#f0fdf4" if role == "倾诉者" else "#eff6ff"
                        label_color = "#166534" if role == "倾诉者" else "#1e40af"
                        
                        st.markdown(
                            f"""
                            <div style='display: flex; flex-direction: column; align-items: {"flex-start" if align=="left" else "flex-end"}; margin: 10px 0;'>
                                <div style='font-size: 0.8em; color: {label_color}; margin-bottom: 2px;'>
                                    {role} [{time_str}]{merge_indicator}
                                </div>
                                <div style='background: {bg_color}; color: #1a1a1a; padding: 10px; border-radius: 10px; max-width: 80%; box-shadow: 0 1px 2px rgba(0,0,0,0.1);'>
                                    {text}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    # Fallback to legacy separate columns (very old format)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**倾听者**")
                        for msg in conversation.get("listener", []):
                            st.caption(f"[{msg.get('time')}] {msg.get('text')}")
                    with col2:
                        st.markdown("**倾诉者**")
                        for msg in conversation.get("speaker", []):
                            st.caption(f"[{msg.get('time')}] {msg.get('text')}")
