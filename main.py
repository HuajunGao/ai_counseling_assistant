# DLL path setup for Windows - MUST be before any torch imports
import os
import sys

if os.environ.get("KMP_DUPLICATE_LIB_OK") is None:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

_CUDA_DLL_DIRS = []
if sys.platform == "win32":
    venv_root = os.path.abspath(os.path.join(os.path.dirname(sys.executable), ".."))
    site_packages = os.path.join(venv_root, "Lib", "site-packages")
    current_path = os.environ.get("PATH", "")
    
    for rel_path in ("nvidia\\cudnn\\bin", "nvidia\\cublas\\bin"):
        dll_dir = os.path.join(site_packages, rel_path)
        if os.path.isdir(dll_dir) and dll_dir not in current_path:
            os.add_dll_directory(dll_dir)
            _CUDA_DLL_DIRS.append(dll_dir)
    
    if _CUDA_DLL_DIRS:
        os.environ["PATH"] = ";".join(_CUDA_DLL_DIRS) + ";" + current_path

from nicegui import ui, app
import asyncio
import time
import queue
import logging
from core.audio_capture import AudioCapture
from core.transcriber import Transcriber
from core.llm_engine import SuggestionEngine
from core.audio_test import list_all_devices, AudioTester
from core.dual_capture import DualStreamCapture, list_devices as list_dual_devices
from ui import components

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global State
audio_capture = AudioCapture()
transcriber = None
suggestion_engine = SuggestionEngine()

# Queues
asr_input_queue = audio_capture.audio_queue
asr_output_queue = queue.Queue()

# UI State
is_recording = False
selected_device = None
current_partial_text = ""
level_tester = {'instance': None}  # For live level display

def main():
    # Load devices using soundcard library (same as record_benchmark.py)
    all_devices = list_all_devices()
    device_options = {d['id']: f"{d['name'][:35]} [{d['type'][:10]}]" for d in all_devices}
    
    # Default to first loopback device
    default_device = all_devices[0]['id'] if all_devices else None

    @ui.page('/')
    def index():
        global is_recording, transcriber, selected_device
        
        # Use a wide layout
        ui.query('body').style('background-color: #f3f4f6')
        
        with ui.header().classes('bg-white text-gray-800 border-b p-4 flex items-center justify-between'):
            ui.label('AI Counseling Copilot').classes('text-2xl font-bold text-indigo-600')
            
            with ui.row().classes('items-center gap-4'):
                # Device Selector (now simplified)
                device_select = ui.select(device_options, value=default_device, label='Audio Device').classes('w-80')
                
                # Live Level Meter
                with ui.column().classes('gap-0'):
                    ui.label('Level').classes('text-xs text-gray-500')
                    level_bar = ui.linear_progress(value=0, show_value=False).classes('w-32 h-3')
                
                # Test Button
                test_btn = ui.button('🔊 Test', icon='hearing').props('flat dense')
                
                # Controls
                start_btn = ui.button('Start Session', on_click=lambda: start_session(device_select.value), icon='mic').props('color=green')
                stop_btn = ui.button('Stop', on_click=lambda: stop_session(), icon='stop').props('color=red').bind_visibility_from(globals(), 'is_recording')
                start_btn.bind_visibility_from(globals(), 'is_recording', backward=lambda x: not x)
        
        # Level meter update logic
        def start_level_test():
            device_id = device_select.value
            if not device_id:
                ui.notify('Please select a device', type='warning')
                return
            
            level_tester['instance'] = AudioTester(device_id, duration=10.0, use_loopback=False)
            level_tester['instance'].start()
            ui.notify('Testing audio level for 10 seconds...', type='info')
            
            def update_level():
                if level_tester['instance']:
                    status = level_tester['instance'].get_status()
                    level_bar.set_value(min(status['current_rms'] * 10, 1.0))
                    if not status['running']:
                        level_bar.set_value(0)
            
            ui.timer(0.1, update_level, active=True)
        
        test_btn.on('click', start_level_test)

        with ui.row().classes('w-full h-[calc(100vh-100px)] p-4 gap-4'):
            # LEFT: Transcript
            with ui.card().classes('w-2/3 h-full flex flex-col'):
                ui.label('Live Transcript').classes('text-lg font-bold mb-2 text-gray-700')
                transcript_container = ui.scroll_area().classes('flex-grow w-full border rounded p-4 bg-white')
            
            # RIGHT: Suggestions
            with ui.card().classes('w-1/3 h-full flex flex-col bg-slate-50'):
                ui.label('AI Suggestions').classes('text-lg font-bold mb-2 text-gray-700')
                suggestion_container = ui.scroll_area().classes('flex-grow w-full p-2')
                
                # Manual Trigger Button (Optional)
                ui.button('Refresh Suggestions', on_click=lambda: generate_suggestion_task(suggestion_container), icon='refresh').classes('w-full mt-2')

        # --- Logic ---

        def start_session(device_id):
            global is_recording, transcriber
            if device_id is None:
                ui.notify('Please select an audio device', type='warning')
                return
            
            logger.info(f"Starting session on device {device_id}")
            try:
                # Start Audio
                audio_capture.start_stream(device_id)
                
                # Start Transcriber
                transcriber = Transcriber(asr_input_queue, asr_output_queue)
                transcriber.start()
                
                is_recording = True
                ui.notify('Session Started', type='positive')
            except Exception as e:
                ui.notify(f'Error starting: {e}', type='negative')
                logger.error(f"Start error: {e}")

        def stop_session():
            global is_recording, transcriber
            logger.info("Stopping session")
            
            audio_capture.stop_stream()
            if transcriber:
                transcriber.stop()
                transcriber = None # Allow thread to join naturally or garbage collect logic
            
            is_recording = False
            ui.notify('Session Topped', type='info')

        async def update_transcript_ui():
            """Consumes ASR queue and updates UI."""
            while True:
                try:
                    # Non-blocking get
                    try:
                        event = asr_output_queue.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.1)
                        continue

                except Exception as e:
                    logger.error(f"UI Update error: {e}")
                    await asyncio.sleep(1)

        # Active transcript label (for partials)
        active_transcript_label = ui.label('').classes('text-gray-500 italic p-2')
        transcript_container.move(active_transcript_label) # Ensure it's at the bottom

        async def process_queue():
            nonlocal transcript_container, active_transcript_label
            global current_partial_text
            
            try:
                # Process up to 10 events per tick to stay responsive but not block
                for _ in range(10): 
                    if asr_output_queue.empty():
                        break
                        
                    event = asr_output_queue.get_nowait()
                    
                    if event['type'] == 'partial':
                        text = event.get('text', '')
                        current_partial_text = text
                        active_transcript_label.set_text(text + "...")
                        
                    elif event['type'] == 'commit':
                        committed_text = event.get('text') or current_partial_text
                        if committed_text.strip():
                            with transcript_container:
                                components.transcript_row(committed_text, time.time(), is_final=True)
                                suggestion_engine.update_transcript(committed_text)
                            active_transcript_label.move(transcript_container)
                            active_transcript_label.set_text('')
                            current_partial_text = ""
                            transcript_container.scroll_to_bottom()
                            
                            # Auto-trigger AI if enough context?
                            # For MVP: Let's stick to manual or timer based.
            except Exception as e:
                logger.error(f"Queue process error: {e}")
        
        ui.timer(0.1, process_queue)

        async def generate_suggestion_task(container):
            ui.notify('Generating suggestions...')
            loop = asyncio.get_event_loop()
            # Run blocking LLM call in executor
            response = await loop.run_in_executor(None, suggestion_engine.generate_suggestions)
            
            if response:
                with container:
                    components.suggestion_card(response)
                    container.scroll_to_bottom()
            else:
                ui.notify('No context to suggest on')

    @ui.page('/test')
    def test_page():
        """Audio device test page with live level meter and playback."""
        all_devices = list_all_devices()
        # Show all devices from list_all_devices (already filtered to loopback + input)
        test_devices = {d['id']: f"{d['id']}: {d['name'][:40]} ({d['api'][:10]})" 
                        for d in all_devices}
        
        tester = {'instance': None}
        
        ui.query('body').style('background-color: #1e293b')
        
        with ui.card().classes('mx-auto mt-8 p-6 max-w-2xl'):
            ui.label('🎤 Audio Device Test').classes('text-2xl font-bold text-center mb-4')
            
            with ui.row().classes('w-full items-end gap-4 mb-4'):
                device_select = ui.select(test_devices, label='Select Device').classes('flex-grow')
                loopback_toggle = ui.switch('Loopback Mode', value=True).tooltip('Enable for Speaker devices (capture system audio)')
            
            # Level meter
            ui.label('Level Meter:').classes('text-sm text-gray-600 mt-2')
            level_bar = ui.linear_progress(value=0).classes('w-full h-4')
            level_label = ui.label('RMS: 0.0000').classes('text-xs text-gray-500')
            
            # Status
            status_label = ui.label('Ready').classes('text-sm text-blue-600 mt-2')
            
            # Buttons
            with ui.row().classes('w-full gap-2 mt-4'):
                start_btn = ui.button('🔴 Start Test (5s)', icon='mic').props('color=red')
                stop_btn = ui.button('⬛ Stop', icon='stop').props('color=grey')
                play_btn = ui.button('▶️ Play Recording', icon='play_arrow').props('color=green')
            
            # Timer for updates
            update_timer = {'ref': None}
            
            def update_ui():
                if tester['instance']:
                    status = tester['instance'].get_status()
                    level_bar.set_value(min(status['current_rms'] * 10, 1.0))
                    level_label.set_text(f"RMS: {status['current_rms']:.4f} (Peak: {status['peak_rms']:.4f})")
                    
                    if status['error']:
                        status_label.set_text(f"❌ Error: {status['error']}")
                        status_label.classes(remove='text-blue-600', add='text-red-600')
                    elif status['running']:
                        status_label.set_text('🔴 Recording...')
                        status_label.classes(remove='text-blue-600', add='text-red-600')
                    elif status['has_recording']:
                        status_label.set_text('✅ Recording complete! Click Play to hear it.')
                        status_label.classes(remove='text-red-600', add='text-green-600')
                        if update_timer['ref']:
                            update_timer['ref'].deactivate()
            
            def start_test():
                device_id = device_select.value
                if device_id is None:
                    ui.notify('Please select a device', type='warning')
                    return
                
                tester['instance'] = AudioTester(device_id, duration=5.0, use_loopback=loopback_toggle.value)
                tester['instance'].start()
                status_label.set_text('🔴 Recording...')
                status_label.classes(remove='text-green-600', add='text-red-600')
                
                update_timer['ref'] = ui.timer(0.1, update_ui)
            
            def stop_test():
                if tester['instance']:
                    tester['instance'].stop()
                    status_label.set_text('⬛ Stopped')
                    if update_timer['ref']:
                        update_timer['ref'].deactivate()
            
            def play_recording():
                if tester['instance'] and tester['instance'].recorded_audio is not None:
                    ui.notify('Playing recording...')
                    tester['instance'].play_recording()
                else:
                    ui.notify('No recording available', type='warning')
            
            start_btn.on('click', start_test)
            stop_btn.on('click', stop_test)
            play_btn.on('click', play_recording)
            
            ui.separator()
            ui.label('Instructions:').classes('text-sm font-bold mt-2')
            ui.markdown('''
1. Select a device from the dropdown
2. Enable **Loopback Mode** for speaker devices (to capture system audio)
3. Click **Start Test** and play some audio
4. Watch the level meter - it should move if audio is detected
5. Click **Play Recording** to hear what was captured
            ''').classes('text-xs')

    @ui.page('/conversation')
    def conversation_page():
        """Dual-stream conversation transcription page."""
        devices = list_dual_devices()
        mic_options = {m['id']: f"🎤 {m['name'][:30]}" for m in devices['mics']}
        speaker_options = {s['id']: f"🔊 {s['name'][:30]}" for s in devices['speakers']}
        
        state = {
            'capture': None,
            'mic_transcriber': None,
            'loopback_transcriber': None,
            'running': False,
        }
        
        ui.query('body').style('background-color: #0f172a')
        
        with ui.column().classes('w-full min-h-screen p-4'):
            # Header
            with ui.row().classes('w-full items-center justify-between mb-4'):
                ui.label('💬 对话转录').classes('text-2xl font-bold text-white')
                ui.link('← 返回主页', '/').classes('text-blue-400')
            
            # Device Selection
            with ui.card().classes('w-full p-4 mb-4'):
                with ui.row().classes('w-full gap-4 items-end'):
                    mic_select = ui.select(mic_options, label='麦克风 (我)', value=0 if mic_options else None).classes('flex-grow')
                    speaker_select = ui.select(speaker_options, label='扬声器 (对方)', value=0 if speaker_options else None).classes('flex-grow')
                    
                    with ui.column().classes('gap-1'):
                        ui.label('Mic Level').classes('text-xs text-gray-500')
                        mic_level = ui.linear_progress(value=0, show_value=False).classes('w-24 h-2')
                    with ui.column().classes('gap-1'):
                        ui.label('Speaker Level').classes('text-xs text-gray-500')
                        speaker_level = ui.linear_progress(value=0, show_value=False).classes('w-24 h-2')
                    
                    start_btn = ui.button('▶️ 开始', icon='play_arrow').props('color=green')
                    stop_btn = ui.button('⏹️ 停止', icon='stop').props('color=red')
            
            # Transcript Columns
            with ui.row().classes('w-full flex-grow gap-4'):
                # Left: 我 (My mic)
                with ui.card().classes('w-1/2 h-[600px] flex flex-col bg-blue-900/30'):
                    ui.label('🧑 我').classes('text-xl font-bold text-blue-300 mb-2')
                    my_transcript = ui.scroll_area().classes('flex-grow w-full bg-blue-950/50 rounded p-2')
                
                # Right: 对方 (Speaker)
                with ui.card().classes('w-1/2 h-[600px] flex flex-col bg-green-900/30'):
                    ui.label('👤 对方').classes('text-xl font-bold text-green-300 mb-2')
                    other_transcript = ui.scroll_area().classes('flex-grow w-full bg-green-950/50 rounded p-2')
        
        # Output queues for transcription results
        mic_output_queue = queue.Queue()
        loopback_output_queue = queue.Queue()
        
        def start_session():
            if state['running']:
                return
            
            mic_id = mic_select.value
            speaker_id = speaker_select.value
            
            if mic_id is None or speaker_id is None:
                ui.notify('请选择麦克风和扬声器', type='warning')
                return
            
            # Create capture with its own internal queues
            state['capture'] = DualStreamCapture(mic_id, speaker_id)
            state['capture'].start()
            
            # Start transcribers using the capture's internal queues
            state['mic_transcriber'] = Transcriber(state['capture'].mic_queue, mic_output_queue)
            state['mic_transcriber'].start()
            
            state['loopback_transcriber'] = Transcriber(state['capture'].loopback_queue, loopback_output_queue)
            state['loopback_transcriber'].start()
            
            state['running'] = True
            ui.notify('开始录制...', type='positive')
        
        def stop_session():
            if not state['running']:
                return
            
            if state['capture']:
                state['capture'].stop()
            if state['mic_transcriber']:
                state['mic_transcriber'].stop()
            if state['loopback_transcriber']:
                state['loopback_transcriber'].stop()
            
            state['running'] = False
            ui.notify('已停止', type='info')
        
        def update_ui():
            # Update levels
            if state['capture']:
                levels = state['capture'].get_levels()
                mic_level.set_value(min(levels['mic_rms'] * 10, 1.0))
                speaker_level.set_value(min(levels['loopback_rms'] * 10, 1.0))
            
            # Process mic transcripts
            while not mic_output_queue.empty():
                try:
                    item = mic_output_queue.get_nowait()
                    if item.get('text'):
                        with my_transcript:
                            ui.label(item['text']).classes('text-blue-200 text-sm mb-1')
                        my_transcript.scroll_to(percent=1.0)
                except:
                    break
            
            # Process loopback transcripts
            while not loopback_output_queue.empty():
                try:
                    item = loopback_output_queue.get_nowait()
                    if item.get('text'):
                        with other_transcript:
                            ui.label(item['text']).classes('text-green-200 text-sm mb-1')
                        other_transcript.scroll_to(percent=1.0)
                except:
                    break
        
        start_btn.on('click', start_session)
        stop_btn.on('click', stop_session)
        ui.timer(0.2, update_ui)

    ui.run(title='AI Counseling Assistant', reload=False, port=8080)

if __name__ in {"__main__", "__mp_main__"}:
    main()
