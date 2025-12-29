from nicegui import ui, app
import asyncio
import time
import queue
import logging
from core.audio_capture import AudioCapture
from core.transcriber import Transcriber
from core.llm_engine import SuggestionEngine
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

def main():
    # Load devices
    input_devices = audio_capture.list_devices()
    device_options = {d['id']: f"{d['name']} ({d['hostapi']})" for d in input_devices}
    
    # Default to first device or a likely candidate
    default_device = input_devices[0]['id'] if input_devices else None

    @ui.page('/')
    def index():
        global is_recording, transcriber, selected_device
        
        # Use a wide layout
        ui.query('body').style('background-color: #f3f4f6')
        
        with ui.header().classes('bg-white text-gray-800 border-b p-4 flex items-center justify-between'):
            ui.label('AI Counseling Copilot').classes('text-2xl font-bold text-indigo-600')
            
            with ui.row().classes('items-center gap-4'):
                # Device Selector
                device_select = ui.select(device_options, value=default_device, label='Audio Input Device').classes('w-64')
                
                # Controls
                start_btn = ui.button('Start Session', on_click=lambda: start_session(device_select.value), icon='mic').props('color=green')
                stop_btn = ui.button('Stop', on_click=lambda: stop_session(), icon='stop').props('color=red').bind_visibility_from(globals(), 'is_recording')
                start_btn.bind_visibility_from(globals(), 'is_recording', backward=lambda x: not x)

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

                    if event['type'] == 'partial':
                        # Update partial view (could be improved to replace last element)
                        # For MVP: We just show finalized segments. Partial handling is tricky in append-only UI.
                        # Wait, user wanted real-time.
                        # Improved logic: If partial, update a "temp" label. If final/commit, Add permanent row.
                        pass # Ignore partials for simple append log, or handle them.
                        # Let's trust "commit" events or just append "partial" if it's long enough.
                        
                        # Actually Transcriber.py gives 'partial' events continuously. 
                        # We need to render them.
                        pass 

                    # Re-reading Transcriber code: It emits type='partial' with text. 
                    # And type='commit' when silence.
                    # BUT Transcriber doesn't send "final" text for the segment in 'commit'.
                    # It relies on 'partial' being the latest text.
                    
                    if event['type'] == 'partial':
                        # This is the latest text for the current line
                        # We should update the LAST element if it's "pending", or create a new one.
                        # Simplification: Just log it for now.
                        # Real implementation: Use a mutable UI element or clear/redraw the "current" line.
                        
                        # Let's just append for MVP to see it working, then refine.
                        # Or better: Maintain a "Status Label" for current speech.
                        pass

                    # REVISION: Let's do a proper "Pending" line.
                    # Since NiceGUI is declarative, we can bind a variable.
                    
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
                        text = event['text']
                        current_partial_text = text
                        active_transcript_label.set_text(text + "...")
                        
                        # Use partials for context if really needed, but maybe too noisy
                        # suggestion_engine.update_transcript(text) 
                        
                    elif event['type'] == 'commit':
                        if current_partial_text.strip():
                            # Commit the line
                            with transcript_container:
                                components.transcript_row(current_partial_text, time.time(), is_final=True)
                                # Add to engine history
                                suggestion_engine.update_transcript(current_partial_text)
                            
                            # Move active label to bottom again
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

    ui.run(title='AI Counseling Assistant', reload=False, port=8080)

if __name__ in {"__main__", "__mp_main__"}:
    main()
