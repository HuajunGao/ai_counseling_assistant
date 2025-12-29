import threading
import queue
import time
import numpy as np
import logging
from faster_whisper import WhisperModel
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Transcriber(threading.Thread):
    def __init__(self, input_queue: queue.Queue, output_queue: queue.Queue, model_size=config.WHISPER_MODEL_SIZE, device=config.DEVICE, compute_type=config.COMPUTE_TYPE):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = False
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        
        # Audio buffer configuration
        self.sample_rate = config.SAMPLE_RATE
        self.audio_buffer = np.array([], dtype=np.float32)
        self.new_data_threshold = 1.0 # Process every 1 second of new data
        self.last_process_time = time.time()
        self.silence_threshold = 0.01 # RMS threshold for silence
        self.silence_duration = 0
        self.max_silence_to_commit = 1.5 # Seconds of silence to commit the segment

    def load_model(self):
        logger.info(f"Loading Whisper model: {self.model_size} on {self.device}...")
        try:
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            # Fallback to cpu/int8 if cuda fails
            if self.device == "cuda":
                logger.warning("Falling back to CPU...")
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

    def run(self):
        self.load_model()
        self.running = True
        
        while self.running:
            try:
                # Get all available chunks
                chunks = []
                while True:
                    try:
                        chunk = self.input_queue.get_nowait()
                        chunks.append(chunk)
                    except queue.Empty:
                        break
                
                if chunks:
                    data = np.concatenate(chunks)
                    self.audio_buffer = np.concatenate((self.audio_buffer, data))

                # Check amplitude for VAD (Simple RMS)
                if len(self.audio_buffer) > 0:
                    current_rms = np.sqrt(np.mean(self.audio_buffer[-int(self.sample_rate*0.5):]**2)) if len(self.audio_buffer) > self.sample_rate*0.5 else 0
                    if current_rms < self.silence_threshold:
                        self.silence_duration += (time.time() - self.last_process_time)
                    else:
                        self.silence_duration = 0

                # Process condition: Enough new data OR silence commit
                time_since_last = time.time() - self.last_process_time
                if (len(self.audio_buffer) > 0) and (time_since_last > self.new_data_threshold or self.silence_duration > self.max_silence_to_commit):
                    
                    self.transcribe_buffer()
                    self.last_process_time = time.time()

                    # If silence committed, clear buffer
                    if self.silence_duration > self.max_silence_to_commit:
                         # Send finalization signal/clear buffer
                         # For MVP: Just clearing buffer is risky if we cut words, but simplest
                         self.audio_buffer = np.array([], dtype=np.float32)
                         self.silence_duration = 0
                         self.output_queue.put({"type": "commit"})
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in transcription loop: {e}")
                time.sleep(1)

    def transcribe_buffer(self):
        if len(self.audio_buffer) < self.sample_rate * 0.5: # Skip very short audio
            return

        logger.info(f"BUFFER SIZE: {len(self.audio_buffer)} - STARTING INFERENCE") # DEBUG
        try:
            segments, info = self.model.transcribe(
                self.audio_buffer, 
                beam_size=5, 
                language="zh", # Prioritize Chinese as per req, or "auto"
                vad_filter=True
            )
            logger.info("INFERENCE RETURNED GENERATOR") # DEBUG
            
            text = " ".join([segment.text for segment in segments])
            logger.info(f"INFERENCE DONE. TEXT: {text[:20]}...") # DEBUG
        except Exception as e:
            logger.error(f"TRANSCRIPTION FAILED: {e}")
            return
        if text.strip():
            self.output_queue.put({
                "type": "partial",
                "text": text,
                "timestamp": time.time()
            })

    def stop(self):
        self.running = False
