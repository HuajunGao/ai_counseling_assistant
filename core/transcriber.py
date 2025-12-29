import threading
import queue
import time
import numpy as np
import logging
import config
from core.asr_providers import create_asr_provider

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Transcriber(threading.Thread):
    def __init__(self, input_queue: queue.Queue, output_queue: queue.Queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = False
        self.provider = None

        # Audio buffer configuration
        self.sample_rate = config.SAMPLE_RATE
        self.chunk_ms = config.ASR_CHUNK_MS
        self.chunk_frames = int(self.sample_rate * (self.chunk_ms / 1000.0))
        self.audio_buffer = np.array([], dtype=np.float32)

        # VAD/dynamic chunking
        self.vad_enabled = config.ASR_VAD_ENABLED
        self.vad_threshold = config.ASR_VAD_THRESHOLD
        self.dynamic_chunks = config.ASR_DYNAMIC_CHUNKS
        self.silence_ms = config.ASR_SILENCE_MS
        self.min_segment_ms = config.ASR_MIN_SEGMENT_MS
        self.max_segment_ms = config.ASR_MAX_SEGMENT_MS
        self.silence_accum_ms = 0.0
        self.segment_parts = []
        self.segment_frames = 0

    def load_model(self):
        try:
            self.provider = create_asr_provider(config)
            logger.info("ASR provider loaded: %s", config.ASR_BACKEND)
        except Exception as e:
            logger.error(f"Error loading ASR provider: {e}")
            raise

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

                while self.audio_buffer.shape[0] >= self.chunk_frames:
                    chunk = self.audio_buffer[: self.chunk_frames]
                    self.audio_buffer = self.audio_buffer[self.chunk_frames :]
                    self.handle_chunk(chunk)

                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in transcription loop: {e}")
                time.sleep(1)

    def handle_chunk(self, chunk: np.ndarray) -> None:
        chunk_rms = rms_energy(chunk)

        if self.dynamic_chunks:
            if chunk_rms < self.vad_threshold:
                if self.segment_parts:
                    self.silence_accum_ms += self.chunk_ms
                    segment_ms = (self.segment_frames / self.sample_rate) * 1000.0
                    if self.silence_accum_ms >= self.silence_ms and segment_ms >= self.min_segment_ms:
                        segment = np.concatenate(self.segment_parts)
                        self.emit_segment(segment)
                        self.reset_segment()
                return

            self.silence_accum_ms = 0.0
            self.segment_parts.append(chunk)
            self.segment_frames += chunk.shape[0]
            segment_ms = (self.segment_frames / self.sample_rate) * 1000.0
            if segment_ms >= self.max_segment_ms:
                segment = np.concatenate(self.segment_parts)
                self.emit_segment(segment)
                self.reset_segment()
            return

        if self.vad_enabled and chunk_rms < self.vad_threshold:
            return
        self.emit_segment(chunk)

    def emit_segment(self, segment: np.ndarray) -> None:
        if segment.size == 0:
            return
        try:
            text = self.provider.transcribe(segment, self.sample_rate)
        except Exception as e:
            logger.error(f"TRANSCRIPTION FAILED: {e}")
            return
        if text.strip():
            self.output_queue.put(
                {
                    "type": "commit",
                    "text": text.strip(),
                    "timestamp": time.time(),
                }
            )

    def reset_segment(self) -> None:
        self.segment_parts = []
        self.segment_frames = 0
        self.silence_accum_ms = 0.0

    def stop(self):
        self.running = False

        if self.dynamic_chunks and self.segment_parts:
            try:
                segment = np.concatenate(self.segment_parts)
                self.emit_segment(segment)
            except Exception:
                pass


def rms_energy(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal ** 2)))
