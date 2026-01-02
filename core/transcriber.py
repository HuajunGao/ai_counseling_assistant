import threading
import queue
import time
import numpy as np
import logging
import torch
import config
from core.asr_providers import create_asr_provider

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SileroVAD:
    def __init__(self, threshold=0.5):
        self.model = None
        self.utils = None
        self.threshold = threshold
        self.sampling_rate = 16000
        self._load_model()

    def _load_model(self):
        try:
            # Load Silero VAD from torch hub
            # process_trust_remote_code=True is needed for recent torch versions
            self.model, self.utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False, trust_repo=True
            )
            (self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks) = (
                self.utils
            )
            self.model.eval()
            logger.info("Silero VAD model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            self.model = None

    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int) -> bool:
        try:
            if self.model is None:
                return rms_energy(audio_chunk) > 0.01

            # Silero VAD v4+ requires processing in fixed chunks of 512 samples (for 16k) or 256 (for 8k)
            # We must iterate over the incoming chunk

            target_chunk_len = 512 if sample_rate == 16000 else 256

            # Ensure contiguous and tensor conversion
            if not audio_chunk.flags["C_CONTIGUOUS"]:
                audio_chunk = np.ascontiguousarray(audio_chunk)

            # Flatten to 1D
            audio_chunk = audio_chunk.flatten()

            # We can batch this or loop.
            # For simplicity and handling RNN state correctly (though we reset often here or share state?),
            # The simple model() call is usually stateless or resets.
            # Actually, Silero VAD is stateful.
            # But here we are just checking "is this block speech?".
            # We will process subs-chunks and if average prob > threshold we return True.

            num_samples = len(audio_chunk)
            # Pad if needed to match multiple of target_chunk_len, or just drop remainder
            # Dropping remainder is fine for VAD trigger purpose

            num_windows = num_samples // target_chunk_len
            if num_windows == 0:
                pass  # Chunk too small, maybe fallback or just return False

            speech_probs = []

            # Reset model state (if exposed) or just feed.
            # Note: Using model(x, sr) usually updates internal state.
            # Ideally we should keep state between external chunks, but Transcriber resets often.
            # For now, let's just feed sub-chunks.

            tensor_chunk = torch.from_numpy(audio_chunk)

            with torch.no_grad():
                for i in range(0, num_windows * target_chunk_len, target_chunk_len):
                    window = tensor_chunk[i : i + target_chunk_len]
                    if window.ndim == 1:
                        window = window.unsqueeze(0)
                    prob = self.model(window, sample_rate).item()
                    speech_probs.append(prob)

            if not speech_probs:
                return False

            # Decision: if max prob > threshold, or average?
            # Max prob is safer to trigger speech.
            return max(speech_probs) >= self.threshold

        except Exception:
            # Fallback to RMS silently or log if needed, avoid spamming stdout
            # import traceback
            # traceback.print_exc()
            return rms_energy(audio_chunk) > 0.01


class Transcriber(threading.Thread):
    def __init__(self, input_queue: queue.Queue, output_queue: queue.Queue, config=None, preloaded_provider=None):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = False
        self.provider = None
        self.preloaded_provider = preloaded_provider  # Store pre-loaded provider

        # Use provided config or fall back to global config module
        import config as default_config

        self.config = config if config is not None else default_config

        # Audio buffer configuration
        self.sample_rate = self.config.SAMPLE_RATE
        self.chunk_ms = self.config.ASR_CHUNK_MS
        self.chunk_frames = int(self.sample_rate * (self.chunk_ms / 1000.0))
        self.audio_buffer = np.array([], dtype=np.float32)

        # VAD/dynamic chunking
        self.vad_enabled = self.config.ASR_VAD_ENABLED
        self.vad_threshold = getattr(self.config, "ASR_VAD_THRESHOLD", 0.01)  # Default to config or low fallback
        self.silero_threshold = 0.4  # Default for Silero

        self.dynamic_chunks = self.config.ASR_DYNAMIC_CHUNKS
        self.silence_ms = self.config.ASR_SILENCE_MS
        self.min_segment_ms = self.config.ASR_MIN_SEGMENT_MS
        self.max_segment_ms = self.config.ASR_MAX_SEGMENT_MS
        self.silence_accum_ms = 0.0
        self.segment_parts = []
        self.segment_frames = 0
        self.segment_start_time = None  # Track when current segment started

        # Context Management
        self.last_text = ""
        self.vad_model = None
        self.punc_model = None
        self.punc_kwargs = None

    def load_model(self):
        try:
            # Use pre-loaded provider if available
            if self.preloaded_provider:
                self.provider = self.preloaded_provider
                logger.info("Using pre-loaded ASR provider")
            else:
                self.provider = create_asr_provider(self.config)
                logger.info("ASR provider loaded: %s", self.config.ASR_BACKEND)

            if self.vad_enabled:
                logger.info("Loading Silero VAD...")
                self.vad_model = SileroVAD(threshold=self.silero_threshold)

            if self.config.ASR_PUNC_POSTPROCESS and self.config.FUNASR_PUNC_MODEL:
                try:
                    from funasr import AutoModel

                    logger.info("Loading punctuation model for postprocess...")
                    model = AutoModel(
                        model=self.config.FUNASR_PUNC_MODEL,
                        device=self.config.FUNASR_DEVICE,
                        disable_update=self.config.FUNASR_DISABLE_UPDATE,
                        trust_remote_code=self.config.FUNASR_TRUST_REMOTE_CODE,
                    )
                    self.punc_model = model
                    self.punc_kwargs = getattr(model, "kwargs", {})
                except Exception as exc:
                    logger.warning("Failed to load punctuation model: %s", exc)

        except Exception as e:
            logger.error(f"Error loading ASR provider or VAD: {e}")
            raise

    def run(self):
        self.load_model()
        self.running = True
        logger.info(f"Transcriber started. Dynamic Chunks: {self.dynamic_chunks}, VAD: {self.vad_enabled}")

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

                time.sleep(0.01)  # Reduced sleep for better responsiveness

            except Exception as e:
                logger.error(f"Error in transcription loop: {e}")
                time.sleep(1)

    def is_speech(self, chunk: np.ndarray) -> bool:
        if not self.vad_enabled:
            return True

        # Use Silero if available
        if self.vad_model:
            return self.vad_model.is_speech(chunk, self.sample_rate)

        # Fallback to RMS
        return rms_energy(chunk) > self.vad_threshold

    def handle_chunk(self, chunk: np.ndarray) -> None:
        # Determine if this chunk has speech
        has_speech = self.is_speech(chunk)

        if self.dynamic_chunks:
            if not has_speech:
                # Silence detected
                if self.segment_parts:
                    self.silence_accum_ms += self.chunk_ms
                    segment_ms = (self.segment_frames / self.sample_rate) * 1000.0

                    # If silence is long enough AND segment is long enough -> Commit
                    if self.silence_accum_ms >= self.silence_ms and segment_ms >= self.min_segment_ms:
                        segment = np.concatenate(self.segment_parts)
                        self.emit_segment(segment)
                        self.reset_segment()
                # If no segment parts, we just ignore the silence
                return

            # Speech detected
            self.silence_accum_ms = 0.0
            if not self.segment_parts:
                # First chunk of new segment - record start time
                self.segment_start_time = time.time()
            self.segment_parts.append(chunk)
            self.segment_frames += chunk.shape[0]
            segment_ms = (self.segment_frames / self.sample_rate) * 1000.0

            # If segment is too long, force commit
            if segment_ms >= self.max_segment_ms:
                segment = np.concatenate(self.segment_parts)
                self.emit_segment(segment)
                self.reset_segment()
            return

        # Static chunks mode
        if not has_speech:
            return
        self.emit_segment(chunk)

    def emit_segment(self, segment: np.ndarray) -> None:
        if segment.size == 0:
            return
        try:
            # Pass last_text as prompt for context
            text = self.provider.transcribe(segment, self.sample_rate, prompt=self.last_text)
        except Exception as e:
            logger.error(f"TRANSCRIPTION FAILED: {e}")
            return

        if text.strip() and self.punc_model is not None and not _has_punctuation(text):
            try:
                punc_res = self.punc_model.inference(text, model=self.punc_model.model, kwargs=self.punc_kwargs)
                if isinstance(punc_res, list) and punc_res:
                    punc_text = (punc_res[0].get("text") or "").strip()
                    if punc_text:
                        text = punc_text
            except Exception as exc:
                logger.warning("Punctuation postprocess failed: %s", exc)

        if text.strip():
            # Update last_text for next context (keep it simple, maybe just the last sentence)
            self.last_text = text.strip()

            # Calculate latency from segment start to now
            now = time.time()
            latency = now - self.segment_start_time if self.segment_start_time else 0.0

            self.output_queue.put(
                {
                    "type": "commit",
                    "text": text.strip(),
                    "timestamp": now,
                    "latency": latency,
                }
            )

    def reset_segment(self) -> None:
        self.segment_parts = []
        self.segment_frames = 0
        self.silence_accum_ms = 0.0
        self.segment_start_time = None

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
    return float(np.sqrt(np.mean(signal**2)))


def _has_punctuation(text: str) -> bool:
    punc_chars = set("\u3002\uff01\uff1f\uff1b\uff0c\u3001\uff1a,.!?;:")
    return any(ch in punc_chars for ch in text)
