"""
Dual-stream audio capture for conversation transcription.
Captures both microphone (我) and speaker/loopback (对方) simultaneously.
"""

import threading
import queue
import time
import warnings
import sys
import numpy as np
import logging

# Suppress soundcard warnings
warnings.filterwarnings("ignore", message=".*discontinuity.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Windows COM initialization for Streamlit compatibility
if sys.platform == "win32":
    try:
        import pythoncom

        pythoncom.CoInitialize()
    except ImportError:
        pass

try:
    import soundcard as sc
except ImportError:
    sc = None

logger = logging.getLogger(__name__)


class DualStreamCapture:
    """Captures audio from both microphone and speaker loopback."""

    def __init__(self, mic_idx: int, loopback_idx: int, sample_rate: int = 16000, mode: str = "dual"):
        self.mic_idx = mic_idx
        self.loopback_idx = loopback_idx
        self.sample_rate = sample_rate
        self.mode = mode  # "dual", "mic_only", or "speaker_only"

        self.mic_queue = queue.Queue()
        self.loopback_queue = queue.Queue()

        self.running = False
        self.mic_rms = 0.0
        self.loopback_rms = 0.0

        self._mic_thread = None
        self._loopback_thread = None

    def start(self):
        """Start capturing from enabled sources based on mode."""
        if sc is None:
            raise RuntimeError("soundcard library not installed")

        self.running = True
        
        # Start mic thread if mode is dual or mic_only
        if self.mode in ("dual", "mic_only"):
            self._mic_thread = threading.Thread(target=self._capture_mic, daemon=True)
            self._mic_thread.start()
            logger.info(f"Mic capture started (mode: {self.mode})")
        
        # Start loopback thread if mode is dual or speaker_only
        if self.mode in ("dual", "speaker_only"):
            self._loopback_thread = threading.Thread(target=self._capture_loopback, daemon=True)
            self._loopback_thread.start()
            logger.info(f"Loopback capture started (mode: {self.mode})")
        
        logger.info(f"Audio capture started in {self.mode} mode")

    def stop(self):
        """Stop capturing."""
        self.running = False
        if self._mic_thread and self._mic_thread.is_alive():
            self._mic_thread.join(timeout=2.0)
            logger.info("Mic capture stopped")
        if self._loopback_thread and self._loopback_thread.is_alive():
            self._loopback_thread.join(timeout=2.0)
            logger.info("Loopback capture stopped")
        logger.info(f"Audio capture stopped ({self.mode} mode)")

    def _capture_mic(self):
        """Capture from microphone."""
        # Initialize COM for this thread on Windows
        if sys.platform == "win32":
            try:
                import pythoncom

                pythoncom.CoInitialize()
            except:
                pass

        mics = sc.all_microphones(include_loopback=False)
        if self.mic_idx >= len(mics):
            logger.error(f"Mic {self.mic_idx} not found")
            return

        mic = mics[self.mic_idx]
        logger.info(f"Opening mic device {self.mic_idx}: {mic.name}")

        try:
            with mic.recorder(samplerate=self.sample_rate, channels=1) as recorder:
                logger.info(f"Mic recorder opened successfully: {mic.name}")
                while self.running:
                    chunk = recorder.record(numframes=int(self.sample_rate * 0.1))
                    if chunk.ndim == 2:
                        chunk = np.mean(chunk, axis=1)
                    self.mic_rms = float(np.sqrt(np.mean(chunk**2)))
                    self.mic_queue.put(chunk.astype(np.float32))
                logger.info(f"Mic recorder closed normally: {mic.name}")
        except Exception as e:
            logger.error(f"Mic capture error for {mic.name}: {e}", exc_info=True)

    def _capture_loopback(self):
        """Capture from speaker loopback."""
        # Initialize COM for this thread on Windows
        if sys.platform == "win32":
            try:
                import pythoncom

                pythoncom.CoInitialize()
            except:
                pass

        speakers = sc.all_speakers()
        if self.loopback_idx >= len(speakers):
            logger.error(f"Speaker {self.loopback_idx} not found")
            return

        speaker = speakers[self.loopback_idx]

        # Find loopback mic
        loopback_mics = [m for m in sc.all_microphones(include_loopback=True) if getattr(m, "isloopback", False)]

        loopback_mic = None
        for lm in loopback_mics:
            if lm.name == speaker.name or speaker.name.lower() in lm.name.lower():
                loopback_mic = lm
                break
        if loopback_mic is None and loopback_mics:
            loopback_mic = loopback_mics[0]

        if loopback_mic is None:
            logger.error("No loopback mic found")
            return

        logger.info(f"Opening loopback device: {loopback_mic.name}")

        try:
            with loopback_mic.recorder(samplerate=self.sample_rate, channels=1) as recorder:
                logger.info(f"Loopback recorder opened successfully: {loopback_mic.name}")
                while self.running:
                    chunk = recorder.record(numframes=int(self.sample_rate * 0.1))
                    if chunk.ndim == 2:
                        chunk = np.mean(chunk, axis=1)
                    self.loopback_rms = float(np.sqrt(np.mean(chunk**2)))
                    self.loopback_queue.put(chunk.astype(np.float32))
                logger.info(f"Loopback recorder closed normally: {loopback_mic.name}")
        except Exception as e:
            logger.error(f"Loopback capture error for {loopback_mic.name}: {e}", exc_info=True)

    def get_levels(self) -> dict:
        """Get current audio levels."""
        return {
            "mic_rms": self.mic_rms if self.mode in ("dual", "mic_only") else 0.0,
            "loopback_rms": self.loopback_rms if self.mode in ("dual", "speaker_only") else 0.0,
        }


def list_devices():
    """List available devices for dual capture."""
    if sc is None:
        return {"mics": [], "speakers": []}

    mics = sc.all_microphones(include_loopback=False)
    speakers = sc.all_speakers()

    return {
        "mics": [{"id": i, "name": m.name} for i, m in enumerate(mics)],
        "speakers": [{"id": i, "name": s.name} for i, s in enumerate(speakers)],
    }
