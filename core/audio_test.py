"""
Audio device test utilities for the UI.
Uses soundcard library for WASAPI loopback support.
"""

import threading
import time
import numpy as np
import sounddevice as sd
import logging

try:
    import soundcard as sc
except ImportError:
    sc = None

logger = logging.getLogger(__name__)


def list_all_devices():
    """List devices using soundcard library only (matches record_benchmark.py)."""
    devices = []

    if sc is None:
        return devices

    # Output devices (for loopback capture)
    speakers = sc.all_speakers()
    for idx, spk in enumerate(speakers):
        devices.append(
            {
                "id": f"loopback:{idx}",
                "name": spk.name,
                "type": "Loopback (抓取电脑声音)",
                "is_loopback": True,
            }
        )

    # Input devices (microphones)
    microphones = sc.all_microphones(include_loopback=False)
    for idx, mic in enumerate(microphones):
        devices.append(
            {
                "id": f"mic:{idx}",
                "name": mic.name,
                "type": "Microphone (麦克风)",
                "is_loopback": False,
            }
        )

    return devices


def get_loopback_devices():
    """Get devices that support loopback capture (soundcard speakers)."""
    if sc is None:
        return []
    speakers = sc.all_speakers()
    return [{"id": f"loopback:{i}", "name": s.name} for i, s in enumerate(speakers)]


class AudioTester:
    """Records audio and calculates RMS for live level display."""

    def __init__(self, device_id: str, duration: float = 5.0, use_loopback: bool = False):
        self.device_id = device_id  # Format: "sc:0" or "sd:25"
        self.duration = duration
        self.use_loopback = use_loopback
        self.sample_rate = 16000
        self.channels = 1

        self.running = False
        self.current_rms = 0.0
        self.peak_rms = 0.0
        self.recorded_audio = None
        self._thread = None
        self._error = None

    def start(self):
        """Start recording in a background thread."""
        self.running = True
        self.current_rms = 0.0
        self.peak_rms = 0.0
        self.recorded_audio = None
        self._error = None
        self._thread = threading.Thread(target=self._record, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop recording."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _record(self):
        """Recording thread."""
        try:
            parts = self.device_id.split(":")
            device_type = parts[0]
            idx = int(parts[1])

            if device_type == "loopback":
                self._record_loopback(idx)
            else:  # mic
                self._record_microphone(idx)
        except Exception as e:
            logger.error(f"Recording error: {e}")
            self._error = str(e)
            self.running = False

    def _record_microphone(self, mic_idx: int):
        """Record from a standard input device using soundcard."""
        if sc is None:
            raise RuntimeError("soundcard library not installed")

        microphones = sc.all_microphones(include_loopback=False)
        if mic_idx >= len(microphones):
            raise RuntimeError(f"Mic {mic_idx} not found. Available: {len(microphones)}")

        mic = microphones[mic_idx]
        audio_buffer = []

        with mic.recorder(samplerate=self.sample_rate, channels=self.channels) as recorder:
            start_time = time.time()
            while self.running and (time.time() - start_time) < self.duration:
                chunk = recorder.record(numframes=int(self.sample_rate * 0.1))
                if chunk.ndim == 2:
                    chunk = np.mean(chunk, axis=1)
                audio_buffer.append(chunk)

                rms = np.sqrt(np.mean(chunk**2))
                self.current_rms = float(rms)
                self.peak_rms = max(self.peak_rms, self.current_rms)

        self.running = False
        if audio_buffer:
            self.recorded_audio = np.concatenate(audio_buffer)

    def _record_loopback(self, speaker_idx: int):
        """Record from an output device using soundcard loopback."""
        if sc is None:
            raise RuntimeError("soundcard library not installed. Run: pip install soundcard")

        speakers = sc.all_speakers()
        if speaker_idx >= len(speakers):
            raise RuntimeError(f"Speaker {speaker_idx} not found. Available: {len(speakers)}")

        speaker = speakers[speaker_idx]

        # Find matching loopback mic
        microphones = sc.all_microphones(include_loopback=True)
        loopback_mics = [m for m in microphones if getattr(m, "isloopback", False)]

        loopback_mic = None
        for mic in loopback_mics:
            if mic.name == speaker.name or speaker.name.lower() in mic.name.lower():
                loopback_mic = mic
                break

        if loopback_mic is None and loopback_mics:
            loopback_mic = loopback_mics[0]

        if loopback_mic is None:
            raise RuntimeError("No loopback microphone found.")

        logger.info(f"Using loopback mic: {loopback_mic.name}")
        audio_buffer = []

        with loopback_mic.recorder(samplerate=self.sample_rate, channels=self.channels) as recorder:
            start_time = time.time()
            while self.running and (time.time() - start_time) < self.duration:
                chunk = recorder.record(numframes=int(self.sample_rate * 0.1))  # 100ms chunks
                if chunk.ndim == 2:
                    chunk = np.mean(chunk, axis=1)
                audio_buffer.append(chunk)

                rms = np.sqrt(np.mean(chunk**2))
                self.current_rms = float(rms)
                self.peak_rms = max(self.peak_rms, self.current_rms)

        self.running = False
        if audio_buffer:
            self.recorded_audio = np.concatenate(audio_buffer)

    def play_recording(self):
        """Play the recorded audio."""
        if self.recorded_audio is None or len(self.recorded_audio) == 0:
            return False
        try:
            sd.play(self.recorded_audio.astype(np.float32), self.sample_rate)
            sd.wait()
            return True
        except Exception as e:
            logger.error(f"Playback error: {e}")
            return False

    def get_status(self) -> dict:
        """Get current status for UI updates."""
        return {
            "running": self.running,
            "current_rms": self.current_rms,
            "peak_rms": self.peak_rms,
            "has_recording": self.recorded_audio is not None and len(self.recorded_audio) > 0,
            "error": self._error,
        }
