import sounddevice as sd
import numpy as np
import queue
import logging
from typing import List, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioCapture:
    def __init__(self, audio_queue=None, sample_rate=16000, native_sample_rate=None, channels=1, block_size=4096):
        self.sample_rate = sample_rate
        # If native_sample_rate is not provided, use sample_rate (assuming hardware supports it)
        # If hardware requires 48k but we need 16k, setup decimation
        self.native_sample_rate = native_sample_rate if native_sample_rate else sample_rate
        self.channels = channels
        self.block_size = block_size
        self.audio_queue = audio_queue if audio_queue is not None else queue.Queue()
        self.stream: Optional[sd.InputStream] = None
        self.running = False
        
        # Simple integer decimation check
        self.decimation_factor = 1
        if self.native_sample_rate != self.sample_rate:
            if self.native_sample_rate % self.sample_rate == 0:
                self.decimation_factor = int(self.native_sample_rate // self.sample_rate)
                logger.info(f"Resampling enabled: Native {self.native_sample_rate}Hz -> Target {self.sample_rate}Hz (Decimation factor: {self.decimation_factor})")
            else:
                logger.warning(f"Complex resampling ({self.native_sample_rate}->{self.sample_rate}) requested but not implemented. Using native rate (Whisper might fail).")
                self.decimation_factor = 1

    def list_devices(self) -> List[dict]:
        """List available audio input devices."""
        devices = sd.query_devices()
        input_devices = []
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                input_devices.append({
                    'id': i,
                    'name': dev['name'],
                    'hostapi': dev['hostapi'],
                    'max_input_channels': dev['max_input_channels']
                })
        return input_devices

    def _callback(self, indata, frames, time, status):
        """Callback for sounddevice stream."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        if self.running:
            # Copy data to avoid buffer issues, flatten if necessary
            # faster-whisper expects float32
            # Apply decimation if configured
            data = indata.copy()
            
            # Apply decimation if configured
            data = indata.copy()
            
            # Diagnostic: Check volume levels once per second-ish
            if data.size > 0:
                rms = np.sqrt(np.mean(data**2))
                if rms > 0.001 and np.random.rand() < 0.05: # Log occasionally when sound is present
                     logger.info(f"Audio Level (RMS): {rms:.4f}")
                elif np.random.rand() < 0.005: # Log occasionally even if silent
                     logger.info(f"Audio Level (RMS-SILENCE): {rms:.6f}")
            
            if self.decimation_factor > 1:
                data = data[::self.decimation_factor]
            
            self.audio_queue.put(data)

    def start_stream(self, device_id: int):
        """Start recording from the specified device."""
        if self.running:
            logger.info("Stream already running")
            return

        try:
            self.stream = sd.InputStream(
                device=device_id,
                samplerate=self.native_sample_rate,
                channels=self.channels,
                blocksize=self.block_size,
                callback=self._callback,
                dtype="float32"
            )
            self.stream.start()
            self.running = True
            logger.info(f"Started audio stream on device {device_id}")
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            raise

    def stop_stream(self):
        """Stop the audio stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.running = False
        logger.info("Stopped audio stream")

    def start(self, device_id=None):
        """Alias for start_stream to match Thread API."""
        if device_id is None:
            # Use sounddevice default input device
            device_id = sd.default.device[0]
        self.start_stream(device_id)

    def stop(self):
        """Alias for stop_stream."""
        self.stop_stream()

    def get_audio_chunk(self):
        """Retrieve a chunk of audio from the queue."""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None
