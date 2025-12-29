import sys
import os
import logging
import traceback
import sounddevice as sd
import soundfile as sf
import numpy as np

# Fix for MKL/OpenMP conflict (silent crash)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model():
    print("--- Faster-Whisper Model Load Test ---", flush=True)
    import numpy
    print(f"NumPy Version: {numpy.__version__}", flush=True)
    
    try:
        from faster_whisper import WhisperModel
        print("faster_whisper imported.", flush=True)
    except Exception as e:
        print(f"Failed to import faster_whisper: {e}", flush=True)
        return

    model_size = "tiny"
    device = "cpu"
    compute_type = "int8"
    
    print(f"Loading {model_size} on {device} ({compute_type})...", flush=True)
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Model loaded successfully!", flush=True)
    except Exception as e:
        print(f"Failed to load model: {e}", flush=True)
        traceback.print_exc()

    print("Test finished.", flush=True)

if __name__ == "__main__":
    test_model()
