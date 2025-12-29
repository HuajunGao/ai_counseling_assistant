import os
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# Audio Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 4096  # Audio chunk size
AUDIO_QUEUE_SIZE = 1000

# ASR Configuration
ASR_BACKEND = os.getenv("ASR_BACKEND", "local")  # local/openai/funasr/azure
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "zh")  # zh/auto
ASR_CHUNK_MS = int(os.getenv("ASR_CHUNK_MS", "1000"))
ASR_DYNAMIC_CHUNKS = _env_bool("ASR_DYNAMIC_CHUNKS", True)
ASR_VAD_ENABLED = _env_bool("ASR_VAD_ENABLED", True)
ASR_VAD_THRESHOLD = float(os.getenv("ASR_VAD_THRESHOLD", "0.01"))
ASR_SILENCE_MS = int(os.getenv("ASR_SILENCE_MS", "600"))
ASR_MIN_SEGMENT_MS = int(os.getenv("ASR_MIN_SEGMENT_MS", "800"))
ASR_MAX_SEGMENT_MS = int(os.getenv("ASR_MAX_SEGMENT_MS", "12000"))

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")  # tiny/base/small/medium/large-v3
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
DEVICE = os.getenv("DEVICE", "cuda")

OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")

FUNASR_MODEL = os.getenv(
    "FUNASR_MODEL",
    "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
)
FUNASR_DEVICE = os.getenv("FUNASR_DEVICE", "cuda")
FUNASR_VAD_MODEL = os.getenv("FUNASR_VAD_MODEL", "")
FUNASR_PUNC_MODEL = os.getenv("FUNASR_PUNC_MODEL", "")
FUNASR_DISABLE_UPDATE = _env_bool("FUNASR_DISABLE_UPDATE", True)
FUNASR_TRUST_REMOTE_CODE = _env_bool("FUNASR_TRUST_REMOTE_CODE", False)

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama") # "ollama", "openai", or "none"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Azure Speech (ASR)
SPEECH_KEY = os.getenv("SPEECH_KEY", "")
SPEECH_REGION = os.getenv("SPEECH_REGION", "")
SPEECH_ENDPOINT = os.getenv("SPEECH_ENDPOINT", "")

# App Configuration
LOG_DIR = "logs"
