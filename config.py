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
ASR_BACKEND = os.getenv("ASR_BACKEND", "funasr")  # local/openai/funasr/azure
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "zh")  # zh/auto
ASR_CHUNK_MS = int(os.getenv("ASR_CHUNK_MS", "1000"))
ASR_DYNAMIC_CHUNKS = _env_bool("ASR_DYNAMIC_CHUNKS", True)
ASR_VAD_ENABLED = _env_bool("ASR_VAD_ENABLED", True)
ASR_VAD_THRESHOLD = float(os.getenv("ASR_VAD_THRESHOLD", "0.01"))
ASR_SILENCE_MS = int(os.getenv("ASR_SILENCE_MS", "900"))
ASR_MIN_SEGMENT_MS = int(os.getenv("ASR_MIN_SEGMENT_MS", "5000"))
ASR_MAX_SEGMENT_MS = int(os.getenv("ASR_MAX_SEGMENT_MS", "15000"))
ASR_PUNC_POSTPROCESS = _env_bool("ASR_PUNC_POSTPROCESS", True)

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3")  # tiny/base/small/medium/large-v3
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")  # float16 for GPU, int8 for CPU
DEVICE = os.getenv("DEVICE", "cuda")  # cuda for GPU, cpu for fallback

OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")

FUNASR_MODEL = os.getenv(
    "FUNASR_MODEL",
    "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
)
FUNASR_DEVICE = os.getenv("FUNASR_DEVICE", "cuda")
FUNASR_VAD_MODEL = os.getenv("FUNASR_VAD_MODEL", "")
FUNASR_PUNC_MODEL = os.getenv("FUNASR_PUNC_MODEL", "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch")
FUNASR_DISABLE_UPDATE = _env_bool("FUNASR_DISABLE_UPDATE", True)
FUNASR_TRUST_REMOTE_CODE = _env_bool("FUNASR_TRUST_REMOTE_CODE", False)

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "ollama", "openai", or "none"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# AI Suggestion Settings
AI_SUGGESTION_INTERVAL = int(os.getenv("AI_SUGGESTION_INTERVAL", "30"))  # seconds
AI_CONTEXT_LENGTH = int(os.getenv("AI_CONTEXT_LENGTH", "10"))  # lines
SYSTEM_PROMPT_FILE = os.getenv("SYSTEM_PROMPT_FILE", "prompts/counseling_system.txt")

# Available models for UI selection
OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
ASR_BACKENDS = ["funasr", "whisper", "openai", "azure"]  # Transcription backend options

# Default device name for auto-selection
DEFAULT_DEVICE_NAME = os.getenv("DEFAULT_DEVICE_NAME", "Echo Cancelling Speakerphone")

# Azure Speech (ASR)
SPEECH_KEY = os.getenv("SPEECH_KEY", "")
SPEECH_REGION = os.getenv("SPEECH_REGION", "")
SPEECH_ENDPOINT = os.getenv("SPEECH_ENDPOINT", "")

# App Configuration
LOG_DIR = "logs"
