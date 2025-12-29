import os
from dotenv import load_dotenv

load_dotenv()

# Audio Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 4096  # Audio chunk size
AUDIO_QUEUE_SIZE = 1000

# ASR Configuration
WHISPER_MODEL_SIZE = "tiny" # or "base", "small", "medium", "large-v3"
COMPUTE_TYPE = "float16" # "float16" if GPU supports it
DEVICE = "cuda" # "cuda" or "cpu"

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama") # "ollama" or "openai"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# App Configuration
LOG_DIR = "logs"
