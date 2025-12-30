import logging
import os
from typing import List, Optional

import config
from core.llm_providers import create_llm_provider, OpenAIProvider

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_system_prompt(filepath: str) -> str:
    """Load system prompt from file."""
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception as e:
        logger.warning(f"Could not load system prompt from {filepath}: {e}")

    # Default fallback prompt
    return """You are an expert counseling supervisor and copilot.
Your goal is to assist the human counselor by providing real-time suggestions.
Keep suggestions CONCISE (1-2 sentences). Use the same language as the conversation."""


class SuggestionEngine:
    def __init__(self, model: str = None, context_length: int = None):
        self.model = model or config.OPENAI_MODEL
        self.context_length = context_length or config.AI_CONTEXT_LENGTH
        self.provider = None
        self.system_prompt = load_system_prompt(config.SYSTEM_PROMPT_FILE)
        self._init_provider()

    def _init_provider(self):
        """Initialize or reinitialize the LLM provider."""
        try:
            if config.OPENAI_API_KEY:
                self.provider = OpenAIProvider(
                    api_key=config.OPENAI_API_KEY, model=self.model, base_url=config.OPENAI_BASE_URL or None
                )
            else:
                self.provider = create_llm_provider(config)
        except Exception as e:
            logger.error(f"Failed to init LLM provider: {e}")
            self.provider = None

    def set_model(self, model: str):
        """Change the model at runtime."""
        if model != self.model:
            self.model = model
            self._init_provider()

    def set_context_length(self, length: int):
        """Set how many lines of context to use."""
        self.context_length = length

    def generate_suggestions(self, context: str) -> str:
        """Generate suggestions based on conversation context."""
        if not self.provider:
            return ""

        if not context or not context.strip():
            return ""

        full_prompt = f"{self.system_prompt}\n\nRecent Conversation:\n{context}\n\nProvide your suggestions:"

        try:
            return self.provider.generate(full_prompt)
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {e}")
            return ""
