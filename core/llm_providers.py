import abc
import logging

import requests

logger = logging.getLogger(__name__)


class LLMProvider(abc.ABC):
    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from the LLM based on the prompt."""


class NullProvider(LLMProvider):
    def generate(self, prompt: str) -> str:
        return ""


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return ""


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError(f"openai package not installed: {exc}") from exc
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI provider.")
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate text using OpenAI chat completions with optional system prompt."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Debug logging - print full prompt
            logger.debug("\n" + "="*50)
            logger.debug("=== AI Request ===")
            for msg in messages:
                content = msg['content']
                logger.debug(f"[{msg['role']}]:\n{content}")
            logger.debug("="*50 + "\n")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return ""


def create_llm_provider(config) -> LLMProvider:
    provider_type = config.LLM_PROVIDER.lower()
    if provider_type == "openai":
        if not config.OPENAI_API_KEY:
            logger.warning("OpenAI API Key not found, falling back to Ollama.")
            return OllamaProvider(config.OLLAMA_BASE_URL, config.OLLAMA_MODEL)
        return OpenAIProvider(config.OPENAI_API_KEY, config.OPENAI_MODEL, config.OPENAI_BASE_URL or None)
    if provider_type == "none":
        return NullProvider()
    return OllamaProvider(config.OLLAMA_BASE_URL, config.OLLAMA_MODEL)
