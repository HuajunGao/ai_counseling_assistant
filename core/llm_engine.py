import abc
import os
import requests
import json
import logging
from typing import List, Dict, Optional
import config

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider(abc.ABC):
    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from the LLM based on the prompt."""
        pass

class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return f"Error: {e}"

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        if not OpenAI:
            raise ImportError("openai package not installed. Please install it to use OpenAI provider.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return f"Error: {e}"

class SuggestionEngine:
    def __init__(self):
        self.provider = self._get_provider()
        self.transcript_buffer: List[str] = []
        self.max_buffer_lines = 20 # Keep last 20 lines for immediate context
        
        # System Prompt / Persona
        self.system_prompt = """
You are an expert counseling supervisor and copilot. 
Your goal is to assist the human counselor by providing real-time suggestions based on the conversation transcript.
Do NOT act as the counselor talking to the client. Act as a whisperer to the counselor.

Output Format:
1. **Suggested Response**: A gentle, empathetic phrase the counselor could say.
2. **Suggested Question**: A question to deepen understanding or clarify.
3. **Observation/Risk**: (Optional) Note any emotional shifts, boundaries, or risks.

Keep suggestions CONCISE (1-2 sentences).
"""

    def _get_provider(self) -> LLMProvider:
        provider_type = config.LLM_PROVIDER.lower()
        if provider_type == "openai":
            if not config.OPENAI_API_KEY:
                logger.warning("OpenAI API Key not found. Falling back to Ollama.")
                return OllamaProvider(config.OLLAMA_BASE_URL, config.OLLAMA_MODEL)
            return OpenAIProvider(config.OPENAI_API_KEY, config.OPENAI_MODEL)
        else:
            return OllamaProvider(config.OLLAMA_BASE_URL, config.OLLAMA_MODEL)

    def update_transcript(self, text: str):
        """Append new text to the transcript buffer."""
        self.transcript_buffer.append(text)
        if len(self.transcript_buffer) > self.max_buffer_lines:
            self.transcript_buffer.pop(0)

    def generate_suggestions(self) -> str:
        """Generate suggestions based on current transcript."""
        if not self.transcript_buffer:
            return ""

        transcript_text = "\n".join(self.transcript_buffer)
        full_prompt = f"{self.system_prompt}\n\nRecent Transcript:\n{transcript_text}\n\nSuggestions:"
        
        return self.provider.generate(full_prompt)

    def set_provider(self, provider_type: str, api_key: str = None, model: str = None):
        """Runtime configuration of provider (optional for UI)."""
        # Logic to switch provider at runtime if needed
        pass
