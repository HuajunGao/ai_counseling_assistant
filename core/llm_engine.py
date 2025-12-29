import logging
from typing import List

import config
from core.llm_providers import create_llm_provider

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SuggestionEngine:
    def __init__(self):
        self.provider = create_llm_provider(config)
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
