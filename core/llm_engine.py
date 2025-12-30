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

    def generate_suggestions(self, speaker_transcript: list, listener_transcript: list, user_question: str = "") -> str:
        """Generate suggestions based on conversation context.
        
        Args:
            speaker_transcript: List of {"time": str, "text": str} from 倾诉者 (client)
            listener_transcript: List of {"time": str, "text": str} from 倾听者 (counselor)
            user_question: Optional question from the counselor to AI
        """
        if not self.provider:
            return ""

        # Build structured context as JSON
        import json
        
        context_data = {
            "倾诉者": [{"time": item.get("time", ""), "text": item.get("text", "")} 
                      for item in speaker_transcript[-self.context_length:]],
            "倾听者": [{"time": item.get("time", ""), "text": item.get("text", "")} 
                      for item in listener_transcript[-self.context_length:]],
        }
        
        if not context_data["倾诉者"] and not context_data["倾听者"]:
            return ""
        
        # Build user message
        user_content = f"对话上下文:\n```json\n{json.dumps(context_data, ensure_ascii=False, indent=2)}\n```"
        
        if user_question and user_question.strip():
            user_content += f"\n\n倾听者的问题: {user_question.strip()}"
        else:
            user_content += "\n\n请根据上下文提供建议。"

        try:
            return self.provider.generate(user_content, system_prompt=self.system_prompt)
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {e}")
            return ""

    def generate_session_summary(self, speaker_transcript: list, listener_transcript: list) -> str:
        """Generate a summary of the entire session for archival purposes.
        
        Args:
            speaker_transcript: List of {"time": str, "text": str} from 倾诉者 (client)
            listener_transcript: List of {"time": str, "text": str} from 倾听者 (counselor)
        
        Returns:
            A comprehensive summary of the session
        """
        if not self.provider:
            return "无法生成总结：AI 服务未配置"

        import json
        
        # Use all transcripts for summary (not limited by context_length)
        context_data = {
            "倾诉者": [{"time": item.get("time", ""), "text": item.get("text", "")} 
                      for item in speaker_transcript],
            "倾听者": [{"time": item.get("time", ""), "text": item.get("text", "")} 
                      for item in listener_transcript],
        }
        
        if not context_data["倾诉者"] and not context_data["倾听者"]:
            return "无对话内容，无法生成总结"
        
        # Summary-specific system prompt
        summary_system_prompt = """你是一位专业的心理咨询记录整理助手。请根据提供的对话内容，生成一份简洁但全面的会话总结。

总结应包含：
1. 来访者（倾诉者）的主要议题和关注点
2. 咨询师（倾听者）的主要干预和回应
3. 会话中的关键时刻或洞察
4. 后续建议（如有）

请使用中文，保持专业、客观的语气。总结长度控制在200-400字。"""

        user_content = f"请总结以下咨询对话：\n```json\n{json.dumps(context_data, ensure_ascii=False, indent=2)}\n```"

        try:
            return self.provider.generate(user_content, system_prompt=summary_system_prompt)
        except Exception as e:
            logger.error(f"Failed to generate session summary: {e}")
            return f"生成总结时出错：{str(e)}"

