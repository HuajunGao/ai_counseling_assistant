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


def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory.
    
    Args:
        prompt_name: Name of the prompt file (without .txt extension)
        
    Returns:
        Content of the prompt file, or a default fallback
    """
    from pathlib import Path
    
    # Get project root (parent of 'core' directory)
    project_root = Path(__file__).parent.parent
    prompt_path = project_root / "prompts" / f"{prompt_name}.txt"
    
    try:
        if prompt_path.exists():
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        else:
            logger.warning(f"Prompt file not found: {prompt_path}")
    except Exception as e:
        logger.error(f"Failed to load prompt '{prompt_name}': {e}")
    
    # Return empty string as fallback
    return ""


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
            "倾诉者": [
                {"time": item.get("time", ""), "text": item.get("text", "")}
                for item in speaker_transcript[-self.context_length :]
            ],
            "倾听者": [
                {"time": item.get("time", ""), "text": item.get("text", "")}
                for item in listener_transcript[-self.context_length :]
            ],
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
            "倾诉者": [{"time": item.get("time", ""), "text": item.get("text", "")} for item in speaker_transcript],
            "倾听者": [{"time": item.get("time", ""), "text": item.get("text", "")} for item in listener_transcript],
        }

        if not context_data["倾诉者"] and not context_data["倾听者"]:
            return "无对话内容，无法生成总结"

        # Load system prompt from file
        summary_system_prompt = load_prompt("session_summary")
        if not summary_system_prompt:
            # Fallback to hardcoded prompt
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

    def generate_visitor_description(self, speaker_transcript: list, listener_transcript: list, previous_profile: dict = None) -> dict:
        """Generate or update visitor profile based on conversation.
        
        Args:
            speaker_transcript: List of {"time": str, "text": str} from 倾诉者 (client)
            listener_transcript: List of {"time": str, "text": str} from 倾听者 (counselor)
            previous_profile: Optional existing profile with 'description' and 'personal_info'
            
        Returns:
            Dictionary with 'description' and 'personal_info' fields
        """
        if not self.provider:
            return {
                "description": "一位来访者",
                "personal_info": {"age": None, "gender": None, "occupation": None, "background": ""}
            }

        import json

        # Prepare conversation context
        context_data = {
            "倾诉者": [{"time": item.get("time", ""), "text": item.get("text", "")} for item in speaker_transcript],
            "倾听者": [{"time": item.get("time", ""), "text": item.get("text", "")} for item in listener_transcript],
        }

        # Load system prompt from file
        system_prompt = load_prompt("visitor_profile")
        if not system_prompt:
            # Fallback
            system_prompt = "你是一位专业的心理咨询记录整理助手。请根据对话内容，为这位来访者生成档案信息。"

        # Construct user message with previous profile if exists
        user_parts = []
        if previous_profile:
            user_parts.append(f"## 过往档案信息\n\n之前的描述：{previous_profile.get('description', '无')}")
            prev_info = previous_profile.get('personal_info', {})
            if prev_info:
                user_parts.append(f"之前的个人信息：\n{json.dumps(prev_info, ensure_ascii=False, indent=2)}")
        
        user_parts.append(f"## 当前会话对话内容\n\n```json\n{json.dumps(context_data, ensure_ascii=False, indent=2)}\n```")
        user_content = "\n\n".join(user_parts)

        try:
            response = self.provider.generate(user_content, system_prompt=system_prompt)
            # Parse JSON response
            profile_data = json.loads(response)
            return profile_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse visitor profile JSON: {e}. Response: {response}")
            return {
                "description": "一位寻求帮助的来访者",
                "personal_info": {"age": None, "gender": None, "occupation": None, "background": ""}
            }
    def proofread_transcript(self, speaker_transcript: list, listener_transcript: list) -> list:
        """Correct ASR errors in transcripts based on context.
        
        Args:
            speaker_transcript: List of {"time": str, "text": str}
            listener_transcript: List of {"time": str, "text": str}
            
        Returns:
            A combined list of chronological messages with corrected text.
        """
        if not self.provider:
            return []

        import json

        # Combine transcripts into a chronological list
        combined = []
        for msg in speaker_transcript:
            combined.append({**msg, "role": "倾诉者", "type": "speaker"})
        for msg in listener_transcript:
            combined.append({**msg, "role": "倾听者", "type": "listener"})
        
        # Sort by time
        combined.sort(key=lambda x: x.get("time", ""))

        if not combined:
            return []

        # Load proofreading system prompt
        proofread_system_prompt = load_prompt("proofread_asr")
        if not proofread_system_prompt:
            proofread_system_prompt = "你是一个专业的心理咨询速记员。请修正以下对话中的 ASR 同音错别字，保持原意不变。返回 JSON 数组。"

        # Prepare messages for LLM
        # We only send the role and text for proofreading to save tokens and avoid confusion
        proofread_input = [{"role": msg["role"], "text": msg["text"]} for msg in combined]
        user_content = f"对话文本 (JSON):\n```json\n{json.dumps(proofread_input, ensure_ascii=False, indent=2)}\n```"

        try:
            response = self.provider.generate(user_content, system_prompt=proofread_system_prompt)
            # Parse JSON response
            corrected_items = json.loads(response)
            
            # Map corrections back to original combined list
            for i, corrected in enumerate(corrected_items):
                if i < len(combined):
                    combined[i]["corrected_text"] = corrected.get("text", combined[i]["text"])
            
            return combined
        except Exception as e:
            logger.error(f"Failed to proofread transcript: {e}")
            # Fallback: return original text as corrected_text
            for item in combined:
                item["corrected_text"] = item["text"]
            return combined
