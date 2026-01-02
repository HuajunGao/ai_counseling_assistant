"""Pydantic models for LLM structured outputs."""

from typing import Optional, List
from pydantic import BaseModel, Field


class PersonalInfo(BaseModel):
    """Personal information extracted from counseling sessions."""
    age: Optional[str] = Field(None, description="年龄或年龄段，如 '28岁' 或 '30多岁'")
    gender: Optional[str] = Field(None, description="性别，如 '女' 或 '男'")
    occupation: Optional[str] = Field(None, description="职业，如 '软件工程师'")
    background: str = Field("", description="背景信息的自然语言描述")


class VisitorProfile(BaseModel):
    """Visitor profile generated from counseling sessions."""
    description: str = Field(..., description="来访者的一句话描述（不超过30字）")
    personal_info: PersonalInfo = Field(..., description="个人基本信息")


class ProofreadMessage(BaseModel):
    """A single proofread message in the conversation."""
    role: str = Field(..., description="角色：倾诉者或倾听者")
    text: str = Field(..., description="纠错后的文本内容")
    merged_from: Optional[List[int]] = Field(None, description="如果是合并的消息，记录原始索引")


class ProofreadResult(BaseModel):
    """Result of ASR proofreading containing corrected messages."""
    messages: List[ProofreadMessage] = Field(..., description="纠错后的消息列表")
