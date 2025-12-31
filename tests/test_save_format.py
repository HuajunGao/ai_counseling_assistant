
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath("."))

from core.session_storage import save_session

def create_mock_session():
    visitor_id = "mock_visitor_20251231"
    
    # Sample listener (counselor) transcript with timestamps
    listener_transcript = [
        {"time": "18:10:00", "text": "你好，今天感觉怎么样？", "timestamp": 1735638600.0},
        {"time": "18:10:15", "text": "能跟我多说说这种压力的来源吗？", "timestamp": 1735638615.0}
    ]
    
    # Sample speaker (client) transcript with timestamps
    speaker_transcript = [
        {"time": "18:10:05", "text": "感觉挺压抑的，工作上有很多事情处理不完。", "timestamp": 1735638605.0},
        {"time": "18:10:25", "text": "主要是最近的项目进度有点赶，老板要求很高。", "timestamp": 1735638625.0}
    ]
    
    summary = "来访者感到工作压力大，主要源于项目进度紧和领导的高要求。咨询师引导其深入探讨压力来源。"
    visitor_description = "一位正面临职场压力、寻求压力管理协助的职场人士。"
    
    filepath = save_session(
        visitor_id=visitor_id,
        listener_transcript=listener_transcript,
        speaker_transcript=speaker_transcript,
        summary=summary,
        visitor_description=visitor_description
    )
    
    print(f"Mock session saved to: {filepath}")
    return filepath

if __name__ == "__main__":
    create_mock_session()
