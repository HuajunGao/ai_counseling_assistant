"""Test cumulative visitor profile generation."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.llm_engine import SuggestionEngine
from core.session_storage import get_visitor_profile, update_visitor_profile
import json

def test_cumulative_profile():
    # Initialize suggestion engine
    engine = SuggestionEngine()
    
    visitor_id = "test_visitor_cumulative"
    
    # Session 1: Initial conversation
    print("=" * 60)
    print("Session 1: Initial conversation")
    print("=" * 60)
    
    speaker_1 = [
        {"time": "10:00", "text": "我最近工作压力特别大，总是失眠。"},
        {"time": "10:01", "text": "我是一名软件工程师，28岁。"},
    ]
    listener_1 = [
        {"time": "10:00", "text": "能说说具体是什么让你感到压力吗？"},
    ]
    
    # Generate first profile (no previous profile)
    profile_1 = engine.generate_visitor_description(
        speaker_transcript=speaker_1,
        listener_transcript=listener_1,
        previous_profile=None
    )
    
    print("\nGenerated Profile 1:")
    print(json.dumps(profile_1, ensure_ascii=False, indent=2))
    
    # Save profile
    update_visitor_profile(visitor_id, profile_1)
    
    # Session 2: Additional information revealed
    print("\n" + "=" * 60)
    print("Session 2: Additional information revealed")
    print("=" * 60)
    
    speaker_2 = [
        {"time": "15:00", "text": "最近和父母的关系也不太好，他们总是催我结婚。"},
        {"time": "15:02", "text": "我读的是计算机专业，现在在一家互联网公司工作。"},
    ]
    listener_2 = [
        {"time": "15:01", "text": "原生家庭的压力也在影响你吗？"},
    ]
    
    # Load existing profile
    existing_profile = get_visitor_profile(visitor_id)
    print("\nExisting Profile:")
    print(json.dumps(existing_profile, ensure_ascii=False, indent=2))
    
    # Generate updated profile (cumulative)
    profile_2 = engine.generate_visitor_description(
        speaker_transcript=speaker_2,
        listener_transcript=listener_2,
        previous_profile=existing_profile
    )
    
    print("\nGenerated Profile 2 (Cumulative):")
    print(json.dumps(profile_2, ensure_ascii=False, indent=2))
    
    # Save updated profile
    update_visitor_profile(visitor_id, profile_2)
    
    # Final check
    final_profile = get_visitor_profile(visitor_id)
    print("\n" + "=" * 60)
    print("Final Visitor Profile:")
    print("=" * 60)
    print(json.dumps(final_profile, ensure_ascii=False, indent=2))
    
    print("\n✅ Test completed! Check if profile accumulated information from both sessions.")

if __name__ == "__main__":
    test_cumulative_profile()
