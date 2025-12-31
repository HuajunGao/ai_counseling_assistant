"""
Session storage module for saving and managing counseling session records.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_sessions_dir() -> Path:
    """Get the sessions directory path (relative to project root)."""
    # Get the project root (parent of 'core' directory)
    project_root = Path(__file__).parent.parent
    sessions_dir = project_root / "sessions"
    return sessions_dir


def generate_default_visitor_id() -> str:
    """Generate default visitor ID in yyyymmddHHMMSS format."""
    return datetime.now().strftime("%Y%m%d%H%M%S")


def get_visitor_ids() -> list:
    """Get list of existing visitor IDs (folder names in sessions directory)."""
    sessions_dir = get_sessions_dir()
    if not sessions_dir.exists():
        return []

    visitor_ids = []
    for item in sessions_dir.iterdir():
        if item.is_dir():
            visitor_ids.append(item.name)

    return sorted(visitor_ids)


def get_sessions_for_visitor(visitor_id: str) -> list:
    """Get all session files for a given visitor ID."""
    visitor_dir = get_sessions_dir() / visitor_id
    if not visitor_dir.exists():
        return []

    sessions = []
    for item in visitor_dir.iterdir():
        if item.is_file() and item.suffix == ".json" and item.name != "visitor_profile.json":
            sessions.append(item.name)

    # Sort by modification time to ensure chronological order
    sessions.sort(key=lambda x: (visitor_dir / x).stat().st_mtime)
    return sessions


def save_session(
    visitor_id: str,
    listener_transcript: list,
    speaker_transcript: list,
    summary: str,
    visitor_description: Optional[dict] = None,
    private_notes: Optional[str] = None,
    start_time: Optional[datetime] = None,
) -> str:
    """
    Save a counseling session to disk.

    Args:
        visitor_id: The visitor/client ID (folder name)
        listener_transcript: List of {"time": str, "text": str} from 倾听者 (counselor)
        speaker_transcript: List of {"time": str, "text": str} from 倾诉者 (client)
        summary: AI-generated session summary
        visitor_description: Optional dictionary containing description and personal_info
        private_notes: Optional counselor's private notes
        start_time: Optional session start time (defaults to now)

    Returns:
        Path to the saved session file
    """
    sessions_dir = get_sessions_dir()
    visitor_dir = sessions_dir / visitor_id

    # Create directories if they don't exist
    visitor_dir.mkdir(parents=True, exist_ok=True)

    # Generate session file name with timestamp
    now = datetime.now()
    if start_time is None:
        start_time = now

    session_id = now.strftime("%Y%m%d_%H%M%S")
    filename = now.strftime("%Y-%m-%d_%H-%M") + ".json"
    filepath = visitor_dir / filename

    # If file already exists (same minute), add seconds to make unique
    if filepath.exists():
        filename = now.strftime("%Y-%m-%d_%H-%M-%S") + ".json"
        filepath = visitor_dir / filename

    # Merge transcripts into a chronological dialogue
    dialogue = []
    for msg in listener_transcript:
        new_msg = msg.copy()
        new_msg["role"] = "倾听者"
        dialogue.append(new_msg)
    
    for msg in speaker_transcript:
        new_msg = msg.copy()
        new_msg["role"] = "倾诉者"
        dialogue.append(new_msg)
    
    # Sort by timestamp
    dialogue.sort(key=lambda x: x.get("timestamp", 0))

    # Prepare session data
    session_data = {
        "session_id": session_id,
        "visitor_id": visitor_id,
        "timestamp": now.isoformat(),
        "conversation": {
            "dialogue": dialogue,  # New chronological format
            "listener": listener_transcript, 
            "speaker": speaker_transcript
        },
        "summary": summary,
        "private_notes": private_notes,
    }

    # Write to file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)

    # Update visitor profile if description is provided
    if visitor_description:
        update_visitor_profile(visitor_id, visitor_description)

    return str(filepath)


def get_visitor_profile(visitor_id: str) -> dict:
    """Get visitor profile data."""
    profile_path = get_sessions_dir() / visitor_id / "visitor_profile.json"
    if profile_path.exists():
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                profile = json.load(f)
                # Ensure all required fields exist with defaults
                if "personal_info" not in profile:
                    profile["personal_info"] = {"age": None, "gender": None, "occupation": None, "background": ""}
                if "last_updated" not in profile:
                    profile["last_updated"] = None
                if "session_count" not in profile:
                    profile["session_count"] = 0
                return profile
        except Exception:
            pass
    
    # Default profile
    return {
        "description": "一位寻求帮助的来访者",
        "personal_info": {
            "age": None,
            "gender": None,
            "occupation": None,
            "background": ""
        },
        "last_updated": None,
        "session_count": 0
    }


def update_visitor_profile(visitor_id: str, data: dict):
    """Update visitor profile data with cumulative merging.
    
    Args:
        data: Dictionary that may contain 'description' and 'personal_info'
    """
    from datetime import datetime
    
    visitor_dir = get_sessions_dir() / visitor_id
    visitor_dir.mkdir(parents=True, exist_ok=True)
    profile_path = visitor_dir / "visitor_profile.json"
    
    # Load existing profile
    profile = get_visitor_profile(visitor_id)
    
    # Update description if provided
    if "description" in data:
        profile["description"] = data["description"]
    
    # Merge personal_info if provided
    if "personal_info" in data:
        new_info = data["personal_info"]
        existing_info = profile.get("personal_info", {})
        
        # Merge each field (only update if new value is not None/empty)
        for key in ["age", "gender", "occupation", "background"]:
            new_value = new_info.get(key)
            if new_value:
                # For background, append rather than replace
                if key == "background" and existing_info.get("background"):
                    # Check if info is already included
                    if new_value not in existing_info["background"]:
                        existing_info["background"] = f"{existing_info['background']}; {new_value}"
                    else:
                        existing_info["background"] = existing_info["background"]
                else:
                    existing_info[key] = new_value
        
        profile["personal_info"] = existing_info
    
    # Update metadata
    profile["last_updated"] = datetime.now().isoformat()
    profile["session_count"] = profile.get("session_count", 0) + 1
    
    # Save
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


def save_visitor_profile(visitor_id: str, profile_data: dict):
    """Save visitor profile data from manual edits.
    
    Args:
        visitor_id: The visitor ID
        profile_data: Dictionary containing 'description' and 'personal_info'
    """
    from datetime import datetime
    
    visitor_dir = get_sessions_dir() / visitor_id
    visitor_dir.mkdir(parents=True, exist_ok=True)
    profile_path = visitor_dir / "visitor_profile.json"
    
    # Load existing to preserve fields like session_count
    profile = get_visitor_profile(visitor_id)
    
    # Overwrite with new data
    if "description" in profile_data:
        profile["description"] = profile_data["description"]
    
    if "personal_info" in profile_data:
        profile["personal_info"] = profile_data["personal_info"]
    
    # Update timestamp
    profile["last_updated"] = datetime.now().isoformat()
    # Note: We do NOT increment session_count here as this is a manual edit
    
    # Save
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


def load_session(filepath: str) -> dict:
    """Load a session file from disk."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
