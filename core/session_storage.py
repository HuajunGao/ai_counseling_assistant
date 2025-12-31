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
        if item.is_file() and item.suffix == ".json":
            sessions.append(item.name)

    return sorted(sessions)


def save_session(
    visitor_id: str,
    listener_transcript: list,
    speaker_transcript: list,
    summary: str,
    start_time: Optional[datetime] = None,
) -> str:
    """
    Save a counseling session to disk.

    Args:
        visitor_id: The visitor/client ID (folder name)
        listener_transcript: List of {"time": str, "text": str} from 倾听者 (counselor)
        speaker_transcript: List of {"time": str, "text": str} from 倾诉者 (client)
        summary: AI-generated session summary
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
    }

    # Write to file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)

    return str(filepath)


def load_session(filepath: str) -> dict:
    """Load a session file from disk."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
