import os
import uuid
import threading
from typing import Tuple


def get_workspace_root() -> str:
    # Default to current working directory if not running inside the given workspace path
    return os.getcwd()


# Thread-safe counter for additional uniqueness
_session_counter = 0
_counter_lock = threading.Lock()


def _get_unique_session_id(session_ts: int) -> str:
    """Generate a unique session ID that prevents conflicts in concurrent requests."""
    global _session_counter
    with _counter_lock:
        _session_counter += 1
        # Combine timestamp, counter, and short UUID for guaranteed uniqueness
        unique_id = f"{session_ts}_{_session_counter}_{uuid.uuid4().hex[:8]}"
        return unique_id


def create_session_directories(
    workspace_root: str,
    patient_id: str,
    session_ts: int,
) -> Tuple[str, str, str, str]:
    """
    Creates unique session directories that won't conflict between concurrent requests:
      {workspace_root}/data/sessions/{patient_id}/{unique_session_id}/
        - media/
        - frames/
        - outputs/
    """
    unique_session_id = _get_unique_session_id(session_ts)
    session_dir = os.path.join(
        workspace_root, "data", "sessions", patient_id, unique_session_id
    )
    media_dir = os.path.join(session_dir, "media")
    frames_dir = os.path.join(session_dir, "frames")
    outputs_dir = os.path.join(session_dir, "outputs")
    
    # Create directories with proper error handling for concurrent access
    try:
        os.makedirs(media_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(outputs_dir, exist_ok=True)
    except OSError as e:
        # If directory creation fails, try with a different UUID
        if "File exists" in str(e):
            unique_session_id = _get_unique_session_id(session_ts)
            session_dir = os.path.join(
                workspace_root, "data", "sessions", patient_id, unique_session_id
            )
            media_dir = os.path.join(session_dir, "media")
            frames_dir = os.path.join(session_dir, "frames")
            outputs_dir = os.path.join(session_dir, "outputs")
            os.makedirs(media_dir, exist_ok=True)
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(outputs_dir, exist_ok=True)
        else:
            raise
    
    return session_dir, media_dir, frames_dir, outputs_dir

