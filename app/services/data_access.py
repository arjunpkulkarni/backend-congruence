import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sessions")
PATIENTS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "patients.json")


def _resolve_data_root() -> str:
    """Return the absolute path to data/sessions/."""
    return os.path.abspath(DATA_ROOT)


def _load_patients_metadata() -> Dict[str, Dict[str, Any]]:
    """Load patient metadata from patients.json."""
    patients_path = os.path.abspath(PATIENTS_FILE)
    if not os.path.exists(patients_path):
        logger.warning("patients.json not found at %s", patients_path)
        return {}
    
    try:
        with open(patients_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to load patients.json: %s", exc)
        return {}


def _load_patients_from_db() -> Dict[str, Dict[str, Any]]:    
    from app.services.database import get_conversation_db
    
    db = get_conversation_db()
    
    if not db.is_enabled():
        logger.debug("Database not enabled, using patients.json")
        return _load_patients_metadata()
    
    try:        
        response = db.client.table("patients").select("*").execute()
        
        if not response.data:
            logger.warning("No patients in database, falling back to patients.json")
            return _load_patients_metadata()
        
        patients_dict = {}
        for patient in response.data:
            patient_id = str(patient["id"])
            patients_dict[patient_id] = {
                "name": patient.get("name", "Unknown"),
                "dob": patient.get("date_of_birth"),
                "contact_email": patient.get("contact_email"),
                "contact_phone": patient.get("contact_phone"),
                "therapist_id": patient.get("therapist_id"),
                "clinic_id": patient.get("clinic_id"),
                "created_at": patient.get("created_at"),
                "updated_at": patient.get("updated_at"),
            }
        
        logger.info(f"Loaded {len(patients_dict)} patients from Supabase")
        return patients_dict
        
    except Exception as exc:
        logger.error(f"Failed to load patients from database: {exc}, using patients.json")
        return _load_patients_metadata()


def _read_json(path: str) -> Optional[Dict[str, Any] | List[Any]]:
    """Safely read and parse a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.debug("File not found: %s", path)
        return None
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON in %s: %s", path, exc)
        return None


def _read_text(path: str) -> Optional[str]:
    """Safely read a text file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.debug("File not found: %s", path)
        return None


def _ts_to_iso(ts: int) -> str:
    """Convert a unix timestamp to ISO-8601 date string."""
    try:
        return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(ts)


def list_patients() -> List[Dict[str, Any]]:
    """
    List all patients from database, including those with and without sessions.

    Returns a list of dicts:
        [{"patient_id": "...", "name": "...", "session_count": N, "latest_session": <ts>}, ...]
    """
    root = _resolve_data_root()
    patients_dict: Dict[str, Dict[str, Any]] = {}
    patients_metadata = _load_patients_from_db()  # Load from database

    # First, add all patients from database (even without sessions)
    for patient_id, metadata in patients_metadata.items():
        patients_dict[patient_id] = {
            "patient_id": patient_id,
            "name": metadata.get("name", patient_id),
            "mrn": None,  # MRN not in database
            "session_count": 0,
            "latest_session": None,
            "latest_session_date": None,
        }

    # Then, update with session data for patients who have sessions on disk
    if os.path.isdir(root):
        for entry in sorted(os.listdir(root)):
            patient_dir = os.path.join(root, entry)
            if not os.path.isdir(patient_dir):
                continue

            session_timestamps = _list_session_timestamps(patient_dir)
            if not session_timestamps:
                continue

            # Update or add patient with session info
            metadata = patients_metadata.get(entry, {})
            patients_dict[entry] = {
                "patient_id": entry,
                "name": metadata.get("name", entry),
                "mrn": None,
                "session_count": len(session_timestamps),
                "latest_session": max(session_timestamps),
                "latest_session_date": _ts_to_iso(max(session_timestamps)),
            }

    return list(patients_dict.values())


def _list_session_timestamps(patient_dir: str) -> List[int]:
    """Return sorted list of session timestamps for a patient directory."""
    timestamps: List[int] = []
    for name in os.listdir(patient_dir):
        full_path = os.path.join(patient_dir, name)
        if os.path.isdir(full_path):
            try:
                timestamps.append(int(name))
            except ValueError:
                continue
    return sorted(timestamps)

def list_sessions(patient_id: str) -> List[Dict[str, Any]]:    
    from app.services.database import get_conversation_db
    
    sessions: List[Dict[str, Any]] = []
    
    db = get_conversation_db()
    if db.is_enabled():
        try:
            response = db.client.table("session_videos")\
                .select("*, session_analysis(*)")\
                .eq("patient_id", patient_id)\
                .order("created_at", desc=True)\
                .execute()
            
            if response.data:
                logger.info(f"Found {len(response.data)} sessions in Supabase for patient {patient_id}")
                
                for video in response.data:
                    analysis_list = video.get("session_analysis", [])
                    analysis_data = analysis_list[0] if analysis_list else {}
                    
                    sessions.append({
                        "session_id": video["id"],
                        "session_date": video.get("created_at"),
                        "patient_id": patient_id,
                        "title": video.get("title"),
                        "has_summary": bool(analysis_data.get("summary")),
                        "has_notes": analysis_data.get("key_moments") is not None,
                        "has_transcript": video.get("status") == "analyzed",
                        "duration": video.get("duration_seconds"),
                        "overall_congruence": analysis_data.get("avg_tecs"),
                        "status": video.get("status"),
                        "video_path": video.get("video_path"),
                    })
                
                # If we found sessions in database, return them
                if sessions:
                    return sessions
                    
        except Exception as e:
            logger.warning(f"Failed to load sessions from Supabase for {patient_id}: {e}")
            # Continue to filesystem fallback
    
    # Fallback to filesystem (for legacy data or local processing)
    root = _resolve_data_root()
    patient_dir = os.path.join(root, patient_id)

    if not os.path.isdir(patient_dir):
        logger.debug("Patient directory not found on filesystem: %s", patient_dir)
        return []

    timestamps = _list_session_timestamps(patient_dir)
    
    for ts in reversed(timestamps):
        session_dir = os.path.join(patient_dir, str(ts))
        outputs_dir = os.path.join(session_dir, "outputs")

        # Read session summary for quick metadata
        summary = _read_json(os.path.join(outputs_dir, "session_summary.json"))
        has_notes = os.path.isfile(os.path.join(outputs_dir, "therapist_notes.json"))
        has_transcript = os.path.isfile(os.path.join(outputs_dir, "transcript.txt"))

        sessions.append({
            "session_id": ts,
            "session_date": _ts_to_iso(ts),
            "patient_id": patient_id,
            "has_summary": summary is not None,
            "has_notes": has_notes,
            "has_transcript": has_transcript,
            "duration": summary.get("duration") if summary else None,
            "overall_congruence": summary.get("overall_congruence") if summary else None,
        })

    return sessions


def get_session_summary(patient_id: str, session_id: int) -> Optional[Dict[str, Any]]:
    """
    Read the session summary for a specific session.
    Tries Supabase first, then falls back to filesystem.

    Contains: overall_congruence, incongruent_moments, emotion_distribution, metrics.
    """
    from app.services.database import get_conversation_db
    
    # Try Supabase first
    db = get_conversation_db()
    if db.is_enabled():
        try:
            # session_id is now a UUID for Supabase sessions
            response = db.client.table("session_analysis")\
                .select("*")\
                .eq("session_video_id", str(session_id))\
                .single()\
                .execute()
            
            if response.data:
                data = response.data
                return {
                    "patient_id": patient_id,
                    "session_id": session_id,
                    "session_date": data.get("created_at"),
                    "overall_congruence": data.get("avg_tecs"),
                    "duration": data.get("duration_seconds"),
                    "summary": data.get("summary"),
                    "emotion_distribution": data.get("emotion_timeline"),
                    "incongruent_moments": data.get("key_moments", []),
                    "metrics": {
                        "avg_tecs": data.get("avg_tecs"),
                        "congruence_data": data.get("congruence_data"),
                    }
                }
        except Exception as e:
            logger.warning(f"Failed to load session summary from Supabase: {e}")
    
    # Fallback to filesystem
    root = _resolve_data_root()
    path = os.path.join(root, patient_id, str(session_id), "outputs", "session_summary.json")
    data = _read_json(path)
    if data and isinstance(data, dict):
        data["session_date"] = _ts_to_iso(int(session_id)) if isinstance(session_id, int) else data.get("session_date")
    return data


def get_session_transcript(
    patient_id: str,
    session_id: int,
    include_segments: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Read the transcript (full text + optional timed segments) for a session.

    Returns:
        {
            "text": "full transcript text...",
            "segments": [{"start": 1.07, "end": 4.51, "text": "..."}],
            "segment_count": 6
        }
    """
    root = _resolve_data_root()
    outputs_dir = os.path.join(root, patient_id, str(session_id), "outputs")

    text = _read_text(os.path.join(outputs_dir, "transcript.txt"))
    if text is None:
        return None

    result: Dict[str, Any] = {
        "patient_id": patient_id,
        "session_id": session_id,
        "text": text,
    }

    if include_segments:
        segments = _read_json(os.path.join(outputs_dir, "transcript_segments.json"))
        result["segments"] = segments or []
        result["segment_count"] = len(segments) if segments else 0

    return result


def get_therapist_notes(patient_id: str, session_id: int) -> Optional[Dict[str, Any]]:
    """
    Read the structured therapist notes for a session.
    Tries Supabase first, then falls back to filesystem.

    Contains: session_overview, key_themes, emotional_analysis,
              clinical_observations, risk_assessment, recommendations.
    """
    from app.services.database import get_conversation_db
    
    # Try Supabase first
    db = get_conversation_db()
    if db.is_enabled():
        try:
            # Get notes from session_notes table
            response = db.client.table("session_notes")\
                .select("*")\
                .eq("session_video_id", str(session_id))\
                .execute()
            
            if response.data:
                # Combine all notes for this session
                notes_data = {
                    "patient_id": patient_id,
                    "session_id": session_id,
                    "session_overview": {},
                    "key_themes": [],
                    "notes": []
                }
                
                for note in response.data:
                    notes_data["notes"].append({
                        "content": note.get("content"),
                        "created_at": note.get("created_at"),
                        "file_path": note.get("file_path"),
                    })
                
                # Also get analysis data which may contain structured notes
                analysis = db.client.table("session_analysis")\
                    .select("summary, key_moments")\
                    .eq("session_video_id", str(session_id))\
                    .single()\
                    .execute()
                
                if analysis.data:
                    notes_data["session_overview"]["summary"] = analysis.data.get("summary")
                    notes_data["key_themes"] = analysis.data.get("key_moments", [])
                
                return notes_data
                
        except Exception as e:
            logger.warning(f"Failed to load therapist notes from Supabase: {e}")
    
    # Fallback to filesystem
    root = _resolve_data_root()
    path = os.path.join(root, patient_id, str(session_id), "outputs", "therapist_notes.json")
    data = _read_json(path)
    if data and isinstance(data, dict):
        data["patient_id"] = patient_id
        data["session_id"] = session_id
    return data


def get_simplified_notes(patient_id: str, session_id: int) -> Optional[str]:
    """Read the simplified clinical notes (markdown) for a session."""
    root = _resolve_data_root()
    path = os.path.join(root, patient_id, str(session_id), "outputs", "simplified_notes.md")
    return _read_text(path)


def get_congruence_timeline(
    patient_id: str,
    session_id: int,
    resolution: str = "1hz",
) -> Optional[List[Dict[str, Any]]]:
    """
    Read the congruence timeline for a session.

    Args:
        resolution: "1hz" for 1-second merged timeline, "10hz" for full detail.
    """
    root = _resolve_data_root()
    outputs_dir = os.path.join(root, patient_id, str(session_id), "outputs")

    if resolution == "10hz":
        filename = "timeline.json"
    else:
        filename = "timeline_1hz.json"

    data = _read_json(os.path.join(outputs_dir, filename))
    return data if isinstance(data, list) else None


def get_spikes(patient_id: str, session_id: int) -> Optional[List[Dict[str, Any]]]:
    """Read detected micro-spikes for a session."""
    root = _resolve_data_root()
    path = os.path.join(root, patient_id, str(session_id), "outputs", "spikes.json")
    data = _read_json(path)
    return data if isinstance(data, list) else None


def get_intensity_timeline(patient_id: str, session_id: int) -> Optional[List[Dict[str, Any]]]:
    """Read the intensity timeline from simplified analysis."""
    root = _resolve_data_root()
    path = os.path.join(root, patient_id, str(session_id), "outputs", "intensity_timeline.json")
    data = _read_json(path)
    return data if isinstance(data, list) else None


def get_incongruence_markers(patient_id: str, session_id: int) -> Optional[List[Dict[str, Any]]]:
    """Read incongruence markers from simplified analysis."""
    root = _resolve_data_root()
    path = os.path.join(root, patient_id, str(session_id), "outputs", "incongruence_markers.json")
    data = _read_json(path)
    return data if isinstance(data, list) else None


# ---------------------------------------------------------------------------
# Cross-session / analytics queries
# ---------------------------------------------------------------------------

def get_patient_history(patient_id: str, limit: int = 10) -> Dict[str, Any]:
    """
    Aggregate a patient's session history for the agent to reason over.

    Returns a compact summary with congruence trend, session dates, and
    the latest clinical notes.
    """
    sessions = list_sessions(patient_id)
    if not sessions:
        return {"patient_id": patient_id, "sessions": [], "message": "No sessions found"}

    # Limit to most recent N sessions
    sessions = sessions[:limit]

    # Build congruence trend
    congruence_trend = []
    for s in sessions:
        if s.get("overall_congruence") is not None:
            congruence_trend.append({
                "session_id": s["session_id"],
                "date": s["session_date"],
                "congruence": s["overall_congruence"],
                "duration": s.get("duration"),
            })

    # Get latest notes
    latest_notes = None
    for s in sessions:
        if s.get("has_notes"):
            latest_notes = get_therapist_notes(patient_id, s["session_id"])
            if latest_notes:
                break

    return {
        "patient_id": patient_id,
        "total_sessions": len(list_sessions(patient_id)),
        "sessions_returned": len(sessions),
        "sessions": sessions,
        "congruence_trend": congruence_trend,
        "latest_notes_summary": _summarize_notes(latest_notes) if latest_notes else None,
    }


def get_practice_analytics_data() -> Dict[str, Any]:
    """
    Compute practice-wide analytics across all patients and sessions.

    Returns aggregate stats: total patients, total sessions,
    average congruence, recent activity, etc.
    """
    patients = list_patients()

    total_sessions = sum(p["session_count"] for p in patients)
    all_congruence_scores: List[float] = []

    # Sample recent sessions for aggregate stats (last 20 sessions across all patients)
    recent_sessions: List[Dict[str, Any]] = []
    for patient in patients:
        sessions = list_sessions(patient["patient_id"])
        for s in sessions[:5]:  # Up to 5 per patient
            recent_sessions.append(s)
            if s.get("overall_congruence") is not None:
                all_congruence_scores.append(s["overall_congruence"])

    # Sort all recent sessions by session_id (timestamp) descending
    recent_sessions.sort(key=lambda x: x.get("session_id", 0), reverse=True)
    recent_sessions = recent_sessions[:20]

    avg_congruence = (
        sum(all_congruence_scores) / len(all_congruence_scores)
        if all_congruence_scores
        else None
    )

    return {
        "total_patients": len(patients),
        "total_sessions": total_sessions,
        "average_congruence": round(avg_congruence, 4) if avg_congruence else None,
        "patients": patients,
        "recent_sessions": recent_sessions,
    }

def _summarize_notes(notes: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a compact summary from full therapist notes."""
    overview = notes.get("session_overview", {})
    themes = notes.get("key_themes", [])
    risk = notes.get("risk_assessment", {})
    recommendations = notes.get("recommendations", {})

    return {
        "summary": overview.get("summary", ""),
        "engagement_level": overview.get("engagement_level", ""),
        "overall_tone": overview.get("overall_tone", ""),
        "theme_count": len(themes),
        "themes": [t.get("theme", "") for t in themes[:5]],
        "has_risk_flags": bool(
            risk.get("suicide_self_harm", {}).get("indicators", "").lower()
            not in ("none", "no", "", "unclear", "unknown")
        ),
        "follow_up_count": len(recommendations.get("follow_up_actions", [])),
    }


def find_latest_session(patient_id: str) -> Optional[int]:
    """Return the latest session timestamp for a patient, or None."""
    sessions = list_sessions(patient_id)
    return sessions[0]["session_id"] if sessions else None


def resolve_session(patient_id: str, session_id: Optional[int] = None) -> Optional[int]:
    """
    Resolve a session_id: if None, default to the latest session.
    Returns the session_id or None if the patient has no sessions.
    """
    if session_id is not None:
        return session_id
    return find_latest_session(patient_id)


def find_patient_by_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Find a patient by name (case-insensitive partial match).
    
    Returns patient info including patient_id, or None if not found.
    """
    patients_metadata = _load_patients_from_db()  # Changed to use database
    name_lower = name.lower().strip()
    
    matches: List[Dict[str, Any]] = []
    
    for patient_id, info in patients_metadata.items():
        patient_name = info.get("name", "").lower()
        
        # Match on name only (no MRN in database)
        if name_lower in patient_name or patient_name in name_lower:
            matches.append({
                "patient_id": patient_id,
                **info
            })
    
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        # Return all matches for disambiguation
        return {
            "multiple_matches": True,
            "matches": matches,
            "count": len(matches)
        }
    
    return None


def search_patients(query: str) -> List[Dict[str, Any]]:
    """
    Search for patients by name or patient_id.
    
    Returns a list of matching patients with their metadata.
    """
    patients_metadata = _load_patients_from_db()  
    query_lower = query.lower().strip()
    
    matches: List[Dict[str, Any]] = []
    
    for patient_id, info in patients_metadata.items():
        patient_name = info.get("name", "").lower()
        pid_lower = patient_id.lower()
        
        # Match on name or patient_id (no MRN in database)
        if query_lower in patient_name or query_lower in pid_lower:
            matches.append({
                "patient_id": patient_id,
                **info
            })
    
    return matches
