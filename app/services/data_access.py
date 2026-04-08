"""
Data Access Layer for Congruence Ops Agent.

All data is stored in and retrieved from Supabase.
No filesystem fallbacks - this is a cloud-first architecture.

Tables used:
- patients: Patient demographic and contact info
- session_videos: Video metadata, transcripts, status
- session_analysis: Congruence scores, summaries, key moments
- session_notes: Clinical notes (structured and simplified)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def _load_patients_from_db() -> Dict[str, Dict[str, Any]]:    
    """Load all patients from Supabase only."""
    from app.services.database import get_conversation_db
    
    db = get_conversation_db()
    
    if not db.is_enabled():
        logger.error("Supabase not enabled - cannot load patients")
        return {}
    
    try:        
        response = db.client.table("patients").select("*").execute()
        
        if not response.data:
            logger.warning("No patients in Supabase database")
            return {}
        
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
        logger.error(f"Failed to load patients from Supabase: {exc}")
        return {}


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
    List all patients from Supabase database, including session counts.

    Returns a list of dicts:
        [{"patient_id": "...", "name": "...", "session_count": N, "latest_session": <uuid>}, ...]
    """
    from app.services.database import get_conversation_db
    
    db = get_conversation_db()
    if not db.is_enabled():
        logger.error("Supabase not enabled - cannot list patients")
        return []
    
    try:
        # Get all patients
        patients_response = db.client.table("patients").select("*").execute()
        
        if not patients_response.data:
            logger.warning("No patients found in Supabase")
            return []
        
        patients_list = []
        
        for patient in patients_response.data:
            patient_id = str(patient["id"])
            
            # Get session count for this patient
            sessions_response = db.client.table("session_videos")\
                .select("id, created_at", count="exact")\
                .eq("patient_id", patient_id)\
                .order("created_at", desc=True)\
                .execute()
            
            session_count = sessions_response.count or 0
            latest_session = None
            latest_session_date = None
            
            if sessions_response.data and len(sessions_response.data) > 0:
                latest_session = sessions_response.data[0]["id"]
                latest_session_date = sessions_response.data[0]["created_at"]
            
            patients_list.append({
                "patient_id": patient_id,
                "name": patient.get("name", patient_id),
                "mrn": None,
                "session_count": session_count,
                "latest_session": latest_session,
                "latest_session_date": latest_session_date,
            })
        
        return patients_list
        
    except Exception as e:
        logger.error(f"Failed to list patients from Supabase: {e}")
        return []


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
    """List all sessions for a patient from Supabase."""
    from app.services.database import get_conversation_db
    
    db = get_conversation_db()
    if not db.is_enabled():
        logger.error("Supabase not enabled - cannot list sessions")
        return []
    
    try:
        response = db.client.table("session_videos")\
            .select("*, session_analysis(*)")\
            .eq("patient_id", patient_id)\
            .order("created_at", desc=True)\
            .execute()
        
        if not response.data:
            logger.info(f"No sessions found in Supabase for patient {patient_id}")
            return []
        
        logger.info(f"Found {len(response.data)} sessions in Supabase for patient {patient_id}")
        
        sessions = []
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
        
        return sessions
                
    except Exception as e:
        logger.error(f"Failed to load sessions from Supabase for {patient_id}: {e}")
        return []


def get_session_summary(patient_id: str, session_id: Union[int, str]) -> Optional[Dict[str, Any]]:
    """
    Read the session summary for a specific session from Supabase.

    Contains: overall_congruence, incongruent_moments, emotion_distribution, metrics.
    """
    from app.services.database import get_conversation_db
    
    db = get_conversation_db()
    if not db.is_enabled():
        logger.error("Supabase not enabled - cannot get session summary")
        return None
    
    try:
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
        
        logger.warning(f"No session summary found in Supabase for session {session_id}")
        return None
        
    except Exception as e:
        logger.error(f"Failed to load session summary from Supabase: {e}")
        return None


def get_session_transcript(
    patient_id: str,
    session_id: Union[int, str],
    include_segments: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Read the transcript (full text + optional timed segments) for a session from Supabase.

    Returns:
        {
            "text": "full transcript text...",
            "segments": [{"start": 1.07, "end": 4.51, "text": "..."}],
            "segment_count": 6
        }
    """
    from app.services.database import get_conversation_db
    
    db = get_conversation_db()
    if not db.is_enabled():
        logger.error("Supabase not enabled - cannot get transcript")
        return None
    
    try:
        # Get transcript from session_videos table
        response = db.client.table("session_videos")\
            .select("transcript_text, transcript_segments")\
            .eq("id", str(session_id))\
            .single()\
            .execute()
        
        if not response.data:
            logger.warning(f"No transcript found in Supabase for session {session_id}")
            return None
        
        if not response.data.get("transcript_text"):
            logger.warning(f"Transcript text is empty for session {session_id}")
            return None
        
        result = {
            "patient_id": patient_id,
            "session_id": session_id,
            "text": response.data.get("transcript_text"),
        }
        
        if include_segments and response.data.get("transcript_segments"):
            segments = response.data.get("transcript_segments", [])
            result["segments"] = segments
            result["segment_count"] = len(segments)
        
        return result
            
    except Exception as e:
        logger.error(f"Failed to load transcript from Supabase: {e}")
        return None


def get_therapist_notes(patient_id: str, session_id: Union[int, str]) -> Optional[Dict[str, Any]]:
    """
    Read the structured therapist notes for a session from Supabase.

    Contains: session_overview, key_themes, emotional_analysis,
              clinical_observations, risk_assessment, recommendations.
    """
    from app.services.database import get_conversation_db
    
    db = get_conversation_db()
    if not db.is_enabled():
        logger.error("Supabase not enabled - cannot get therapist notes")
        return None
    
    try:
        # Get notes from session_notes table
        response = db.client.table("session_notes")\
            .select("*")\
            .eq("session_video_id", str(session_id))\
            .execute()
        
        # Combine all notes for this session
        notes_data = {
            "patient_id": patient_id,
            "session_id": session_id,
            "session_overview": {},
            "key_themes": [],
            "notes": []
        }
        
        if response.data:
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
        
        # Return notes even if empty (to distinguish from error)
        return notes_data if (notes_data["notes"] or notes_data["key_themes"]) else None
            
    except Exception as e:
        logger.error(f"Failed to load therapist notes from Supabase: {e}")
        return None


def get_simplified_notes(patient_id: str, session_id: Union[int, str]) -> Optional[str]:
    """
    Read the simplified clinical notes (markdown) for a session from Supabase.
    """
    from app.services.database import get_conversation_db
    
    db = get_conversation_db()
    if not db.is_enabled():
        logger.error("Supabase not enabled - cannot get simplified notes")
        return None
    
    try:
        response = db.client.table("session_notes")\
            .select("content")\
            .eq("session_video_id", str(session_id))\
            .eq("note_type", "simplified")\
            .execute()
        
        if response.data and len(response.data) > 0:
            # Return the most recent simplified note
            return response.data[0].get("content")
        
        logger.warning(f"No simplified notes found in Supabase for session {session_id}")
        return None
            
    except Exception as e:
        logger.error(f"Failed to load simplified notes from Supabase: {e}")
        return None


def get_congruence_timeline(
    patient_id: str,
    session_id: Union[int, str],
    resolution: str = "1hz",
) -> Optional[List[Dict[str, Any]]]:
    """
    Read the congruence timeline for a session from Supabase.

    Args:
        resolution: "1hz" for 1-second merged timeline, "10hz" for full detail.
    """
    from app.services.database import get_conversation_db
    
    db = get_conversation_db()
    if not db.is_enabled():
        logger.error("Supabase not enabled - cannot get congruence timeline")
        return None
    
    try:
        # Get congruence data from session_analysis table
        response = db.client.table("session_analysis")\
            .select("congruence_data")\
            .eq("session_video_id", str(session_id))\
            .single()\
            .execute()
        
        if response.data and response.data.get("congruence_data"):
            data = response.data.get("congruence_data")
            return data if isinstance(data, list) else None
        
        logger.warning(f"No congruence timeline found in Supabase for session {session_id}")
        return None
        
    except Exception as e:
        logger.error(f"Failed to load congruence timeline from Supabase: {e}")
        return None


def get_spikes(patient_id: str, session_id: Union[int, str]) -> Optional[List[Dict[str, Any]]]:
    """Read detected micro-spikes for a session from Supabase."""
    from app.services.database import get_conversation_db
    
    db = get_conversation_db()
    if not db.is_enabled():
        logger.error("Supabase not enabled - cannot get spikes")
        return None
    
    try:
        # Spikes are stored in key_moments in session_analysis
        response = db.client.table("session_analysis")\
            .select("key_moments")\
            .eq("session_video_id", str(session_id))\
            .single()\
            .execute()
        
        if response.data and response.data.get("key_moments"):
            data = response.data.get("key_moments")
            return data if isinstance(data, list) else None
        
        logger.warning(f"No spikes found in Supabase for session {session_id}")
        return None
        
    except Exception as e:
        logger.error(f"Failed to load spikes from Supabase: {e}")
        return None


def get_intensity_timeline(patient_id: str, session_id: Union[int, str]) -> Optional[List[Dict[str, Any]]]:
    """Read the intensity timeline from simplified analysis from Supabase."""
    from app.services.database import get_conversation_db
    
    db = get_conversation_db()
    if not db.is_enabled():
        logger.error("Supabase not enabled - cannot get intensity timeline")
        return None
    
    try:
        # Intensity timeline is stored in emotion_timeline in session_analysis
        response = db.client.table("session_analysis")\
            .select("emotion_timeline")\
            .eq("session_video_id", str(session_id))\
            .single()\
            .execute()
        
        if response.data and response.data.get("emotion_timeline"):
            data = response.data.get("emotion_timeline")
            return data if isinstance(data, list) else None
        
        logger.warning(f"No intensity timeline found in Supabase for session {session_id}")
        return None
        
    except Exception as e:
        logger.error(f"Failed to load intensity timeline from Supabase: {e}")
        return None


def get_incongruence_markers(patient_id: str, session_id: Union[int, str]) -> Optional[List[Dict[str, Any]]]:
    """Read incongruence markers from simplified analysis from Supabase."""
    from app.services.database import get_conversation_db
    
    db = get_conversation_db()
    if not db.is_enabled():
        logger.error("Supabase not enabled - cannot get incongruence markers")
        return None
    
    try:
        # Incongruence markers are stored in key_moments in session_analysis
        response = db.client.table("session_analysis")\
            .select("key_moments")\
            .eq("session_video_id", str(session_id))\
            .single()\
            .execute()
        
        if response.data and response.data.get("key_moments"):
            data = response.data.get("key_moments")
            return data if isinstance(data, list) else None
        
        logger.warning(f"No incongruence markers found in Supabase for session {session_id}")
        return None
        
    except Exception as e:
        logger.error(f"Failed to load incongruence markers from Supabase: {e}")
        return None


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


def get_multiple_session_summaries(patient_id: str, num_sessions: int = 3) -> Dict[str, Any]:
    """
    Get summaries for the last N sessions for a patient.
    
    Returns:
        {
            "patient_id": str,
            "sessions_count": int,
            "sessions": [
                {
                    "session_id": str,
                    "session_date": str,
                    "title": str,
                    "summary": str,
                    "overall_congruence": float,
                    "key_themes": list,
                    "duration": int
                },
                ...
            ]
        }
    """
    sessions = list_sessions(patient_id)
    if not sessions:
        return {
            "patient_id": patient_id,
            "sessions_count": 0,
            "sessions": [],
            "message": "No sessions found for this patient"
        }
    
    # Limit to most recent N sessions
    recent_sessions = sessions[:num_sessions]
    
    summaries = []
    for session in recent_sessions:
        sid = session["session_id"]
        
        # Get summary and notes for each session
        summary = get_session_summary(patient_id, sid)
        notes = get_therapist_notes(patient_id, sid)
        
        session_data = {
            "session_id": sid,
            "session_date": session.get("session_date"),
            "title": session.get("title", "Untitled Session"),
            "duration": session.get("duration"),
            "overall_congruence": session.get("overall_congruence"),
        }
        
        # Add summary data if available
        if summary:
            session_data["summary"] = summary.get("summary", "")
            session_data["incongruent_moments_count"] = len(summary.get("incongruent_moments", []))
            session_data["emotion_distribution"] = summary.get("emotion_distribution")
        
        # Add key themes from notes if available
        if notes:
            themes = notes.get("key_themes", [])
            session_data["key_themes"] = [
                {
                    "theme": t.get("theme", ""),
                    "description": t.get("description", "")[:200]  # Truncate for brevity
                }
                for t in themes[:3]  # Top 3 themes per session
            ]
            
            # Add risk assessment summary
            risk = notes.get("risk_assessment", {})
            if risk:
                session_data["risk_flags"] = {
                    "suicide_self_harm": risk.get("suicide_self_harm", {}).get("level", "none"),
                    "harm_to_others": risk.get("harm_to_others", {}).get("level", "none"),
                }
        
        summaries.append(session_data)
    
    return {
        "patient_id": patient_id,
        "sessions_count": len(summaries),
        "total_sessions_available": len(sessions),
        "sessions": summaries
    }


def find_latest_session(patient_id: str):
    """Return the latest session ID (int or UUID string) for a patient, or None."""
    sessions = list_sessions(patient_id)
    return sessions[0]["session_id"] if sessions else None


def resolve_session(patient_id: str, session_id=None):
    """
    Resolve a session_id: if None, default to the latest session.
    Returns the session_id (int or UUID string) or None if the patient has no sessions.
    """
    if session_id is not None:
        return session_id
    return find_latest_session(patient_id)


def find_patient_by_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Find a patient by name (case-insensitive partial match) from Supabase.
    
    Returns patient info including patient_id, or None if not found.
    """
    patients_metadata = _load_patients_from_db()
    
    if not patients_metadata:
        logger.error("No patients loaded from Supabase")
        return None
    
    name_lower = name.lower().strip()
    matches: List[Dict[str, Any]] = []
    
    for patient_id, info in patients_metadata.items():
        patient_name = info.get("name", "").lower()
        
        # Match on name only
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
    Search for patients by name or patient_id from Supabase.
    
    Returns a list of matching patients with their metadata.
    """
    patients_metadata = _load_patients_from_db()
    
    if not patients_metadata:
        logger.error("No patients loaded from Supabase")
        return []
    
    query_lower = query.lower().strip()
    matches: List[Dict[str, Any]] = []
    
    for patient_id, info in patients_metadata.items():
        patient_name = info.get("name", "").lower()
        pid_lower = patient_id.lower()
        
        # Match on name or patient_id
        if query_lower in patient_name or query_lower in pid_lower:
            matches.append({
                "patient_id": patient_id,
                **info
            })
    
    return matches


# ---------------------------------------------------------------------------
# Note Style Management Functions (MVP)
# ---------------------------------------------------------------------------

def save_note_style(
    user_id: str,
    note_name: str,
    note_text: str,
    file_type: str,
    validation_info: Optional[Dict[str, Any]] = None,
    style_analysis: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Save note style to database.
    
    Args:
        user_id: User identifier
        note_name: Name for the note style
        note_text: Extracted note text
        file_type: File type (pdf, docx, txt)
        validation_info: Validation results
        style_analysis: Style analysis results
    
    Returns:
        Note style ID if successful, None otherwise
    """
    from app.services.database import get_conversation_db
    
    db = get_conversation_db()
    if not db.is_enabled():
        logger.error("Database not enabled - cannot save note style")
        return None
    
    try:
        # For MVP: deactivate existing note styles for this user (one style per user)
        db.client.table("user_note_styles")\
            .update({"is_active": False})\
            .eq("user_id", user_id)\
            .execute()
        
        # Insert new note style
        insert_data = {
            "user_id": user_id,
            "note_name": note_name,
            "note_text": note_text,
            "file_type": file_type,
            "is_active": True
        }
        
        if validation_info:
            insert_data["validation_info"] = validation_info
        
        if style_analysis:
            insert_data["style_analysis"] = style_analysis
        
        result = db.client.table("user_note_styles").insert(insert_data).execute()
        
        if result.data:
            note_id = result.data[0]["id"]
            logger.info(f"✅ Note style saved for user {user_id[:8]}... (ID: {note_id})")
            return note_id
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to save note style for user {user_id[:8]}...: {e}")
        return None

def get_active_note_style(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get active note style for user.
    
    Args:
        user_id: User identifier
    
    Returns:
        Note style data if found, None otherwise
    """
    from app.services.database import get_conversation_db
    
    db = get_conversation_db()
    if not db.is_enabled():
        logger.warning("Database not enabled - cannot get note style")
        return None
    
    try:
        result = db.client.table("user_note_styles")\
            .select("*")\
            .eq("user_id", user_id)\
            .eq("is_active", True)\
            .single()\
            .execute()
        
        if result.data:
            logger.debug(f"Found active note style for user {user_id[:8]}...")
            return result.data
        
        return None
        
    except Exception as e:
        logger.debug(f"No active note style found for user {user_id[:8]}...: {e}")
        return None

def list_note_styles(user_id: str) -> List[Dict[str, Any]]:
    """
    List all note styles for user.
    
    Args:
        user_id: User identifier
    
    Returns:
        List of note style records
    """
    from app.services.database import get_conversation_db
    
    db = get_conversation_db()
    if not db.is_enabled():
        logger.warning("Database not enabled - cannot list note styles")
        return []
    
    try:
        result = db.client.table("user_note_styles")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .execute()
        
        note_styles = result.data or []
        logger.info(f"Found {len(note_styles)} note styles for user {user_id[:8]}...")
        return note_styles
        
    except Exception as e:
        logger.error(f"Failed to list note styles for user {user_id[:8]}...: {e}")
        return []

def get_note_style_by_id(user_id: str, note_style_id: str) -> Optional[Dict[str, Any]]:
    """
    Get specific note style by ID.
    
    Args:
        user_id: User identifier (for security)
        note_style_id: Note style ID
    
    Returns:
        Note style data if found and owned by user, None otherwise
    """
    from app.services.database import get_conversation_db
    
    db = get_conversation_db()
    if not db.is_enabled():
        logger.warning("Database not enabled - cannot get note style")
        return None
    
    try:
        result = db.client.table("user_note_styles")\
            .select("*")\
            .eq("id", note_style_id)\
            .eq("user_id", user_id)\
            .single()\
            .execute()
        
        return result.data if result.data else None
        
    except Exception as e:
        logger.error(f"Failed to get note style {note_style_id} for user {user_id[:8]}...: {e}")
        return None

def set_active_note_style(user_id: str, note_style_id: str) -> bool:
    """
    Set a specific note style as active for user.
    
    Args:
        user_id: User identifier
        note_style_id: Note style ID to activate
    
    Returns:
        True if successful, False otherwise
    """
    from app.services.database import get_conversation_db
    
    db = get_conversation_db()
    if not db.is_enabled():
        logger.error("Database not enabled - cannot set active note style")
        return False
    
    try:
        # First verify the note style exists and belongs to user
        note_style = get_note_style_by_id(user_id, note_style_id)
        if not note_style:
            logger.warning(f"Note style {note_style_id} not found for user {user_id[:8]}...")
            return False
        
        # Deactivate all note styles for user
        db.client.table("user_note_styles")\
            .update({"is_active": False})\
            .eq("user_id", user_id)\
            .execute()
        
        # Activate the specified note style
        result = db.client.table("user_note_styles")\
            .update({"is_active": True})\
            .eq("id", note_style_id)\
            .eq("user_id", user_id)\
            .execute()
        
        if result.data:
            logger.info(f"✅ Set note style {note_style_id} as active for user {user_id[:8]}...")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to set active note style for user {user_id[:8]}...: {e}")
        return False

def delete_note_style(user_id: str, note_style_id: str) -> bool:
    """
    Delete a note style.
    
    Args:
        user_id: User identifier (for security)
        note_style_id: Note style ID to delete
    
    Returns:
        True if successful, False otherwise
    """
    from app.services.database import get_conversation_db
    
    db = get_conversation_db()
    if not db.is_enabled():
        logger.error("Database not enabled - cannot delete note style")
        return False
    
    try:
        # Verify ownership before deletion
        note_style = get_note_style_by_id(user_id, note_style_id)
        if not note_style:
            logger.warning(f"Note style {note_style_id} not found for user {user_id[:8]}...")
            return False
        
        # Delete the note style
        result = db.client.table("user_note_styles")\
            .delete()\
            .eq("id", note_style_id)\
            .eq("user_id", user_id)\
            .execute()
        
        logger.info(f"✅ Deleted note style {note_style_id} for user {user_id[:8]}...")
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete note style {note_style_id} for user {user_id[:8]}...: {e}")
        return False
