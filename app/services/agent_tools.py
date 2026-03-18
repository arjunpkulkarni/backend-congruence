"""
Real tool implementations for the Congruence Ops Agent.

Each tool wraps the data_access layer and returns a formatted string
that the LLM can reason over.  Tools follow the LangChain @tool decorator
pattern and accept a single JSON-serialisable input.
"""

import json
import logging
from typing import Optional

from langchain_core.tools import tool

from app.services.data_access import (
    get_congruence_timeline,
    get_incongruence_markers,
    get_intensity_timeline,
    get_patient_history,
    get_practice_analytics_data,
    get_session_summary,
    get_session_transcript,
    get_simplified_notes,
    get_spikes,
    get_therapist_notes,
    list_patients,
    list_sessions,
    resolve_session,
    find_patient_by_name,
    search_patients,
    get_multiple_session_summaries,
)
from app.services.agent_intent import format_evidence_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_compact(obj, max_len: int = 3000) -> str:
    """Serialize to compact JSON, truncated to max_len chars."""
    raw = json.dumps(obj, indent=2, default=str)
    if len(raw) > max_len:
        return raw[:max_len] + "\n... (truncated)"
    return raw


# ---------------------------------------------------------------------------
# Patient / Record Tools
# ---------------------------------------------------------------------------

@tool
def find_patient(query: str) -> str:
    """
    Find a patient by their name, MRN, or partial patient ID.
    Use this tool when the user mentions a patient by name instead of ID.
    Example queries: "Rob Wazowski", "MRN-001", "demo"
    Returns the patient_id and metadata needed for other tools.
    """
    result = find_patient_by_name(query)
    
    if result is None:
        # Try broader search
        matches = search_patients(query)
        if not matches:
            return json.dumps({
                "status": "not_found",
                "message": f"No patient found matching '{query}'",
                "suggestion": "Try list_all_patients to see available patients"
            })
        result = {"multiple_matches": True, "matches": matches, "count": len(matches)}
    
    if isinstance(result, dict) and result.get("multiple_matches"):
        return json.dumps({
            "status": "multiple_matches",
            "count": result["count"],
            "matches": result["matches"],
            "message": f"Found {result['count']} patients matching '{query}'. Please be more specific or use the patient_id directly."
        })
    
    return json.dumps({
        "status": "found",
        "patient_id": result["patient_id"],
        "name": result.get("name"),
        "mrn": result.get("mrn"),
        "message": f"Found patient: {result.get('name', result['patient_id'])}"
    })


@tool
def get_patient_record(query: str) -> str:
    """
    Retrieve patient record, session history, and congruence trend.
    The query should contain the patient_id. If you don't know the
    patient_id, call list_all_patients first.
    Example query: "4e3c1260-9e27-4cc8-9720-114e068d03f1"
    """
    patient_id = query.strip()
    if not patient_id:
        return "Error: please provide a patient_id."

    history = get_patient_history(patient_id)
    if not history.get("sessions"):
        return f"No records found for patient '{patient_id}'. Use list_all_patients to see available patients."

    return _json_compact(history)


@tool
def list_all_patients(query: str) -> str:
    """
    List all patients in the system with their session counts and latest activity.
    No specific query needed — pass any string (e.g. 'all').
    """
    patients = list_patients()
    if not patients:
        return "No patients found in the system."
    return _json_compact({"patients": patients, "total": len(patients)})


# ---------------------------------------------------------------------------
# Transcript Tool
# ---------------------------------------------------------------------------

@tool
def get_session_transcript_tool(query: str) -> str:
    """
    Get the transcript from a therapy session.
    Query format: 'patient_id' or 'patient_id session_id'.
    If session_id is omitted the latest session is used.
    Example: '4e3c1260-9e27-4cc8-9720-114e068d03f1 1768675235'
    """
    parts = query.strip().split()
    if not parts:
        return "Error: provide patient_id (and optionally session_id)."

    patient_id = parts[0]
    # Handle both UUID strings and integer timestamps
    if len(parts) > 1:
        try:
            session_id = int(parts[1])
        except ValueError:
            # It's a UUID string, keep as is
            session_id = parts[1]
    else:
        session_id = None
    session_id = resolve_session(patient_id, session_id)

    if session_id is None:
        return f"No sessions found for patient '{patient_id}'."

    transcript = get_session_transcript(patient_id, session_id)
    if transcript is None:
        return f"No transcript available for patient '{patient_id}' session {session_id}."

    return _json_compact(transcript)


# ---------------------------------------------------------------------------
# Clinical Note Generation Tool
# ---------------------------------------------------------------------------

@tool
def generate_clinical_note(query: str) -> str:
    """
    Retrieve existing clinical notes for a session, including the
    structured therapist notes and simplified notes.
    Query format: 'patient_id' or 'patient_id session_id'.
    """
    parts = query.strip().split()
    if not parts:
        return "Error: provide patient_id (and optionally session_id)."

    patient_id = parts[0]
    # Handle both UUID strings and integer timestamps
    if len(parts) > 1:
        try:
            session_id = int(parts[1])
        except ValueError:
            # It's a UUID string, keep as is
            session_id = parts[1]
    else:
        session_id = None
    session_id = resolve_session(patient_id, session_id)

    if session_id is None:
        return f"No sessions found for patient '{patient_id}'."

    # Gather both note types
    notes = get_therapist_notes(patient_id, session_id)
    simplified = get_simplified_notes(patient_id, session_id)
    summary = get_session_summary(patient_id, session_id)

    if notes is None and simplified is None:
        return f"No clinical notes available for patient '{patient_id}' session {session_id}."

    result = {
        "patient_id": patient_id,
        "session_id": session_id,
    }
    if notes:
        result["therapist_notes"] = notes
    if simplified:
        result["simplified_notes_preview"] = simplified[:1500]
    if summary:
        result["session_summary"] = {
            "duration": summary.get("duration"),
            "overall_congruence": summary.get("overall_congruence"),
            "incongruent_moments_count": len(summary.get("incongruent_moments", [])),
            "emotion_distribution": summary.get("emotion_distribution"),
        }

    return _json_compact(result, max_len=4000)


@tool
def get_multiple_sessions_summary(query: str) -> str:
    """
    Retrieve summaries for the last N sessions for a patient.
    Useful when the user asks for "last 3 sessions", "recent sessions", etc.
    Query format: 'patient_id' or 'patient_id N' where N is the number of sessions.
    Example: 'sophia-uuid 3' or just 'sophia-uuid' (defaults to 3 sessions)
    """
    parts = query.strip().split()
    if not parts:
        return "Error: provide patient_id (and optionally number of sessions)."
    
    patient_id = parts[0]
    num_sessions = int(parts[1]) if len(parts) > 1 else 3
    
    # Validate num_sessions
    if num_sessions < 1 or num_sessions > 10:
        return "Error: number of sessions must be between 1 and 10."
    
    result = get_multiple_session_summaries(patient_id, num_sessions)
    
    if result.get("sessions_count", 0) == 0:
        return f"No sessions found for patient '{patient_id}'."
    
    return _json_compact(result, max_len=6000)


# ---------------------------------------------------------------------------
# ICD-10 Suggestion Tool
# ---------------------------------------------------------------------------

@tool
def suggest_icd10_codes(query: str) -> str:
    """
    Suggest ICD-10 diagnostic codes based on session data and clinical notes.
    Query format: 'patient_id' or 'patient_id session_id'.
    Analyzes therapist notes, risk assessment, and emotional patterns
    to suggest relevant diagnostic codes.
    """
    parts = query.strip().split()
    if not parts:
        return "Error: provide patient_id (and optionally session_id)."

    patient_id = parts[0]
    # Handle both UUID strings and integer timestamps
    if len(parts) > 1:
        try:
            session_id = int(parts[1])
        except ValueError:
            # It's a UUID string, keep as is
            session_id = parts[1]
    else:
        session_id = None
    session_id = resolve_session(patient_id, session_id)

    if session_id is None:
        return f"No sessions found for patient '{patient_id}'."

    notes = get_therapist_notes(patient_id, session_id)
    summary = get_session_summary(patient_id, session_id)

    if notes is None and summary is None:
        return f"No clinical data available for code suggestion. Process a session first."

    # Build context for the LLM to reason about ICD-10 codes
    context = {
        "patient_id": patient_id,
        "session_id": session_id,
        "note": (
            "ICD-10 code suggestions require clinical judgment. "
            "The following data is provided for the agent to reason about possible codes. "
            "These are NOT automated diagnoses."
        ),
    }

    if notes:
        risk = notes.get("risk_assessment", {})
        observations = notes.get("clinical_observations", {})
        themes = notes.get("key_themes", [])
        context["risk_assessment"] = risk
        context["clinical_observations"] = observations
        context["themes"] = [t.get("theme", "") for t in themes]

    if summary:
        context["overall_congruence"] = summary.get("overall_congruence")
        context["emotion_distribution"] = summary.get("emotion_distribution")
        context["incongruent_moments"] = summary.get("incongruent_moments", [])

    return _json_compact(context, max_len=3000)


# ---------------------------------------------------------------------------
# Insurance / Billing Tools (stub-ish but with real data context)
# ---------------------------------------------------------------------------

@tool
def generate_insurance_packet(query: str) -> str:
    """
    Generate an insurance authorization packet for a patient.
    Query format: 'patient_id' or 'patient_id session_id'.
    Gathers session summary, notes, and congruence data to build
    documentation that supports medical necessity.
    """
    parts = query.strip().split()
    if not parts:
        return "Error: provide patient_id (and optionally session_id)."

    patient_id = parts[0]
    # Handle both UUID strings and integer timestamps
    if len(parts) > 1:
        try:
            session_id = int(parts[1])
        except ValueError:
            # It's a UUID string, keep as is
            session_id = parts[1]
    else:
        session_id = None
    session_id = resolve_session(patient_id, session_id)

    if session_id is None:
        return f"No sessions found for patient '{patient_id}'."

    summary = get_session_summary(patient_id, session_id)
    notes = get_therapist_notes(patient_id, session_id)
    history = get_patient_history(patient_id, limit=5)

    packet = {
        "patient_id": patient_id,
        "session_id": session_id,
        "packet_type": "prior_authorization",
        "status": "draft",
        "session_summary": {
            "duration": summary.get("duration") if summary else None,
            "overall_congruence": summary.get("overall_congruence") if summary else None,
            "incongruent_moments": len(summary.get("incongruent_moments", [])) if summary else 0,
        } if summary else None,
        "clinical_justification": {
            "themes": [t.get("theme", "") for t in (notes or {}).get("key_themes", [])],
            "risk_flags": bool(notes and notes.get("risk_assessment")),
            "session_count": history.get("total_sessions", 0),
        },
        "note": (
            "This is a draft insurance packet. A clinician should review and "
            "complete this before submission. Full integration with insurance "
            "APIs will be available in a future iteration."
        ),
    }

    return _json_compact(packet)


@tool
def check_claim_status(query: str) -> str:
    """
    Check insurance claim status for a patient.
    Query should be a patient_id or claim reference number.
    Note: Real insurance API integration is planned for a future iteration.
    """
    return json.dumps({
        "query": query.strip(),
        "status": "pending_integration",
        "message": (
            "Insurance claim status checking requires integration with insurance "
            "clearinghouse APIs (e.g. Availity, Change Healthcare). This will be "
            "implemented in a future iteration. For now, check your practice "
            "management system directly."
        ),
    })


# ---------------------------------------------------------------------------
# Scheduling / Intake Tools (stubs with helpful context)
# ---------------------------------------------------------------------------

@tool
def schedule_appointment(query: str) -> str:
    """
    Schedule a patient appointment.
    Query should include patient_id and desired date/time.
    Note: Real scheduling integration is planned for a future iteration.
    """
    return json.dumps({
        "query": query.strip(),
        "status": "pending_integration",
        "message": (
            "Appointment scheduling requires integration with your practice "
            "management system (e.g. SimplePractice, TherapyNotes). This will "
            "be implemented in a future iteration."
        ),
    })


@tool
def send_intake_form(query: str) -> str:
    """
    Send intake forms to a patient via email.
    Query should include patient_id or email address.
    Note: Real form delivery is planned for a future iteration.
    """
    return json.dumps({
        "query": query.strip(),
        "status": "pending_integration",
        "message": (
            "Intake form delivery requires email/SMS integration. "
            "This will be implemented in a future iteration."
        ),
    })


# ---------------------------------------------------------------------------
# Evidence Search Tool (NEW - for "show me proof" queries)
# ---------------------------------------------------------------------------

@tool
def search_clinical_evidence(query: str) -> str:
    """
    Search for specific evidence/mentions in patient records, notes, and transcripts.
    Use this when the user asks for PROOF, EVIDENCE, or specific MENTIONS of conditions/topics.
    
    Query format: 'search_term patient_id' or just 'search_term' to search all patients.
    Examples: "OCD", "anxiety Rob Wazowski", "sleep problems"
    
    Returns exact quotes with sources, timestamps, and patient info.
    """
    parts = query.strip().split()
    if not parts:
        return json.dumps({"error": "Please provide search terms"})
    
    # Try to extract patient_id from query
    patient_id = None
    search_terms = []
    
    for part in parts:
        # Check if it's a UUID-like patient_id
        if len(part) > 30 or part in ["4e3c1260-9e27-4cc8-9720-114e068d03f1", "demo", "dev"]:
            patient_id = part
        else:
            search_terms.append(part.lower())
    
    search_query = " ".join(search_terms)
    
    # Get patients to search
    if patient_id:
        patients_to_search = [{"patient_id": patient_id}]
    else:
        patients_to_search = list_patients()
    
    # Search for evidence
    evidence_found = []
    
    for patient in patients_to_search:
        pid = patient["patient_id"]
        patient_name = patient.get("name", pid)
        
        # Get ALL sessions (not just latest)
        sessions = list_sessions(pid)
        if not sessions:
            continue
        
        # Filter sessions by title if search terms match
        # This prioritizes sessions with matching titles (e.g., "OCD" session)
        matching_sessions = []
        for session in sessions:
            session_title = session.get("title", "").lower()
            # If any search term is in the session title, prioritize this session
            if any(term in session_title for term in search_terms):
                matching_sessions.append(session)
        
        # If no title matches, search all sessions
        if not matching_sessions:
            matching_sessions = sessions
        
        # Now search within matching sessions
        for session in matching_sessions:
            sid = session["session_id"]
            session_title = session.get("title", "")
            session_date = session.get("session_date", "")
            
            # Search in clinical notes
            notes = get_therapist_notes(pid, sid)
            if notes:
                # Search in key themes
                for theme in notes.get("key_themes", []):
                    theme_text = json.dumps(theme).lower()
                    if any(term in theme_text for term in search_terms):
                        evidence_found.append({
                            "text": theme.get("description", ""),
                            "source": f"Clinical Notes - Theme: {theme.get('theme', 'Unknown')}",
                            "patient": patient_name,
                            "patient_id": pid,
                            "session_id": sid,
                            "session_title": session_title,
                            "session_date": session_date,
                            "type": "clinical_note"
                        })
                
                # Search in clinical observations
                observations = notes.get("clinical_observations", {})
                for key, items in observations.items():
                    if isinstance(items, list):
                        for item in items:
                            item_lower = str(item).lower()
                            if any(term in item_lower for term in search_terms):
                                evidence_found.append({
                                    "text": item,
                                    "source": f"Clinical Notes - {key.replace('_', ' ').title()}",
                                    "patient": patient_name,
                                    "patient_id": pid,
                                    "session_id": sid,
                                    "session_title": session_title,
                                    "session_date": session_date,
                                    "type": "clinical_observation"
                                })
                
                # Search in risk assessment
                risk = notes.get("risk_assessment", {})
                risk_text = json.dumps(risk).lower()
                if any(term in risk_text for term in search_terms):
                    for risk_type, risk_data in risk.items():
                        if isinstance(risk_data, dict):
                            evidence_text = risk_data.get("evidence", "")
                            if any(term in evidence_text.lower() for term in search_terms):
                                evidence_found.append({
                                    "text": evidence_text,
                                    "source": f"Clinical Notes - Risk Assessment: {risk_type.replace('_', ' ').title()}",
                                    "patient": patient_name,
                                    "patient_id": pid,
                                    "session_id": sid,
                                    "session_title": session_title,
                                    "session_date": session_date,
                                    "type": "risk_assessment"
                                })
            
            # Search in transcript
            transcript = get_session_transcript(pid, sid, include_segments=True)
            if transcript and transcript.get("text"):
                transcript_text = transcript["text"].lower()
                if any(term in transcript_text for term in search_terms):
                    # Find relevant segments
                    for segment in transcript.get("segments", []):
                        segment_text = segment.get("text", "").lower()
                        if any(term in segment_text for term in search_terms):
                            evidence_found.append({
                                "text": segment.get("text", "").strip(),
                                "source": "Session Transcript",
                                "patient": patient_name,
                                "patient_id": pid,
                                "session_id": sid,
                                "session_title": session_title,
                                "session_date": session_date,
                                "timestamp": f"{segment.get('start', 0):.1f}s - {segment.get('end', 0):.1f}s",
                                "type": "transcript"
                            })
    
    # Format response
    if not evidence_found:
        return json.dumps({
            "status": "no_evidence",
            "query": search_query,
            "patients_searched": len(patients_to_search),
            "message": f"No evidence found for '{search_query}' in {len(patients_to_search)} patient(s)"
        })
    
    # Return formatted evidence
    return format_evidence_response(evidence_found, search_query)


# ---------------------------------------------------------------------------
# Practice Analytics Tool
# ---------------------------------------------------------------------------

@tool
def get_practice_analytics(query: str) -> str:
    """
    Generate practice-wide metrics and analytics including total patients,
    sessions, average congruence scores, and recent activity.
    No specific query needed — pass any string (e.g. 'overview').
    """
    analytics = get_practice_analytics_data()
    return _json_compact(analytics, max_len=4000)


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    search_clinical_evidence,  # NEW - Evidence mode tool
    find_patient,
    list_all_patients,
    get_patient_record,
    get_session_transcript_tool,
    generate_clinical_note,
    get_multiple_sessions_summary,  # NEW - Multi-session summary tool
    suggest_icd10_codes,
    generate_insurance_packet,
    check_claim_status,
    schedule_appointment,
    send_intake_form,
    get_practice_analytics,
]

TOOL_MAP = {t.name: t for t in ALL_TOOLS}
