"""
Real-time session fact extraction service.
Called automatically after each session is processed.

Extracts structured clinical facts from session data using LLM and saves to session_facts table.
"""

import json
import logging
from typing import Optional, Dict, Any

from openai import OpenAI

from app.services.database import get_conversation_db

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """
You are a clinical data extraction assistant. Extract ONLY facts explicitly stated in the content.

Session Content:
{content}

Extract these fields (use null if not mentioned):
{{
  "symptoms_json": {{"symptom_name": "severity"}},
  "interventions_json": [{{"type": "CBT/DBT/Psychodynamic/etc", "focus": "brief description"}}],
  "homework_json": [{{"task": "description", "assigned": true}}],
  "adherence_json": {{"medication": "good/partial/poor/unknown", "homework": "good/partial/poor/unknown"}},
  "risk_json": {{"suicide": "none/low/moderate/high", "self_harm": "none/low/moderate/high", "harm_to_others": "none/low/moderate/high"}},
  "stressors_json": ["stressor1", "stressor2"],
  "progress_markers_json": {{"scale_name": score}},
  "uncertainty_json": ["unclear_item1", "unclear_item2"]
}}

CRITICAL RULES:
- Only extract what's explicitly stated
- Use "unknown" if unclear
- Don't infer or assume
- Be conservative
- Use null for missing sections
"""


def extract_and_save_session_facts(
    session_video_id: str,
    patient_id: str,
    source_text: str
) -> bool:
    """
    Extract structured facts from session data and save to session_facts table.
    
    Args:
        session_video_id: UUID of the session_videos record
        patient_id: UUID of the patient
        source_text: Text content (therapist notes, transcript, or session analysis JSON)
    
    Returns:
        True if successful, False otherwise
    """
    db = get_conversation_db()
    if not db.is_enabled():
        logger.warning("Supabase not enabled, skipping fact extraction")
        return False
    
    # Check if facts already exist
    try:
        existing = db.client.table("session_facts")\
            .select("id")\
            .eq("session_video_id", session_video_id)\
            .execute()
        
        if existing.data:
            logger.info(f"Facts already exist for session {session_video_id[:8]}..., skipping")
            return True
    except Exception as e:
        logger.error(f"Error checking existing facts: {e}")
        return False
    
    # Validate source text
    if not source_text or len(source_text.strip()) < 50:
        logger.warning(f"Source text too short ({len(source_text)} chars), skipping fact extraction")
        return False
    
    try:
        client = OpenAI()
        
        # Call LLM to extract facts
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You extract clinical facts conservatively. Only extract what's explicitly stated."},
                {"role": "user", "content": EXTRACTION_PROMPT.format(content=source_text[:4000])}  # Limit to 4k chars
            ],
            response_format={"type": "json_object"},
            temperature=0.1  # Low temperature for consistency
        )
        
        facts = json.loads(response.choices[0].message.content)
        
        # Add required fields
        facts["session_video_id"] = session_video_id
        facts["patient_id"] = patient_id
        
        # Ensure all expected fields exist (with defaults)
        facts.setdefault("symptoms_json", {})
        facts.setdefault("interventions_json", [])
        facts.setdefault("homework_json", [])
        facts.setdefault("adherence_json", {})
        facts.setdefault("risk_json", {})
        facts.setdefault("stressors_json", [])
        facts.setdefault("progress_markers_json", {})
        facts.setdefault("uncertainty_json", [])
        
        # Save to database
        db.client.table("session_facts").insert(facts).execute()
        
        logger.info(f"✅ Extracted and saved facts for session {session_video_id[:8]}...")
        logger.debug(f"Extracted facts: {json.dumps(facts, indent=2)}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to extract session facts for {session_video_id[:8]}...: {e}")
        return False


def extract_facts_from_therapist_notes(
    session_video_id: str,
    patient_id: str,
    therapist_notes: Dict[str, Any]
) -> bool:
    """
    Extract facts from structured therapist notes.
    
    Args:
        session_video_id: UUID of the session_videos record
        patient_id: UUID of the patient
        therapist_notes: Structured notes dict from generate_therapist_notes()
    
    Returns:
        True if successful
    """
    # Convert structured notes to text for extraction
    source_text = json.dumps(therapist_notes, indent=2)
    return extract_and_save_session_facts(session_video_id, patient_id, source_text)


def extract_facts_from_analysis(
    session_video_id: str,
    patient_id: str,
    session_summary: Dict[str, Any]
) -> bool:
    """
    Extract facts from session_analysis summary.
    
    Args:
        session_video_id: UUID of the session_videos record
        patient_id: UUID of the patient
        session_summary: Session analysis dict with incongruent_moments
    
    Returns:
        True if successful
    """
    # Convert analysis to text for extraction
    source_text = json.dumps(session_summary, indent=2)
    return extract_and_save_session_facts(session_video_id, patient_id, source_text)
