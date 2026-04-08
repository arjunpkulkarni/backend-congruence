import json
import logging
import os
from typing import Any, Dict, List, Optional

from app.services.llm import analyze_text_emotion_with_llm
from app.services.prompts import (
    PROMPT_SINGLE_RELIABLE_EXTRACTION,
    build_single_extraction_message,
)

logger = logging.getLogger("emotion_api.notes")


def _get_notes_client():
    """
    Get OpenAI client for notes generation.
    Returns (client, model) or (None, None) if unavailable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        logger.warning("OPENAI_API_KEY not found in environment")
        return None, None

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        logger.error("Failed to import OpenAI: %s", e)
        return None, None

    try:
        client = OpenAI(api_key=api_key.strip())
        model = "gpt-4o"
        return client, model
    except Exception as e:
        logger.error("Failed to initialize OpenAI client: %s", e)
        return None, None


def generate_therapist_notes(
    transcript_text: str,
    transcript_segments: Optional[List[Dict[str, Any]]] = None,
    session_summary: Optional[Dict[str, Any]] = None,
    patient_id: Optional[str] = None,
    use_sequential: bool = True,
) -> Optional[Dict[str, Any]]:    
    return _generate_notes_single_call(
        transcript_text=transcript_text,
        transcript_segments=transcript_segments,
        session_summary=session_summary,
        patient_id=patient_id,
    )

def generate_therapist_notes_with_style(
    transcript_text: str,
    transcript_segments: Optional[List[Dict[str, Any]]] = None,
    session_summary: Optional[Dict[str, Any]] = None,
    patient_id: Optional[str] = None,
    user_id: Optional[str] = None,
    use_note_style: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Generate therapist notes with optional style matching.
    
    Args:
        transcript_text: Session transcript
        transcript_segments: Transcript segments with timestamps
        session_summary: Session analysis summary
        patient_id: Patient identifier
        user_id: User ID for note style lookup
        use_note_style: Whether to use uploaded note style
    
    Returns:
        Generated notes with style matching if requested
    """
    
    if use_note_style and user_id:
        # Get user's active note style
        from app.services.data_access import get_active_note_style
        note_style = get_active_note_style(user_id)
        
        if note_style:
            logger.info(f"Using note style '{note_style['note_name']}' for user {user_id[:8]}...")
            return _generate_notes_with_style_matching(
                transcript_text=transcript_text,
                transcript_segments=transcript_segments,
                session_summary=session_summary,
                patient_id=patient_id,
                reference_note=note_style["note_text"],
                style_info={
                    "note_style_id": note_style["id"],
                    "note_name": note_style["note_name"],
                    "file_type": note_style["file_type"],
                    "style_analysis": note_style.get("style_analysis")
                }
            )
        else:
            logger.warning(f"No active note style found for user {user_id[:8]}..., falling back to standard generation")
    
    # Fall back to regular note generation
    return _generate_notes_single_call(
        transcript_text=transcript_text,
        transcript_segments=transcript_segments,
        session_summary=session_summary,
        patient_id=patient_id,
    )


def _generate_notes_single_call(
    transcript_text: str,
    transcript_segments: Optional[List[Dict[str, Any]]] = None,
    session_summary: Optional[Dict[str, Any]] = None,
    patient_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Single-call reliable extraction for therapist session data.
    
    Focuses on what LLMs can reliably extract:
    - Factual content (medications, symptoms, timeline)
    - Speaker separation
    - Clinical template formatting
    """
    logger.info("Starting single-call reliable extraction for patient_id=%s", patient_id)

    notes_client, notes_model = _get_notes_client()

    if notes_client is None or notes_model is None:
        logger.warning("Notes OpenAI client not available")
        return None
    
    if not transcript_text or not transcript_text.strip():
        logger.warning("Empty transcript provided")
        return None

    # Optional: Get emotion data for context (but don't rely on it)
    logger.info("Pre-processing: Analyzing transcript with LLM for emotional context...")
    
    llm_analysis = analyze_text_emotion_with_llm(
        text=transcript_text,
        model=None,
        ensemble_size=1,
        temperature=0.2,
    )

    emotion_data_summary = _build_emotion_data_summary(llm_analysis, session_summary)
    
    # Build session metadata
    duration_str = "unknown"
    if session_summary and "duration" in session_summary:
        duration_seconds = session_summary.get("duration", 0)
        duration_str = f"{duration_seconds:.0f} seconds (~{duration_seconds/60:.1f} minutes)"
    
    has_timestamps = bool(transcript_segments)
        
    logger.info("Performing single reliable extraction call...")
    
    # Build user message for single extraction
    user_message = build_single_extraction_message(
        transcript_text=transcript_text,
        duration_str=duration_str,
        has_timestamps=has_timestamps,
        emotion_data_summary=emotion_data_summary,
    )
    
    # Single LLM call for all reliable extraction
    extraction_output = _call_llm_step(
        notes_client,
        notes_model,
        system_prompt=PROMPT_SINGLE_RELIABLE_EXTRACTION,
        user_message=user_message,
        step_name="Reliable Extraction",
        temperature=0.2,
    )
    
    if not extraction_output:
        logger.error("Reliable extraction failed")
        return None
    
    # Log what was extracted
    facts = extraction_output.get("extracted_facts", {})
    logger.info("Extraction complete:")
    logger.info("  - Medications: %d", len(facts.get("medications", [])))
    logger.info("  - Symptoms: %d", len(facts.get("symptoms", [])))
    logger.info("  - Timeline events: %d", len(facts.get("timeline_events", [])))
    logger.info("  - Life events: %d", len(facts.get("life_events", [])))
    
    summary = extraction_output.get("discussion_summary", {})
    logger.info("  - Main topics: %d", len(summary.get("main_topics", [])))
    logger.info("  - Patient concerns: %d", len(summary.get("patient_concerns", [])))
    
    logger.info("Single-call reliable extraction completed successfully")
    return extraction_output

def _generate_notes_with_style_matching(
    transcript_text: str,
    transcript_segments: Optional[List[Dict[str, Any]]] = None,
    session_summary: Optional[Dict[str, Any]] = None,
    patient_id: Optional[str] = None,
    reference_note: str = None,
    style_info: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Generate notes matching the style of a reference note.
    
    This is the core MVP functionality - LLM mimics structure, tone, and format
    of the user's uploaded note while staying factual to the transcript.
    """
    
    logger.info("Generating notes with style matching for patient_id=%s", patient_id)

    notes_client, notes_model = _get_notes_client()
    if notes_client is None or notes_model is None:
        logger.warning("Notes OpenAI client not available")
        return None
    
    if not transcript_text or not transcript_text.strip():
        logger.warning("Empty transcript provided")
        return None

    if not reference_note or not reference_note.strip():
        logger.warning("Empty reference note provided, falling back to standard generation")
        return _generate_notes_single_call(transcript_text, transcript_segments, session_summary, patient_id)

    # Prepare reference note for prompt (limit size)
    from app.services.note_style import prepare_style_context
    reference_context = prepare_style_context(reference_note, max_chars=2000)
    
    # Build session context
    duration_str = "unknown"
    if session_summary and "duration" in session_summary:
        duration_seconds = session_summary.get("duration", 0)
        duration_str = f"{duration_seconds:.0f} seconds (~{duration_seconds/60:.1f} minutes)"
    
    # Build the style-matching prompt
    style_matching_prompt = f"""You are formatting a clinical note to match a specific clinician's documentation style.

Here is a reference note written by the clinician:

---REFERENCE NOTE---
{reference_context}
---END REFERENCE---

Your task: Generate a clinical note that matches the EXACT style of the reference note.

Match these elements:
- Structure (section headers, ordering)
- Tone (concise vs detailed, formal vs conversational)  
- Level of detail (brief vs comprehensive)
- Phrasing patterns and terminology
- Section presence (if reference has "Family History", include it)

CRITICAL RULES:
- Do NOT invent clinical information
- Only use information from the provided transcript
- If a section from the reference note has no corresponding transcript information, write "Not discussed" or leave blank
- Match the style and format, not the content
- Stay factual - no clinical interpretations or assessments beyond what's explicitly stated

Session Information:
- Duration: {duration_str}
- Patient ID: {patient_id or 'Unknown'}

Here is the transcript to document:

---TRANSCRIPT---
{transcript_text}
---END TRANSCRIPT---

Generate a note in the same format and style as the reference note. Focus on matching structure, tone, and level of detail while staying completely factual to the transcript content."""

    try:
        response = notes_client.chat.completions.create(
            model=notes_model,
            messages=[
                {"role": "system", "content": "You are a clinical documentation assistant that matches writing styles perfectly while maintaining factual accuracy."},
                {"role": "user", "content": style_matching_prompt}
            ],
            temperature=0.3,  # Some creativity for style matching, but not too much
            max_tokens=2500
        )
        
        note_content = response.choices[0].message.content
        
        # Return in a structured format compatible with existing system
        from datetime import datetime
        return {
            "format": "style_matched",
            "content": note_content,
            "style_source": "user_uploaded",
            "patient_id": patient_id,
            "generated_at": datetime.now().isoformat(),
            "style_info": style_info,
            "session_metadata": {
                "duration": duration_str,
                "transcript_length": len(transcript_text),
                "reference_note_length": len(reference_note)
            }
        }
        
    except Exception as e:
        logger.error(f"Style-matched note generation failed: {e}")
        # Fall back to standard generation on error
        logger.info("Falling back to standard note generation due to style matching error")
        return _generate_notes_single_call(transcript_text, transcript_segments, session_summary, patient_id)


def _build_emotion_data_summary(llm_analysis: Optional[Dict[str, Any]], session_summary: Optional[Dict[str, Any]]) -> str:
    """Build emotion data summary for prompts."""
    emotion_data_summary = []
    if llm_analysis:
        emotion_data_summary.append("LLM Transcript Analysis Results:")
        emotion_data_summary.append(f"- Emotion distribution: {llm_analysis.get('emotion_distribution', {})}")
        emotion_data_summary.append(f"- Valence: {llm_analysis.get('valence', 0.0):.3f}")
        emotion_data_summary.append(f"- Arousal: {llm_analysis.get('arousal', 0.0):.3f}")
        emotion_data_summary.append(f"- Communication style: {llm_analysis.get('style', 'unknown')}")
        if "speakers" in llm_analysis:
            emotion_data_summary.append(f"- Speakers detected: {len(llm_analysis['speakers'])}")
        if "incongruence_reason" in llm_analysis:
            emotion_data_summary.append(f"- Incongruence flagged: {llm_analysis['incongruence_reason']}")

    if session_summary:
        emotion_data_summary.append("\nSession Emotion Distribution:")
        emotion_dist = session_summary.get("emotion_distribution", {})
        for modality in ["text", "face", "audio"]:
            if modality in emotion_dist:
                emotion_data_summary.append(f"- {modality}: {emotion_dist[modality]}")

    return "\n".join(emotion_data_summary) if emotion_data_summary else "None provided"


def _call_llm_step(
    client,
    model: str,
    system_prompt: str,
    user_message: str,
    step_name: str,
    temperature: float = 0.3,
    max_tokens: int = 2000,
) -> Optional[Dict[str, Any]]:
    """Call LLM for a single pipeline step with error handling."""
    try:
        logger.debug("Calling OpenAI API for %s (temp=%.2f)", step_name, temperature)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        
        response_text = response.choices[0].message.content
        if not response_text:
            logger.warning("%s returned empty content", step_name)
            return None
        
        try:
            result = json.loads(response_text)
            logger.debug("%s returned %d top-level keys", step_name, len(result))
            return result
        except json.JSONDecodeError as e:
            logger.error("%s returned invalid JSON: %s", step_name, e)
            logger.debug("Raw response: %s", response_text[:500])
            return None
        
    except Exception as e:
        logger.exception("%s failed: %s", step_name, e)
        return None


# Old pipeline merge function removed - no longer needed with single-call approach


def save_therapist_notes(
    notes: Optional[Dict[str, Any]],
    output_path: str,
) -> bool:

    if not notes:
        logger.warning("Cannot save therapist notes: notes content is empty")
        return False
    
    try:
        logger.info("Saving therapist notes to: %s", output_path)
        
        markdown_content = _convert_notes_to_markdown(notes)
        
        # Ensure we have the correct paths for both JSON and MD
        if output_path.endswith('.json'):
            json_path = output_path
            md_path = output_path.replace('.json', '.md')
        elif output_path.endswith('.md'):
            md_path = output_path
            json_path = output_path.replace('.md', '.json')
        else:
            # No extension provided, add both
            json_path = output_path + '.json'
            md_path = output_path + '.md'
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)
        
        logger.info("Therapist notes saved successfully")
        return True
    except Exception as e:
        logger.exception("Failed to save therapist notes: %s", e)
        return False


def _convert_notes_to_markdown(notes: Dict[str, Any]) -> str:
    """Convert SOAP note output to readable markdown format."""
    lines = ["# 📝 SOAP Clinical Note", ""]
    lines.append("*Generated clinical documentation - requires clinician review and completion*")
    lines.append("")
    
    # Handle style-matched format
    if notes.get("format") == "style_matched":
        lines.append("## Style-Matched Clinical Note")
        lines.append("")
        lines.append(f"**Style Source:** {notes.get('style_info', {}).get('note_name', 'User uploaded note')}")
        lines.append("")
        lines.append("### Generated Note Content")
        lines.append("```")
        lines.append(notes.get("content", "No content available"))
        lines.append("```")
        lines.append("")
        
        # Session metadata for style-matched notes
        session_meta = notes.get("session_metadata", {})
        if session_meta:
            lines.append("### Session Information")
            if "duration" in session_meta:
                lines.append(f"**Duration:** {session_meta['duration']}")
            if "transcript_length" in session_meta:
                lines.append(f"**Transcript Length:** {session_meta['transcript_length']} characters")
            lines.append("")
        
        return "\n".join(lines)
    
    # Handle fallback format
    if notes.get("format") == "fallback":
        lines.append("**Note:** This is a fallback format due to parsing issues.")
        lines.append("")
        lines.append(notes.get("raw_content", "No content available"))
        return "\n".join(lines)
    
    # Handle new SOAP format
    soap_note = notes.get("soap_note", {})
    if soap_note:
        lines.append("## SOAP Note")
        lines.append("")
        
        # SUBJECTIVE
        subjective = soap_note.get("subjective", {})
        if subjective:
            lines.append("### 🗣️ SUBJECTIVE")
            lines.append("*Patient's personal experiences, feelings, and perspectives*")
            lines.append("")
            
            if subjective.get("chief_complaint"):
                lines.append("**Chief Complaint:**")
                lines.append(f"{subjective['chief_complaint']}")
                lines.append("")
            
            if subjective.get("history_present_illness"):
                lines.append("**History of Present Illness:**")
                lines.append(f"{subjective['history_present_illness']}")
                lines.append("")
            
            if subjective.get("current_medications"):
                lines.append("**Current Medications:**")
                for med in subjective["current_medications"]:
                    if isinstance(med, dict):
                        med_str = med.get("medication", "Unknown medication")
                        if med.get("dosage"):
                            med_str += f" ({med['dosage']})"
                        lines.append(f"- {med_str}")
                        if med.get("patient_report"):
                            lines.append(f"  - Patient reports: \"{med['patient_report']}\"")
                    else:
                        lines.append(f"- {med}")
                lines.append("")
            
            if subjective.get("psychosocial_factors"):
                lines.append("**Psychosocial Factors:**")
                lines.append(f"{subjective['psychosocial_factors']}")
                lines.append("")
            
            if subjective.get("patient_perspective"):
                lines.append("**Patient's Perspective:**")
                lines.append(f"{subjective['patient_perspective']}")
                lines.append("")
        
        # OBJECTIVE
        objective = soap_note.get("objective", {})
        if objective:
            lines.append("### 👁️ OBJECTIVE")
            lines.append("*Observable clinical data and mental status examination*")
            lines.append("")
            
            mse = objective.get("mental_status_exam", {})
            if mse:
                lines.append("**Mental Status Examination:**")
                for key, value in mse.items():
                    if value and value != "Not discussed in this session":
                        formatted_key = key.replace("_", " ").title()
                        lines.append(f"- **{formatted_key}:** {value}")
                lines.append("")
            
            if objective.get("clinical_observations"):
                lines.append("**Clinical Observations:**")
                lines.append(f"{objective['clinical_observations']}")
                lines.append("")
        
        # ASSESSMENT
        assessment = soap_note.get("assessment", {})
        if assessment:
            lines.append("### 🔍 ASSESSMENT")
            lines.append("*Clinical analysis and diagnostic impressions*")
            lines.append("")
            
            if assessment.get("clinical_impressions"):
                lines.append("**Clinical Impressions:**")
                lines.append(f"{assessment['clinical_impressions']}")
                lines.append("")
            
            if assessment.get("problem_list"):
                lines.append("**Problem List:**")
                for problem in assessment["problem_list"]:
                    if isinstance(problem, dict):
                        problem_str = problem.get("problem", "Unknown problem")
                        if problem.get("priority"):
                            problem_str += f" (Priority: {problem['priority']})"
                        if problem.get("status"):
                            problem_str += f" - Status: {problem['status']}"
                        lines.append(f"- {problem_str}")
                    else:
                        lines.append(f"- {problem}")
                lines.append("")
            
            if assessment.get("risk_assessment"):
                lines.append("**Risk Assessment:**")
                lines.append(f"{assessment['risk_assessment']}")
                lines.append("")
            
            if assessment.get("progress_notes"):
                lines.append("**Progress Notes:**")
                lines.append(f"{assessment['progress_notes']}")
                lines.append("")
        
        # PLAN
        plan = soap_note.get("plan", {})
        if plan:
            lines.append("### 📋 PLAN")
            lines.append("*Treatment interventions and next steps*")
            lines.append("")
            
            if plan.get("therapeutic_interventions"):
                lines.append("**Therapeutic Interventions:**")
                for intervention in plan["therapeutic_interventions"]:
                    lines.append(f"- {intervention}")
                lines.append("")
            
            if plan.get("homework_assignments"):
                lines.append("**Homework/Assignments:**")
                for homework in plan["homework_assignments"]:
                    lines.append(f"- {homework}")
                lines.append("")
            
            if plan.get("medication_plan"):
                lines.append("**Medication Plan:**")
                lines.append(f"{plan['medication_plan']}")
                lines.append("")
            
            follow_up = plan.get("follow_up", {})
            if follow_up:
                lines.append("**Follow-up:**")
                if follow_up.get("next_appointment"):
                    lines.append(f"- Next appointment: {follow_up['next_appointment']}")
                if follow_up.get("frequency"):
                    lines.append(f"- Session frequency: {follow_up['frequency']}")
                if follow_up.get("monitoring"):
                    lines.append(f"- Monitoring: {follow_up['monitoring']}")
                lines.append("")
            
            if plan.get("referrals"):
                lines.append("**Referrals:**")
                for referral in plan["referrals"]:
                    lines.append(f"- {referral}")
                lines.append("")
            
            if plan.get("patient_education"):
                lines.append("**Patient Education:**")
                lines.append(f"{plan['patient_education']}")
                lines.append("")
    
    # Session metadata
    metadata = notes.get("session_metadata", {})
    if metadata:
        lines.append("## Session Information")
        if "duration_seconds" in metadata and metadata["duration_seconds"] > 0:
            duration = metadata["duration_seconds"]
            lines.append(f"**Duration:** {duration} seconds (~{duration/60:.1f} minutes)")
        if "session_type" in metadata:
            lines.append(f"**Session Type:** {metadata['session_type']}")
        if "primary_focus" in metadata:
            lines.append(f"**Primary Focus:** {metadata['primary_focus']}")
        if "extraction_confidence" in metadata:
            lines.append(f"**Extraction Confidence:** {metadata['extraction_confidence']}")
        lines.append("")
    
    # Clinical summary
    clinical_summary = notes.get("clinical_summary", {})
    if clinical_summary:
        lines.append("## Clinical Summary")
        lines.append("")
        
        if clinical_summary.get("key_themes"):
            lines.append("**Key Therapeutic Themes:**")
            for theme in clinical_summary["key_themes"]:
                lines.append(f"- {theme}")
            lines.append("")
        
        if clinical_summary.get("patient_goals"):
            lines.append("**Patient Goals:**")
            for goal in clinical_summary["patient_goals"]:
                lines.append(f"- {goal}")
            lines.append("")
        
        if clinical_summary.get("clinician_observations"):
            lines.append("**Clinician Observations:**")
            for obs in clinical_summary["clinician_observations"]:
                lines.append(f"- {obs}")
            lines.append("")
        
        if clinical_summary.get("session_outcome"):
            lines.append("**Session Outcome:**")
            lines.append(f"{clinical_summary['session_outcome']}")
            lines.append("")
    
    lines.append("---")
    lines.append("*This SOAP note requires clinician review and completion. All clinical assessments and treatment plans should be verified by qualified healthcare professionals.*")
    
    return "\n".join(lines)
