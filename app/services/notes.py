import json
import logging
import os
from typing import Any, Dict, List, Optional

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
        model = os.getenv("OPENAI_NOTES_MODEL", "gpt-4o-mini")
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
    Single-call extraction: transcript in, SOAP note + summary out.
    No emotion data, no prior session data — fully isolated per session.
    """
    def _ndbg(msg, data=None):
        # #region agent log
        import json as _j, time as _t
        try:
            with open("/Users/arjunkulkarni/Desktop/BCONGRUENCE/.cursor/debug-adf136.log","a") as _f:
                _f.write(_j.dumps({"sessionId":"adf136","hypothesisId":"H4","location":"notes.py","message":msg,"data":data or {},"timestamp":int(_t.time()*1000)})+"\n")
        except Exception:
            pass
        # #endregion

    # #region agent log
    import time as _ntime; _notes_start = _ntime.time()
    _ndbg("notes_generation_start", {"patient_id": patient_id, "transcript_len": len(transcript_text or "")})
    # #endregion
    logger.info("Starting note generation for patient_id=%s", patient_id)

    notes_client, notes_model = _get_notes_client()

    if notes_client is None or notes_model is None:
        logger.warning("Notes OpenAI client not available")
        return None

    if not transcript_text or not transcript_text.strip():
        logger.warning("Empty transcript provided")
        return None

    # Build session duration string from current session only
    duration_str = "unknown"
    if session_summary and "duration" in session_summary:
        duration_seconds = session_summary.get("duration", 0)
        duration_str = f"{duration_seconds:.0f} seconds (~{duration_seconds/60:.1f} minutes)"

    # Build user message — transcript only, no emotion data, no history
    user_message = build_single_extraction_message(
        transcript_text=transcript_text,
        duration_str=duration_str,
        has_timestamps=bool(transcript_segments),
        emotion_data_summary="",  # intentionally empty — no external data in prompt
    )

    # Single isolated LLM call
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

    # #region agent log
    _ndbg("notes_generation_done", {"duration_s": round(_ntime.time() - _notes_start, 2), "has_soap": "soap_note" in extraction_output, "has_summary": "transcript_summary" in extraction_output, "keys": list(extraction_output.keys())})
    # #endregion
    logger.info("Note generation completed successfully")
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
            max_tokens=4000
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
                "reference_note_length": len(reference_note),
            },
            "_transcript_text": transcript_text,
            "_transcript_segments": transcript_segments,
        }
        
    except Exception as e:
        logger.error(f"Style-matched note generation failed: {e}")
        # Fall back to standard generation on error
        logger.info("Falling back to standard note generation due to style matching error")
        return _generate_notes_single_call(transcript_text, transcript_segments, session_summary, patient_id)




def _call_llm_step(
    client,
    model: str,
    system_prompt: str,
    user_message: str,
    step_name: str,
    temperature: float = 0.3,
    max_tokens: int = 4000,
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
        
        # Exclude internal keys (transcript data) from JSON — already saved separately
        json_notes = {k: v for k, v in notes.items() if not k.startswith("_")}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_notes, f, ensure_ascii=False, indent=2)
        
        logger.info("Therapist notes saved successfully")
        return True
    except Exception as e:
        logger.exception("Failed to save therapist notes: %s", e)
        return False


def _convert_notes_to_markdown(notes: Dict[str, Any]) -> str:
    """Convert notes to markdown matching real psychiatric progress note format."""
    lines = []

    # Handle style-matched format
    if notes.get("format") == "style_matched":
        lines.append("# Progress Note")
        lines.append("")
        lines.append(notes.get("content", "No content available"))
        lines.append("")
        _append_transcript_summary(lines, notes)
        _append_full_transcript(lines, notes)
        return "\n".join(lines)

    # Handle legacy SOAP format (backwards compatibility)
    if "soap_note" in notes and "identifying_data" not in notes:
        return _convert_legacy_soap_to_markdown(notes)

    # ── Progress Note ──
    lines.append("# Progress Note")
    lines.append("")

    lines.append("## Identifying Data")
    lines.append("")
    lines.append(notes.get("identifying_data", "Not discussed in this session."))
    lines.append("")

    lines.append("## Subjective")
    lines.append("")
    lines.append(notes.get("subjective", "Not discussed in this session."))
    lines.append("")

    lines.append("## Mental Status Exam")
    lines.append("")
    lines.append(notes.get("mental_status_exam", "Not discussed in this session."))
    lines.append("")

    lines.append("## Assessment & Plan")
    lines.append("")
    lines.append(notes.get("assessment_and_plan", "Not discussed in this session."))
    lines.append("")

    # ── Transcript Summary ──
    _append_transcript_summary(lines, notes)

    # ── Full Transcript ──
    _append_full_transcript(lines, notes)

    return "\n".join(lines)


def _convert_legacy_soap_to_markdown(notes: Dict[str, Any]) -> str:
    """Backwards-compatible renderer for old SOAP-structured notes."""
    lines = []
    lines.append("# Progress Note")
    lines.append("")

    soap = notes.get("soap_note", {})

    subj = soap.get("subjective", {})
    lines.append("## Subjective")
    lines.append("")
    lines.append(subj.get("chief_complaint", ""))
    if subj.get("patient_perspective"):
        lines.append(subj["patient_perspective"])
    lines.append("")

    obj = soap.get("objective", {})
    lines.append("## Mental Status Exam")
    lines.append("")
    lines.append(obj.get("mental_status_exam", "Not discussed in this session."))
    lines.append("")

    assess = soap.get("assessment", {})
    plan = soap.get("plan", {})
    lines.append("## Assessment & Plan")
    lines.append("")
    parts = []
    if assess.get("clinical_interpretation"):
        parts.append(assess["clinical_interpretation"])
    if assess.get("diagnosis") and assess["diagnosis"] != "Not stated":
        parts.append(f"Diagnosis: {assess['diagnosis']}.")
    if plan.get("treatment_plan"):
        parts.append(plan["treatment_plan"])
    if plan.get("medications") and plan["medications"] != "Not discussed in this session":
        parts.append(f"Medications: {plan['medications']}.")
    if plan.get("next_steps"):
        parts.append(plan["next_steps"])
    lines.append(" ".join(parts) if parts else "Not discussed in this session.")
    lines.append("")

    _append_transcript_summary(lines, notes)
    _append_full_transcript(lines, notes)

    return "\n".join(lines)


def _append_transcript_summary(lines: list, notes: Dict[str, Any]) -> None:
    """Append the Short Transcript Summary section."""
    lines.append("---")
    lines.append("")
    lines.append("# Short Transcript Summary")
    lines.append("")

    summary = notes.get("transcript_summary", {})

    key_themes = summary.get("key_themes", [])
    lines.append("- **Key themes:**")
    for theme in (key_themes or ["None identified"]):
        lines.append(f"  - {theme}")

    major_events = summary.get("major_events", [])
    lines.append("- **Major events:**")
    for event in (major_events or ["None identified"]):
        lines.append(f"  - {event}")

    lines.append(f"- **Emotional tone:** {summary.get('emotional_tone', 'Not assessed')}")

    decisions = summary.get("decisions_made", [])
    lines.append("- **Decisions made:**")
    for decision in (decisions or ["None identified"]):
        lines.append(f"  - {decision}")
    lines.append("")


def _append_full_transcript(lines: list, notes: Dict[str, Any]) -> None:
    """Append the Full Transcript section (raw + timestamps)."""
    transcript_text = notes.get("_transcript_text", "")
    transcript_segments = notes.get("_transcript_segments", [])

    lines.append("---")
    lines.append("")
    lines.append("# Full Transcript")
    lines.append("")

    if transcript_segments:
        for seg in transcript_segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "").strip()
            if text:
                lines.append(f"[{_fmt_ts(start)} - {_fmt_ts(end)}] {text}")
        lines.append("")
    elif transcript_text:
        lines.append(transcript_text)
        lines.append("")
    else:
        lines.append("*No transcript available.*")
        lines.append("")


def _fmt_ts(seconds: float) -> str:
    """Format seconds as MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"
