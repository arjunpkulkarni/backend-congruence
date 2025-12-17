"""
Therapist Notes Generation Service

This module generates comprehensive therapist notes from session transcripts
and analysis data using a separate OpenAI API key.
"""

import json
from typing import Any, Dict, List, Optional


# ==============================================================================
# CONFIGURATION - ADD YOUR NOTES API KEY HERE
# ==============================================================================
NOTES_API_KEY = "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"  # ← ADD YOUR OPENAI API KEY HERE
# ==============================================================================


def _get_notes_client():
    """
    Initialize OpenAI client for notes generation using separate API key.
    Returns (client, model) or (None, None) if unavailable.
    """
    api_key = NOTES_API_KEY
    if not api_key or not api_key.strip():
        return None, None
    
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None, None
    
    client = OpenAI(api_key=api_key.strip())
    model = "gpt-4o"  # Using GPT-4 for higher quality notes
    return client, model


def generate_therapist_notes(
    transcript_text: str,
    transcript_segments: Optional[List[Dict[str, Any]]] = None,
    session_summary: Optional[Dict[str, Any]] = None,
    patient_id: Optional[str] = None,
) -> Optional[str]:
    """
    Generate comprehensive therapist notes from session transcript and analysis.
    
    Args:
        transcript_text: Full transcript text
        transcript_segments: List of transcript segments with timestamps
        session_summary: Session summary with congruence metrics and emotion analysis
        patient_id: Patient identifier
    
    Returns:
        Formatted therapist notes as a string, or None if generation fails
    """
    client, model = _get_notes_client()
    
    if client is None or model is None:
        # API key not configured - return None silently
        return None
    
    if not transcript_text or not transcript_text.strip():
        return None
    
    # Build context from session summary
    context_parts = []
    
    if patient_id:
        context_parts.append(f"Patient ID: {patient_id}")
    
    if session_summary:
        duration = session_summary.get("duration", 0)
        congruence = session_summary.get("overall_congruence", 0)
        num_incongruent = session_summary.get("metrics", {}).get("num_incongruent_segments", 0)
        
        context_parts.append(f"Session Duration: {duration:.1f} seconds")
        context_parts.append(f"Overall Congruence Score: {congruence:.2f}")
        context_parts.append(f"Incongruent Moments: {num_incongruent}")
        
        # Add incongruent moments summary
        incongruent_moments = session_summary.get("incongruent_moments", [])
        if incongruent_moments:
            context_parts.append("\nKey Incongruent Moments:")
            for i, moment in enumerate(incongruent_moments[:5], 1):  # Top 5
                start = moment.get("start", 0)
                end = moment.get("end", 0)
                reason = moment.get("reason", "")
                context_parts.append(f"  {i}. [{start:.1f}s - {end:.1f}s]: {reason}")
        
        # Add emotion distribution summary
        emotion_dist = session_summary.get("emotion_distribution", {})
        if emotion_dist:
            context_parts.append("\nEmotion Distribution:")
            for modality in ["text", "face", "audio"]:
                if modality in emotion_dist:
                    emotions = emotion_dist[modality]
                    # Get top 3 emotions
                    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                    emotion_str = ", ".join([f"{e}: {v:.2f}" for e, v in sorted_emotions])
                    context_parts.append(f"  {modality.capitalize()}: {emotion_str}")
    
    context = "\n".join(context_parts)
    
    # System prompt for therapist notes
    system_prompt = """You are an experienced clinical psychologist assistant specializing in therapy session analysis.

Your task is to generate comprehensive, professional therapist notes from session transcripts and emotional analysis data.

The notes should include:

1. **SESSION OVERVIEW**
   - Brief summary of session duration and overall tone
   - Patient engagement and participation level

2. **KEY THEMES & TOPICS**
   - Main topics discussed during the session
   - Recurring themes or patterns
   - Important statements or revelations

3. **EMOTIONAL ANALYSIS**
   - Predominant emotions observed (from transcript, facial expressions, and vocal tone)
   - Emotional shifts or notable changes during the session
   - Incongruent moments where verbal and non-verbal communication didn't align

4. **CLINICAL OBSERVATIONS**
   - Notable behavioral patterns
   - Potential areas of concern
   - Strengths and coping mechanisms observed

5. **RECOMMENDATIONS**
   - Topics to explore in future sessions
   - Therapeutic interventions to consider
   - Follow-up actions

**IMPORTANT GUIDELINES:**
- Write in professional, clinical language suitable for medical records
- Be objective and evidence-based in observations
- Maintain patient confidentiality and dignity
- Use precise terminology appropriate for mental health professionals
- Include timestamps for significant moments when relevant
- Be concise but comprehensive (aim for 500-800 words)
- If speakers are identified (Therapist/Client), note interaction dynamics

Format the notes clearly with markdown headers and bullet points for readability."""

    user_content = f"""Generate therapist notes for this session.

SESSION CONTEXT:
{context}

FULL TRANSCRIPT:
{transcript_text}

Please generate comprehensive therapist notes following the format specified."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,  # Lower temperature for more consistent, professional output
            max_tokens=2000,  # Allow for comprehensive notes
        )
        
        notes = response.choices[0].message.content
        return notes.strip() if notes else None
        
    except Exception as e:
        # Log error but don't fail the pipeline
        print(f"Warning: Failed to generate therapist notes: {e}")
        return None


def save_therapist_notes(
    notes: Optional[str],
    output_path: str,
) -> bool:
    """
    Save therapist notes to a file.
    
    Args:
        notes: Generated notes text
        output_path: Path to save the notes file
    
    Returns:
        True if successful, False otherwise
    """
    if not notes:
        return False
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(notes)
        return True
    except Exception:
        return False



