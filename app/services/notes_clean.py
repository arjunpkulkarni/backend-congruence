import json
import logging
import os
from typing import Any, Dict, List, Optional

from app.services.llm import analyze_text_emotion_with_llm
from app.services.prompts import (
    PROMPT_1_EXTRACTION,
    PROMPT_2_EMOTION,
    PROMPT_3_CLINICAL,
    PROMPT_4_RECOMMENDATIONS,
    build_user_message_step1,
    build_user_message_step2,
    build_user_message_step3,
    build_user_message_step4,
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
    """
    Generate therapist notes using sequential 4-prompt pipeline.
    
    Args:
        transcript_text: Full transcript text
        transcript_segments: Optional segments with timestamps
        session_summary: Optional session summary with emotion data
        patient_id: Optional patient identifier
        use_sequential: Use sequential pipeline (always True for now)
    
    Returns:
        Structured therapist notes dict or None if failed
    """
    return _generate_notes_sequential(
        transcript_text=transcript_text,
        transcript_segments=transcript_segments,
        session_summary=session_summary,
        patient_id=patient_id,
    )


def _generate_notes_sequential(
    transcript_text: str,
    transcript_segments: Optional[List[Dict[str, Any]]] = None,
    session_summary: Optional[Dict[str, Any]] = None,
    patient_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Sequential 4-prompt pipeline for therapist notes generation.
    
    Pipeline:
    1. Extract objective facts (timestamps, quotes, topics)
    2. Analyze emotions (patterns, incongruence)
    3. Synthesize clinical observations (behaviors, risks)
    4. Generate recommendations (summary, next steps)
    """
    logger.info("Starting sequential 4-prompt pipeline for patient_id=%s", patient_id)

    notes_client, notes_model = _get_notes_client()

    if notes_client is None or notes_model is None:
        logger.warning("Notes OpenAI client not available")
        return None
    
    if not transcript_text or not transcript_text.strip():
        logger.warning("Empty transcript provided")
        return None

    # Pre-process: Optional LLM analysis for text emotions
    logger.info("Pre-processing: Analyzing transcript with LLM for emotional content...")
    
    llm_analysis = analyze_text_emotion_with_llm(
        text=transcript_text,
        model=None,
        ensemble_size=1,
        temperature=0.2,
    )

    if llm_analysis:
        logger.info("LLM transcript analysis completed")
        logger.info("  - Emotions: %s", llm_analysis.get("emotion_distribution", {}))
        logger.info("  - Valence: %.3f, Arousal: %.3f", 
                   llm_analysis.get("valence", 0.0),
                   llm_analysis.get("arousal", 0.0))
    else:
        logger.warning("LLM transcript analysis failed")

    # Build emotion data summary
    emotion_data_summary = _build_emotion_data_summary(llm_analysis, session_summary)
    
    # Determine session metadata
    duration_str = "unknown"
    if session_summary and "duration" in session_summary:
        duration_seconds = session_summary.get("duration", 0)
        duration_str = f"{duration_seconds:.0f} seconds (~{duration_seconds/60:.1f} minutes)"
    
    has_timestamps = bool(transcript_segments)
    
    # =================================================================
    # STEP 1: DATA EXTRACTION
    # =================================================================
    logger.info("Pipeline Step 1/4: Extracting factual data...")
    
    step1_user_msg = build_user_message_step1(
        transcript_text=transcript_text,
        duration_str=duration_str,
        has_timestamps=has_timestamps,
        emotion_data_summary=emotion_data_summary,
    )
    
    step1_output = _call_llm_step(
        notes_client,
        notes_model,
        system_prompt=PROMPT_1_EXTRACTION,
        user_message=step1_user_msg,
        step_name="Step 1: Data Extraction",
        temperature=0.2,
    )
    
    if not step1_output:
        logger.error("Step 1 (Data Extraction) failed")
        return None
    
    logger.info("Step 1 complete: %d topics, %d emotional datapoints",
                len(step1_output.get("key_topics", [])),
                len(step1_output.get("emotional_datapoints", [])))
    
    # =================================================================
    # STEP 2: EMOTIONAL ANALYSIS
    # =================================================================
    logger.info("Pipeline Step 2/4: Analyzing emotional patterns...")
    
    step2_user_msg = build_user_message_step2(
        step1_output=step1_output,
        emotion_data_summary=emotion_data_summary,
    )
    
    step2_output = _call_llm_step(
        notes_client,
        notes_model,
        system_prompt=PROMPT_2_EMOTION,
        user_message=step2_user_msg,
        step_name="Step 2: Emotional Analysis",
        temperature=0.3,
    )
    
    if not step2_output:
        logger.error("Step 2 (Emotional Analysis) failed")
        return None
    
    logger.info("Step 2 complete: %d predominant emotions, %d incongruence moments",
                len(step2_output.get("predominant_emotions", [])),
                len(step2_output.get("incongruence_analysis", [])))
    
    # =================================================================
    # STEP 3: CLINICAL SYNTHESIS
    # =================================================================
    logger.info("Pipeline Step 3/4: Generating clinical observations...")
    
    step3_user_msg = build_user_message_step3(
        step1_output=step1_output,
        step2_output=step2_output,
    )
    
    step3_output = _call_llm_step(
        notes_client,
        notes_model,
        system_prompt=PROMPT_3_CLINICAL,
        user_message=step3_user_msg,
        step_name="Step 3: Clinical Synthesis",
        temperature=0.3,
    )
    
    if not step3_output:
        logger.error("Step 3 (Clinical Synthesis) failed")
        return None
    
    risk = step3_output.get("risk_assessment", {})
    risk_suicide = risk.get("suicide_self_harm", {}).get("indicators", "unknown")
    logger.info("Step 3 complete: %d behavioral patterns, %d concerns, risk_suicide=%s",
                len(step3_output.get("behavioral_patterns", [])),
                len(step3_output.get("areas_of_concern", [])),
                risk_suicide)
    
    # =================================================================
    # STEP 4: RECOMMENDATIONS & FINAL SYNTHESIS
    # =================================================================
    logger.info("Pipeline Step 4/4: Compiling recommendations...")
    
    step4_user_msg = build_user_message_step4(
        step1_output=step1_output,
        step2_output=step2_output,
        step3_output=step3_output,
    )
    
    step4_output = _call_llm_step(
        notes_client,
        notes_model,
        system_prompt=PROMPT_4_RECOMMENDATIONS,
        user_message=step4_user_msg,
        step_name="Step 4: Recommendations",
        temperature=0.3,
    )
    
    if not step4_output:
        logger.error("Step 4 (Recommendations) failed")
        return None
    
    logger.info("Step 4 complete: %d themes, %d follow-up actions",
                len(step4_output.get("key_themes", [])),
                len(step4_output.get("recommendations", {}).get("follow_up_actions", [])))
    
    # =================================================================
    # MERGE OUTPUTS INTO FINAL NOTES
    # =================================================================
    final_notes = _merge_pipeline_outputs(step1_output, step2_output, step3_output, step4_output)
    
    logger.info("Sequential pipeline completed successfully")
    return final_notes


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


def _merge_pipeline_outputs(
    step1: Dict[str, Any],
    step2: Dict[str, Any],
    step3: Dict[str, Any],
    step4: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge outputs from 4-step pipeline into final notes structure."""
    final = {
        "session_overview": step4.get("session_overview", {}),
        "key_themes": step4.get("key_themes", []),
        "emotional_analysis": {
            "predominant_emotions": step2.get("predominant_emotions", []),
            "emotional_shifts": step2.get("emotional_shifts", []),
            "incongruence_moments": step2.get("incongruence_analysis", []),
        },
        "clinical_observations": {
            "behavioral_patterns": step3.get("behavioral_patterns", []),
            "areas_of_concern": step3.get("areas_of_concern", []),
            "strengths_and_coping": step3.get("strengths_and_coping", []),
        },
        "risk_assessment": step3.get("risk_assessment", {}),
        "recommendations": step4.get("recommendations", {}),
        "interaction_dynamics": step4.get("interaction_dynamics", {}),
    }
    
    return final


def save_therapist_notes(
    notes: Optional[Dict[str, Any]],
    output_path: str,
) -> bool:
    """
    Save therapist notes to a file.
    
    Args:
        notes: Generated notes dictionary (structured format)
        output_path: Path to save the notes file
    
    Returns:
        True if successful, False otherwise
    """
    if not notes:
        logger.warning("Cannot save therapist notes: notes content is empty")
        return False
    
    try:
        logger.info("Saving therapist notes to: %s", output_path)
        
        # Convert structured notes to readable markdown for file storage
        markdown_content = _convert_notes_to_markdown(notes)
        
        # Save both markdown and JSON versions
        md_path = (output_path.replace('.json', '.md')
                  if output_path.endswith('.json') else output_path)
        json_path = (output_path.replace('.md', '.json')
                    if output_path.endswith('.md')
                    else output_path.replace('.md', '') + '.json')
        
        # Save markdown version
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        # Save JSON version
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)
        
        logger.info("Therapist notes saved successfully")
        return True
    except Exception as e:
        logger.exception("Failed to save therapist notes: %s", e)
        return False


def _convert_notes_to_markdown(notes: Dict[str, Any]) -> str:
    """Convert structured notes dictionary to readable markdown format."""
    lines = ["# Therapist Session Notes", ""]
    
    # Handle error/fallback format
    if notes.get("format") == "fallback":
        lines.append("**Note:** This is a fallback format due to parsing issues.")
        lines.append("")
        lines.append(notes.get("raw_content", "No content available"))
        return "\n".join(lines)
    
    # Session Overview
    if "session_overview" in notes:
        overview = notes["session_overview"]
        lines.append("## Session Overview")
        lines.append("")
        if "summary" in overview:
            lines.append(overview["summary"])
            lines.append("")
        if "duration" in overview:
            lines.append(f"**Duration:** {overview['duration']}")
        if "engagement_level" in overview:
            lines.append(f"**Engagement Level:** {overview['engagement_level']}")
        if "overall_tone" in overview:
            lines.append(f"**Overall Tone:** {overview['overall_tone']}")
        lines.append("")
    
    # Key Themes
    if "key_themes" in notes and notes["key_themes"]:
        lines.append("## Key Themes & Topics")
        lines.append("")
        for i, theme in enumerate(notes["key_themes"], 1):
            lines.append(f"### {i}. {theme.get('theme', 'Unnamed Theme')}")
            lines.append("")
            if "description" in theme:
                lines.append(theme["description"])
                lines.append("")
            if "evidence" in theme and theme["evidence"]:
                lines.append("**Evidence:**")
                for evidence in theme["evidence"]:
                    lines.append(f"- {evidence}")
                lines.append("")
    
    # Emotional Analysis
    if "emotional_analysis" in notes:
        ea = notes["emotional_analysis"]
        lines.append("## Emotional Analysis")
        lines.append("")
        
        if "predominant_emotions" in ea and ea["predominant_emotions"]:
            lines.append("### Predominant Emotions")
            lines.append("")
            for emotion in ea["predominant_emotions"]:
                lines.append(f"**{emotion.get('emotion', 'Unknown')}** ({emotion.get('source', 'unknown')} - {emotion.get('intensity', 'unknown')} intensity)")
                if "context" in emotion:
                    lines.append(f"- {emotion['context']}")
                lines.append("")
        
        if "emotional_shifts" in ea and ea["emotional_shifts"]:
            lines.append("### Emotional Shifts")
            lines.append("")
            for shift in ea["emotional_shifts"]:
                lines.append(f"**[{shift.get('timestamp', 'Unknown time')}]** {shift.get('from_emotion', '?')} → {shift.get('to_emotion', '?')}")
                if "trigger" in shift:
                    lines.append(f"- Trigger: {shift['trigger']}")
                lines.append("")
        
        if "incongruence_moments" in ea and ea["incongruence_moments"]:
            lines.append("### Incongruence Moments")
            lines.append("")
            for moment in ea["incongruence_moments"]:
                lines.append(f"**[{moment.get('timestamp', 'Unknown time')}]**")
                if "verbal" in moment:
                    lines.append(f"- Verbal: {moment['verbal']}")
                if "nonverbal" in moment:
                    lines.append(f"- Non-verbal: {moment['nonverbal']}")
                if "significance" in moment:
                    lines.append(f"- Significance: {moment['significance']}")
                lines.append("")
    
    # Clinical Observations
    if "clinical_observations" in notes:
        co = notes["clinical_observations"]
        lines.append("## Clinical Observations")
        lines.append("")
        
        if "behavioral_patterns" in co and co["behavioral_patterns"]:
            lines.append("### Behavioral Patterns")
            for pattern in co["behavioral_patterns"]:
                lines.append(f"- {pattern}")
            lines.append("")
        
        if "areas_of_concern" in co and co["areas_of_concern"]:
            lines.append("### Areas of Concern")
            for concern in co["areas_of_concern"]:
                lines.append(f"- {concern}")
            lines.append("")
        
        if "strengths_and_coping" in co and co["strengths_and_coping"]:
            lines.append("### Strengths & Coping Mechanisms")
            for strength in co["strengths_and_coping"]:
                lines.append(f"- {strength}")
            lines.append("")
    
    # Risk Assessment
    if "risk_assessment" in notes:
        risk = notes["risk_assessment"]
        lines.append("## Risk Assessment")
        lines.append("")
        
        if "suicide_self_harm" in risk:
            ssh = risk["suicide_self_harm"]
            lines.append("### Suicide/Self-Harm Risk")
            lines.append(f"**Indicators:** {ssh.get('indicators', 'unclear')}")
            lines.append(f"**Evidence:** {ssh.get('evidence', 'none provided')}")
            if ssh.get("protective_factors"):
                lines.append("**Protective Factors:**")
                for factor in ssh["protective_factors"]:
                    lines.append(f"- {factor}")
            if ssh.get("recommended_actions"):
                lines.append("**Recommended Actions:**")
                for action in ssh["recommended_actions"]:
                    lines.append(f"- {action}")
            lines.append("")
        
        if "harm_to_others" in risk:
            hto = risk["harm_to_others"]
            lines.append("### Harm to Others Risk")
            lines.append(f"**Indicators:** {hto.get('indicators', 'unclear')}")
            lines.append(f"**Evidence:** {hto.get('evidence', 'none provided')}")
            if hto.get("recommended_actions"):
                lines.append("**Recommended Actions:**")
                for action in hto["recommended_actions"]:
                    lines.append(f"- {action}")
            lines.append("")
        
        if "substance_use" in risk:
            su = risk["substance_use"]
            lines.append("### Substance Use Risk")
            lines.append(f"**Indicators:** {su.get('indicators', 'unclear')}")
            lines.append(f"**Evidence:** {su.get('evidence', 'none provided')}")
            if su.get("recommended_actions"):
                lines.append("**Recommended Actions:**")
                for action in su["recommended_actions"]:
                    lines.append(f"- {action}")
            lines.append("")
    
    # Recommendations
    if "recommendations" in notes:
        rec = notes["recommendations"]
        lines.append("## Recommendations")
        lines.append("")
        
        if "future_topics" in rec and rec["future_topics"]:
            lines.append("### Future Topics to Explore")
            for topic in rec["future_topics"]:
                lines.append(f"- {topic}")
            lines.append("")
        
        if "interventions" in rec and rec["interventions"]:
            lines.append("### Therapeutic Interventions")
            for intervention in rec["interventions"]:
                lines.append(f"- {intervention}")
            lines.append("")
        
        if "follow_up_actions" in rec and rec["follow_up_actions"]:
            lines.append("### Follow-up Actions")
            for action in rec["follow_up_actions"]:
                lines.append(f"- {action}")
            lines.append("")
    
    # Interaction Dynamics
    if "interaction_dynamics" in notes:
        dynamics = notes["interaction_dynamics"]
        lines.append("## Interaction Dynamics")
        lines.append("")
        if "therapist_approach" in dynamics:
            lines.append(f"**Therapist Approach:** {dynamics['therapist_approach']}")
            lines.append("")
        if "client_responsiveness" in dynamics:
            lines.append(f"**Client Responsiveness:** {dynamics['client_responsiveness']}")
            lines.append("")
        if "rapport_quality" in dynamics:
            lines.append(f"**Rapport Quality:** {dynamics['rapport_quality']}")
            lines.append("")
    
    return "\n".join(lines)
