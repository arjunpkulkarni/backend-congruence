import json
import logging
import os
from typing import Any, Dict, List, Optional

from app.services.llm import analyze_text_emotion_with_llm

logger = logging.getLogger("emotion_api.notes")


def _get_notes_client():
    """
    Get OpenAI client with hardcoded API key for notes generation.
    Returns (client, model) or (None, None) if unavailable.
    """
    api_key = os.getenv("NOTES_API_KEY")
    if not api_key or not api_key.strip():
        return None, None

    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None, None

    client = OpenAI(api_key=api_key.strip())
    model = "gpt-4o"
    return client, model


def generate_therapist_notes(
    transcript_text: str,
    transcript_segments: Optional[List[Dict[str, Any]]] = None,
    session_summary: Optional[Dict[str, Any]] = None,
    patient_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:

    logger.info("Starting therapist notes generation for patient_id=%s", patient_id)

    notes_client, notes_model = _get_notes_client()

    if notes_client is None or notes_model is None:
        logger.warning("Notes OpenAI client not available")
        return None
    
    if not transcript_text or not transcript_text.strip():
        logger.warning("Empty transcript provided")
        return None

    logger.info("Step 1: Analyzing transcript with LLM for emotional content and speaker diarization...")

    # FIRST CALL: Analyze the transcript with LLM
    llm_analysis = analyze_text_emotion_with_llm(
        text=transcript_text,
        model=None,  # Use default model
        ensemble_size=1,
        temperature=0.2,
    )

    if llm_analysis:
        logger.info("LLM transcript analysis completed successfully")
        logger.info("  - Emotion distribution: %s", llm_analysis.get("emotion_distribution", {}))
        logger.info("  - Valence: %.3f", llm_analysis.get("valence", 0.0))
        logger.info("  - Arousal: %.3f", llm_analysis.get("arousal", 0.0))
        logger.info("  - Style: %s", llm_analysis.get("style", "unknown"))

        if "speakers" in llm_analysis:
            logger.info("  - Detected %d speakers", len(llm_analysis["speakers"]))
            for i, speaker in enumerate(llm_analysis.get("speakers", [])[:3]):  # Log first 3
                logger.debug("    Speaker %d: %s", i + 1,
                            speaker.get("speaker", "unknown"))

        if "incongruence_reason" in llm_analysis:
            logger.info("  - Incongruence detected: %s", llm_analysis["incongruence_reason"])
    else:
        logger.warning("LLM transcript analysis failed or returned None")

    context_parts = []

    if patient_id:
        context_parts.append(f"Patient ID: {patient_id}")
    
    # Add LLM analysis results to context
    if llm_analysis:
        context_parts.append("\n## LLM Transcript Analysis:")

        # Add overall emotion distribution
        emotions = llm_analysis.get("emotion_distribution", {})
        if emotions:
            sorted_emotions = sorted(emotions.items(),
                                   key=lambda x: x[1], reverse=True)[:3]
            emotion_str = ", ".join([f"{e}: {v:.2f}"
                                   for e, v in sorted_emotions])
            context_parts.append(f"Overall Emotions: {emotion_str}")

        # Add valence/arousal/style
        context_parts.append(
            f"Valence: {llm_analysis.get('valence', 0.0):.3f} (range: -1 to +1)")
        context_parts.append(
            f"Arousal: {llm_analysis.get('arousal', 0.0):.3f} (range: 0 to 1)")
        context_parts.append(
            f"Communication Style: {llm_analysis.get('style', 'unknown')}")

        # Add speaker information if available
        speakers = llm_analysis.get("speakers", [])
        if speakers:
            context_parts.append(f"\nDetected {len(speakers)} speaker(s):")
            for speaker in speakers:
                speaker_label = speaker.get("speaker", "Unknown")
                speaker_text_preview = speaker.get("text", "")[:100]
                context_parts.append(f"  - {speaker_label}: \"{speaker_text_preview}...\"")

                # Add speaker-specific emotions if available
                speaker_emotions = speaker.get("emotion_distribution", {})
                if speaker_emotions:
                    sorted_sp_emotions = sorted(
                        speaker_emotions.items(),
                        key=lambda x: x[1], reverse=True)[:2]
                    sp_emotion_str = ", ".join(
                        [f"{e}: {v:.2f}" for e, v in sorted_sp_emotions])
                    context_parts.append(f"    Emotions: {sp_emotion_str}")

        # Add incongruence if detected
        if "incongruence_reason" in llm_analysis:
            context_parts.append(
                f"\nIncongruence Note: {llm_analysis['incongruence_reason']}")

    if session_summary:
        duration = session_summary.get("duration", 0)
        congruence = session_summary.get("overall_congruence", 0)
        num_incongruent = session_summary.get("metrics", {}).get("num_incongruent_segments", 0)

        context_parts.append(f"Session Duration: {duration:.1f} seconds")
        context_parts.append(
            f"Overall Congruence Score: {congruence:.2f}")
        context_parts.append(f"Incongruent Moments: {num_incongruent}")

        # Add incongruent moments summary
        incongruent_moments = session_summary.get("incongruent_moments", [])
        if incongruent_moments:
            context_parts.append("\nKey Incongruent Moments:")
            for i, moment in enumerate(incongruent_moments[:5], 1):  # Top 5
                start = moment.get("start", 0)
                end = moment.get("end", 0)
                reason = moment.get("reason", "")
                context_parts.append(
                    f"  {i}. [{start:.1f}s - {end:.1f}s]: {reason}")

        emotion_dist = session_summary.get("emotion_distribution", {})
        if emotion_dist:
            context_parts.append("\nEmotion Distribution:")
            for modality in ["text", "face", "audio"]:
                if modality in emotion_dist:
                    emotions = emotion_dist[modality]
                    # Get top 3 emotions
                    sorted_emotions = sorted(
                        emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                    emotion_str = ", ".join(
                        [f"{e}: {v:.2f}" for e, v in sorted_emotions])
                    context_parts.append(
                        f"  {modality.capitalize()}: {emotion_str}")

    context = "\n".join(context_parts)
    logger.info("Step 2: Generating clinical therapist notes with enriched context...")
    logger.debug("Context includes %d lines of analysis data",
                 len(context_parts))
    
    system_prompt = """You are an experienced clinical documentation assistant \
for licensed mental health clinicians. Your job is to transform therapy session \
transcripts PLUS any provided emotional/behavioral signals (e.g., vocal affect \
markers, facial affect probabilities, arousal/valence trends) into clinically \
useful progress notes that are objective, evidence-linked, and actionable.

ROLE & SCOPE (IMPORTANT):
- You do not diagnose unless a diagnosis is explicitly provided in the input.
- You do not provide medical advice to the client; you generate documentation for clinician use.
- You use a trauma-informed, culturally humble, non-stigmatizing style.
- You do not invent facts. If evidence is missing, say "insufficient evidence".

CORE CLINICAL RESTRAINT RULE:
If a therapist could reasonably say "this was just joking," you are NOT allowed to assign motive, discomfort, or pathology. Describe behavior only, never interpret intent.

MICRO-LANGUAGE PRECISION:
- Use "characterized by" or "consistent with" instead of "indicating" (avoids inference)
- Never use "apparent intent", "seeming to", "suggesting intent" (no intent assignments)
- For engagement: use behavioral descriptions like "oriented toward playful, non-literal speech" or "verbally responsive with joking content" instead of interpretive labels like "playful"

CLINICAL VALUE REQUIREMENTS (DO THIS OR YOUR OUTPUT IS WRONG):
1) Evidence anchoring: Every key theme, concern, strength, shift, and \
incongruence must be supported by (a) a short quote or paraphrase from \
transcript AND (b) a timestamp or time-range if available.
2) Clinical formulations: Provide brief hypotheses using structured language \
(e.g., "may suggest", "consistent with", "could reflect"), and list alternative \
explanations when appropriate. Do not present hypotheses as facts. DO NOT assign \
motives, internal states, or intentions unless explicitly stated by the client. \
Describe observable behaviors only.
3) Risk & safety: Always scan for self-harm, suicidality, violence, abuse, \
substance risk, severe impairment. If not present, explicitly state "No risk \
indicators identified in provided data". If ambiguous, state what is missing.
4) Functional impact: Only include functional impact if it is explicitly \
evidenced in transcript or provided signals. If not evidenced, state \
"insufficient evidence to assess functional impact" (do NOT infer).
5) Interventions: Extract what the therapist actually did (e.g., reflections, \
CBT reframes, MI, grounding, psychoeducation). If not present, say "Therapist \
interventions not clearly evidenced".
6) Next steps: Recommend ONLY concrete, evidence-based next steps. NO generic \
therapeutic interventions (e.g., "build rapport", "explore feelings", "use \
motivational interviewing"). Only specific actions tied to observed patterns. \
For brief sessions, provide maximum 1 concrete next step.

MULTI-MODAL EMOTION RULES:
- If emotional analysis data is provided, integrate it; if absent, rely only on transcript.
- Distinguish verbal content from observed affect (vocal/facial) and note congruence ONLY when both are available and clearly interpretable.
- INCONGRUENCE STRICT RULES:
  * NEVER label incongruence for playful, joking, or explicitly non-literal content
  * Playful joking + positive affect = CONGRUENT (not incongruent)
  * Only flag incongruence when there is genuine mismatch between serious content and contradictory affect
  * If incongruence data is not available or not clinically significant, use empty array []
- Incongruence moments must include: timestamp, exact verbal line, nonverbal signal description, clinical significance + alternative explanations.

CONFIDENTIALITY & MINIMUM NECESSARY:
- Remove or mask identifying details (names, addresses, employers, phone numbers). Use [CLIENT], [THERAPIST], [PARTNER], etc.
- Do not include gratuitous detail unrelated to clinical care.

OUTPUT FORMAT (STRICT):
- You MUST return ONLY valid JSON with the exact schema below.
- Do not include markdown, commentary, or extra keys.
- Use double quotes for all strings. No trailing commas.

CRITICAL: VALID JSON + REQUIRED KEYS
- You MUST include every top-level key exactly as in the schema (even if data is insufficient).
- When a section is not supported by evidence OR disallowed by data sufficiency rules:
  - Use empty arrays [] for list fields.
  - Use "insufficient evidence due to session brevity" (or "not indicated in provided data") for string fields.
  - Do NOT fabricate content to fill required keys.

DATA SUFFICIENCY RULE (MANDATORY):
Before generating content, assess session duration and evidence density.

If session duration < 2 minutes OR fewer than 3 distinct client turns:
- Content caps:
  - key_themes: max 1 theme (descriptive only, no interpretations)
  - recommendations: max 1 concrete next step only
- You MUST set these sections to EMPTY for brief sessions:
  - emotional_analysis.predominant_emotions = []
  - emotional_analysis.emotional_shifts = []
  - emotional_analysis.incongruence_moments = []
  - clinical_observations.strengths_and_coping = []
  - recommendations.interventions = [] (no therapeutic interventions for insufficient data)
- Add note: "Insufficient evidence due to session brevity" in relevant sections
- interaction_dynamics: minimal, descriptive content only
- If therapist speech is absent: "Therapist speech not present in provided data; insufficient evidence due to session brevity"

CONSTRAINTS:
- key_themes: max 1 when duration < 2 minutes; otherwise max 3
- emotional_analysis: EMPTY arrays for sessions < 2 minutes
- clinical_observations.areas_of_concern: max 1 and ONLY if risk/impairment is evidenced
- recommendations.future_topics: max 1 for brief sessions (concrete, behavioral focus)
- recommendations.interventions: EMPTY array for sessions < 2 minutes (insufficient evidence)
- recommendations.follow_up_actions: max 1 for brief sessions (concrete steps only)

ANTI-PATHOLOGIZING LANGUAGE RULE:
Use descriptive, non-judgmental language. Avoid loaded clinical terms that assign intent or pathology.

BANNED TERMS (use descriptive alternatives):
- "deceptive" → "non-literal" or "playful"
- "manipulative" → "indirect communication style"
- "testing boundaries" → "exploring interaction patterns"
- "resistance" → "hesitant engagement"
- "denial" → "does not acknowledge"
- "indicating" → "characterized by" or "consistent with"
- "apparent intent" → delete phrase entirely
- "seeming to" → use direct behavioral description
- "suggesting intent" → describe observable behavior only
- "identity disturbance"
- "reality testing"
- "authentic communication" challenges

CORE RULE: If a therapist could reasonably say "this was just joking," you are not allowed to assign motive, discomfort, or pathology.

Use concrete behavioral descriptions only. No interpretive labels about internal states.

QUALITY CHECK BEFORE YOU OUTPUT:
- Did you include timestamps wherever possible?
- Did you avoid diagnosis unless provided?
- Did you avoid invented content?
- Did you avoid inferring functional impact when not evidenced?
- Did each theme have evidence?

JSON SCHEMA:
{
  "session_overview": {
    "summary": "2-3 sentence clinical summary focusing on presenting concerns + session work + outcome",
    "duration": "e.g., 50 minutes (or 'unknown')",
    "engagement_level": "behavioral indicators only (e.g., 'verbally responsive', 'oriented toward non-literal content', 'minimal verbal output')",
    "overall_tone": "brief behavioral tone description (avoid interpretive labels)"
  },
  "key_themes": [
    {
      "theme": "Theme name",
      "description": "Clinically framed description of the theme and its functional impact",
      "evidence": ["[timestamp] short quote/paraphrase", "[timestamp] short quote/paraphrase"]
    }
  ],
  "emotional_analysis": {
    "predominant_emotions": [
      {
        "emotion": "Emotion label",
        "source": "text|facial|vocal|mixed",
        "intensity": "low|medium|high",
        "context": "What was happening + evidence"
      }
    ],
    "emotional_shifts": [
      {
        "timestamp": "Time in session",
        "from_emotion": "Prior state",
        "to_emotion": "New state",
        "trigger": "Transcript-linked trigger"
      }
    ],
    "incongruence_moments": [
      {
        "timestamp": "Time in session",
        "verbal": "Exact quote or close paraphrase",
        "nonverbal": "Observed affect/tone markers",
        "significance": "Why this might matter clinically + alternative explanations"
      }
    ]
  },
  "clinical_observations": {
    "behavioral_patterns": ["Observed pattern + evidence pointer"],
    "areas_of_concern": ["Concern + functional impact + evidence pointer"],
    "strengths_and_coping": ["Strength/coping strategy + evidence pointer"]
  },
  "risk_assessment": {
    "suicide_self_harm": {
      "indicators": "present|absent|unclear",
      "evidence": "Evidence or 'not indicated in provided data'",
      "protective_factors": ["If present, list"],
      "recommended_actions": ["If present/unclear: clinician actions"]
    },
    "harm_to_others": {
      "indicators": "present|absent|unclear",
      "evidence": "Evidence or 'not indicated in provided data'",
      "recommended_actions": ["If present/unclear"]
    },
    "substance_use": {
      "indicators": "present|absent|unclear",
      "evidence": "Evidence or 'not indicated in provided data'",
      "recommended_actions": ["If present/unclear"]
    }
  },
  "recommendations": {
    "future_topics": ["Concrete next topics tied to observed behavior (e.g., 'Establish baseline communication style', 'Clarify treatment goals')"],
    "interventions": ["ONLY if sufficient evidence exists - specific techniques matched to clear patterns. For brief sessions: use empty array []"],
    "follow_up_actions": ["Concrete, actionable steps (e.g., 'Schedule longer session', 'Obtain collateral information', 'Complete intake assessment')"]
  },
  "interaction_dynamics": {
    "therapist_approach": "What therapist did (techniques), with evidence if possible",
    "client_responsiveness": "Behavioral description of client responses (e.g., 'verbally responsive', 'minimal engagement', 'oriented toward joking content'), with evidence",
    "rapport_quality": "Brief alliance assessment grounded in observed interaction behaviors"
  }
}

Prioritize accuracy over completeness.

Remember: output ONLY JSON matching the schema. If transcript lacks timestamps, infer approximate sequence (early/mid/late) and state "timestamp unavailable".
"""

    # Prepare emotional/multimodal data summary
    emotion_data_summary = []
    if llm_analysis:
        emotion_data_summary.append("LLM Transcript Analysis Results:")
        emotion_data_summary.append(f"- Emotion distribution: {llm_analysis.get('emotion_distribution', {})}")
        emotion_data_summary.append(f"- Valence: {llm_analysis.get('valence', 0.0):.3f}")
        emotion_data_summary.append(f"- Arousal: {llm_analysis.get('arousal', 0.0):.3f}")
        emotion_data_summary.append(f"- Communication style: {llm_analysis.get('style', 'unknown')}")
        if "speakers" in llm_analysis:
            emotion_data_summary.append(
                f"- Speakers detected: {len(llm_analysis['speakers'])}")
        if "incongruence_reason" in llm_analysis:
            emotion_data_summary.append(
                f"- Incongruence flagged: {llm_analysis['incongruence_reason']}")

    if session_summary:
        emotion_data_summary.append("\nSession Emotion Distribution:")
        emotion_dist = session_summary.get("emotion_distribution", {})
        for modality in ["text", "face", "audio"]:
            if modality in emotion_dist:
                emotion_data_summary.append(
                    f"- {modality}: {emotion_dist[modality]}")

    emotion_data_text = ("\n".join(emotion_data_summary)
                         if emotion_data_summary else "None provided")
    
    # Determine duration
    duration_str = "unknown"
    if session_summary and "duration" in session_summary:
        duration_seconds = session_summary.get("duration", 0)
        duration_str = (f"{duration_seconds:.0f} seconds "
                       f"(~{duration_seconds/60:.1f} minutes)")

    # Check for timestamps and speakers
    has_timestamps = bool(transcript_segments)
    has_speakers = bool(llm_analysis and llm_analysis.get("speakers"))

    emotion_types = []
    if llm_analysis:
        emotion_types.append("LLM text analysis")
    if session_summary and "emotion_distribution" in session_summary:
        if "face" in session_summary["emotion_distribution"]:
            emotion_types.append("facial affect")
        if "audio" in session_summary["emotion_distribution"]:
            emotion_types.append("vocal affect")
    
    user_content = f"""Generate clinician-facing therapy progress notes using the system instructions.

INPUTS PROVIDED:
- Session duration: {duration_str}
- Timestamps available: {"yes" if has_timestamps else "no"}
- Speakers labeled: {"yes" if has_speakers else "no"}
- Emotional signals provided: {', '.join(emotion_types) if emotion_types else 'none'}

SESSION CONTEXT:
{context}

TRANSCRIPT:
{transcript_text}

EMOTIONAL / MULTIMODAL DATA (if any):
{emotion_data_text}

CONSTRAINTS:
- If something is not in the transcript or data, write "insufficient evidence".
- Mask identifying details.
- Include evidence pointers for every key claim.

Return ONLY valid JSON."""

    try:
        logger.info(
            "Calling OpenAI API (model: %s) for therapist notes generation...",
            notes_model)
        logger.debug("System prompt: %d chars, User content: %d chars",
                     len(system_prompt), len(user_content))
        
        response = notes_client.chat.completions.create(
            model=notes_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,  # Lower temperature for more consistent, professional output
            max_tokens=2500,  # Allow for comprehensive structured notes
            response_format={"type": "json_object"}  # Force JSON response
        )
        
        logger.info("OpenAI API call successful")
        
        notes_text = response.choices[0].message.content
        if not notes_text:
            logger.warning("OpenAI returned empty notes content")
            return None
        
        # Parse the JSON response
        try:
            notes_dict = json.loads(notes_text)
            logger.info(
                "Therapist notes generated successfully "
                "(structured JSON with %d top-level keys)",
                len(notes_dict))
            logger.debug("Notes sections: %s", list(notes_dict.keys()))
            return notes_dict
        except json.JSONDecodeError as e:
            logger.error("Failed to parse notes JSON: %s", e)
            logger.debug("Raw response: %s", notes_text[:500])
            # Return as fallback plain text in a structured format
            return {
                "error": "Failed to parse structured notes",
                "raw_content": notes_text,
                "format": "fallback"
            }
        
    except Exception as e:
        logger.exception("Failed to generate therapist notes: %s", e)
        return None


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
        
        logger.info(
            "Therapist notes saved successfully "
            "(markdown: %d bytes, json: %d bytes)",
            len(markdown_content), len(json.dumps(notes)))
        return True
    except Exception as e:
        logger.exception("Failed to save therapist notes: %s", e)
        return False


def _convert_notes_to_markdown(notes: Dict[str, Any]) -> str:
    """
    Convert structured notes dictionary to readable markdown format.
    """
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
            lines.append(
                f"**Engagement Level:** {overview['engagement_level']}")
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
                lines.append(
                    f"**{emotion.get('emotion', 'Unknown')}** "
                    f"({emotion.get('source', 'unknown')} - "
                    f"{emotion.get('intensity', 'unknown')} intensity)")
                if "context" in emotion:
                    lines.append(f"- {emotion['context']}")
                lines.append("")
        
        if "emotional_shifts" in ea and ea["emotional_shifts"]:
            lines.append("### Emotional Shifts")
            lines.append("")
            for shift in ea["emotional_shifts"]:
                lines.append(
                    f"**[{shift.get('timestamp', 'Unknown time')}]** "
                    f"{shift.get('from_emotion', '?')} → "
                    f"{shift.get('to_emotion', '?')}")
                if "trigger" in shift:
                    lines.append(f"- Trigger: {shift['trigger']}")
                lines.append("")
        
        if "incongruence_moments" in ea and ea["incongruence_moments"]:
            lines.append("### Incongruence Moments")
            lines.append("")
            for moment in ea["incongruence_moments"]:
                lines.append(
                    f"**[{moment.get('timestamp', 'Unknown time')}]**")
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

