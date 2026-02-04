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
    Get OpenAI client with hardcoded API key for notes generation.
    Returns (client, model) or (None, None) if unavailable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
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
    use_sequential: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Generate therapist notes using sequential 4-prompt pipeline.
    
    Args:
        transcript_text: Full transcript text
        transcript_segments: Optional segments with timestamps
        session_summary: Optional session summary with emotion data
        patient_id: Optional patient identifier
        use_sequential: Use new sequential pipeline (default: True)
    
    Returns:
        Structured therapist notes dict or None if failed
    """
    if use_sequential:
        return _generate_notes_sequential(
            transcript_text=transcript_text,
            transcript_segments=transcript_segments,
            session_summary=session_summary,
            patient_id=patient_id,
        )
    else:
        logger.warning("Non-sequential mode not implemented, using sequential")
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

    # Build emotion data summary for all pipeline steps
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

CORE OUTPUT QUALITY RULE:
BE SPECIFIC. Use concrete quotes, timestamps, and quantitative data. Avoid vague statements like "various emotions" or "some concerns". Name specific emotions with their intensities and sources. Cite specific transcript content with timestamps. Report specific metrics when provided (e.g., "text valence: +0.25, facial valence: -0.40").

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
- If emotional analysis data is provided (emotion distributions, valence, arousal, incongruence scores), ACTIVELY INTEGRATE IT into your analysis
- When emotion data exists, report specific findings:
  * Name the predominant emotions detected and their sources (text/facial/vocal)
  * Note valence and arousal levels when provided
  * Describe mismatches between modalities when present
- Distinguish verbal content from observed affect (vocal/facial) and compare them explicitly
- INCONGRUENCE ANALYSIS REQUIREMENTS:
  * If incongruence metrics/moments are provided in the input data, analyze them specifically
  * Include timestamp, exact verbal content, specific nonverbal signals, and clinical interpretation
  * NEVER label incongruence for playful, joking, or explicitly non-literal content
  * Playful joking + positive affect = CONGRUENT (not incongruent)  
  * Only flag incongruence when there is genuine mismatch between serious content and contradictory affect
  * For each incongruence moment, provide alternative explanations (e.g., "could reflect emotional suppression, cultural display rules, or momentary distraction")
  * If incongruence data is available, include it - don't skip just because session is brief
  * If incongruence data is not available OR not clinically significant, use empty array []
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
Before generating content, assess BOTH session duration AND available data quality/completeness.

IMPORTANT: Duration thresholds are GUIDELINES, not absolute cutoffs. Always prioritize actual data availability over duration alone.

For brief sessions (< 2 minutes) OR < 3 distinct client turns:
- Apply extra scrutiny, but DO NOT automatically empty sections if quality data exists
- Check what multimodal data is actually provided (text analysis, facial affect, vocal affect, incongruence metrics)
- If emotional analysis data IS provided (emotion distributions, valence/arousal, incongruence scores):
  * USE IT - analyze the patterns you observe
  * Report specific emotions detected with their sources (text/facial/vocal)
  * Include incongruence moments if flagged in the data
  * Be concrete about what the data shows
- If emotional data is NOT provided or genuinely insufficient:
  * Then use empty arrays and "insufficient evidence" notes

Content guidelines for brief sessions:
- key_themes: max 2 themes when data supports it (focus on most salient patterns)
- emotional_analysis: 
  * predominant_emotions: include if emotion data provided (max 2-3 emotions)
  * emotional_shifts: include only if clear shifts are evidenced
  * incongruence_moments: ALWAYS include if incongruence data provided and clinically relevant
- clinical_observations:
  * behavioral_patterns: describe observed patterns (max 2)
  * strengths_and_coping: only if explicitly evidenced
- recommendations:
  * future_topics: max 2 concrete topics
  * interventions: empty array for brief sessions (insufficient evidence for treatment planning)
  * follow_up_actions: max 2 concrete actions
- interaction_dynamics: descriptive only, note limitations
- If therapist speech absent: "Therapist contributions not captured in provided data"

CONSTRAINTS:
- key_themes: max 2 for brief sessions; max 3-4 for longer sessions
- emotional_analysis: analyze available data, don't skip just due to duration
- clinical_observations.areas_of_concern: max 2 and ONLY if risk/impairment is evidenced
- recommendations.interventions: empty for sessions < 3 minutes (insufficient for treatment planning)
- NEVER fabricate data, but DO analyze data that exists

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
    "summary": "2-3 sentence SPECIFIC clinical summary: what client actually discussed (key topics with brief quotes), what emotional patterns were observed (cite specific emotions/incongruence if detected), and concrete next steps. Be factual and evidence-based.",
    "duration": "e.g., 50 minutes, 86 seconds, etc. (or 'unknown')",
    "engagement_level": "Specific behavioral indicators with evidence (e.g., 'Verbally engaged; provided unprompted self-disclosure about career and family', 'Minimal verbal output; single-word responses predominant')",
    "overall_tone": "Specific behavioral tone with supporting evidence (e.g., 'Serious verbal tone when discussing both positive (career) and challenging (family) topics; facial affect showed consistent sadness despite positive content')"
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
        "emotion": "Specific emotion label (e.g., sadness, anxiety, joy)",
        "source": "text|facial|vocal|mixed (be specific about which modality detected this)",
        "intensity": "low|medium|high (based on quantitative data if available)",
        "context": "SPECIFIC context: what client was discussing + supporting quote with timestamp"
      }
    ],
    "emotional_shifts": [
      {
        "timestamp": "Specific time in session (e.g., '32s', 'mid-session')",
        "from_emotion": "Prior emotional state (be specific)",
        "to_emotion": "New emotional state (be specific)",
        "trigger": "Specific transcript content or topic that triggered the shift + quote"
      }
    ],
    "incongruence_moments": [
      {
        "timestamp": "Specific time in session with range if available (e.g., '12-18s')",
        "verbal": "Exact quote showing what client said",
        "nonverbal": "Specific affect markers observed (e.g., 'facial: sadness 0.65, vocal: flat/low arousal', or 'text valence +0.3, facial valence -0.4')",
        "significance": "Specific clinical interpretation: what mismatch suggests (e.g., 'positive content about career paired with sadness in facial affect') + 2-3 alternative explanations (e.g., 'could reflect ambivalence, emotional suppression, or cultural display rules')"
      }
    ]
  },
  "clinical_observations": {
    "behavioral_patterns": ["Specific observed pattern + evidence with timestamp/quote (e.g., 'Client exhibited serious communication style while discussing career excitement [12-32s: \"I'm really excited for that\"], suggesting possible ambivalence or guarded optimism')"],
    "areas_of_concern": ["Specific concern + functional impact if evidenced + supporting evidence with timestamp/quote (e.g., 'Client mentioned past family difficulties [32-86s] but did not elaborate; may indicate avoidance or need for safety-building before disclosure')"],
    "strengths_and_coping": ["Specific strength/coping strategy + evidence with timestamp/quote (only include if explicitly demonstrated in session)"]
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
    "future_topics": ["Specific next topics tied directly to session content (e.g., 'Explore client's stated family challenges with brother [referenced at 32s] when therapeutic alliance is established', 'Assess client's emotional relationship with career transition given detected ambivalence')"],
    "interventions": ["ONLY if sufficient evidence exists (typically requires sessions >3 minutes with clear patterns). For brief/intake sessions: use empty array []. When included, be specific: match intervention to observed pattern with rationale"],
    "follow_up_actions": ["Specific, actionable steps tied to session findings (e.g., 'Schedule full intake session (45-50 min) to establish baseline and treatment goals', 'Assess whether detected incongruence pattern (positive content + negative affect) persists in longer interactions', 'Complete trauma screening given reference to family difficulties')"]
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

