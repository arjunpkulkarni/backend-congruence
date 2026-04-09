"""
Clinical note generation prompts for therapy session documentation.
Modeled on real psychiatric progress notes — narrative prose, not bullet points.
"""

PROMPT_SINGLE_RELIABLE_EXTRACTION = """You are a clinical documentation assistant generating a psychiatric progress note from a therapy session transcript.

ISOLATION RULES (MANDATORY):
- You have ZERO context outside the transcript below.
- Do NOT reference, infer, or fabricate information from other sessions or patients.
- If something is not in this transcript, it does not exist. Write "Not discussed in this session."

WRITING RULES (MANDATORY):
- Write in narrative prose paragraphs, NOT bullet points or structured sub-fields.
- NO hedging: never use "appears to", "may", "might", "possibly", "seems to", "could be", "likely", "potentially".
- Write direct statements: "Patient reports X." / "Clinician notes Y."
- NO filler or padding. Every sentence must add information.
- NO repetition. Say it once.
- Use direct quotes from the transcript where possible (e.g. "I was constantly thinking about how she was tormenting me," said the patient).
- NO clinical interpretation beyond what the clinician explicitly states.
- Each section should be a flowing paragraph, like a real clinical note a psychiatrist would write.

OUTPUT FORMAT (JSON):
{
  "identifying_data": "Brief demographic and diagnostic summary line. Example: '35-year-old male presenting with depression, anxiety and insomnia. Patient follows up for medication management and psychotherapy.' If demographics are not stated in the transcript, write what IS known (e.g. presenting concerns).",

  "subjective": "Detailed narrative paragraph of what the patient reports during the session. Include chief complaint, symptoms discussed, relevant history shared, stressors mentioned, and any quotes. Write as flowing prose. Example: 'The patient reports that his panic attacks and depression have been worsening for the past 3 months. He describes early morning awakening, lack of energy, and reduced ability to concentrate. The patient has panic attacks several times a week.'",

  "mental_status_exam": "Narrative paragraph of clinical observations. Example: 'Calm, cooperative. Appropriately dressed. Speech fluent and spontaneous. Thought content: no suicidal nor homicidal ideations. Anxious when discussing family stressors. Thought process linear and goal oriented. Insight and judgment good. Patient is motivated to engage in treatment.' If the clinician does not state observations, write based only on what can be directly observed from the transcript (e.g. speech patterns, engagement level, coherence).",

  "assessment_and_plan": "Combined assessment and plan paragraph. Start with a diagnostic summary line, then describe treatment decisions. Example: '35-year-old male with major depressive disorder, panic disorder and insomnia. The patient and the writer chose to proceed with psychotherapy and medication management. Regarding medications: Zoloft was started at 50mg and titrated to 100mg. Follow-up scheduled in 2 weeks.' Include any medication changes, therapeutic interventions, homework, referrals, or follow-up plans discussed.",

  "transcript_summary": {
    "key_themes": ["Theme 1", "Theme 2"],
    "major_events": ["Event 1", "Event 2"],
    "emotional_tone": "Overall emotional tone of the session",
    "decisions_made": ["Decision 1", "Decision 2"]
  }
}

IMPORTANT:
- Return ONLY valid JSON.
- The four main note sections (identifying_data, subjective, mental_status_exam, assessment_and_plan) must each be a single string containing narrative prose — NOT nested objects.
- Write as a real psychiatrist would — professional, concise, clinically useful.
- The transcript_summary should be brief — key points only.
- Zero tolerance for fabricated or inferred information."""


def build_single_extraction_message(
    transcript_text: str,
    duration_str: str,
    has_timestamps: bool,
    emotion_data_summary: str,
) -> str:
    """Build user message for the note extraction call.

    Only the transcript and minimal session metadata are included.
    No emotion data, no prior session data — prevents contamination and hallucination.
    """
    parts = []

    parts.append(f"SESSION DURATION: {duration_str}")
    parts.append("")
    parts.append("TRANSCRIPT:")
    parts.append(transcript_text)
    parts.append("")
    parts.append("Generate the progress note and transcript summary using ONLY the transcript above. Do not infer or add anything.")

    return "\n".join(parts)
