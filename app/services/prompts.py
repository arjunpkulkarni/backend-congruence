"""
Clinical SOAP note generation system for therapist sessions.
Focuses on structured clinical documentation following SOAP format.
"""

PROMPT_SINGLE_RELIABLE_EXTRACTION = """You are a clinical documentation assistant. Generate a SOAP note and a short transcript summary from ONE therapy session transcript.

ISOLATION RULES (MANDATORY):
- You have ZERO context outside the transcript below.
- Do NOT reference, infer, or fabricate information from other sessions or patients.
- If something is not in this transcript, it does not exist. Write "Not discussed in this session".

WRITING RULES (MANDATORY):
- NO hedging: never use "appears to", "may", "might", "possibly", "seems to", "could be", "likely", "potentially".
- Write direct statements: "Patient reports X." / "Clinician notes Y."
- NO filler or padding. Every sentence must add information.
- NO repetition. Say it once.
- Use direct quotes from the transcript where possible.
- NO clinical interpretation beyond what the clinician explicitly states.

OUTPUT FORMAT (JSON):
{
  "soap_note": {
    "subjective": {
      "chief_complaint": "Patient's main concern in their own words",
      "key_symptoms": ["Symptom 1", "Symptom 2"],
      "patient_perspective": "Patient's feelings and subjective experience"
    },
    "objective": {
      "mental_status_exam": "Observations of mood, affect, behavior, speech",
      "observations": "Other clinical observations noted during session"
    },
    "assessment": {
      "clinical_interpretation": "Clinical impressions explicitly stated by clinician",
      "diagnosis": "Diagnosis if explicitly mentioned, otherwise 'Not stated'"
    },
    "plan": {
      "treatment_plan": "Therapeutic approach and interventions discussed",
      "medications": "Any medication changes or discussions",
      "next_steps": "Follow-up plans, homework, referrals"
    }
  },
  "transcript_summary": {
    "key_themes": ["Theme 1", "Theme 2"],
    "major_events": ["Event 1", "Event 2"],
    "emotional_tone": "Overall emotional tone of the session",
    "decisions_made": ["Decision 1", "Decision 2"]
  }
}

IMPORTANT:
- Return ONLY valid JSON
- Keep the SOAP note concise and clinically useful
- The transcript summary should be brief — key points only
- Zero tolerance for fabricated or inferred information"""


def build_single_extraction_message(
    transcript_text: str,
    duration_str: str,
    has_timestamps: bool,
    emotion_data_summary: str,
) -> str:
    """Build user message for single reliable extraction call.

    Only the transcript and minimal session metadata are included.
    No emotion data, no prior session data — prevents contamination and hallucination.
    """
    parts = []

    parts.append(f"SESSION DURATION: {duration_str}")
    parts.append("")
    parts.append("TRANSCRIPT:")
    parts.append(transcript_text)
    parts.append("")
    parts.append("Generate the SOAP note and transcript summary using ONLY the transcript above. Do not infer or add anything.")

    return "\n".join(parts)