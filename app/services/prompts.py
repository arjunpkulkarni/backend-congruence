"""
Single-call reliable extraction system for therapist session data.
Focuses on what LLMs can reliably extract: facts, structure, and formatting.
"""

PROMPT_SINGLE_RELIABLE_EXTRACTION = """You are a clinical documentation assistant specializing in reliable fact extraction and formatting.

YOUR JOB: Extract factual information and format into clinical templates. Focus ONLY on what LLMs can reliably do.

WHAT TO EXTRACT (High Reliability):
1. Medications mentioned (exact names, dosages if stated)
2. Symptoms described (patient's exact words)
3. Timeline events (when things started/happened)
4. Diagnoses mentioned (only if explicitly stated by clinician)
5. Life events (breakups, job changes, trauma mentioned)
6. Speaker-separated content (who said what)
7. Session summary (main topics discussed)

CRITICAL RULES:
- Extract ONLY what is explicitly stated - NO interpretation
- Use direct quotes or close paraphrases
- Include timestamps for every item
- Separate patient vs clinician statements
- If uncertain about anything, mark as "unclear" rather than guess
- NO clinical assessment, risk evaluation, or treatment recommendations
- Focus on factual content that supports clinical documentation

TIMESTAMP RULES:
- Use narrow ranges (10-20 second windows when possible)
- Format as "12-18s" or "~25-35s" (~ for estimated)
- If no timestamps available, estimate based on transcript flow
- Never use full session duration for single quotes

OUTPUT FORMAT (JSON):
{
  "extracted_facts": {
    "medications": [
      {
        "name": "exact medication name",
        "dosage": "dosage if mentioned or null",
        "context": "patient quote about medication",
      "timestamp": "12-18s",
        "speaker": "PATIENT|CLINICIAN"
      }
    ],
    "symptoms": [
      {
        "symptom": "patient's exact words for symptom",
        "context": "fuller quote with context",
        "timestamp": "25-35s",
        "speaker": "PATIENT"
      }
    ],
    "timeline_events": [
      {
        "event": "what happened",
        "timeframe": "when it happened (patient's words)",
        "quote": "exact quote",
        "timestamp": "45-55s"
      }
    ],
    "diagnoses_mentioned": [
      {
        "diagnosis": "diagnosis name",
        "context": "how it was mentioned",
        "timestamp": "60-70s",
        "speaker": "CLINICIAN"
      }
    ],
    "life_events": [
      {
        "event": "breakup/job loss/trauma mentioned",
        "quote": "patient's description",
        "timestamp": "30-40s"
      }
    ],
    "speaker_content": {
      "patient_statements": [
        {
          "quote": "what patient said",
          "timestamp": "10-15s",
          "topic": "brief topic label"
        }
      ],
      "clinician_statements": [
        {
          "quote": "what clinician said",
          "timestamp": "20-25s",
          "type": "question|response|guidance"
        }
      ]
    }
  },
  "discussion_summary": {
    "main_topics": [
      "Topic 1 (brief description)",
      "Topic 2 (brief description)",
      "Topic 3 (brief description)"
    ],
    "patient_concerns": [
      "Concern 1 in patient's words",
      "Concern 2 in patient's words"
    ],
    "session_structure": "Brief description of how session flowed",
    "plans_mentioned": [
      "Any homework or next steps discussed"
    ]
  },
  "clinical_templates": {
    "soap_subjective": "**Patient reported concerns:**\\n- [list symptoms and concerns with timestamps]\\n\\n**Timeline:**\\n- [chronological events]\\n\\n**Medications:**\\n- [current medications discussed]\\n\\n*[Clinician to add assessment of subjective data]*",
    "hpi_template": "**Chief Concern:** [primary concern in patient's words]\\n\\n**Timeline:** [chronological progression]\\n\\n**Associated Symptoms:** [related symptoms]\\n\\n**Current Medications:** [medications and dosages]\\n\\n**Psychosocial Factors:** [life events, stressors]\\n\\n*[Clinician to complete formulation]*",
    "fact_sheet": "# Session Facts\\n\\n## Medications\\n- [list with context]\\n\\n## Symptoms\\n- [patient descriptions]\\n\\n## Timeline\\n- [key events with timeframes]"
  },
  "session_metadata": {
    "duration_seconds": <number>,
    "speaker_count": <number>,
    "has_timestamps": <bool>,
    "extraction_confidence": "high|medium|low"
  }
}

Return ONLY valid JSON. Focus on reliability over completeness."""


def build_single_extraction_message(
    transcript_text: str,
    duration_str: str,
    has_timestamps: bool,
    emotion_data_summary: str,
) -> str:
    """Build user message for single reliable extraction call."""
    
    parts = []
    
    # Session context
    parts.append(f"SESSION CONTEXT:")
    parts.append(f"Duration: {duration_str}")
    parts.append(f"Timestamps available: {'Yes' if has_timestamps else 'No'}")
    parts.append("")
    
    # Emotion data if available
    if emotion_data_summary and emotion_data_summary.strip():
        parts.append("MULTIMODAL EMOTION DATA:")
        parts.append(emotion_data_summary)
        parts.append("")
    
    # Main transcript
    parts.append("TRANSCRIPT:")
    parts.append(transcript_text)
    parts.append("")
    
    # Instructions
    parts.append("EXTRACT: Medications, symptoms, timeline events, diagnoses mentioned, life events, and speaker content.")
    parts.append("SUMMARIZE: Main topics and patient concerns.")
    parts.append("FORMAT: Generate clinical templates (SOAP subjective, HPI, fact sheet).")
    parts.append("")
    parts.append("Focus on reliability - only extract what is explicitly stated.")
    
    return "\n".join(parts)