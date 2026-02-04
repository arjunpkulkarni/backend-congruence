#!/usr/bin/env python3
"""Test the sequential 4-prompt pipeline with fake data"""

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.services.notes import generate_therapist_notes

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_fake_data():
    """Create realistic fake therapy session data"""
    transcript = """
Therapist: Hi, thanks for coming in today. How have you been feeling?

Client: I've been okay, I guess. Work has been really exciting actually. I got promoted to lead the new project.

Therapist: That's wonderful news! Congratulations.

Client: Yeah, I mean, it's a great opportunity. My manager said I'm perfect for it.

Therapist: I notice you said 'I guess' and 'I mean' - sounds like maybe there's some hesitation?

Client: Well, there's just a lot going on. My brother called last week and we had another argument. Same old family stuff.

Therapist: Do you want to talk more about your brother?

Client: Not really. I'd rather focus on the positive. The promotion is a big deal.

Therapist: Of course. How are you feeling about the increased responsibility?

Client: I'm ready for it. I've been preparing for years.
"""
    
    segments = [
        {"start": 0.0, "end": 5.0, "text": "Hi, thanks for coming in today. How have you been feeling?", "speaker": "THERAPIST"},
        {"start": 6.0, "end": 15.0, "text": "I've been okay, I guess. Work has been really exciting actually. I got promoted to lead the new project.", "speaker": "CLIENT"},
        {"start": 16.0, "end": 20.0, "text": "That's wonderful news! Congratulations.", "speaker": "THERAPIST"},
        {"start": 21.0, "end": 28.0, "text": "Yeah, I mean, it's a great opportunity. My manager said I'm perfect for it.", "speaker": "CLIENT"},
        {"start": 29.0, "end": 37.0, "text": "I notice you said 'I guess' and 'I mean' - sounds like maybe there's some hesitation?", "speaker": "THERAPIST"},
        {"start": 38.0, "end": 48.0, "text": "Well, there's just a lot going on. My brother called last week and we had another argument. Same old family stuff.", "speaker": "CLIENT"},
        {"start": 49.0, "end": 53.0, "text": "Do you want to talk more about your brother?", "speaker": "THERAPIST"},
        {"start": 54.0, "end": 60.0, "text": "Not really. I'd rather focus on the positive. The promotion is a big deal.", "speaker": "CLIENT"},
        {"start": 61.0, "end": 67.0, "text": "Of course. How are you feeling about the increased responsibility?", "speaker": "THERAPIST"},
        {"start": 68.0, "end": 72.0, "text": "I'm ready for it. I've been preparing for years.", "speaker": "CLIENT"},
    ]
    
    summary = {
        "duration": 72.0,
        "overall_congruence": 0.68,
        "incongruent_moments": [
            {"start": 6.0, "end": 15.0, "reason": "Positive career content with flat vocal affect"},
            {"start": 21.0, "end": 28.0, "reason": "Enthusiastic words with hesitant vocal tone"}
        ],
        "emotion_distribution": {
            "text": {"joy": 0.40, "neutral": 0.35, "sadness": 0.15, "anxiety": 0.10},
            "face": {"neutral": 0.50, "sadness": 0.30, "joy": 0.15, "surprise": 0.05},
            "audio": {"neutral": 0.45, "sadness": 0.30, "joy": 0.20, "anxiety": 0.05}
        },
        "metrics": {"avg_tecs": 0.68, "num_incongruent_segments": 2}
    }
    
    return transcript, segments, summary


def main():
    logger.info("=" * 80)
    logger.info("TESTING SEQUENTIAL 4-PROMPT PIPELINE")
    logger.info("=" * 80)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set. Please set it first:")
        logger.error("export OPENAI_API_KEY='your-key-here'")
        return False
    
    # Create fake data
    transcript, segments, summary = create_fake_data()
    logger.info("Created fake test data: %.0f seconds", summary["duration"])
    
    # Run pipeline
    logger.info("\nRunning sequential pipeline...\n")
    
    notes = generate_therapist_notes(
        transcript_text=transcript,
        transcript_segments=segments,
        session_summary=summary,
        patient_id="test_patient_123",
    )
    
    if not notes:
        logger.error("\n❌ Pipeline failed to generate notes")
        return False
    
    # Validate output
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATING OUTPUT")
    logger.info("=" * 80)
    
    required = ["session_overview", "key_themes", "emotional_analysis",
                "clinical_observations", "risk_assessment", "recommendations",
                "interaction_dynamics"]
    
    for section in required:
        if section in notes:
            logger.info("✓ %s: present", section)
        else:
            logger.error("✗ %s: MISSING", section)
            return False
    
    # Show sample output
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLE OUTPUT")
    logger.info("=" * 80)
    
    overview = notes.get("session_overview", {})
    logger.info("\nSummary:\n%s\n", overview.get("summary", "N/A"))
    
    themes = notes.get("key_themes", [])
    logger.info("Key Themes (%d):", len(themes))
    for i, theme in enumerate(themes, 1):
        logger.info("  %d. %s", i, theme.get("theme", "N/A"))
    
    emotions = notes.get("emotional_analysis", {}).get("predominant_emotions", [])
    logger.info("\nPredominant Emotions (%d):", len(emotions))
    for emotion in emotions:
        logger.info("  - %s (%s, %s)", emotion.get("emotion"), emotion.get("source"), emotion.get("intensity"))
    
    incongruence = notes.get("emotional_analysis", {}).get("incongruence_moments", [])
    logger.info("\nIncongruence Moments (%d):", len(incongruence))
    for moment in incongruence:
        logger.info("  - [%s] %s", moment.get("timestamp", "?"), moment.get("verbal", "?")[:60])
    
    risk = notes.get("risk_assessment", {})
    logger.info("\nRisk Assessment:")
    for risk_type in ["suicide_self_harm", "harm_to_others", "substance_use"]:
        indicators = risk.get(risk_type, {}).get("indicators", "N/A")
        logger.info("  - %s: %s", risk_type, indicators)
    
    # Save output
    output_file = Path("test_sequential_output.json")
    with open(output_file, 'w') as f:
        json.dump(notes, f, indent=2)
    logger.info("\n✓ Full output saved to: %s", output_file)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ TEST PASSED - Sequential pipeline working!")
    logger.info("=" * 80)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
