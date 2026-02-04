#!/usr/bin/env python3
"""
Process testvideo2.mov through the full pipeline and test sequential notes generation.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.video_processing import (
    extract_audio_with_ffmpeg,
    extract_frames_with_ffmpeg,
)
from app.services.analysis import (
    analyze_frames_with_deepface,
    analyze_audio_with_vesper,
    merge_timelines,
    detect_micro_spikes,
)
from app.services.transcription import transcribe_audio_with_faster_whisper
from app.services.congruence_engine import (
    build_congruence_timeline,
    build_session_summary,
)
from app.services.notes import generate_therapist_notes, save_therapist_notes
from app.utils.paths import create_session_directories

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_video(video_path: str, patient_id: str, session_id: str):
    """Process video through full pipeline"""
    
    logger.info("="*80)
    logger.info(f"PROCESSING: {video_path}")
    logger.info("="*80)
    
    # Create session directories
    timestamp = int(time.time())
    workspace_root = str(Path(__file__).parent)
    session_dir, media_dir, frames_dir, outputs_dir = create_session_directories(
        workspace_root, patient_id, timestamp
    )
    
    session_dirs = {
        'session': Path(session_dir),
        'media': Path(media_dir),
        'frames': Path(frames_dir),
        'outputs': Path(outputs_dir),
    }
    
    logger.info(f"Session directory: {session_dir}")
    
    # Copy video to media directory
    import shutil
    video_dest = session_dirs['media'] / "input.mp4"
    shutil.copy(video_path, video_dest)
    logger.info(f"✓ Video copied to: {video_dest}")
    
    # Extract audio
    logger.info("\n[1/7] Extracting audio...")
    audio_path = str(session_dirs['media'] / "audio.wav")
    extract_audio_with_ffmpeg(str(video_dest), audio_path)
    logger.info(f"✓ Audio extracted: {audio_path}")
    
    # Extract frames
    logger.info("\n[2/7] Extracting frames...")
    extract_frames_with_ffmpeg(str(video_dest), str(session_dirs['frames']))
    logger.info(f"✓ Frames extracted to: {session_dirs['frames']}")
    
    # Analyze frames (facial emotions)
    logger.info("\n[3/7] Analyzing facial emotions...")
    facial_timeline = analyze_frames_with_deepface(str(session_dirs['frames']), silent=True)
    logger.info(f"✓ Analyzed {len(facial_timeline)} frames")
    
    # Analyze audio (vocal emotions)
    logger.info("\n[4/7] Analyzing vocal emotions...")
    audio_timeline = analyze_audio_with_vesper(audio_path)
    logger.info(f"✓ Analyzed {len(audio_timeline)} audio segments")
    
    # Merge timelines
    logger.info("\n[5/7] Merging emotion timelines...")
    merged_timeline = merge_timelines(facial_timeline, audio_timeline)
    merged_timeline = detect_micro_spikes(merged_timeline)
    logger.info(f"✓ Merged timeline: {len(merged_timeline)} entries")
    
    # Transcribe audio
    logger.info("\n[6/7] Transcribing audio...")
    transcript_text, segments = transcribe_audio_with_faster_whisper(audio_path)
    logger.info(f"✓ Transcribed {len(segments)} segments")
    
    # Build congruence timeline
    logger.info("\n[7/7] Computing congruence...")
    congruence_timeline = build_congruence_timeline(
        merged_timeline=merged_timeline,
        transcript_segments=segments,
        spikes=None,
        target_hz=10.0,
    )
    
    session_summary = build_session_summary(
        congruence_timeline=congruence_timeline,
        patient_id=patient_id,
        session_id=timestamp,
        transcript_segments=segments,
    )
    logger.info(f"✓ Overall congruence: {session_summary['overall_congruence']:.3f}")
    logger.info(f"✓ Incongruent moments: {len(session_summary['incongruent_moments'])}")
    
    # Save intermediate outputs
    outputs_dir = session_dirs['outputs']
    
    with open(outputs_dir / "timeline.json", 'w') as f:
        json.dump(congruence_timeline, f, indent=2)
    
    with open(outputs_dir / "session_summary.json", 'w') as f:
        json.dump(session_summary, f, indent=2)
    
    with open(outputs_dir / "transcript_segments.json", 'w') as f:
        json.dump({"segments": segments}, f, indent=2)
    
    with open(outputs_dir / "transcript.txt", 'w') as f:
        f.write(transcript_text)
    
    logger.info(f"\n✓ Intermediate outputs saved to: {outputs_dir}")
    
    # Generate therapist notes with NEW SEQUENTIAL PIPELINE
    logger.info("\n" + "="*80)
    logger.info("GENERATING THERAPIST NOTES (Sequential 4-Prompt Pipeline)")
    logger.info("="*80 + "\n")
    
    notes = generate_therapist_notes(
        transcript_text=transcript_text,
        transcript_segments=segments,
        session_summary=session_summary,
        patient_id=patient_id,
    )
    
    if notes:
        # Save notes
        notes_path = str(outputs_dir / "therapist_notes.json")
        save_therapist_notes(notes, notes_path)
        
        logger.info("\n" + "="*80)
        logger.info("✅ SUCCESS - NOTES GENERATED")
        logger.info("="*80)
        logger.info(f"\nSession directory: {session_dirs['session']}")
        logger.info(f"Notes saved to: {outputs_dir}/therapist_notes.json")
        logger.info(f"              : {outputs_dir}/therapist_notes.md")
        
        # Print summary
        overview = notes.get("session_overview", {})
        logger.info(f"\n📊 SUMMARY:")
        logger.info(f"Duration: {overview.get('duration', 'N/A')}")
        logger.info(f"Themes: {len(notes.get('key_themes', []))}")
        logger.info(f"Emotions: {len(notes.get('emotional_analysis', {}).get('predominant_emotions', []))}")
        logger.info(f"Incongruence: {len(notes.get('emotional_analysis', {}).get('incongruence_moments', []))}")
        
        risk = notes.get("risk_assessment", {})
        logger.info(f"\n🛡️  RISK ASSESSMENT:")
        logger.info(f"Suicide/Self-harm: {risk.get('suicide_self_harm', {}).get('indicators', 'N/A')}")
        logger.info(f"Harm to others: {risk.get('harm_to_others', {}).get('indicators', 'N/A')}")
        logger.info(f"Substance use: {risk.get('substance_use', {}).get('indicators', 'N/A')}")
        
        return True
    else:
        logger.error("\n❌ FAILED - Notes generation failed")
        return False


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set. Please set it first:")
        logger.error("export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    VIDEO_PATH = "data/media/testvideo2.mov"
    
    if not Path(VIDEO_PATH).exists():
        logger.error(f"Video not found: {VIDEO_PATH}")
        sys.exit(1)
    
    success = process_video(
        video_path=VIDEO_PATH,
        patient_id="testvideo2_test",
        session_id="sequential_pipeline_test"
    )
    
    sys.exit(0 if success else 1)
