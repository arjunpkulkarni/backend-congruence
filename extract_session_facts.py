#!/usr/bin/env python3
"""
Extract structured facts from existing session notes using LLM
Includes confidence scoring and dry-run mode for testing
"""

import sys
import os
import json
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from app.services.database import get_conversation_db

EXTRACTION_PROMPT = """
You are a clinical data extraction assistant. Extract ONLY facts from the session analysis.

Session Analysis (JSON from congruence analysis):
{note_text}

The analysis contains "incongruent_moments" which describe emotional/verbal mismatches.
Extract clinical facts from these descriptions:

{{
  "symptoms_json": {{"symptom_name": "severity"}},
  "stressors_json": ["work stress", "relationship issues", etc],
  "risk_json": {{"suicide": "none/low/moderate/high", "self_harm": "none/low/moderate/high"}},
  "progress_markers_json": {{"overall_congruence": score}}
}}

RULES:
- Extract symptoms mentioned in incongruent_moments (e.g., "anxiety", "depression", "self-criticism")
- Extract stressors mentioned (e.g., "work", "relationships")
- Infer risk level from content (mentions of "suicide", "self-harm", "hopelessness")
- Use overall_congruence score as a progress marker
- Leave interventions_json, homework_json, adherence_json as null (not in analysis data)
- Be conservative - only extract what's clearly stated
"""

def calculate_confidence(facts: dict, note_text: str) -> float:
    """Simple confidence scoring based on extraction quality."""
    score = 1.0
    
    # Penalize if too many "unknown" values
    unknown_count = str(facts).lower().count("unknown")
    score -= (unknown_count * 0.05)
    
    # Penalize if extracted facts are longer than note (hallucination indicator)
    if len(str(facts)) > len(note_text) * 0.5:
        score -= 0.2
    
    # Penalize if note is very short but lots of facts extracted
    if len(note_text) < 200 and len(str(facts)) > 500:
        score -= 0.3
    
    # Bonus if specific scales mentioned
    if facts.get("progress_markers_json"):
        score += 0.1
    
    return max(0.0, min(1.0, score))


def extract_facts_for_session(
    client: OpenAI,
    db,
    session_id: str,
    patient_id: str,
    note_text: str,
    dry_run: bool = True
) -> Optional[Dict[str, Any]]:
    """Extract facts with LLM and return with confidence score."""
    
    if not note_text or len(note_text.strip()) < 50:
        print(f"⚠️  Session {session_id[:8]}...: Note too short ({len(note_text)} chars), skipping")
        return None
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You extract clinical facts conservatively. Only extract what's explicitly stated."},
                {"role": "user", "content": EXTRACTION_PROMPT.format(note_text=note_text[:4000])}  # Limit to 4k chars
            ],
            response_format={"type": "json_object"},
            temperature=0.1  # Low temperature for consistency
        )
        
        facts = json.loads(response.choices[0].message.content)
        
        # Calculate confidence score
        confidence = calculate_confidence(facts, note_text)
        
        if dry_run:
            print(f"\n{'='*70}")
            print(f"Session: {session_id[:8]}...")
            print(f"Confidence: {confidence:.0%}")
            print(f"Note length: {len(note_text)} chars")
            print(f"Note preview: {note_text[:150]}...")
            print(f"\nExtracted Facts:")
            print(json.dumps(facts, indent=2))
            print(f"{'='*70}")
            return None
        else:
            # Save to database
            facts["session_video_id"] = session_id  # Link to session_videos
            facts["patient_id"] = patient_id  # Required field
            # Note: confidence_score column may not exist, remove it for now
            db.client.table("session_facts").insert(facts).execute()
            
            emoji = "✅" if confidence > 0.7 else "⚠️"
            print(f"{emoji} Session {session_id[:8]}...: Extracted (confidence: {confidence:.0%})")
            return facts
            
    except Exception as e:
        print(f"❌ Error extracting session {session_id[:8]}...: {e}")
        return None


def backfill_session_facts(dry_run: bool = True, limit: Optional[int] = 10):
    """
    Backfill session facts from existing session_videos.
    
    Args:
        dry_run: If True, just print extractions without saving
        limit: Number of sessions to process (use None for all)
    """
    
    db = get_conversation_db()
    
    if not db.is_enabled():
        print("❌ Supabase not enabled. Check environment variables.")
        return
    
    client = OpenAI()
    
    print("🔍 Fetching session_videos without facts...\n")
    
    # Get session_videos with transcripts
    # Note: We'll use session_analysis.summary as the "note text" if transcript_text is empty
    sessions_query = db.client.table("session_videos")\
        .select("id, patient_id, title, transcript_text, session_analysis(summary)")
    
    if limit:
        sessions_query = sessions_query.limit(limit)
    
    sessions = sessions_query.execute()
    
    # Filter out sessions that already have facts
    sessions_to_process = []
    for video in sessions.data:
        existing_facts = db.client.table("session_facts")\
            .select("id")\
            .eq("session_video_id", video["id"])\
            .execute()
        
        if not existing_facts.data:
            # Use transcript_text if available, otherwise use summary from analysis
            note_text = video.get("transcript_text")
            if not note_text:
                analysis = video.get("session_analysis", [])
                if analysis and len(analysis) > 0:
                    note_text = analysis[0].get("summary", "")
            
            if note_text and len(note_text.strip()) > 50:
                sessions_to_process.append({
                    "id": video["id"],
                    "patient_id": video["patient_id"],
                    "title": video.get("title", "Untitled"),
                    "raw_note_text": note_text
                })
    
    print(f"📊 Found {len(sessions_to_process)} sessions without facts")
    
    if dry_run:
        print("\n🔍 DRY RUN MODE - No data will be saved\n")
    else:
        print("\n💾 LIVE MODE - Data will be saved to database\n")
    
    if not sessions_to_process:
        print("✅ All sessions already have facts!")
        return
    
    processed = 0
    for i, session in enumerate(sessions_to_process, 1):
        print(f"\n[{i}/{len(sessions_to_process)}] Processing session {session['id'][:8]}...")
        
        result = extract_facts_for_session(
            client,
            db,
            session["id"],
            session["patient_id"],
            session["raw_note_text"],
            dry_run=dry_run
        )
        
        if result:
            processed += 1
    
    print(f"\n{'='*70}")
    if dry_run:
        print(f"🔍 Dry run complete! Processed {processed}/{len(sessions_to_process)} sessions")
        print(f"\nTo save to database, run:")
        print(f"  python extract_session_facts.py --live")
    else:
        print(f"✅ Extraction complete! Saved {processed}/{len(sessions_to_process)} session facts")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract session facts from notes")
    parser.add_argument("--live", action="store_true", help="Save to database (default is dry-run)")
    parser.add_argument("--limit", type=int, default=10, help="Number of sessions to process (default: 10)")
    parser.add_argument("--all", action="store_true", help="Process all sessions")
    
    args = parser.parse_args()
    
    limit = None if args.all else args.limit
    dry_run = not args.live
    
    print("🚀 Starting session facts extraction...\n")
    backfill_session_facts(dry_run=dry_run, limit=limit)
