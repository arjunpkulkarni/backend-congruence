#!/usr/bin/env python3
"""
No migration needed! 

The 'sessions' table is for appointments/scheduling.
The 'session_videos' table already has all the data we need.
The 'session_facts' table links directly to session_videos via session_video_id.

This script just validates that session_videos has the necessary data.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

from app.services.database import get_conversation_db

def validate_session_videos():
    db = get_conversation_db()
    
    if not db.is_enabled():
        print("❌ Supabase not enabled. Check SUPABASE_URL and SUPABASE_KEY environment variables.")
        return
    
    print("🔍 Validating session_videos data...\n")
    
    # Get all session_videos
    videos = db.client.table("session_videos")\
        .select("id, patient_id, therapist_id, created_at, transcript_text, signed_status, status, title")\
        .execute()
    
    print(f"📊 Found {len(videos.data)} session videos\n")
    
    has_transcript = 0
    missing_transcript = 0
    
    for i, video in enumerate(videos.data, 1):
        video_id = video["id"]
        patient_id = video["patient_id"]
        title = video.get("title", "Untitled")
        transcript = video.get("transcript_text")
        
        if transcript and len(transcript.strip()) > 50:
            has_transcript += 1
            status = "✅"
        else:
            missing_transcript += 1
            status = "⚠️"
        
        transcript_len = len(transcript) if transcript else 0
        print(f"{status} [{i}/{len(videos.data)}] {title[:30]:30} | Patient: {patient_id[:8]}... | Transcript: {transcript_len:6} chars")
    
    print(f"\n{'='*70}")
    print(f"✅ Validation Complete!")
    print(f"   Videos with transcripts: {has_transcript}")
    print(f"   Videos without transcripts: {missing_transcript}")
    print(f"   Total: {len(videos.data)}")
    print(f"\n📝 Note: session_facts will link directly to session_videos (no separate sessions table needed)")
    print(f"{'='*70}")

if __name__ == "__main__":
    print("🚀 Validating session_videos...\n")
    validate_session_videos()
