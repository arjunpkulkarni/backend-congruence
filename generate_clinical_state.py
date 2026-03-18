#!/usr/bin/env python3
"""
Generate patient_clinical_state from session_facts
Aggregates recent session data into a living clinical summary
"""

import sys
import os
from collections import Counter
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

from app.services.database import get_conversation_db

def generate_clinical_state_for_patient(db, patient_id: str, verbose: bool = True):
    """Aggregate session facts into clinical state for one patient."""
    
    # Get recent sessions (last 3 months)
    three_months_ago = (datetime.now() - timedelta(days=90)).isoformat()
    
    # Get session_videos with their facts
    videos = db.client.table("session_videos")\
        .select("id, created_at, session_facts(*)")\
        .eq("patient_id", patient_id)\
        .gte("created_at", three_months_ago)\
        .order("created_at", desc=True)\
        .execute()
    
    if not videos.data:
        if verbose:
            print(f"⚠️  Patient {patient_id[:8]}...: No recent sessions")
        return False
    
    # Aggregate data
    all_symptoms = []
    all_stressors = []
    all_uncertainties = []
    risk_levels = {"suicide": [], "self_harm": [], "harm_to_others": []}
    
    sessions_with_facts = 0
    
    for video in videos.data:
        if not video.get("session_facts") or len(video["session_facts"]) == 0:
            continue
        
        sessions_with_facts += 1
        facts = video["session_facts"][0]
        
        # Collect symptoms
        if facts.get("symptoms_json") and isinstance(facts["symptoms_json"], dict):
            all_symptoms.extend(facts["symptoms_json"].keys())
        
        # Collect stressors
        if facts.get("stressors_json") and isinstance(facts["stressors_json"], list):
            all_stressors.extend(facts["stressors_json"])
        
        # Collect uncertainties
        if facts.get("uncertainty_json") and isinstance(facts["uncertainty_json"], list):
            all_uncertainties.extend(facts["uncertainty_json"])
        
        # Collect risk levels
        if facts.get("risk_json") and isinstance(facts["risk_json"], dict):
            for risk_type in ["suicide", "self_harm", "harm_to_others"]:
                if risk_type in facts["risk_json"]:
                    risk_levels[risk_type].append(facts["risk_json"][risk_type])
    
    if sessions_with_facts == 0:
        if verbose:
            print(f"⚠️  Patient {patient_id[:8]}...: {len(videos.data)} sessions but no facts extracted yet")
        return False
    
    # Get most common items
    symptom_counts = Counter(all_symptoms)
    stressor_counts = Counter(all_stressors)
    
    # Calculate recent trends (simplified)
    recent_trends = {
        "sessions_analyzed": sessions_with_facts,
        "date_range": f"{three_months_ago[:10]} to {datetime.now().date()}",
        "total_sessions_in_period": len(videos.data)
    }
    
    # Add risk trend if available
    if risk_levels["suicide"]:
        recent_trends["suicide_risk_latest"] = risk_levels["suicide"][0]
    if risk_levels["self_harm"]:
        recent_trends["self_harm_risk_latest"] = risk_levels["self_harm"][0]
    
    # Build clinical state
    state = {
        "patient_id": patient_id,
        "active_problems_json": [s for s, _ in symptom_counts.most_common(10)],
        "ongoing_themes_json": [s for s, _ in stressor_counts.most_common(5)],
        "recent_trends_json": recent_trends,
        "unresolved_followups_json": list(set(all_uncertainties))[:5],
        "last_updated_at": datetime.now().isoformat()
    }
    
    # Upsert (insert or update)
    try:
        db.client.table("patient_clinical_state")\
            .upsert(state, on_conflict="patient_id")\
            .execute()
        
        if verbose:
            print(f"✅ Patient {patient_id[:8]}...: Updated clinical state")
            print(f"   - {len(state['active_problems_json'])} active problems")
            print(f"   - {len(state['ongoing_themes_json'])} ongoing themes")
            print(f"   - {sessions_with_facts} sessions with facts")
        
        return True
        
    except Exception as e:
        print(f"❌ Patient {patient_id[:8]}...: Error updating clinical state: {e}")
        return False


def generate_all_clinical_states():
    """Generate clinical state for all patients."""
    
    db = get_conversation_db()
    
    if not db.is_enabled():
        print("❌ Supabase not enabled. Check environment variables.")
        return
    
    print("🔍 Fetching all patients...\n")
    
    patients = db.client.table("patients").select("id, name").execute()
    
    print(f"📊 Found {len(patients.data)} patients\n")
    
    updated = 0
    skipped = 0
    
    for i, patient in enumerate(patients.data, 1):
        patient_id = patient["id"]
        patient_name = patient.get("name", "Unknown")
        
        print(f"[{i}/{len(patients.data)}] {patient_name} ({patient_id[:8]}...):")
        
        success = generate_clinical_state_for_patient(db, patient_id, verbose=True)
        
        if success:
            updated += 1
        else:
            skipped += 1
        
        print()  # Blank line between patients
    
    print(f"{'='*70}")
    print(f"✅ Clinical State Generation Complete!")
    print(f"   Updated: {updated}")
    print(f"   Skipped: {skipped}")
    print(f"   Total: {len(patients.data)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    print("🚀 Starting clinical state generation...\n")
    generate_all_clinical_states()
