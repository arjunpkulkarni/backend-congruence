"""
Patient clinical state aggregation service.
Updates the rolling clinical snapshot after each session.

Aggregates session_facts from recent sessions into a living clinical summary per patient.
"""

import logging
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from app.services.database import get_conversation_db

logger = logging.getLogger(__name__)


def update_patient_clinical_state(patient_id: str) -> bool:
    """
    Aggregate recent session_facts into patient_clinical_state.
    Uses upsert to update existing record or create new one.
    
    This function:
    1. Fetches all session_facts from the last 3 months
    2. Aggregates symptoms, stressors, risk levels
    3. Identifies trends and patterns
    4. Upserts into patient_clinical_state table
    
    Args:
        patient_id: UUID of the patient
    
    Returns:
        True if successful, False otherwise
    """
    db = get_conversation_db()
    if not db.is_enabled():
        logger.warning("Supabase not enabled, skipping clinical state update")
        return False
    
    try:
        # Get recent sessions (last 3 months)
        three_months_ago = (datetime.now() - timedelta(days=90)).isoformat()
        
        videos = db.client.table("session_videos")\
            .select("id, created_at, session_facts(*)")\
            .eq("patient_id", patient_id)\
            .gte("created_at", three_months_ago)\
            .order("created_at", desc=True)\
            .execute()
        
        if not videos.data:
            logger.info(f"No recent sessions for patient {patient_id[:8]}...")
            return False
        
        # Aggregate data from session_facts
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
            logger.info(f"Patient {patient_id[:8]}...: {len(videos.data)} sessions but no facts extracted yet")
            return False
        
        # Get most common items
        symptom_counts = Counter(all_symptoms)
        stressor_counts = Counter(all_stressors)
        
        # Calculate recent trends
        recent_trends = {
            "sessions_analyzed": sessions_with_facts,
            "date_range": f"{three_months_ago[:10]} to {datetime.now().date()}",
            "total_sessions_in_period": len(videos.data)
        }
        
        # Add risk trends if available
        if risk_levels["suicide"]:
            recent_trends["suicide_risk_latest"] = risk_levels["suicide"][0]
            recent_trends["suicide_risk_history"] = risk_levels["suicide"][:5]  # Last 5
        if risk_levels["self_harm"]:
            recent_trends["self_harm_risk_latest"] = risk_levels["self_harm"][0]
            recent_trends["self_harm_risk_history"] = risk_levels["self_harm"][:5]
        
        # Build clinical state
        state = {
            "patient_id": patient_id,
            "active_problems_json": [s for s, _ in symptom_counts.most_common(10)],
            "ongoing_themes_json": [s for s, _ in stressor_counts.most_common(5)],
            "recent_trends_json": recent_trends,
            "unresolved_followups_json": list(set(all_uncertainties))[:5],
            "last_updated_at": datetime.now().isoformat()
        }
        
        # Upsert (insert or update based on patient_id)
        db.client.table("patient_clinical_state")\
            .upsert(state, on_conflict="patient_id")\
            .execute()
        
        logger.info(f"✅ Updated clinical state for patient {patient_id[:8]}...")
        logger.info(f"   - {len(state['active_problems_json'])} active problems")
        logger.info(f"   - {len(state['ongoing_themes_json'])} ongoing themes")
        logger.info(f"   - {sessions_with_facts} sessions with facts")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to update clinical state for patient {patient_id[:8]}...: {e}")
        return False


def get_patient_clinical_state(patient_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve the current clinical state for a patient.
    
    Args:
        patient_id: UUID of the patient
    
    Returns:
        Clinical state dict or None if not found
    """
    db = get_conversation_db()
    if not db.is_enabled():
        return None
    
    try:
        response = db.client.table("patient_clinical_state")\
            .select("*")\
            .eq("patient_id", patient_id)\
            .single()\
            .execute()
        
        return response.data if response.data else None
        
    except Exception as e:
        logger.error(f"Failed to get clinical state for patient {patient_id[:8]}...: {e}")
        return None
