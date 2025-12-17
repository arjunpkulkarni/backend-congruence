"""
Simplified Analysis: 3 Core Clinical Signals

This module implements a therapist-friendly analysis system with only 3 signals:
1. Emotional Intensity Timeline - continuous intensity without emotion labels
2. Incongruence Markers - timestamped moments of valence/arousal mismatch
3. Repetition/Stuckness Indicators - pattern similarity across sessions

No complex emotion theory. No predictions. Just observable signals.
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import json
import os
from pathlib import Path


# =============================================================================
# SIGNAL 1: EMOTIONAL INTENSITY TIMELINE
# =============================================================================

def compute_facial_intensity(face_emotions: Dict[str, float]) -> float:
    """
    Compute facial emotional intensity as 1 - neutral.
    Higher value = more emotional activation, regardless of type.
    """
    neutral = float(face_emotions.get("neutral", 0.0))
    return max(0.0, min(1.0, 1.0 - neutral))


def compute_vocal_intensity(audio_emotions: Dict[str, float]) -> float:
    """
    Compute vocal emotional intensity as 1 - neutral.
    In practice, could also use prosody features (pitch variance, volume, etc.)
    """
    neutral = float(audio_emotions.get("neutral", 0.0))
    return max(0.0, min(1.0, 1.0 - neutral))


def compute_combined_intensity(
    face_intensity: float, 
    vocal_intensity: float,
    face_weight: float = 0.6,
    vocal_weight: float = 0.4
) -> float:
    """
    Combine facial tension and vocal intensity into single physiological intensity signal.
    Face weighted slightly higher as it's often more reliable.
    """
    return face_weight * face_intensity + vocal_weight * vocal_intensity


def build_intensity_timeline(
    merged_timeline: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Build a simple intensity timeline from face + audio signals.
    
    Returns:
        List of {"t": seconds, "intensity": 0.0-1.0, "spike": bool}
    """
    timeline = []
    prev_intensity = None
    
    for entry in sorted(merged_timeline, key=lambda x: x.get("t", 0)):
        t = float(entry.get("t", 0.0))
        face = entry.get("face", {}) or {}
        audio = entry.get("audio", {}) or {}
        
        face_int = compute_facial_intensity(face)
        vocal_int = compute_vocal_intensity(audio)
        combined_int = compute_combined_intensity(face_int, vocal_int)
        
        # Detect intensity spikes (rapid increases)
        is_spike = False
        if prev_intensity is not None:
            intensity_jump = combined_int - prev_intensity
            if intensity_jump > 0.15:  # 15% jump in intensity
                is_spike = True
        
        timeline.append({
            "t": t,
            "intensity": round(combined_int, 3),
            "face_intensity": round(face_int, 3),
            "vocal_intensity": round(vocal_int, 3),
            "spike": is_spike
        })
        
        prev_intensity = combined_int
    
    return timeline


# =============================================================================
# SIGNAL 2: INCONGRUENCE MARKERS
# =============================================================================

def compute_valence(emotion_dist: Dict[str, float]) -> float:
    """
    Compute valence (positive/negative) from emotion distribution.
    Returns value in range [-1, 1] where:
        +1 = very positive (joy, surprise)
        -1 = very negative (sadness, anger, fear, disgust)
         0 = neutral
    """
    joy = float(emotion_dist.get("joy", 0.0))
    surprise = float(emotion_dist.get("surprise", 0.0))
    sadness = float(emotion_dist.get("sadness", 0.0))
    anger = float(emotion_dist.get("anger", 0.0))
    fear = float(emotion_dist.get("fear", 0.0))
    disgust = float(emotion_dist.get("disgust", 0.0))
    
    positive = joy + 0.5 * surprise
    negative = sadness + anger + fear + disgust
    
    return positive - negative


def detect_valence_mismatch(
    text_valence: float,
    face_valence: float,
    audio_valence: float,
    intensity: float,
    has_text: bool = True,
    threshold: float = 0.4
) -> Tuple[bool, str]:
    """
    Detect incongruence: verbal positivity/negativity doesn't match physiology.
    
    Returns:
        (is_incongruent, reason_type)
    """
    # Only flag if there's meaningful emotional intensity
    if intensity < 0.15:
        return False, ""
    
    # Average non-verbal valence
    nonverbal_valence = 0.5 * (face_valence + audio_valence)
    
    # If we have text, check text-based incongruence
    if has_text and abs(text_valence) > 0.1:  # Only if text has meaningful valence
        # Type 1: Positive words, negative body/voice (classic "I'm fine" when not fine)
        if text_valence > threshold and nonverbal_valence < -threshold:
            return True, "positive_words_negative_physiology"
        
        # Type 2: Negative words, positive/flat physiology (intellectualizing pain)
        if text_valence < -threshold and nonverbal_valence > 0:
            return True, "negative_words_flat_physiology"
        
        # Type 3: High-stakes content but emotional flattening (dissociation indicator)
        if intensity < 0.1 and abs(text_valence) > 0.3:
            return True, "emotional_flattening"
    
    # Type 4: Face/voice mismatch (works without text)
    # Detect when face shows strong emotion but voice is flat/opposite
    face_audio_gap = abs(face_valence - audio_valence)
    if face_audio_gap > 0.7 and intensity > 0.3:
        # Determine which direction
        if face_valence > 0.3 and audio_valence < 0.0:
            return True, "smiling_but_voice_shows_stress"
        elif face_valence < -0.3 and audio_valence > 0.0:
            return True, "negative_face_but_calm_voice"
    
    return False, ""


def detect_emotional_flattening(
    intensity_window: List[float],
    text_presence: List[bool],
    threshold: float = 0.12
) -> bool:
    """
    Detect sustained low intensity during speaking (potential dissociation/avoidance).
    """
    if len(intensity_window) < 3:
        return False
    
    # Only check when person is speaking
    speaking_intensities = [
        intensity for intensity, speaking in zip(intensity_window, text_presence)
        if speaking
    ]
    
    if len(speaking_intensities) < 3:
        return False
    
    avg_intensity = sum(speaking_intensities) / len(speaking_intensities)
    return avg_intensity < threshold


def build_incongruence_markers(
    merged_timeline: List[Dict[str, Any]],
    intensity_timeline: List[Dict[str, Any]],
    transcript_segments: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Detect and mark incongruent moments with explanations.
    
    Returns:
        List of {
            "start": seconds,
            "end": seconds,
            "type": "positive_words_negative_physiology" | "negative_words_flat_physiology" | "emotional_flattening",
            "explanation": human-readable string,
            "snippet": transcript text if available,
            "metrics": {"text_valence": ..., "face_valence": ..., "audio_valence": ..., "intensity": ...}
        }
    """
    # Build lookup structures
    intensity_by_t = {int(e["t"]): e["intensity"] for e in intensity_timeline}
    
    # Analyze transcript segments for text valence and map to seconds
    # Import here to avoid circular dependencies
    text_valence_by_t = {}  # Map second -> text_valence
    if transcript_segments:
        try:
            from app.services.llm import batch_analyze_text_emotions
            texts = [str(seg.get("text", "")).strip() for seg in transcript_segments]
            analyses = batch_analyze_text_emotions(texts, max_workers=5)
            
            # Map valence to timeline seconds
            for seg, analysis in zip(transcript_segments, analyses):
                if analysis and "valence" in analysis:
                    valence = float(analysis.get("valence", 0.0))
                    start_t = int(seg.get("start", 0))
                    end_t = int(seg.get("end", 0))
                    # Assign this valence to all seconds in the segment
                    for t in range(start_t, end_t + 1):
                        text_valence_by_t[t] = valence
        except Exception as e:
            # If LLM analysis fails, continue without text valence
            import logging
            logging.warning(f"Text valence analysis failed: {e}")
            pass
    
    markers = []
    in_incongruent_run = False
    run_start_idx = 0
    current_run_type = ""
    
    for i, entry in enumerate(sorted(merged_timeline, key=lambda x: x.get("t", 0))):
        t = int(entry.get("t", 0))
        
        face = entry.get("face", {}) or {}
        audio = entry.get("audio", {}) or {}
        
        # Extract text valence from our precomputed lookup
        text_valence = text_valence_by_t.get(t, 0.0)
        has_text = t in text_valence_by_t
        
        # Compute valences
        face_valence = compute_valence(face)
        audio_valence = compute_valence(audio)
        intensity = intensity_by_t.get(t, 0.0)
        
        # Detect incongruence (works with or without text)
        is_incongruent, incongruence_type = detect_valence_mismatch(
            text_valence, face_valence, audio_valence, intensity, has_text
        )
        
        # Track runs of incongruence
        if is_incongruent and not in_incongruent_run:
            in_incongruent_run = True
            run_start_idx = i
            current_run_type = incongruence_type
        elif not is_incongruent and in_incongruent_run:
            in_incongruent_run = False
            # Save the run if it's at least 2 seconds
            if i > run_start_idx + 1:
                markers.append(_create_incongruence_marker(
                    merged_timeline[run_start_idx:i],
                    current_run_type,
                    transcript_segments,
                    text_valence_by_t
                ))
            current_run_type = ""
    
    # Handle trailing run
    if in_incongruent_run and len(merged_timeline) > run_start_idx + 1:
        markers.append(_create_incongruence_marker(
            merged_timeline[run_start_idx:],
            current_run_type,
            transcript_segments,
            text_valence_by_t
        ))
    
    return markers


def _create_incongruence_marker(
    entries: List[Dict[str, Any]],
    marker_type: str,
    transcript_segments: Optional[List[Dict[str, Any]]],
    text_valence_by_t: Optional[Dict[int, float]] = None
) -> Dict[str, Any]:
    """Helper to create a single incongruence marker from a run of entries."""
    start_t = float(entries[0].get("t", 0.0))
    end_t = float(entries[-1].get("t", 0.0))
    
    # Collect text snippet
    snippet = ""
    if transcript_segments:
        parts = []
        for seg in transcript_segments:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", 0.0))
            seg_text = str(seg.get("text", "")).strip()
            if seg_text and not (seg_end <= start_t or seg_start >= end_t):
                parts.append(seg_text)
        snippet = " ".join(parts)[:300]
    
    # Compute average metrics
    face_valences = []
    audio_valences = []
    text_valences = []
    intensities = []
    
    for entry in entries:
        t = int(entry.get("t", 0))
        face = entry.get("face", {}) or {}
        audio = entry.get("audio", {}) or {}
        
        face_valences.append(compute_valence(face))
        audio_valences.append(compute_valence(audio))
        
        # Get text valence from lookup if available
        if text_valence_by_t and t in text_valence_by_t:
            text_valences.append(text_valence_by_t[t])
        
        # Compute intensity for this entry
        face_int = compute_facial_intensity(face)
        vocal_int = compute_vocal_intensity(audio)
        intensities.append(compute_combined_intensity(face_int, vocal_int))
    
    avg_face_val = sum(face_valences) / len(face_valences) if face_valences else 0.0
    avg_audio_val = sum(audio_valences) / len(audio_valences) if audio_valences else 0.0
    avg_text_val = sum(text_valences) / len(text_valences) if text_valences else 0.0
    avg_intensity = sum(intensities) / len(intensities) if intensities else 0.0
    
    # Generate explanation
    explanation = _generate_incongruence_explanation(
        marker_type, avg_text_val, avg_face_val, avg_audio_val, avg_intensity
    )
    
    return {
        "start": round(start_t, 1),
        "end": round(end_t, 1),
        "type": marker_type,
        "explanation": explanation,
        "snippet": snippet,
        "metrics": {
            "text_valence": round(avg_text_val, 3),
            "face_valence": round(avg_face_val, 3),
            "audio_valence": round(avg_audio_val, 3),
            "nonverbal_valence": round(0.5 * (avg_face_val + avg_audio_val), 3),
            "intensity": round(avg_intensity, 3)
        }
    }


def _generate_incongruence_explanation(
    marker_type: str,
    text_val: float,
    face_val: float,
    audio_val: float,
    intensity: float
) -> str:
    """Generate human-readable explanation for incongruence."""
    nonverbal_val = 0.5 * (face_val + audio_val)
    
    if marker_type == "positive_words_negative_physiology":
        return (
            f"Verbal content appears positive (valence: {text_val:+.2f}), "
            f"but facial and vocal cues show tension or negativity (valence: {nonverbal_val:+.2f}). "
            f"Client may be minimizing distress or presenting a positive facade."
        )
    elif marker_type == "negative_words_flat_physiology":
        return (
            f"Verbal content discusses negative topics (valence: {text_val:+.2f}), "
            f"but emotional expression is flat or positive (valence: {nonverbal_val:+.2f}). "
            f"May indicate intellectualization or emotional distance from the content."
        )
    elif marker_type == "emotional_flattening":
        return (
            f"Sustained low emotional intensity (intensity: {intensity:.2f}) during speech. "
            f"Could indicate dissociation, emotional avoidance, or discussion of defended topics."
        )
    elif marker_type == "smiling_but_voice_shows_stress":
        return (
            f"Facial expression appears positive/happy (face valence: {face_val:+.2f}), "
            f"but vocal tone suggests tension or negativity (audio valence: {audio_val:+.2f}). "
            f"This face-voice mismatch may indicate client is putting on a brave face while experiencing distress."
        )
    elif marker_type == "negative_face_but_calm_voice":
        return (
            f"Facial expression shows distress (face valence: {face_val:+.2f}), "
            f"but vocal tone is calm or flat (audio valence: {audio_val:+.2f}). "
            f"May indicate emotional suppression or controlled presentation despite inner distress."
        )
    else:
        return "Mismatch observed between verbal and non-verbal emotional signals."


# =============================================================================
# SIGNAL 3: REPETITION / STUCKNESS INDICATORS
# =============================================================================

def extract_intensity_signature(
    intensity_timeline: List[Dict[str, Any]],
    window_size: int = 5
) -> List[float]:
    """
    Extract key features from intensity timeline for pattern matching.
    
    Uses a smoothed version to capture overall intensity patterns,
    not moment-to-moment noise.
    """
    if not intensity_timeline:
        return []
    
    intensities = [e["intensity"] for e in intensity_timeline]
    
    # Smooth with moving average
    smoothed = []
    for i in range(len(intensities)):
        start = max(0, i - window_size // 2)
        end = min(len(intensities), i + window_size // 2 + 1)
        window = intensities[start:end]
        smoothed.append(sum(window) / len(window))
    
    return smoothed


def compute_pattern_similarity(
    signature1: List[float],
    signature2: List[float]
) -> float:
    """
    Compute similarity between two intensity signatures using Dynamic Time Warping (DTW).
    For simplicity, we use normalized correlation after resampling to same length.
    
    Returns similarity score in [0, 1] where 1 = identical patterns.
    """
    if not signature1 or not signature2:
        return 0.0
    
    # Resample to common length (simple linear interpolation)
    target_len = 20  # Compare patterns at 20-point resolution
    s1 = _resample_signal(signature1, target_len)
    s2 = _resample_signal(signature2, target_len)
    
    # Compute normalized correlation
    mean1 = sum(s1) / len(s1)
    mean2 = sum(s2) / len(s2)
    
    numerator = sum((a - mean1) * (b - mean2) for a, b in zip(s1, s2))
    denom1 = math.sqrt(sum((a - mean1) ** 2 for a in s1))
    denom2 = math.sqrt(sum((b - mean2) ** 2 for b in s2))
    
    if denom1 < 1e-6 or denom2 < 1e-6:
        return 0.0
    
    correlation = numerator / (denom1 * denom2)
    
    # Map correlation from [-1, 1] to [0, 1]
    return (correlation + 1.0) / 2.0


def _resample_signal(signal: List[float], target_len: int) -> List[float]:
    """Resample signal to target length using linear interpolation."""
    if len(signal) <= 1:
        return signal * target_len
    
    resampled = []
    for i in range(target_len):
        # Map i to position in original signal
        pos = i * (len(signal) - 1) / (target_len - 1)
        idx = int(pos)
        alpha = pos - idx
        
        if idx + 1 < len(signal):
            value = (1 - alpha) * signal[idx] + alpha * signal[idx + 1]
        else:
            value = signal[idx]
        
        resampled.append(value)
    
    return resampled


def find_similar_sessions(
    current_session_signature: List[float],
    patient_id: str,
    current_session_id: int,
    sessions_root: str = "data/sessions",
    similarity_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Find previous sessions with similar emotional intensity patterns.
    
    Returns:
        List of {"session_id": ..., "similarity": ..., "date": ...}
    """
    similar_sessions = []
    patient_dir = Path(sessions_root) / patient_id
    
    if not patient_dir.exists():
        return []
    
    # Scan all session directories for this patient
    for session_dir in sorted(patient_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        
        session_id = session_dir.name
        
        # Skip current session
        if session_id == str(current_session_id):
            continue
        
        # Look for intensity timeline from previous analysis
        intensity_file = session_dir / "outputs" / "intensity_timeline.json"
        if not intensity_file.exists():
            continue
        
        try:
            with open(intensity_file, 'r') as f:
                past_timeline = json.load(f)
            
            past_signature = extract_intensity_signature(past_timeline)
            similarity = compute_pattern_similarity(current_session_signature, past_signature)
            
            if similarity >= similarity_threshold:
                similar_sessions.append({
                    "session_id": session_id,
                    "similarity": round(similarity, 3),
                    "date": session_id  # Could parse timestamp to readable date
                })
        except Exception:
            continue
    
    # Sort by similarity descending
    similar_sessions.sort(key=lambda x: x["similarity"], reverse=True)
    
    return similar_sessions


def detect_repetition_patterns(
    intensity_timeline: List[Dict[str, Any]],
    patient_id: str,
    session_id: int,
    sessions_root: str = "data/sessions"
) -> Dict[str, Any]:
    """
    Detect if current session shows similar emotional patterns to previous sessions.
    
    Returns:
        {
            "has_repetition": bool,
            "similar_sessions": [...],
            "observation": "Pattern similar to sessions on [dates]" or ""
        }
    """
    signature = extract_intensity_signature(intensity_timeline)
    similar = find_similar_sessions(
        signature, patient_id, session_id, sessions_root
    )
    
    has_repetition = len(similar) > 0
    
    observation = ""
    if has_repetition:
        dates = [s["session_id"] for s in similar[:3]]  # Top 3 most similar
        observation = f"Pattern similar to sessions: {', '.join(dates)}"
    
    return {
        "has_repetition": has_repetition,
        "similar_sessions": similar,
        "observation": observation
    }


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_simplified_analysis(
    merged_timeline: List[Dict[str, Any]],
    transcript_segments: Optional[List[Dict[str, Any]]],
    patient_id: str,
    session_id: int,
    sessions_root: str = "data/sessions"
) -> Dict[str, Any]:
    """
    Run complete simplified analysis producing 3 core signals.
    
    Returns:
        {
            "intensity_timeline": [...],
            "incongruence_markers": [...],
            "repetition_patterns": {...}
        }
    """
    # Signal 1: Intensity Timeline
    intensity_timeline = build_intensity_timeline(merged_timeline)
    
    # Signal 2: Incongruence Markers
    incongruence_markers = build_incongruence_markers(
        merged_timeline,
        intensity_timeline,
        transcript_segments
    )
    
    # Signal 3: Repetition/Stuckness
    repetition_patterns = detect_repetition_patterns(
        intensity_timeline,
        patient_id,
        session_id,
        sessions_root
    )
    
    return {
        "intensity_timeline": intensity_timeline,
        "incongruence_markers": incongruence_markers,
        "repetition_patterns": repetition_patterns
    }

