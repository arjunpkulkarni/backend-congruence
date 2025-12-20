"""
Simplified Therapist Notes Generator

Generates clean, actionable notes from the 3-signal analysis:
- Intensity overview (spikes/drops, no emotion labels)
- Incongruence observations (with timestamps and context)
- Pattern repetition notes (if applicable)

No jargon. No predictions. Just observable facts.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def generate_intensity_summary(
    intensity_timeline: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate summary statistics and notable moments from intensity timeline.
    
    Returns:
        {
            "avg_intensity": float,
            "peak_intensity": float,
            "peak_time": seconds,
            "notable_spikes": [{"time": ..., "intensity": ...}, ...],
            "low_periods": [{"start": ..., "end": ..., "avg_intensity": ...}, ...]
        }
    """
    if not intensity_timeline:
        return {
            "avg_intensity": 0.0,
            "peak_intensity": 0.0,
            "peak_time": 0.0,
            "notable_spikes": [],
            "low_periods": []
        }
    
    intensities = [e["intensity"] for e in intensity_timeline]
    times = [e["t"] for e in intensity_timeline]
    
    avg_intensity = sum(intensities) / len(intensities)
    peak_intensity = max(intensities)
    peak_idx = intensities.index(peak_intensity)
    peak_time = times[peak_idx]
    
    # Find notable spikes (intensity jumps > 0.15)
    notable_spikes = []
    for entry in intensity_timeline:
        if entry.get("spike", False):
            notable_spikes.append({
                "time": entry["t"],
                "intensity": entry["intensity"]
            })
    
    # Find sustained low periods (avg intensity < 0.15 for at least 5 seconds)
    low_periods = []
    in_low_period = False
    low_start = 0
    low_intensities = []
    
    for i, entry in enumerate(intensity_timeline):
        if entry["intensity"] < 0.15:
            if not in_low_period:
                in_low_period = True
                low_start = entry["t"]
                low_intensities = [entry["intensity"]]
            else:
                low_intensities.append(entry["intensity"])
        else:
            if in_low_period and len(low_intensities) >= 5:
                low_periods.append({
                    "start": low_start,
                    "end": intensity_timeline[i-1]["t"],
                    "avg_intensity": sum(low_intensities) / len(low_intensities)
                })
            in_low_period = False
            low_intensities = []
    
    # Handle trailing low period
    if in_low_period and len(low_intensities) >= 5:
        low_periods.append({
            "start": low_start,
            "end": intensity_timeline[-1]["t"],
            "avg_intensity": sum(low_intensities) / len(low_intensities)
        })
    
    return {
        "avg_intensity": round(avg_intensity, 3),
        "peak_intensity": round(peak_intensity, 3),
        "peak_time": peak_time,
        "notable_spikes": notable_spikes[:10],  # Limit to top 10
        "low_periods": low_periods
    }


def generate_simplified_notes(
    analysis_results: Dict[str, Any],
    patient_id: str,
    session_id: int,
    duration: float
) -> str:
    """
    Generate simplified therapist notes in markdown format.
    
    Args:
        analysis_results: Output from run_simplified_analysis()
        patient_id: Patient identifier
        session_id: Session identifier
        duration: Session duration in seconds
    
    Returns:
        Markdown-formatted therapist notes
    """
    intensity_timeline = analysis_results.get("intensity_timeline", [])
    incongruence_markers = analysis_results.get("incongruence_markers", [])
    repetition_patterns = analysis_results.get("repetition_patterns", {})
    
    intensity_summary = generate_intensity_summary(intensity_timeline)
    
    # Build markdown document
    lines = []
    
    # Header
    lines.append("# Therapist Notes (Simplified Analysis)")
    lines.append("")
    lines.append("## SESSION OVERVIEW")
    lines.append(f"- **Patient ID:** {patient_id}")
    lines.append(f"- **Session ID:** {session_id}")
    lines.append(f"- **Duration:** {duration:.1f} seconds ({format_timestamp(duration)})")
    lines.append(f"- **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    
    # Signal 1: Emotional Intensity
    lines.append("## 1️⃣ EMOTIONAL INTENSITY TIMELINE")
    lines.append("")
    lines.append(f"**Average Intensity:** {intensity_summary['avg_intensity']:.2f} (0 = flat, 1 = highly activated)")
    lines.append(f"**Peak Intensity:** {intensity_summary['peak_intensity']:.2f} at {format_timestamp(intensity_summary['peak_time'])}")
    lines.append("")
    
    # Notable spikes
    if intensity_summary["notable_spikes"]:
        lines.append("**Notable Intensity Spikes:**")
        for spike in intensity_summary["notable_spikes"][:5]:  # Top 5
            lines.append(f"- {format_timestamp(spike['time'])} — Intensity jumped to {spike['intensity']:.2f}")
        lines.append("")
    else:
        lines.append("*No significant intensity spikes observed.*")
        lines.append("")
    
    # Low periods (potential avoidance/dissociation)
    if intensity_summary["low_periods"]:
        lines.append("**Sustained Low Intensity Periods:**")
        lines.append("*(May indicate emotional flatness, avoidance, or discussion of defended topics)*")
        for period in intensity_summary["low_periods"][:5]:
            lines.append(
                f"- {format_timestamp(period['start'])} to {format_timestamp(period['end'])} "
                f"(avg intensity: {period['avg_intensity']:.2f})"
            )
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Signal 2: Incongruence Markers
    lines.append("## 2️⃣ INCONGRUENCE MARKERS")
    lines.append("")
    
    if incongruence_markers:
        lines.append(f"**{len(incongruence_markers)} incongruent moment(s) observed**")
        lines.append("")
        lines.append("*These are timestamps where verbal content and non-verbal signals (face/voice) don't match.*")
        lines.append("")
        
        for i, marker in enumerate(incongruence_markers, 1):
            lines.append(f"### Moment {i}: {format_timestamp(marker['start'])} - {format_timestamp(marker['end'])}")
            lines.append("")
            lines.append(f"**Type:** `{marker['type']}`")
            lines.append("")
            lines.append("**Observation:**")
            lines.append(marker["explanation"])
            lines.append("")
            
            if marker.get("snippet"):
                lines.append("**What was said:**")
                lines.append(f'> "{marker["snippet"]}"')
                lines.append("")
            
            # Show metrics
            metrics = marker.get("metrics", {})
            lines.append("**Measurements:**")
            lines.append(f"- Verbal tone: {metrics.get('text_valence', 0):+.2f} (negative ← 0 → positive)")
            lines.append(f"- Face expression: {metrics.get('face_valence', 0):+.2f}")
            lines.append(f"- Voice tone: {metrics.get('audio_valence', 0):+.2f}")
            lines.append(f"- Overall intensity: {metrics.get('intensity', 0):.2f}")
            lines.append("")
    else:
        lines.append("*No significant incongruence observed. Verbal and non-verbal signals were generally aligned.*")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Signal 3: Repetition/Stuckness
    lines.append("## 3️⃣ PATTERN REPETITION")
    lines.append("")
    
    if repetition_patterns.get("has_repetition"):
        lines.append("**Similar emotional pattern observed in previous sessions.**")
        lines.append("")
        lines.append(repetition_patterns.get("observation", ""))
        lines.append("")
        
        similar_sessions = repetition_patterns.get("similar_sessions", [])
        if similar_sessions:
            lines.append("**Similar Sessions:**")
            for sess in similar_sessions[:5]:  # Top 5
                lines.append(f"- Session `{sess['session_id']}` (similarity: {sess['similarity']:.1%})")
            lines.append("")
            lines.append("*Consider exploring: Is the patient returning to the same emotional territory? "
                        "What themes or triggers might be recurring?*")
    else:
        lines.append("*No similar patterns found in previous sessions.*")
        lines.append("")
        lines.append("This could indicate:")
        lines.append("- New emotional territory being explored")
        lines.append("- Progress/change from previous patterns")
        lines.append("- First session or limited history")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Clinical considerations
    lines.append("## CLINICAL CONSIDERATIONS")
    lines.append("")
    
    # Auto-generate based on findings
    considerations = []
    
    if intensity_summary["avg_intensity"] < 0.15:
        considerations.append(
            "- **Low overall emotional intensity**: Client may be intellectualizing, "
            "avoiding emotional engagement, or discussing less charged topics."
        )
    
    if intensity_summary["avg_intensity"] > 0.4:
        considerations.append(
            "- **High emotional activation**: Session involved emotionally charged content. "
            "Monitor for signs of overwhelm or need for grounding."
        )
    
    if len(incongruence_markers) > 2:
        considerations.append(
            "- **Multiple incongruent moments**: Client may be struggling to express authentic emotions, "
            "presenting a facade, or experiencing internal conflict about what they're discussing."
        )
    
    if repetition_patterns.get("has_repetition"):
        considerations.append(
            "- **Recurring pattern detected**: Client may be stuck on a particular issue or returning to "
            "unresolved themes. Consider exploring what's preventing forward movement."
        )
    
    if intensity_summary["low_periods"]:
        considerations.append(
            "- **Extended flat affect periods**: May indicate dissociation, avoidance of difficult material, "
            "or discussion of heavily defended topics."
        )
    
    if considerations:
        lines.extend(considerations)
    else:
        lines.append("- Session showed typical emotional engagement patterns.")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Footer
    lines.append("## METHODOLOGY NOTE")
    lines.append("")
    lines.append("This analysis is based on three objective signals:")
    lines.append("1. **Emotional intensity** from facial and vocal activation (not emotion categories)")
    lines.append("2. **Incongruence markers** where words and physiology don't match")
    lines.append("3. **Pattern recognition** comparing to previous sessions")
    lines.append("")
    lines.append("*No diagnoses or predictions are made. These are observations to inform clinical judgment.*")
    
    return "\n".join(lines)


def save_simplified_outputs(
    analysis_results: Dict[str, Any],
    notes_markdown: str,
    output_dir: str
) -> None:
    """
    Save simplified analysis outputs to files.
    
    Saves:
    - intensity_timeline.json
    - incongruence_markers.json
    - repetition_patterns.json
    - simplified_notes.md
    """
    import json
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save intensity timeline
    with open(output_path / "intensity_timeline.json", "w") as f:
        json.dump(analysis_results.get("intensity_timeline", []), f, indent=2)
    
    # Save incongruence markers
    with open(output_path / "incongruence_markers.json", "w") as f:
        json.dump(analysis_results.get("incongruence_markers", []), f, indent=2)
    
    # Save repetition patterns
    with open(output_path / "repetition_patterns.json", "w") as f:
        json.dump(analysis_results.get("repetition_patterns", {}), f, indent=2)
    
    # Save markdown notes
    with open(output_path / "simplified_notes.md", "w") as f:
        f.write(notes_markdown)


