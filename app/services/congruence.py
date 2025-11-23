from typing import Any, Dict, List, Optional, Tuple

from .llm import analyze_text_emotion_with_llm


EMOTION_VALENCE: Dict[str, float] = {
    "neutral": 0.0,
    "happy": 0.8,
    "sad": -0.7,
    "angry": -0.8,
    "fear": -0.8,
    "disgust": -0.7,
    "surprise": 0.3,
}


def _weighted_valence(emotions: Dict[str, float]) -> float:
    if not emotions:
        return 0.0
    total = 0.0
    weight = 0.0
    for k, v in emotions.items():
        if k in EMOTION_VALENCE:
            total += float(v) * EMOTION_VALENCE[k]
            weight += float(v)
    if weight == 0.0:
        return 0.0
    return total / weight


def _primary_emotion(emotions: Dict[str, float]) -> Tuple[str, float]:
    if not emotions:
        return "neutral", 0.0
    k = max(emotions, key=lambda x: emotions.get(x, 0.0))
    return k, float(emotions.get(k, 0.0))


def _overlap(a_start: float, a_end: float, t: int) -> bool:
    """
    Check if second 't' (treated as [t, t+1)) overlaps segment [a_start, a_end).
    """
    return (a_start < (t + 1.0)) and (a_end > t)


def bin_transcript_segments_to_seconds(
    segments: Optional[List[Dict[str, Any]]],
    max_t: int,
) -> Dict[int, Dict[str, Any]]:
    """
    Build a dictionary t -> {'text': '...', 'segments': [seg, ...]} where 'text' is a concatenation
    of segment texts overlapping that second.
    """
    by_t: Dict[int, Dict[str, Any]] = {t: {"text": "", "segments": []} for t in range(max_t + 1)}
    if not segments:
        return by_t
    for seg in segments:
        try:
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", 0.0))
            txt = str(seg.get("text", "")).strip()
        except Exception:
            continue
        if not txt:
            continue
        for t in range(max_t + 1):
            if _overlap(s, e, t):
                entry = by_t.setdefault(t, {"text": "", "segments": []})
                if entry["text"]:
                    entry["text"] += " "
                entry["text"] += txt
                entry["segments"].append(seg)
    return by_t


def attach_text_bins_to_timeline(
    merged_timeline: List[Dict[str, Any]],
    transcript_segments: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Attach 'text' field per second into the merged timeline entries.
    """
    if not merged_timeline:
        return []
    max_t = int(max(e.get("t", 0) for e in merged_timeline))
    bins = bin_transcript_segments_to_seconds(transcript_segments, max_t=max_t)
    enriched: List[Dict[str, Any]] = []
    for e in merged_timeline:
        t = int(e.get("t", 0))
        text_info = bins.get(t, {"text": "", "segments": []})
        enriched.append({**e, "text": text_info})
    return enriched


def estimate_text_bins_emotion(
    enriched_timeline: List[Dict[str, Any]],
    use_llm: bool = True,
) -> List[Dict[str, Any]]:
    """
    For each entry with non-empty text, estimate text emotion using LLM (if available) or heuristic fallback.
    Writes results into entry['text']['analysis'] = { valence, arousal, emotions, rationale, source }.
    """
    out: List[Dict[str, Any]] = []
    for e in enriched_timeline:
        text_blob = ""
        text_field = e.get("text")
        if isinstance(text_field, dict):
            text_blob = str(text_field.get("text", "")).strip()
        analysis: Dict[str, Any] = {"valence": 0.0, "arousal": 0.0, "emotions": {}, "rationale": "", "source": "none"}
        if text_blob:
            if use_llm:
                analysis = analyze_text_emotion_with_llm(text_blob)
            else:
                # Force heuristic path by telling LLM layer to fallback when no text
                analysis = analyze_text_emotion_with_llm(text_blob)
        new_e = dict(e)
        new_e.setdefault("text", {})
        if isinstance(new_e["text"], dict):
            new_e["text"]["analysis"] = analysis
        out.append(new_e)
    return out


def compute_congruence_metrics(
    enriched_timeline_with_text: List[Dict[str, Any]],
    spikes: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    For each second compute a set of congruence measures across face, audio and text.
    Returns a list of entries:
      {
        t,
        face, audio, combined, text: {raw, analysis},
        metrics: {
          face_audio_valence_gap,
          text_face_valence_gap,
          text_audio_valence_gap,
          face_top, audio_top, text_top,
          congruence_score
        },
        spikes: [ ... ]  # spikes coinciding at this second
      }
    """
    spikes_by_t: Dict[int, List[Dict[str, Any]]] = {}
    if spikes:
        for s in spikes:
            tt = int(s.get("t", -1))
            if tt >= 0:
                spikes_by_t.setdefault(tt, []).append(s)

    results: List[Dict[str, Any]] = []
    for e in enriched_timeline_with_text:
        t = int(e.get("t", 0))
        face = e.get("face", {}) or {}
        audio = e.get("audio", {}) or {}
        combined = e.get("combined", {}) or {}
        text_field = e.get("text", {}) if isinstance(e.get("text"), dict) else {}
        text_analysis = text_field.get("analysis", {}) if isinstance(text_field, dict) else {}

        # Valence estimations
        v_face = _weighted_valence(face)
        v_audio = _weighted_valence(audio)
        v_text = float(text_analysis.get("valence", 0.0)) if text_analysis else 0.0

        # Primary emotions
        face_top, face_top_p = _primary_emotion(face)
        audio_top, audio_top_p = _primary_emotion(audio)
        text_emotions = text_analysis.get("emotions", {}) if text_analysis else {}
        text_top, text_top_p = _primary_emotion(text_emotions)

        # Gaps
        face_audio_gap = abs(v_face - v_audio)
        text_face_gap = abs(v_text - v_face)
        text_audio_gap = abs(v_text - v_audio)

        # Congruence score: 1 - normalized mean gap over available signals
        gaps = [face_audio_gap]
        if text_field and text_field.get("text"):
            gaps.extend([text_face_gap, text_audio_gap])
        denom = max(1, len(gaps))
        raw_gap = sum(gaps) / denom  # 0..2 potentially, but each gap is within 0..2
        # Normalize: valence is in [-1,1], so max gap is 2. Map to 0..1 discongruence
        discongruence = min(1.0, raw_gap / 2.0)
        congruence_score = float(max(0.0, 1.0 - discongruence))

        result_entry = {
            "t": t,
            "face": face,
            "audio": audio,
            "combined": combined,
            "text": text_field,
            "metrics": {
                "face_audio_valence_gap": face_audio_gap,
                "text_face_valence_gap": text_face_gap,
                "text_audio_valence_gap": text_audio_gap,
                "face_top": {"label": face_top, "p": face_top_p},
                "audio_top": {"label": audio_top, "p": audio_top_p},
                "text_top": {"label": text_top, "p": text_top_p},
                "congruence_score": congruence_score,
            },
            "spikes": spikes_by_t.get(t, []),
        }
        results.append(result_entry)
    return results


def extract_congruence_events(
    congruence_timeline: List[Dict[str, Any]],
    score_threshold: float = 0.4,
    max_events: int = 32,
) -> List[Dict[str, Any]]:
    """
    Pull notable low-congruence moments and attach brief context.
    """
    events: List[Dict[str, Any]] = []
    for e in congruence_timeline:
        score = float(e.get("metrics", {}).get("congruence_score", 1.0))
        if score <= score_threshold:
            txt = ""
            if isinstance(e.get("text"), dict):
                txt = str(e["text"].get("text", ""))[:160]
            events.append(
                {
                    "t": e.get("t", 0),
                    "congruence_score": score,
                    "face_top": e.get("metrics", {}).get("face_top", {}),
                    "audio_top": e.get("metrics", {}).get("audio_top", {}),
                    "text_top": e.get("metrics", {}).get("text_top", {}),
                    "snippet": txt,
                    "spikes": e.get("spikes", []),
                }
            )
    # Sort by ascending score (most incongruent first)
    events.sort(key=lambda x: x.get("congruence_score", 1.0))
    return events[:max_events]


