import os
from typing import Any, Dict, Optional


_DEFAULT_OPENAI_KEY = ""  # Optionally place a local dev key here; prefer env var OPENAI_API_KEY
_DEFAULT_MODEL = "gpt-4o-mini"


def _get_openai_client():
    """
    Lazily import and initialize the OpenAI client if available and key is present.
    Returns (client, model) or (None, None) when unavailable.
    """
    api_key = os.getenv("OPENAI_API_KEY") or _DEFAULT_OPENAI_KEY
    if not api_key:
        return None, None
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None, None
    client = OpenAI(api_key=api_key)
    return client, _DEFAULT_MODEL


def analyze_text_emotion_with_llm(
    text: str,
    model: Optional[str] = None,
    instruction: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ask an LLM to estimate valence [-1, 1], arousal [0, 1], basic emotions, and a short rationale.
    Returns a structured dict. Falls back to a lightweight heuristic if the LLM is unavailable.
    """
    client, default_model = _get_openai_client()
    if model is None:
        model = default_model

    if client is None or model is None or not text.strip():
        return _heuristic_text_emotion(text)

    sys_msg = (
        "You are an affect analysis assistant. Given a short text excerpt, "
        "estimate: (1) valence in [-1,1] negative to positive, (2) arousal in [0,1] calm to excited, "
        "(3) probabilities for basic emotions: neutral, happy, sad, angry, fear, disgust, surprise. "
        "Return a strict JSON object with keys: valence, arousal, emotions (mapping), rationale."
    )
    if instruction:
        sys_msg += f" Additional instruction: {instruction}"

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": text[:4000]},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        import json

        parsed = json.loads(content)
        # Basic validation
        valence = float(parsed.get("valence", 0.0))
        arousal = float(parsed.get("arousal", 0.0))
        emotions = parsed.get("emotions", {}) or {}
        rationale = parsed.get("rationale", "")
        return {
            "valence": max(-1.0, min(1.0, valence)),
            "arousal": max(0.0, min(1.0, arousal)),
            "emotions": {str(k): float(v) for k, v in emotions.items()},
            "rationale": rationale,
            "source": "llm",
        }
    except Exception:
        return _heuristic_text_emotion(text)


_POSITIVE_WORDS = {
    "good",
    "great",
    "happy",
    "glad",
    "love",
    "excited",
    "relieved",
    "calm",
    "fine",
    "okay",
    "ok",
}
_NEGATIVE_WORDS = {
    "bad",
    "sad",
    "angry",
    "mad",
    "upset",
    "anxious",
    "nervous",
    "scared",
    "afraid",
    "disgusted",
    "worried",
}


def _heuristic_text_emotion(text: str) -> Dict[str, Any]:
    """
    Lightweight fallback if LLM isn't available: bag-of-words valence and neutral emotions.
    """
    words = {w.strip(".,!?;:").lower() for w in text.split()} if text else set()
    pos_hits = len(words & _POSITIVE_WORDS)
    neg_hits = len(words & _NEGATIVE_WORDS)
    total = pos_hits + neg_hits
    if total == 0:
        valence = 0.0
    else:
        valence = (pos_hits - neg_hits) / total
        valence = max(-1.0, min(1.0, valence))
    emotions = {
        "neutral": 0.7 if total == 0 else max(0.0, 0.7 - 0.1 * total),
        "happy": max(0.0, 0.2 * pos_hits),
        "sad": max(0.0, 0.2 * neg_hits),
        "angry": 0.0,
        "fear": 0.0,
        "disgust": 0.0,
        "surprise": 0.0,
    }
    # Normalize
    s = sum(emotions.values()) or 1.0
    emotions = {k: float(v) / s for k, v in emotions.items()}
    return {
        "valence": float(valence),
        "arousal": 0.3 if total == 0 else min(1.0, 0.3 + 0.1 * total),
        "emotions": emotions,
        "rationale": "Heuristic estimate based on word list.",
        "source": "heuristic",
    }


