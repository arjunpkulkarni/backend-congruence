# Simplified Analysis System (3-Signal Approach)

## Overview

The simplified analysis system provides **3 core clinical signals** designed for therapist usability:

1. **Emotional Intensity Timeline** - A continuous measure of emotional activation
2. **Incongruence Markers** - Timestamped moments where signals don't match  
3. **Repetition/Stuckness Indicators** - Pattern similarity across sessions

**No emotion labels. No predictions. No diagnosis. Just observable signals.**

---

## Philosophy

### What We Removed
- ❌ 7-emotion classification (joy, sadness, anger, fear, disgust, surprise, neutral)
- ❌ Complex TECS (Tri-Modal Emotional Congruence Score) calculations
- ❌ Cosine similarity between emotion distributions
- ❌ Emotion-specific recommendations
- ❌ Predictive language

### What We Kept
- ✅ Facial analysis (DeepFace) - used only for **intensity extraction**
- ✅ Audio analysis (Vesper) - used only for **intensity extraction**  
- ✅ Text analysis (LLM) - used only for **valence** (positive/negative)
- ✅ Timestamped observations

---

## The 3 Signals

### 1️⃣ Emotional Intensity Timeline

**What it measures:**
- Facial emotional activation (1 - neutral score)
- Vocal emotional activation (1 - neutral score)
- Combined physiological intensity

**Output:**
```json
{
  "t": 3.0,
  "intensity": 0.71,
  "face_intensity": 0.98,
  "vocal_intensity": 0.30,
  "spike": true
}
```

**Clinical utility:**
- Track when client becomes emotionally activated
- Identify intensity spikes (potential breakthrough moments)
- Detect sustained low-intensity periods (possible avoidance/dissociation)

**Key feature:** No emotion labels - just "how activated" not "what emotion"

---

### 2️⃣ Incongruence Markers (The Killer Feature)

**What it detects:**

#### Type A: Text-based incongruence (when transcript available)
- **Positive words + negative physiology** - "I'm fine" while showing stress
- **Negative words + flat physiology** - Intellectualizing difficult content
- **Emotional flattening** - Low intensity during high-valence speech

#### Type B: Face/Voice mismatch (works without transcript)
- **Smiling but voice shows stress** - Positive face, negative/flat voice
- **Negative face but calm voice** - Distressed expression, controlled voice

**Output:**
```json
{
  "start": 3.0,
  "end": 5.0,
  "type": "smiling_but_voice_shows_stress",
  "explanation": "Facial expression appears positive/happy (face valence: +0.93), but vocal tone suggests tension or negativity (audio valence: -0.13). This face-voice mismatch may indicate client is putting on a brave face while experiencing distress.",
  "snippet": "My name is, wait can I give it lies to?",
  "metrics": {
    "text_valence": 0.0,
    "face_valence": 0.93,
    "audio_valence": -0.13,
    "intensity": 0.68
  }
}
```

**Clinical utility:**
- Identify moments where client may be masking true feelings
- Detect dissociation or avoidance patterns
- Flag potential areas to explore in session

**Key feature:** Provides timestamp + context + specific observation (not diagnosis)

---

### 3️⃣ Repetition/Stuckness Indicators

**What it measures:**
- Similarity of intensity patterns across sessions
- Uses resampled signal correlation (normalized to 20-point signature)

**Output:**
```json
{
  "has_repetition": true,
  "similar_sessions": [
    {"session_id": "1764634117", "similarity": 0.87},
    {"session_id": "1764600925", "similarity": 0.81}
  ],
  "observation": "Pattern similar to sessions: 1764634117, 1764600925"
}
```

**Clinical utility:**
- Identify if client is returning to same emotional territory
- Track progress (or lack thereof) across sessions
- No judgment - just pattern observation

**Key feature:** "Pattern similar to sessions on [dates]" - factual, not interpretive

---

## Implementation

### New Files

1. **`app/services/simplified_analysis.py`** - Core analysis logic
   - `build_intensity_timeline()` - Extract intensity from face/audio
   - `build_incongruence_markers()` - Detect mismatches
   - `detect_repetition_patterns()` - Cross-session pattern matching
   - `run_simplified_analysis()` - Main orchestrator

2. **`app/services/simplified_notes.py`** - Output generation
   - `generate_simplified_notes()` - Create markdown therapist notes
   - `save_simplified_outputs()` - Save JSON + markdown files

### Integration

The simplified analysis runs **alongside** the existing complex analysis in `main.py`:

```python
# After existing analysis
simplified_results = run_simplified_analysis(
    merged_timeline=merged_timeline,
    transcript_segments=transcript_segments,
    patient_id=patient_id,
    session_id=session_id
)

simplified_notes = generate_simplified_notes(
    analysis_results=simplified_results,
    patient_id=patient_id,
    session_id=session_id,
    duration=duration
)

save_simplified_outputs(
    analysis_results=simplified_results,
    notes_markdown=simplified_notes,
    output_dir=outputs_dir
)
```

### Outputs

For each session, the following files are generated:

```
data/sessions/{patient_id}/{session_id}/outputs/
  ├── intensity_timeline.json          # NEW: Intensity over time
  ├── incongruence_markers.json        # NEW: Detected mismatches
  ├── repetition_patterns.json         # NEW: Cross-session patterns
  ├── simplified_notes.md              # NEW: Clean therapist notes
  ├── timeline_1hz.json                # OLD: Full emotion timeline
  ├── session_summary.json             # OLD: Complex analysis
  └── therapist_notes.md               # OLD: LLM-generated notes
```

---

## Testing

### Test on existing session:
```bash
python3 test_simplified_analysis.py
```

### Debug incongruence detection:
```bash
python3 test_simplified_debug.py
```

### Compare outputs:
- **Old approach:** `data/sessions/test/1764811542/outputs/therapist_notes.md`
- **New approach:** `data/sessions/test/1764811542/outputs/simplified_notes.md`

---

## Example Output

```markdown
# Therapist Notes (Simplified Analysis)

## 1️⃣ EMOTIONAL INTENSITY TIMELINE

**Average Intensity:** 0.35 (0 = flat, 1 = highly activated)
**Peak Intensity:** 0.72 at 00:09

**Notable Intensity Spikes:**
- 00:03 — Intensity jumped to 0.71
- 00:07 — Intensity jumped to 0.50

## 2️⃣ INCONGRUENCE MARKERS

**3 incongruent moment(s) observed**

### Moment 1: 00:03 - 00:05

**Type:** `smiling_but_voice_shows_stress`

**Observation:**
Facial expression appears positive/happy (face valence: +0.93), but vocal tone 
suggests tension or negativity (audio valence: -0.13). This face-voice mismatch 
may indicate client is putting on a brave face while experiencing distress.

**What was said:**
> "My name is, wait can I give it lies to?"

## 3️⃣ PATTERN REPETITION

*No similar patterns found in previous sessions.*
```

---

## Advantages Over Previous System

### For Therapists:
1. **Faster to read** - 3 signals vs. 7 emotions × 3 modalities
2. **Easier to interpret** - "Intensity spikes" vs. "cosine similarity decreased"
3. **Actionable** - Specific timestamps to review, not aggregate scores
4. **Non-judgmental** - Observable facts, not diagnostic language

### For Clients:
1. **Less stigmatizing** - "Intensity" vs. "Fear/Disgust detected"
2. **More private** - No emotion classification stored
3. **More respectful** - Acknowledges complexity of human emotion

### Technical:
1. **Faster** - Simple computations vs. complex cosine similarities
2. **More robust** - Works without text analysis (face/voice alone)
3. **Scalable** - Pattern matching across unlimited sessions
4. **Maintainable** - Clear, documented logic

---

## Future Enhancements

### Potential additions:
- [ ] Physiological stress indicators (heart rate variability if available)
- [ ] Topic clustering (what themes correlate with incongruence?)
- [ ] Therapist feedback loop (mark false positives to refine thresholds)
- [ ] Multi-client pattern detection (anonymized population insights)

### What NOT to add:
- ❌ More emotion categories
- ❌ Diagnostic predictions  
- ❌ Treatment recommendations
- ❌ Risk scoring

---

## Philosophy: The Three Signals Are Enough

Every therapist understands:
1. **Intensity** - "They got activated here"
2. **Incongruence** - "Something doesn't match"
3. **Repetition** - "We've been here before"

No need for:
- Emotion theory debates (is surprise positive or negative?)
- Statistical complexity (what does TECS really mean?)
- Algorithmic opacity (why did it flag this moment?)

**Simple. Observable. Useful.**

---

## Questions?

Contact the development team or see:
- `/app/services/simplified_analysis.py` - Core logic
- `/app/services/simplified_notes.py` - Output formatting
- `/test_simplified_analysis.py` - Example usage

