# Therapist Notes Generation Setup

## Overview

The pipeline now automatically generates comprehensive therapist notes from session transcripts and analysis data as the final step. This feature uses a **separate OpenAI API key** to generate professional clinical notes.

## Quick Setup

### 1. Add Your API Key

Open the file: `app/services/notes.py`

Find this line (around line 14):

```python
NOTES_API_KEY = ""  # ← ADD YOUR OPENAI API KEY HERE
```

Replace with your OpenAI API key:

```python
NOTES_API_KEY = "sk-proj-your-api-key-here"
```

**That's it!** The feature is now enabled.

### 2. Run the Pipeline

```bash
python3 local_test.py --video path/to/video.mp4 --patient-id patient123
```

The therapist notes will be automatically generated and saved to:
```
data/sessions/{patient_id}/{timestamp}/outputs/therapist_notes.md
```

## What Gets Generated

The therapist notes include:

1. **Session Overview**
   - Duration and overall tone
   - Patient engagement level

2. **Key Themes & Topics**
   - Main discussion topics
   - Recurring patterns
   - Important revelations

3. **Emotional Analysis**
   - Predominant emotions (from text, face, audio)
   - Emotional shifts during session
   - Incongruent moments (when verbal/non-verbal don't match)

4. **Clinical Observations**
   - Behavioral patterns
   - Areas of concern
   - Strengths and coping mechanisms

5. **Recommendations**
   - Topics for future sessions
   - Therapeutic interventions to consider
   - Follow-up actions

## Example Output

```markdown
# SESSION OVERVIEW
This 28.1-second session segment shows moderate engagement...

# KEY THEMES & TOPICS
- Self-disclosure and identity
- Truthfulness and deception
- Financial statements

# EMOTIONAL ANALYSIS
Primary emotions detected:
- Text: Neutral (70.9%), Joy (7.8%)
- Face: Neutral (60.8%), Joy (34.6%)
- Audio: Neutral (70.0%)

Notable incongruent moment at 2.1-28.0s: The speaker's verbal content 
about having "a million dollars" showed neutral text valence (+0.005) 
but negative audio valence (-0.125), suggesting uncertainty...

# CLINICAL OBSERVATIONS
...

# RECOMMENDATIONS
...
```

## Configuration

### Model Selection

By default, the notes generator uses **GPT-4o** for highest quality. You can change this in `app/services/notes.py`:

```python
model = "gpt-4o"  # Options: gpt-4o, gpt-4o-mini, gpt-3.5-turbo
```

### Cost Estimation

For a 2-hour session with ~30,000 words of transcript:
- Input: ~40,000 tokens
- Output: ~2,000 tokens (notes)
- Cost with GPT-4o: ~$0.10/session
- Cost with GPT-4o-mini: ~$0.02/session

### Temperature Setting

The default temperature is **0.3** for consistent, professional output. You can adjust in `notes.py`:

```python
temperature=0.3,  # Range: 0.0-2.0 (lower = more consistent)
```

## Optional: Disable Notes Generation

If you don't want to generate therapist notes:

**Option 1**: Leave the API key empty in `notes.py`:
```python
NOTES_API_KEY = ""  # Empty = disabled
```

**Option 2**: The pipeline will continue normally and skip notes generation if the key is missing.

## Pipeline Integration

The therapist notes generation happens **automatically as the final step**:

1. Video/audio extraction
2. Frame extraction
3. Transcription
4. Facial emotion analysis (DeepFace)
5. Audio emotion analysis (Vesper)
6. Timeline merging & spike detection
7. Congruence analysis
8. Session summary generation
9. **→ Therapist notes generation** ← NEW FINAL STEP
10. Save all outputs

**No other pipeline steps are affected.** If notes generation fails or is disabled, the pipeline completes successfully without notes.

## Verify It's Working

After running a session, check the output:

```bash
# Check if notes were generated
ls data/sessions/testVideo1/*/outputs/therapist_notes.md

# View the notes
cat data/sessions/testVideo1/*/outputs/therapist_notes.md

# Check the JSON output for confirmation
python3 local_test.py --video test.mp4 | grep therapist_notes
# Should show: "therapist_notes_generated": true
```

## Logging

You'll see these log messages during processing:

```
INFO local_test: Generating therapist notes
INFO local_test: Therapist notes generated successfully (1234 chars)
INFO local_test: Therapist notes saved to .../therapist_notes.md
```

If API key is not configured:
```
INFO local_test: Therapist notes generation skipped (API key not configured or generation failed)
```

## Troubleshooting

### Notes not generating?

1. **Check API key**: Ensure `NOTES_API_KEY` is set correctly in `notes.py`
2. **Check API quota**: Verify your OpenAI account has available credits
3. **Check network**: Ensure internet connectivity for API calls
4. **Check logs**: Look for error messages in terminal output

### Empty transcript?

If the transcript is empty, notes won't be generated. Ensure:
- Audio file has speech content
- Whisper transcription is working
- Check `transcript.txt` in outputs folder

### API errors?

```python
Warning: Failed to generate therapist notes: <error message>
```

Common causes:
- Invalid API key
- Insufficient quota/credits
- Rate limiting (wait a moment and retry)
- Network connectivity issues

## Privacy & Security

**Important**: The therapist notes feature sends transcript data to OpenAI's API. Ensure you:

1. Use a separate API key for notes (already configured)
2. Review OpenAI's data usage policies
3. Ensure compliance with HIPAA/privacy regulations in your jurisdiction
4. Consider using a Business/Enterprise OpenAI account for additional privacy guarantees
5. Store notes securely (they may contain sensitive patient information)

## Different API Keys

The system uses **two separate API keys**:

1. **Main Pipeline API Key** (`app/services/llm.py`):
   - Used for emotion analysis
   - Used for incongruence reasons
   - Processes short text segments

2. **Notes API Key** (`app/services/notes.py`):
   - Used ONLY for therapist notes
   - Processes full transcripts
   - Separate billing and quota

This separation allows you to:
- Use different OpenAI accounts/organizations
- Track costs separately
- Apply different usage limits
- Disable notes without affecting main pipeline

## Summary

**To enable therapist notes:**
1. Add your OpenAI API key to `app/services/notes.py` (line 14)
2. Run the pipeline normally
3. Find notes in `outputs/therapist_notes.md`

**The notes generation:**
- ✅ Happens automatically as the final step
- ✅ Uses a separate API key
- ✅ Doesn't change any other pipeline steps
- ✅ Gracefully skips if disabled or fails
- ✅ Generates comprehensive clinical notes
- ✅ Includes timestamps and emotional analysis
- ✅ Provides actionable recommendations



