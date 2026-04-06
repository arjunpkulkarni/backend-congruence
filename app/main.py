import os
import time
import shutil
from typing import Dict, Any, List, Optional
import logging
import contextlib
import glob
import asyncio

from fastapi import FastAPI, HTTPException, Response, Query
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import (
    ProcessSessionRequest,
    ProcessSessionResponse,
    AgentChatRequest,
    AgentChatResponse,
)
from app.models.conversation import (
    ConversationCreate,
    ConversationUpdate,
    Conversation,
    ConversationWithMessages,
    ConversationListItem,
)
from app.services.video_processing import (
    download_video_file,
    download_audio_file,
    extract_audio_with_ffmpeg,
    convert_audio_to_wav,
    extract_frames_with_ffmpeg,
    has_video_stream,
)
from app.services.analysis import (
    analyze_frames_with_deepface,
    analyze_audio_with_vesper,
    merge_timelines,
    detect_micro_spikes,
)
from app.services.transcription import transcribe_audio_with_faster_whisper
from app.services.congruence_engine import (
    build_congruence_timeline,
    build_session_summary,
)
from app.services.simplified_analysis import run_simplified_analysis
from app.services.simplified_notes import (
    generate_simplified_notes,
    save_simplified_outputs,
)
from app.services.notes import generate_therapist_notes, save_therapist_notes
from app.services.fact_extraction import extract_facts_from_therapist_notes, extract_facts_from_analysis
from app.services.clinical_state import update_patient_clinical_state
from app.services.agent import get_agent
from app.services.database import get_conversation_db
from app.services.data_access import (
    list_patients as da_list_patients,
    list_sessions as da_list_sessions,
    get_session_summary as da_get_session_summary,
    get_session_transcript as da_get_session_transcript,
    get_therapist_notes as da_get_therapist_notes,
    get_patient_history as da_get_patient_history,
    get_practice_analytics_data as da_get_practice_analytics,
)
from app.utils.paths import (
    get_workspace_root,
    create_session_directories,
)

import json

logger = logging.getLogger("emotion_api")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(_handler)


app = FastAPI(title="Emotion Analysis API", version="0.2.0")

# Log API key status on startup
api_key = os.getenv("OPENAI_API_KEY")
logger.info("OPENAI_API_KEY present: %s", bool(api_key))
if api_key:
    logger.info("OPENAI_API_KEY length: %d chars", len(api_key))
    logger.info("OPENAI_MODEL: %s", os.getenv("OPENAI_MODEL", "gpt-4o-mini (default)"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# Health / Utility
# =====================================================================

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.options("/api-key-status")
def api_key_status_options():
    """Handle CORS preflight request"""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )


@app.get("/api-key-status")
def api_key_status() -> Dict[str, Any]:
    """Check if OpenAI API key is configured."""
    logger.info("API key status endpoint called")
    key = os.getenv("OPENAI_API_KEY")

    if not key:
        return {
            "configured": False,
            "present": False,
            "message": "OPENAI_API_KEY not configured",
        }

    masked = f"{key[:7]}...{key[-4:]}" if len(key) > 11 else "***"
    return {
        "configured": True,
        "present": True,
        "key_preview": masked,
        "key_length": len(key),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "message": "OpenAI API key is configured",
    }


# =====================================================================
# Session Processing (existing pipeline)
# =====================================================================

@app.post("/process_session", response_model=ProcessSessionResponse)
def process_session(payload: ProcessSessionRequest) -> ProcessSessionResponse:
    """Full video/audio analysis pipeline - supports both video and audio-only inputs."""
    start_time = time.time()
    
    # Determine input type
    has_video = bool(payload.video_url)
    has_audio_only = bool(payload.audio_url and not payload.video_url)
    
    logger.info(
        "process_session START patient_id=%s video_url=%s audio_url=%s mode=%s",
        payload.patient_id,
        payload.video_url,
        payload.audio_url,
        "video" if has_video else "audio-only"
    )
    
    workspace_root = get_workspace_root()
    session_ts = int(time.time())
    session_dir, media_dir, frames_dir, outputs_dir = create_session_directories(
        workspace_root=workspace_root,
        patient_id=payload.patient_id,
        session_ts=session_ts,
    )

    video_path = os.path.join(media_dir, "input.mp4") if has_video else None
    audio_path = os.path.join(media_dir, "audio.wav")

    # 1) Download and process input file
    try:
        if has_video:
            # Video processing path
            download_video_file(video_url=payload.video_url, destination_path=video_path)
            logger.info("Video downloaded to %s", video_path)
        elif has_audio_only:
            # Audio-only processing path
            audio_input_path = os.path.join(media_dir, "input_audio")
            download_audio_file(audio_url=payload.audio_url, destination_path=audio_input_path)
            logger.info("Audio downloaded to %s", audio_input_path)
            # Convert to standard WAV format
            convert_audio_to_wav(input_audio_path=audio_input_path, output_audio_path=audio_path)
            logger.info("Audio converted to %s", audio_path)
    except Exception as exc:
        with contextlib.suppress(Exception):
            if os.path.isdir(session_dir):
                shutil.rmtree(session_dir)
        logger.exception("Input file download failed")
        raise HTTPException(status_code=400, detail=f"Input file download failed: {exc}") from exc

    # 2) Extract audio (only for video input)
    if has_video:
        try:
            extract_audio_with_ffmpeg(input_video_path=video_path, output_audio_path=audio_path)
            logger.info("Audio extracted to %s", audio_path)
        except Exception as exc:
            logger.exception("Audio extraction failed")
            raise HTTPException(status_code=500, detail=f"Audio extraction failed: {exc}") from exc

    # 3) Extract frames (only for video input with actual video streams)
    frame_count = 0
    if has_video:
        try:
            # Check if the video file actually has video streams
            if has_video_stream(video_path):
                extract_frames_with_ffmpeg(
                    input_video_path=video_path,
                    frames_dir=frames_dir,
                    fps=0.3,
                    filename_pattern="frame_%04d.png",
                )
                frame_count = len(glob.glob(os.path.join(frames_dir, "frame_*.png")))
                logger.info("Frames extracted to %s count=%d", frames_dir, frame_count)
            else:
                logger.info("Video file contains no video stream - treating as audio-only")
                frame_count = 0
        except Exception as exc:
            logger.exception("Frame extraction failed")
            # Don't fail the entire pipeline for frame extraction issues
            # Log the error but continue with audio-only processing
            logger.warning("Continuing with audio-only processing due to frame extraction failure")
            frame_count = 0
    else:
        logger.info("Skipping frame extraction for audio-only input")

    logger.info("Starting parallel analysis...")
    parallel_start = time.time()

    async def run_parallel_analysis():
        loop = asyncio.get_event_loop()

        async def async_transcription():
            try:
                t_text, t_segments = await loop.run_in_executor(
                    None, transcribe_audio_with_faster_whisper, audio_path
                )
                if t_text:
                    logger.info("Transcription completed chars=%d segments=%d", len(t_text or ""), len(t_segments or []))
                    return t_text, t_segments
                return None, None
            except Exception:
                logger.info("Transcription skipped or failed (best-effort)")
                return None, None

        async def async_deepface():
            if has_video and frame_count > 0:
                try:
                    result = await loop.run_in_executor(None, analyze_frames_with_deepface, frames_dir)
                    logger.info("DeepFace analysis completed entries=%d", len(result))
                    return result
                except Exception as exc:
                    logger.exception("DeepFace analysis failed")
                    raise HTTPException(status_code=500, detail=f"DeepFace analysis failed: {exc}") from exc
            else:
                logger.info("Skipping DeepFace analysis for audio-only input")
                return None

        async def async_audio_emotion():
            try:
                result = await loop.run_in_executor(None, analyze_audio_with_vesper, audio_path)
                logger.info("Audio emotion timeline entries=%d", len(result))
                return result
            except Exception as exc:
                logger.exception("Audio emotion analysis failed (Vesper required)")
                raise HTTPException(status_code=500, detail=f"Audio emotion analysis failed: {exc}") from exc

        results = await asyncio.gather(
            async_transcription(), async_deepface(), async_audio_emotion(), return_exceptions=False
        )
        return results

    try:
        parallel_results = asyncio.run(run_parallel_analysis())
        (transcript_text, transcript_segments), facial_timeline, audio_timeline = parallel_results
        parallel_duration = time.time() - parallel_start
        logger.info("Parallel analysis completed in %.2fs", parallel_duration)
        if transcript_text:
            logger.info("Transcript text:\n%s", transcript_text)
        if transcript_segments:
            logger.debug("Transcript segments: %s", transcript_segments)
    except Exception as exc:
        logger.exception("Parallel analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis pipeline failed: {exc}") from exc

    merged_timeline = merge_timelines(facial_timeline=facial_timeline, audio_timeline=audio_timeline)
    logger.info("Merged timeline entries=%d", len(merged_timeline))

    merged_timeline = detect_micro_spikes(merged_timeline, threshold=payload.spike_threshold)
    spikes = [e for e in merged_timeline if e.get("micro_spike")]
    logger.info("Detected spikes=%d", len(spikes))

    try:
        def _write_json(path: str, obj: object) -> None:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)

        congruence_timeline_10hz = build_congruence_timeline(
            merged_timeline=merged_timeline,
            transcript_segments=transcript_segments,
            spikes=spikes,
            target_hz=10.0,
        )
        session_summary = build_session_summary(
            congruence_timeline=congruence_timeline_10hz,
            patient_id=payload.patient_id,
            session_id=session_ts,
            transcript_segments=transcript_segments,
        )
        _write_json(os.path.join(outputs_dir, "timeline.json"), congruence_timeline_10hz)
        _write_json(os.path.join(outputs_dir, "timeline_1hz.json"), merged_timeline)
        _write_json(os.path.join(outputs_dir, "spikes.json"), spikes)
        _write_json(os.path.join(outputs_dir, "session_summary.json"), session_summary)
        if transcript_text:
            with open(os.path.join(outputs_dir, "transcript.txt"), "w", encoding="utf-8") as f:
                f.write(transcript_text)
        if transcript_segments:
            _write_json(os.path.join(outputs_dir, "transcript_segments.json"), transcript_segments)
        logger.info("Wrote enriched timeline and session summary to outputs/")

        # Simplified analysis
        logger.info("Running simplified analysis (3 signals)...")
        try:
            simplified_results = run_simplified_analysis(
                merged_timeline=merged_timeline,
                transcript_segments=transcript_segments,
                patient_id=payload.patient_id,
                session_id=session_ts,
                sessions_root=os.path.join(workspace_root, "sessions"),
            )
            duration_seconds = len(merged_timeline)
            simplified_notes_md = generate_simplified_notes(
                analysis_results=simplified_results,
                patient_id=payload.patient_id,
                session_id=session_ts,
                duration=duration_seconds,
            )
            save_simplified_outputs(
                analysis_results=simplified_results,
                notes_markdown=simplified_notes_md,
                output_dir=outputs_dir,
            )
            logger.info("Simplified analysis completed and saved")
        except Exception as exc:
            logger.exception("Simplified analysis failed (non-critical): %s", exc)

        therapist_notes = None
        if transcript_text and locals().get("session_summary"):
            logger.info("Generating therapist notes...")
            try:
                therapist_notes = generate_therapist_notes(
                    transcript_text=transcript_text,
                    transcript_segments=transcript_segments,
                    session_summary=locals().get("session_summary"),
                    patient_id=payload.patient_id,
                )
                if therapist_notes:
                    therapist_notes_path = os.path.join(outputs_dir, "therapist_notes.md")
                    save_therapist_notes(therapist_notes, therapist_notes_path)
                    logger.info("Therapist notes generated and saved (%d chars)", len(therapist_notes))
                else:
                    logger.info("Therapist notes generation skipped")
            except Exception as exc:
                logger.exception("Therapist notes generation failed (non-critical): %s", exc)
        
        # NEW: Extract session facts and update clinical state
        # This happens after all analysis is complete
        session_video_id = None
        if locals().get("session_summary") or therapist_notes:
            logger.info("Post-processing: Extracting session facts...")
            try:
                # Get the session_video_id from Supabase (query by patient_id and timestamp)
                db = get_conversation_db()
                if db.is_enabled():
                    # Find the most recent session_video for this patient (just created)
                    video_query = db.client.table("session_videos")\
                        .select("id")\
                        .eq("patient_id", payload.patient_id)\
                        .order("created_at", desc=True)\
                        .limit(1)\
                        .execute()
                    
                    if video_query.data:
                        session_video_id = video_query.data[0]["id"]
                        
                        # Extract facts from therapist notes or session summary
                        if therapist_notes:
                            extract_facts_from_therapist_notes(
                                session_video_id=session_video_id,
                                patient_id=payload.patient_id,
                                therapist_notes=therapist_notes
                            )
                        elif locals().get("session_summary"):
                            extract_facts_from_analysis(
                                session_video_id=session_video_id,
                                patient_id=payload.patient_id,
                                session_summary=locals().get("session_summary")
                            )
                        
                        logger.info("Session facts extraction completed")
                        
                        # Update patient clinical state
                        logger.info("Updating patient clinical state...")
                        update_patient_clinical_state(patient_id=payload.patient_id)
                        logger.info("Patient clinical state updated")
                    else:
                        logger.warning("Could not find session_video record to link facts")
            except Exception as exc:
                logger.exception("Post-processing (facts/state) failed (non-critical): %s", exc)

    except Exception as exc:
        logger.exception("Failed to write enriched outputs: %s", exc)

    resp = ProcessSessionResponse(
        patient_id=payload.patient_id,
        session_timestamp=session_ts,
        paths={
            "session_dir": session_dir,
            "media_dir": media_dir,
            "frames_dir": frames_dir if has_video else None,
            "audio_path": audio_path,
            "video_path": video_path if has_video else None,
        },
        timeline_json=merged_timeline,
        spikes_json=spikes,
        timeline_10hz=locals().get("congruence_timeline_10hz"),
        session_summary=locals().get("session_summary"),
        notes=therapist_notes,
        transcript_text=transcript_text,
        transcript_segments=transcript_segments,
    )
    duration = time.time() - start_time
    logger.info(
        "process_session END patient_id=%s session_ts=%d duration_s=%.2f",
        payload.patient_id,
        session_ts,
        duration,
    )
    return resp


# =====================================================================
# Congruence Ops Agent (Iteration 2 - real tool calling)
# =====================================================================

@app.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest) -> AgentChatResponse:
    """
    Congruence Ops Agent chat endpoint.

    The agent uses real tools backed by the data access layer to read
    session data, transcripts, clinical notes, and analytics from disk.
    """
    logger.info(
        "Agent chat request: user_id=%s role=%s message_length=%d",
        request.user_id,
        request.role,
        len(request.message),
    )

    try:
        agent = get_agent()
        response = await agent.process_message(request)
        logger.info(
            "Agent response: tools_used=%s actions_count=%d",
            response.tools_used,
            len(response.actions),
        )
        return response
    except Exception as exc:
        logger.exception("Agent chat failed")
        raise HTTPException(status_code=500, detail=f"Agent processing failed: {exc}") from exc


@app.get("/agent/status")
def agent_status() -> Dict[str, Any]:
    """Check agent system status including available tools."""
    try:
        agent = get_agent()
        from app.services.agent_tools import ALL_TOOLS
        db = get_conversation_db()

        return {
            "status": "ready",
            "model": agent.llm.model_name,
            "tools_count": len(ALL_TOOLS),
            "tools": [t.name for t in ALL_TOOLS],
            "database_enabled": db.is_enabled(),
            "message": "Congruence Ops Agent is ready (Iteration 2 - real data access)",
        }
    except Exception as e:
        return {"status": "error", "message": f"Agent initialization failed: {e}"}


# =====================================================================
# Conversation Management API (Database Persistence)
# =====================================================================

@app.post("/conversations", response_model=Conversation)
async def create_conversation(
    data: ConversationCreate,
    user_id: str = Query(..., description="User ID from auth")
) -> Conversation:
    """Create a new conversation."""
    from uuid import UUID
    db = get_conversation_db()
    
    if not db.is_enabled():
        raise HTTPException(status_code=503, detail="Database not configured")
    
    conversation = await db.create_conversation(UUID(user_id), data)
    if not conversation:
        raise HTTPException(status_code=500, detail="Failed to create conversation")
    
    return conversation


@app.get("/conversations", response_model=List[ConversationListItem])
async def list_conversations(
    user_id: str = Query(..., description="User ID from auth"),
    limit: int = Query(50, ge=1, le=100)
) -> List[ConversationListItem]:
    """List all conversations for the current user."""
    from uuid import UUID
    db = get_conversation_db()
    
    if not db.is_enabled():
        return []
    
    return await db.list_conversations(UUID(user_id), limit=limit)


@app.get("/conversations/{conversation_id}", response_model=ConversationWithMessages)
async def get_conversation(
    conversation_id: str,
    user_id: str = Query(..., description="User ID from auth")
) -> ConversationWithMessages:
    """Get a conversation with all its messages."""
    from uuid import UUID
    db = get_conversation_db()
    
    if not db.is_enabled():
        raise HTTPException(status_code=503, detail="Database not configured")
    
    conversation = await db.get_conversation_with_messages(
        UUID(conversation_id),
        UUID(user_id)
    )
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversation


@app.patch("/conversations/{conversation_id}", response_model=Conversation)
async def update_conversation(
    conversation_id: str,
    data: ConversationUpdate,
    user_id: str = Query(..., description="User ID from auth")
) -> Conversation:
    """Update a conversation (e.g., change title or link to patient)."""
    from uuid import UUID
    db = get_conversation_db()
    
    if not db.is_enabled():
        raise HTTPException(status_code=503, detail="Database not configured")
    
    conversation = await db.update_conversation(
        UUID(conversation_id),
        UUID(user_id),
        data
    )
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversation


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user_id: str = Query(..., description="User ID from auth")
) -> Dict[str, str]:
    """Delete a conversation and all its messages."""
    from uuid import UUID
    db = get_conversation_db()
    
    if not db.is_enabled():
        raise HTTPException(status_code=503, detail="Database not configured")
    
    success = await db.delete_conversation(UUID(conversation_id), UUID(user_id))
    
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"status": "deleted", "conversation_id": conversation_id}


# =====================================================================
# Data Access API (Iteration 2)
# =====================================================================

@app.get("/data/patients")
def api_list_patients() -> Dict[str, Any]:
    """List all patients with session counts and latest activity."""
    patients = da_list_patients()
    return {"patients": patients, "total": len(patients)}


@app.get("/data/patients/{patient_id}/sessions")
def api_list_sessions(patient_id: str) -> Dict[str, Any]:
    """List all sessions for a patient, sorted newest first."""
    sessions = da_list_sessions(patient_id)
    if not sessions:
        raise HTTPException(status_code=404, detail=f"No sessions found for patient '{patient_id}'")
    return {"patient_id": patient_id, "sessions": sessions, "total": len(sessions)}


@app.get("/data/patients/{patient_id}/sessions/{session_id}/summary")
def api_get_session_summary(patient_id: str, session_id: int) -> Dict[str, Any]:
    """Get session summary including congruence scores and emotion distributions."""
    summary = da_get_session_summary(patient_id, session_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Session summary not found")
    return summary


@app.get("/data/patients/{patient_id}/sessions/{session_id}/transcript")
def api_get_transcript(
    patient_id: str,
    session_id: int,
    include_segments: bool = Query(True, description="Include timed segments"),
) -> Dict[str, Any]:
    """Get session transcript with optional timed segments."""
    transcript = da_get_session_transcript(patient_id, session_id, include_segments)
    if transcript is None:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return transcript


@app.get("/data/patients/{patient_id}/sessions/{session_id}/notes")
def api_get_notes(patient_id: str, session_id: int) -> Dict[str, Any]:
    """Get structured therapist notes for a session."""
    notes = da_get_therapist_notes(patient_id, session_id)
    if notes is None:
        raise HTTPException(status_code=404, detail="Therapist notes not found")
    return notes


@app.get("/data/patients/{patient_id}/history")
def api_get_patient_history(
    patient_id: str,
    limit: int = Query(10, ge=1, le=50, description="Max sessions to include"),
) -> Dict[str, Any]:
    """Get patient history including congruence trend and latest notes summary."""
    history = da_get_patient_history(patient_id, limit)
    if not history.get("sessions"):
        raise HTTPException(status_code=404, detail=f"No history found for patient '{patient_id}'")
    return history


@app.get("/data/analytics")
def api_get_practice_analytics() -> Dict[str, Any]:
    """Get practice-wide analytics: total patients, sessions, average congruence."""
    return da_get_practice_analytics()
