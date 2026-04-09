import os
import time
import shutil
import uuid
from typing import Dict, Any, List, Optional
import logging
import contextlib
import glob

from fastapi import FastAPI, HTTPException, Response, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import (
    ProcessSessionRequest,
    ProcessSessionResponse,
    AgentChatRequest,
    AgentChatResponse,
    # Note Style Management schemas
    UploadNoteStyleRequest,
    NoteStyleResponse,
    ListNoteStylesResponse,
    GenerateNotesWithStyleRequest,
    GenerateNotesWithStyleResponse,
    SetActiveNoteStyleRequest,
    JobSubmittedResponse,
    JobStatusResponse,
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
    get_video_duration,
)
from app.services.analysis import merge_timelines
from app.services.transcription import transcribe_long_audio_chunked
from app.services.congruence_engine import (
    build_congruence_timeline,
    build_session_summary,
)
from app.services.simplified_analysis import run_simplified_analysis
from app.services.simplified_notes import (
    generate_simplified_notes,
    save_simplified_outputs,
)
from app.services.notes import generate_therapist_notes, generate_therapist_notes_with_style, save_therapist_notes
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


def cleanup_large_files(session_dir: str, video_path: Optional[str], audio_path: str, frames_dir: str) -> None:
    """Clean up large temporary files to save disk space after processing long videos."""
    try:
        # Remove original video file (keep audio for potential reprocessing)
        if video_path and os.path.exists(video_path):
            video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            if video_size_mb > 50:  # Only clean up large files
                os.remove(video_path)
                logger.info("Cleaned up video file: %.1f MB", video_size_mb)
        
        # Remove frame images (can be regenerated if needed)
        if os.path.exists(frames_dir):
            frame_files = glob.glob(os.path.join(frames_dir, "*.png"))
            if len(frame_files) > 20:  # Only clean up if many frames
                for frame_file in frame_files:
                    os.remove(frame_file)
                logger.info("Cleaned up %d frame files", len(frame_files))
        
        # Keep audio file as it's needed for potential reprocessing
        
    except Exception as exc:
        logger.warning("File cleanup failed (non-critical): %s", exc)

import json

logger = logging.getLogger("emotion_api")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(_handler)


# In-memory job store: job_id -> status dict (shared ref with processing_status inside each job)
_jobs: Dict[str, Dict[str, Any]] = {}

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

def _dbg(hypothesis, msg, data=None):
    # #region agent log
    import json as _j
    try:
        with open("/Users/arjunkulkarni/Desktop/BCONGRUENCE/.cursor/debug-adf136.log","a") as _f:
            _f.write(_j.dumps({"sessionId":"adf136","hypothesisId":hypothesis,"location":"main.py","message":msg,"data":data or {},"timestamp":int(time.time()*1000)})+"\n")
    except Exception:
        pass
    # #endregion

def _run_session_job(job_id: str, payload: ProcessSessionRequest) -> None:
    """Background worker: runs the full analysis pipeline and writes results into _jobs[job_id]."""
    processing_status = _jobs[job_id]
    start_time = time.time()
    # #region agent log
    _dbg("H1", "background_job_started", {"job_id": job_id, "patient_id": payload.patient_id})
    # #endregion

    request_id = job_id[:8]

    has_video = bool(payload.video_url)
    has_audio_only = bool(payload.audio_url and not payload.video_url)

    logger.info(
        "[%s] process_session START patient_id=%s video_url=%s audio_url=%s mode=%s",
        request_id,
        payload.patient_id,
        payload.video_url,
        payload.audio_url,
        "video" if has_video else "audio-only",
    )

    processing_status.update({
        "request_id": request_id,
        "stage": "initializing",
        "progress": 0,
        "message": "Starting session processing...",
        "errors": [],
        "recording_saved": False,
        "duration_verified": False,
        "concurrent_safe": True,
    })
    logger.info(
        "[%s] 🔧 PROCESSING OPTIONS: fast_mode=%s no_facial_analysis=%s skip_video_analysis=%s cleanup_files=%s",
        request_id,
        payload.fast_mode,
        payload.no_facial_analysis,
        payload.skip_video_analysis,
        payload.cleanup_files,
    )

    workspace_root = get_workspace_root()
    session_ts = int(time.time())
    session_dir, media_dir, frames_dir, outputs_dir = create_session_directories(
        workspace_root=workspace_root,
        patient_id=payload.patient_id,
        session_ts=session_ts,
    )

    logger.info("[%s] Created unique session directory: %s", request_id, session_dir)

    video_path = os.path.join(media_dir, "input.mp4") if has_video else None
    audio_path = os.path.join(media_dir, "audio.wav")

    actual_duration_seconds = None

    # 1) Download and process input file
    try:
        processing_status.update({"stage": "downloading", "progress": 10, "message": "Downloading media file..."})
        logger.info("📥 PROCESSING STATUS: %s", processing_status["message"])

        if has_video:
            logger.info("Downloading video file (timeout: 10 minutes)...")
            download_video_file(video_url=payload.video_url, destination_path=video_path)

            video_duration = get_video_duration(video_path)
            actual_duration_seconds = video_duration
            video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            logger.info("Video downloaded: %.1f MB, duration: %.1f minutes", video_size_mb, video_duration / 60)

            processing_status.update({
                "recording_saved": True,
                "duration_verified": True,
                "message": f"Video saved successfully - {video_duration/60:.1f} minutes",
            })
            logger.info("✅ RECORDING SAVED: Video file secured locally (%.1f MB, %.1f min)", video_size_mb, video_duration / 60)

            if video_duration > 3600:
                logger.info("Long video detected (%.1f min) - using optimized processing", video_duration / 60)

        elif has_audio_only:
            logger.info("Downloading audio file (timeout: 10 minutes)...")
            audio_input_path = os.path.join(media_dir, "input_audio")
            download_audio_file(audio_url=payload.audio_url, destination_path=audio_input_path)

            convert_audio_to_wav(input_audio_path=audio_input_path, output_audio_path=audio_path, fast_mode=payload.fast_mode)

            from app.services.transcription import get_audio_duration
            audio_duration = get_audio_duration(audio_path)
            actual_duration_seconds = audio_duration
            audio_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            logger.info("Audio processed: %.1f MB, duration: %.1f minutes", audio_size_mb, audio_duration / 60)

            processing_status.update({
                "recording_saved": True,
                "duration_verified": True,
                "message": f"Audio saved successfully - {audio_duration/60:.1f} minutes",
            })
            logger.info("✅ RECORDING SAVED: Audio file secured locally (%.1f MB, %.1f min)", audio_size_mb, audio_duration / 60)

            if audio_duration > 3600:
                logger.info("Long audio detected (%.1f min) - using chunked processing", audio_duration / 60)

    except Exception as exc:
        with contextlib.suppress(Exception):
            if os.path.isdir(session_dir):
                shutil.rmtree(session_dir)
        logger.exception("Input file download failed")
        processing_status.update({"status": "failed", "stage": "failed", "message": f"Input file download failed: {exc}"})
        return

    # 2) Extract audio (only for video input)
    if has_video:
        try:
            processing_status.update({"stage": "audio_extraction", "progress": 20, "message": "Extracting audio from video..."})
            logger.info("📥 PROCESSING STATUS: %s", processing_status["message"])
            extract_audio_with_ffmpeg(input_video_path=video_path, output_audio_path=audio_path, fast_mode=payload.fast_mode)
            logger.info("Audio extracted to %s", audio_path)
        except Exception as exc:
            processing_status["errors"].append(f"Audio extraction failed: {exc}")
            logger.exception("Audio extraction failed")
            processing_status.update({"status": "failed", "stage": "failed", "message": f"Audio extraction failed: {exc}"})
            return

    # 3) Transcription only — facial/voice emotion analysis disabled for speed
    processing_status.update({"stage": "transcription", "progress": 40, "message": "Transcribing audio..."})
    logger.info("📥 PROCESSING STATUS: %s", processing_status["message"])
    transcription_start = time.time()

    transcript_text = None
    transcript_segments = None
    try:
        model_size = "tiny" if payload.fast_mode else "small"
        # #region agent log
        _dbg("H2", "transcription_start", {"audio_path": audio_path, "model_size": model_size})
        # #endregion
        transcript_text, transcript_segments = transcribe_long_audio_chunked(
            audio_path, model_size, "en", True, 10
        )
        transcription_duration = time.time() - transcription_start
        # #region agent log
        _dbg("H2", "transcription_done", {"duration_s": round(transcription_duration, 2), "text_len": len(transcript_text or ""), "segments": len(transcript_segments or [])})
        # #endregion
        logger.info("Transcription completed in %.2fs: chars=%d segments=%d",
                     transcription_duration, len(transcript_text or ""), len(transcript_segments or []))
        if transcript_text:
            logger.info("Transcript text:\n%s", transcript_text)
    except Exception as exc:
        # #region agent log
        _dbg("H2", "transcription_crashed", {"error": str(exc)})
        # #endregion
        logger.exception("Transcription failed")
        processing_status.update({"status": "failed", "stage": "failed", "message": f"Transcription failed: {exc}"})
        return

    # No facial or voice emotion analysis — pass empty timelines downstream
    facial_timeline = None
    audio_timeline = None

    merged_timeline = merge_timelines(facial_timeline=facial_timeline, audio_timeline=audio_timeline)
    logger.info("Merged timeline entries=%d (transcript-only mode)", len(merged_timeline))
    # #region agent log
    _dbg("H3", "merge_timelines_result", {"entries": len(merged_timeline)})
    # #endregion

    spikes: list = []

    # Variables that may not be set if the synthesis block raises
    congruence_timeline_10hz = None
    session_summary = None
    therapist_notes = None

    try:
        processing_status.update({"stage": "synthesis", "progress": 80, "message": "Building session summary and timeline..."})
        logger.info("📥 PROCESSING STATUS: %s", processing_status["message"])

        def _write_json(path: str, obj: object) -> None:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)

        # #region agent log
        _dbg("H3", "build_congruence_timeline_start", {"merged_len": len(merged_timeline), "segments": len(transcript_segments or []), "spikes": len(spikes)})
        # #endregion
        congruence_timeline_10hz = build_congruence_timeline(
            merged_timeline=merged_timeline,
            transcript_segments=transcript_segments,
            spikes=spikes,
            target_hz=10.0,
        )
        # #region agent log
        _dbg("H3", "build_congruence_timeline_done", {"entries": len(congruence_timeline_10hz or [])})
        # #endregion
        session_summary = build_session_summary(
            congruence_timeline=congruence_timeline_10hz,
            patient_id=payload.patient_id,
            session_id=session_ts,
            transcript_segments=transcript_segments,
            actual_duration_seconds=actual_duration_seconds,
        )
        # #region agent log
        _dbg("H3", "build_session_summary_done", {"has_summary": session_summary is not None, "keys": list((session_summary or {}).keys())})
        # #endregion
        _write_json(os.path.join(outputs_dir, "timeline.json"), congruence_timeline_10hz)
        _write_json(os.path.join(outputs_dir, "timeline_1hz.json"), merged_timeline)
        _write_json(os.path.join(outputs_dir, "spikes.json"), spikes)
        _write_json(os.path.join(outputs_dir, "session_summary.json"), session_summary)
        if transcript_text:
            with open(os.path.join(outputs_dir, "transcript.txt"), "w", encoding="utf-8") as f:
                f.write(transcript_text)
        if transcript_segments:
            _write_json(os.path.join(outputs_dir, "transcript_segments.json"), transcript_segments)
        _write_json(os.path.join(outputs_dir, "processing_status.json"), processing_status)
        logger.info("✅ AUTO-SAVE: All outputs secured locally in %s", outputs_dir)
        processing_status.update({"local_backup_saved": True})

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

        if transcript_text and session_summary:
            logger.info("Generating therapist notes...")
            try:
                therapist_notes = generate_therapist_notes(
                    transcript_text=transcript_text,
                    transcript_segments=transcript_segments,
                    session_summary=session_summary,
                    patient_id=payload.patient_id,
                )
                if therapist_notes:
                    # Attach transcript for the Full Transcript section in markdown output
                    therapist_notes["_transcript_text"] = transcript_text or ""
                    therapist_notes["_transcript_segments"] = transcript_segments or []
                    therapist_notes_path = os.path.join(outputs_dir, "therapist_notes.md")
                    save_therapist_notes(therapist_notes, therapist_notes_path)
                    logger.info("Therapist notes generated and saved (%d chars)", len(therapist_notes))
                else:
                    logger.info("Therapist notes generation skipped")
            except Exception as exc:
                logger.exception("Therapist notes generation failed (non-critical): %s", exc)

        session_video_id = None
        if session_summary or therapist_notes:
            logger.info("Post-processing: Uploading to database with retry logic...")
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    db = get_conversation_db()
                    if db.is_enabled():
                        video_query = db.client.table("session_videos")\
                            .select("id")\
                            .eq("patient_id", payload.patient_id)\
                            .order("created_at", desc=True)\
                            .limit(1)\
                            .execute()

                        if video_query.data:
                            session_video_id = video_query.data[0]["id"]
                            if therapist_notes:
                                extract_facts_from_therapist_notes(
                                    session_video_id=session_video_id,
                                    patient_id=payload.patient_id,
                                    therapist_notes=therapist_notes,
                                )
                            elif session_summary:
                                extract_facts_from_analysis(
                                    session_video_id=session_video_id,
                                    patient_id=payload.patient_id,
                                    session_summary=session_summary,
                                )
                            logger.info("✅ DATABASE UPLOAD: Session facts uploaded successfully")
                            logger.info("Updating patient clinical state...")
                            update_patient_clinical_state(patient_id=payload.patient_id)
                            logger.info("✅ DATABASE UPLOAD: Patient clinical state updated")
                            processing_status.update({"database_upload_success": True})
                            break
                        else:
                            logger.warning("Could not find session_video record to link facts")
                            processing_status.update({"database_upload_success": False, "database_error": "No session_video record found"})
                            break
                    else:
                        logger.warning("Database not enabled, skipping upload")
                        processing_status.update({"database_upload_success": False, "database_error": "Database not enabled"})
                        break
                except Exception as exc:
                    logger.warning("Database upload attempt %d/%d failed: %s", attempt + 1, max_retries, exc)
                    processing_status["errors"].append(f"Database upload attempt {attempt + 1} failed: {exc}")
                    if attempt == max_retries - 1:
                        logger.error("❌ DATABASE UPLOAD FAILED: All retry attempts exhausted. Data saved locally.")
                        processing_status.update({
                            "database_upload_success": False,
                            "database_error": f"All {max_retries} retry attempts failed",
                            "local_backup_available": True,
                        })
                    else:
                        time.sleep(2 ** attempt)

    except Exception as exc:
        # #region agent log
        _dbg("H3", "synthesis_block_crashed", {"error": str(exc), "type": type(exc).__name__})
        # #endregion
        logger.exception("Failed to write enriched outputs: %s", exc)

    if payload.cleanup_files:
        cleanup_large_files(session_dir, video_path, audio_path, frames_dir)
        logger.info("Temporary files cleaned up to save disk space")

    processing_status.update({
        "stage": "completed",
        "progress": 100,
        "message": "Session processing completed successfully",
        "duration_verified": actual_duration_seconds is not None,
    })
    logger.info("✅ PROCESSING COMPLETED: %s", processing_status["message"])

    if actual_duration_seconds and session_summary:
        summary_duration = session_summary.get("duration", 0)
        duration_diff = abs(actual_duration_seconds - summary_duration)
        if duration_diff < 5:
            logger.info("✅ DURATION VERIFIED: Actual=%.1fs, Summary=%.1fs (diff=%.1fs)",
                       actual_duration_seconds, summary_duration, duration_diff)
        else:
            logger.warning("⚠️ DURATION MISMATCH: Actual=%.1fs, Summary=%.1fs (diff=%.1fs)",
                          actual_duration_seconds, summary_duration, duration_diff)
            processing_status["errors"].append(f"Duration mismatch: {duration_diff:.1f}s difference")

    # #region agent log
    _dbg("H5", "building_response", {"has_transcript": bool(transcript_text), "has_notes": therapist_notes is not None, "timeline_len": len(merged_timeline), "spikes_len": len(spikes)})
    # #endregion
    try:
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
            timeline_10hz=congruence_timeline_10hz,
            session_summary=session_summary,
            notes=therapist_notes,
            transcript_text=transcript_text,
            transcript_segments=transcript_segments,
            processing_status=processing_status,
        )
    except Exception as exc:
        # #region agent log
        _dbg("H5", "response_construction_failed", {"error": str(exc), "type": type(exc).__name__})
        # #endregion
        processing_status.update({"status": "failed", "stage": "failed", "message": f"Response construction failed: {exc}"})
        return
    duration = time.time() - start_time
    logger.info(
        "[%s] process_session END patient_id=%s session_ts=%d duration_s=%.2f",
        request_id,
        payload.patient_id,
        session_ts,
        duration,
    )

    processing_status.update({"status": "completed", "result": resp.model_dump()})
    # #region agent log
    _dbg("H1", "background_job_completed", {"job_id": job_id, "total_duration_s": round(duration, 2)})
    # #endregion


@app.post("/process_session", response_model=JobSubmittedResponse)
def process_session(payload: ProcessSessionRequest, background_tasks: BackgroundTasks) -> JobSubmittedResponse:
    """Enqueue a background analysis job and return immediately. Poll GET /jobs/{job_id} for results."""
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "queued", "stage": "queued", "progress": 0, "message": "Job queued, processing will begin shortly"}
    background_tasks.add_task(_run_session_job, job_id, payload)
    logger.info("Enqueued session job %s for patient_id=%s", job_id, payload.patient_id)
    return JobSubmittedResponse(job_id=job_id, status="queued", message="Processing started in background. Poll /jobs/{job_id} for status.")


@app.post("/process_session_sync", response_model=ProcessSessionResponse)
def process_session_sync(payload: ProcessSessionRequest) -> ProcessSessionResponse:
    """Blocking variant — runs the full pipeline and returns the complete result."""
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "queued", "stage": "queued", "progress": 0, "message": "Job queued, processing will begin shortly"}
    logger.info("Starting synchronous session job %s for patient_id=%s", job_id, payload.patient_id)
    _run_session_job(job_id, payload)

    job = _jobs[job_id]
    if job.get("status") == "failed":
        raise HTTPException(status_code=500, detail=job.get("message", "Processing failed"))

    result = job.get("result")
    if not result:
        raise HTTPException(status_code=500, detail="Processing completed but produced no result")

    return ProcessSessionResponse(**result)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    """Poll the status of a background processing job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    result = job.get("result")
    return JobStatusResponse(
        job_id=job_id,
        status=job.get("status", "processing"),
        stage=job.get("stage"),
        progress=job.get("progress"),
        message=job.get("message"),
        error=job.get("error"),
        result=ProcessSessionResponse(**result) if result else None,
    )


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


# ---------------------------------------------------------------------------
# Note Style Management API Endpoints (MVP)
# ---------------------------------------------------------------------------

@app.post("/note-styles/upload")
def api_upload_note_style(request: UploadNoteStyleRequest) -> NoteStyleResponse:
    """
    Upload and process a clinical note for style matching.
    
    Extracts text from PDF/DOCX/TXT files and analyzes the writing style
    for future note generation matching.
    """
    import uuid
    request_id = uuid.uuid4().hex[:8]
    logger.info("[%s] Note style upload started for user %s", request_id, request.user_id[:8])
    from app.services.note_style import (
        extract_text_from_file, 
        get_preview_text, 
        validate_note_content, 
        analyze_note_style
    )
    from app.services.data_access import save_note_style
    from datetime import datetime
    
    try:
        # Extract text from uploaded file
        note_text = extract_text_from_file(request.file_content, request.file_type)
        
        # Validate the extracted content
        validation_info = validate_note_content(note_text)
        if not validation_info["is_valid"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid note content: {'; '.join(validation_info['errors'])}"
            )
        
        # Analyze the note style
        style_analysis = analyze_note_style(note_text)
        
        # Save to database
        note_id = save_note_style(
            user_id=request.user_id,
            note_name=request.note_name,
            note_text=note_text,
            file_type=request.file_type,
            validation_info=validation_info,
            style_analysis=style_analysis
        )
        
        if not note_id:
            raise HTTPException(status_code=500, detail="Failed to save note style")
        
        # Generate preview
        preview_text = get_preview_text(note_text)
        
        return NoteStyleResponse(
            id=note_id,
            user_id=request.user_id,
            note_name=request.note_name,
            preview_text=preview_text,
            file_type=request.file_type,
            created_at=datetime.now().isoformat(),
            is_active=True,  # MVP: one style per user, so new uploads are always active
            validation_info=validation_info,
            style_analysis=style_analysis
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Note style upload failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during note processing")

@app.get("/note-styles/{user_id}")
def api_list_note_styles(user_id: str) -> ListNoteStylesResponse:
    """List all note styles for a user."""
    from app.services.data_access import list_note_styles
    from app.services.note_style import get_preview_text
    
    note_styles_data = list_note_styles(user_id)
    
    note_styles = []
    for style_data in note_styles_data:
        preview_text = get_preview_text(style_data.get("note_text", ""))
        
        note_styles.append(NoteStyleResponse(
            id=style_data["id"],
            user_id=style_data["user_id"],
            note_name=style_data["note_name"],
            preview_text=preview_text,
            file_type=style_data["file_type"],
            created_at=style_data["created_at"],
            is_active=style_data.get("is_active", False),
            validation_info=style_data.get("validation_info"),
            style_analysis=style_data.get("style_analysis")
        ))
    
    return ListNoteStylesResponse(
        note_styles=note_styles,
        total_count=len(note_styles)
    )

@app.post("/note-styles/set-active")
def api_set_active_note_style(request: SetActiveNoteStyleRequest) -> Dict[str, str]:
    """Set a specific note style as active for the user."""
    from app.services.data_access import set_active_note_style
    
    success = set_active_note_style(request.user_id, request.note_style_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Note style not found or access denied")
    
    return {"message": "Note style activated successfully", "note_style_id": request.note_style_id}

@app.delete("/note-styles/{user_id}/{note_style_id}")
def api_delete_note_style(user_id: str, note_style_id: str) -> Dict[str, str]:
    """Delete a note style."""
    from app.services.data_access import delete_note_style
    
    success = delete_note_style(user_id, note_style_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Note style not found or access denied")
    
    return {"message": "Note style deleted successfully", "note_style_id": note_style_id}

@app.post("/notes/generate-with-style")
def api_generate_notes_with_style(request: GenerateNotesWithStyleRequest) -> GenerateNotesWithStyleResponse:
    """
    Generate clinical notes with optional style matching.
    
    If use_note_style=True and user has an active note style, 
    the generated notes will match that style. Otherwise, 
    falls back to standard note generation.
    """
    import uuid
    request_id = uuid.uuid4().hex[:8]
    logger.info("[%s] Style-matched note generation started", request_id)
    from datetime import datetime
    
    try:
        # Generate notes with style matching
        notes_result = generate_therapist_notes_with_style(
            transcript_text=request.transcript_text,
            transcript_segments=request.transcript_segments,
            session_summary=request.session_summary,
            patient_id=request.patient_id,
            user_id=request.user_id,
            use_note_style=request.use_note_style
        )
        
        if not notes_result:
            raise HTTPException(status_code=500, detail="Failed to generate notes")
        
        # Handle both style-matched and standard note formats
        if notes_result.get("format") == "style_matched":
            return GenerateNotesWithStyleResponse(
                format="style_matched",
                content=notes_result["content"],
                style_source=notes_result.get("style_source"),
                patient_id=request.patient_id,
                generated_at=notes_result.get("generated_at", datetime.now().isoformat()),
                style_info=notes_result.get("style_info")
            )
        else:
            # Standard format - convert to markdown
            from app.services.notes import _convert_notes_to_markdown
            markdown_content = _convert_notes_to_markdown(notes_result)
            
            return GenerateNotesWithStyleResponse(
                format="standard",
                content=markdown_content,
                style_source=None,
                patient_id=request.patient_id,
                generated_at=datetime.now().isoformat(),
                style_info=None
            )
        
    except Exception as e:
        logger.error(f"Notes generation with style failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during note generation")
