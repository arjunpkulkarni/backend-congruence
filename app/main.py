import os
import time
import shutil
from typing import Dict, Any
import logging
import contextlib
import glob
import asyncio

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import ProcessSessionRequest, ProcessSessionResponse, AgentChatRequest, AgentChatResponse
from app.services.video_processing import (
    download_video_file,
    extract_audio_with_ffmpeg,
    extract_frames_with_ffmpeg,
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
from app.services.agent import get_agent
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


app = FastAPI(title="Emotion Analysis API", version="0.1.0")

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
    """
    Check if OpenAI API key is configured (safe for frontend to call).
    Returns masked key preview for verification without exposing full key.
    """
    logger.info("API key status endpoint called")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return {
            "configured": False,
            "present": False,
            "message": "OPENAI_API_KEY not configured"
        }
    
    # Show only first 7 and last 4 characters for verification
    masked_key = f"{api_key[:7]}...{api_key[-4:]}" if len(api_key) > 11 else "***"
    
    return {
        "configured": True,
        "present": True,
        "key_preview": masked_key,
        "key_length": len(api_key),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "message": "OpenAI API key is configured"
    }


@app.post("/process_session", response_model=ProcessSessionResponse)
def process_session(payload: ProcessSessionRequest) -> ProcessSessionResponse:
    """
    1) Download video
    2) Extract audio via ffmpeg
    3) Extract frames (1 FPS) via ffmpeg
    4) DeepFace on frames -> facial emotions timeline
    5) Vesper audio emotions (if available)
    6) Merge timelines
    7) Detect micro-spikes
    """
    # Prepare session directories under workspace
    start_time = time.time()
    logger.info(
        "process_session START patient_id=%s video_url=%s",
        payload.patient_id,
        payload.video_url,
    )
    workspace_root = get_workspace_root()
    session_ts = int(time.time())
    session_dir, media_dir, frames_dir, outputs_dir = create_session_directories(
        workspace_root=workspace_root,
        patient_id=payload.patient_id,
        session_ts=session_ts,
    )

    video_path = os.path.join(media_dir, "input.mp4")
    audio_path = os.path.join(media_dir, "audio.wav")

    # 1) Download video
    try:
        download_video_file(video_url=payload.video_url, destination_path=video_path)
    except Exception as exc:
        # Best-effort cleanup for partially downloaded files
        with contextlib.suppress(Exception):
            if os.path.isdir(session_dir):
                shutil.rmtree(session_dir)
        logger.exception("Video download failed")
        raise HTTPException(status_code=400, detail=f"Video download failed: {exc}") from exc
    logger.info("Video downloaded to %s", video_path)

    # 2) Extract audio
    try:
        extract_audio_with_ffmpeg(input_video_path=video_path, output_audio_path=audio_path)
    except Exception as exc:
        logger.exception("Audio extraction failed")
        raise HTTPException(status_code=500, detail=f"Audio extraction failed: {exc}") from exc
    logger.info("Audio extracted to %s", audio_path)

    # 3) Extract frames (0.3 FPS = 1 frame per ~3 seconds for therapy analysis)
    try:
        extract_frames_with_ffmpeg(
            input_video_path=video_path,
            frames_dir=frames_dir,
            fps=0.3,
            filename_pattern="frame_%04d.png",
        )
    except Exception as exc:
        logger.exception("Frame extraction failed")
        raise HTTPException(status_code=500, detail=f"Frame extraction failed: {exc}") from exc
    frame_count = len(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    logger.info("Frames extracted to %s count=%d", frames_dir, frame_count)

    logger.info("Starting parallel analysis (transcription + DeepFace + audio emotion)...")
    parallel_start = time.time()
    
    async def run_parallel_analysis():
        loop = asyncio.get_event_loop()
        
        # Async wrapper for transcription
        async def async_transcription():
            try:
                t_text, t_segments = await loop.run_in_executor(
                    None,
                    transcribe_audio_with_faster_whisper,
                    audio_path
                )
                if t_text:
                    logger.info(
                        "Transcription completed chars=%d segments=%d",
                        len(t_text or ""),
                        len(t_segments or []),
                    )
                    return t_text, t_segments
                return None, None
            except Exception:
                logger.info("Transcription skipped or failed (best-effort)")
                return None, None
        
        # Async wrapper for DeepFace
        async def async_deepface():
            try:
                result = await loop.run_in_executor(
                    None,
                    analyze_frames_with_deepface,
                    frames_dir
                )
                logger.info("DeepFace analysis completed entries=%d", len(result))
                return result
            except Exception as exc:
                logger.exception("DeepFace analysis failed")
                raise HTTPException(status_code=500, detail=f"DeepFace analysis failed: {exc}") from exc
        
        async def async_audio_emotion():
            try:
                result = await loop.run_in_executor(
                    None,
                    analyze_audio_with_vesper,
                    audio_path
                )
                logger.info("Audio emotion timeline entries=%d", len(result))
                return result
            except Exception as exc:
                logger.exception("Audio emotion analysis failed (Vesper required)")
                raise HTTPException(status_code=500, detail=f"Audio emotion analysis failed: {exc}") from exc
        
        # Run all three tasks in parallel
        results = await asyncio.gather(
            async_transcription(),
            async_deepface(),
            async_audio_emotion(),
            return_exceptions=False
        )
        
        return results
    
    try:
        parallel_results = asyncio.run(run_parallel_analysis())
        (transcript_text, transcript_segments), facial_timeline, audio_timeline = parallel_results
        
        parallel_duration = time.time() - parallel_start
        logger.info("Parallel analysis completed in %.2fs (saved ~40-60s vs sequential)", parallel_duration)
        
        if transcript_text:
            logger.info("Transcript text:\n%s", transcript_text)
        if transcript_segments:
            logger.debug("Transcript segments: %s", transcript_segments)
            
    except Exception as exc:
        logger.exception("Parallel analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis pipeline failed: {exc}") from exc

    # 6) Merge into a unified timeline
    merged_timeline = merge_timelines(facial_timeline=facial_timeline, audio_timeline=audio_timeline)
    logger.info("Merged timeline entries=%d", len(merged_timeline))

    merged_timeline = detect_micro_spikes(merged_timeline, threshold=payload.spike_threshold)
    spikes = [e for e in merged_timeline if e.get("micro_spike")]
    logger.info("Detected spikes=%d", len(spikes))

    # 8) Build 10Hz congruence signal and session summary and write outputs for UI
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
        timeline_json_path = os.path.join(outputs_dir, "timeline.json")
        timeline_1hz_path = os.path.join(outputs_dir, "timeline_1hz.json")
        spikes_json_path = os.path.join(outputs_dir, "spikes.json")
        session_summary_path = os.path.join(outputs_dir, "session_summary.json")
        _write_json(timeline_json_path, congruence_timeline_10hz)
        _write_json(timeline_1hz_path, merged_timeline)
        _write_json(spikes_json_path, spikes)
        _write_json(session_summary_path, session_summary)
        if transcript_text:
            transcript_txt_path = os.path.join(outputs_dir, "transcript.txt")
            with open(transcript_txt_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)
        if transcript_segments:
            transcript_segments_path = os.path.join(outputs_dir, "transcript_segments.json")
            _write_json(transcript_segments_path, transcript_segments)
        logger.info("Wrote enriched timeline and session summary to outputs/")
        
        # 9) Run simplified analysis (3-signal approach)
        logger.info("Running simplified analysis (3 signals)...")
        try:
            simplified_results = run_simplified_analysis(
                merged_timeline=merged_timeline,
                transcript_segments=transcript_segments,
                patient_id=payload.patient_id,
                session_id=session_ts,
                sessions_root=os.path.join(workspace_root, "sessions")
            )
            
            duration_seconds = len(merged_timeline) 
            simplified_notes_md = generate_simplified_notes(
                analysis_results=simplified_results,
                patient_id=payload.patient_id,
                session_id=session_ts,
                duration=duration_seconds
            )
            
            # Save simplified outputs
            save_simplified_outputs(
                analysis_results=simplified_results,
                notes_markdown=simplified_notes_md,
                output_dir=outputs_dir
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
                    # Save therapist notes to file
                    therapist_notes_path = os.path.join(outputs_dir, "therapist_notes.md")
                    save_therapist_notes(therapist_notes, therapist_notes_path)
                    logger.info("Therapist notes generated and saved (%d chars)", len(therapist_notes))
                else:
                    logger.info("Therapist notes generation skipped (API key not configured or generation failed)")
            except Exception as exc:
                logger.exception("Therapist notes generation failed (non-critical): %s", exc)
        
    except Exception as exc:
        logger.exception("Failed to write enriched outputs: %s", exc)

    resp = ProcessSessionResponse(
        patient_id=payload.patient_id,
        session_timestamp=session_ts,
        paths={
            "session_dir": session_dir,
            "media_dir": media_dir,
            "frames_dir": frames_dir,
            "audio_path": audio_path,
            "video_path": video_path,
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


@app.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest) -> AgentChatResponse:
    """
    Congruence Ops Agent chat endpoint
    
    Processes natural language requests for clinical operations including:
    - Clinical documentation
    - Patient management  
    - Insurance workflows
    - Practice analytics
    """
    logger.info(
        "Agent chat request: user_id=%s role=%s message_length=%d",
        request.user_id,
        request.role, 
        len(request.message)
    )
    
    try:
        agent = get_agent()
        response = await agent.process_message(request)
        
        logger.info(
            "Agent response: tools_used=%s actions_count=%d",
            response.tools_used,
            len(response.actions)
        )
        
        return response
        
    except Exception as exc:
        logger.exception("Agent chat failed")
        raise HTTPException(
            status_code=500, 
            detail=f"Agent processing failed: {exc}"
        ) from exc


@app.get("/agent/status")
def agent_status() -> Dict[str, Any]:
    """Check agent system status"""
    try:
        agent = get_agent()
        return {
            "status": "ready",
            "model": agent.llm.model_name,
            "tools_count": len(agent.tools),
            "message": "Congruence Ops Agent is ready"
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Agent initialization failed: {e}"
        }

