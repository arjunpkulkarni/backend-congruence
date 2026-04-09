from typing import Any, Dict, List, Optional, Tuple
import os
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("emotion_api.transcription")


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode == 0:
            return float(result.stdout.decode().strip())
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# OpenAI Whisper API transcription (fast, cloud-based)
# ---------------------------------------------------------------------------

def _split_audio_into_chunks(audio_path: str, chunk_seconds: int = 120) -> List[Tuple[str, float]]:
    """
    Split audio into chunk files using ffmpeg.
    Returns list of (chunk_path, start_offset_seconds).
    """
    duration = get_audio_duration(audio_path)
    if duration <= 0:
        return [(audio_path, 0.0)]

    chunks: List[Tuple[str, float]] = []
    chunk_idx = 0
    start = 0.0

    while start < duration:
        length = min(chunk_seconds, duration - start)
        chunk_path = f"{audio_path}.chunk_{chunk_idx}.wav"
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-t", str(length),
            "-i", audio_path,
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            chunk_path,
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        chunks.append((chunk_path, start))
        start += chunk_seconds
        chunk_idx += 1

    return chunks


def _transcribe_chunk_openai(chunk_path: str, start_offset: float, language: str = "en") -> Tuple[List[Dict[str, Any]], str]:
    """Transcribe a single chunk via the OpenAI Whisper API. Returns (segments, text)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return [], ""

    from openai import OpenAI
    client = OpenAI(api_key=api_key.strip())

    with open(chunk_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    segments: List[Dict[str, Any]] = []
    text_parts: List[str] = []

    for seg in getattr(resp, "segments", []) or []:
        segments.append({
            "start": float(seg["start"]) + start_offset,
            "end": float(seg["end"]) + start_offset,
            "text": seg.get("text", ""),
        })
        t = seg.get("text", "").strip()
        if t:
            text_parts.append(t)

    full_text = getattr(resp, "text", "") or " ".join(text_parts)
    return segments, full_text.strip()


def transcribe_with_openai_api(
    audio_path: str,
    language: str = "en",
    chunk_seconds: int = 120,
    max_workers: int = 6,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Transcribe audio using the OpenAI Whisper API with parallel chunking.
    ~2-min chunks sent in parallel → merges results by timestamp offset.
    Typical speedup: 10-15 min → 20-40 seconds.
    """
    duration = get_audio_duration(audio_path)

    if duration <= chunk_seconds:
        segments, text = _transcribe_chunk_openai(audio_path, 0.0, language)
        return text, segments

    logger.info("Splitting %.0fs audio into ~%ds chunks for parallel transcription", duration, chunk_seconds)
    chunks = _split_audio_into_chunks(audio_path, chunk_seconds)
    logger.info("Created %d chunks, sending to OpenAI Whisper API in parallel (max_workers=%d)", len(chunks), max_workers)

    all_segments: List[Dict[str, Any]] = []
    all_text_by_offset: List[Tuple[float, str]] = []

    def _process(chunk_info: Tuple[str, float]) -> Tuple[float, List[Dict[str, Any]], str]:
        path, offset = chunk_info
        try:
            segs, txt = _transcribe_chunk_openai(path, offset, language)
            return offset, segs, txt
        finally:
            if path != audio_path and os.path.exists(path):
                os.remove(path)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process, c): c for c in chunks}
        for future in as_completed(futures):
            try:
                offset, segs, txt = future.result()
                all_segments.extend(segs)
                all_text_by_offset.append((offset, txt))
            except Exception as exc:
                logger.warning("Chunk transcription failed: %s", exc)

    all_segments.sort(key=lambda s: s["start"])
    all_text_by_offset.sort(key=lambda x: x[0])
    full_text = " ".join(t for _, t in all_text_by_offset).strip()

    logger.info("OpenAI Whisper API transcription complete: %d chars, %d segments", len(full_text), len(all_segments))
    return full_text, all_segments


# ---------------------------------------------------------------------------
# Local faster-whisper fallback
# ---------------------------------------------------------------------------

def transcribe_audio_with_faster_whisper(
    audio_path: str,
    model_size: str = "small",
    language: Optional[str] = "en",
    fast_mode: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception:
        return "", []

    try:
        if fast_mode and model_size == "small":
            model_size = "base"

        model = WhisperModel(model_size, device="auto", compute_type="auto")

        beam_size = 1 if fast_mode else 5
        vad_filter = False

        segments_iter, _info = model.transcribe(
            audio_path,
            language=language,
            vad_filter=vad_filter,
            beam_size=beam_size,
        )
        segments_list: List[Dict[str, Any]] = []
        full_text_parts: List[str] = []
        for seg in segments_iter:
            segments_list.append(
                {"start": float(seg.start), "end": float(seg.end), "text": seg.text}
            )
            if seg.text:
                full_text_parts.append(seg.text.strip())
        full_text = " ".join(full_text_parts).strip()
        return full_text, segments_list
    except Exception:
        return "", []


# ---------------------------------------------------------------------------
# Public entry point (tries OpenAI API first, falls back to local)
# ---------------------------------------------------------------------------

def transcribe_long_audio_chunked(
    audio_path: str,
    model_size: str = "small",
    language: Optional[str] = "en",
    fast_mode: bool = True,
    chunk_duration_minutes: int = 10,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Transcribe audio, preferring the OpenAI Whisper API for speed (~20-40s)
    and falling back to local faster-whisper if the API key is missing or the call fails.
    """
    lang = language or "en"

    def _tdbg(msg, data=None):
        # #region agent log
        import json as _j
        try:
            with open("/Users/arjunkulkarni/Desktop/BCONGRUENCE/.cursor/debug-adf136.log","a") as _f:
                _f.write(_j.dumps({"sessionId":"adf136","hypothesisId":"H2","location":"transcription.py","message":msg,"data":data or {},"timestamp":int(time.time()*1000)})+"\n")
        except Exception:
            pass
        # #endregion

    if os.getenv("OPENAI_API_KEY"):
        try:
            # #region agent log
            _tdbg("whisper_api_path", {"has_key": True, "audio_path": audio_path})
            # #endregion
            logger.info("Using OpenAI Whisper API for transcription (fast path)")
            text, segments = transcribe_with_openai_api(audio_path, language=lang)
            # #region agent log
            _tdbg("whisper_api_result", {"text_len": len(text or ""), "segments": len(segments or [])})
            # #endregion
            if text:
                return text, segments
            logger.warning("OpenAI Whisper API returned empty result, falling back to local model")
        except Exception as exc:
            # #region agent log
            _tdbg("whisper_api_failed", {"error": str(exc), "type": type(exc).__name__})
            # #endregion
            logger.warning("OpenAI Whisper API failed (%s), falling back to local model", exc)

    logger.info("Using local faster-whisper model (slow path, model=%s)", model_size)
    duration = get_audio_duration(audio_path)

    if duration <= (chunk_duration_minutes * 60):
        return transcribe_audio_with_faster_whisper(audio_path, model_size, lang, fast_mode)

    try:
        from faster_whisper import WhisperModel
    except Exception:
        return "", []

    try:
        if duration > 1800:
            model_size = "tiny" if fast_mode else "base"

        model = WhisperModel(model_size, device="auto", compute_type="auto")

        chunk_duration_seconds = chunk_duration_minutes * 60
        num_chunks = int(duration / chunk_duration_seconds) + 1

        all_segments: List[Dict[str, Any]] = []
        all_text_parts: List[str] = []

        for chunk_idx in range(num_chunks):
            start_time = chunk_idx * chunk_duration_seconds
            end_time = min(start_time + chunk_duration_seconds, duration)

            if start_time >= duration:
                break

            chunk_path = f"{audio_path}.chunk_{chunk_idx}.wav"

            try:
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start_time),
                    "-t", str(end_time - start_time),
                    "-i", audio_path,
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    chunk_path,
                ]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

                beam_size = 1 if fast_mode else 3
                segments_iter, _info = model.transcribe(
                    chunk_path,
                    language=lang,
                    vad_filter=False,
                    beam_size=beam_size,
                )

                for seg in segments_iter:
                    all_segments.append({
                        "start": float(seg.start) + start_time,
                        "end": float(seg.end) + start_time,
                        "text": seg.text,
                    })
                    if seg.text:
                        all_text_parts.append(seg.text.strip())
            finally:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)

        full_text = " ".join(all_text_parts).strip()
        return full_text, all_segments

    except Exception:
        return transcribe_audio_with_faster_whisper(audio_path, model_size, lang, fast_mode)
