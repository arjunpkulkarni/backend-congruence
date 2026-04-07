from typing import Any, Dict, List, Optional, Tuple
import os


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
        # Use base model in fast mode for better speed/accuracy tradeoff
        if fast_mode and model_size == "small":
            model_size = "base"
        
        model = WhisperModel(model_size, device="auto", compute_type="auto")
        
        # Fast mode: lower beam size but DISABLE VAD to capture full audio duration
        # VAD was causing early cutoff, missing parts of the session
        beam_size = 1 if fast_mode else 5
        vad_filter = False  # DISABLED: Was cutting off audio prematurely
        
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


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    try:
        import subprocess
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


def transcribe_long_audio_chunked(
    audio_path: str,
    model_size: str = "small", 
    language: Optional[str] = "en",
    fast_mode: bool = True,
    chunk_duration_minutes: int = 10
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Transcribe long audio files by processing in chunks to avoid memory issues.
    """
    duration = get_audio_duration(audio_path)
    
    # If audio is short enough, use regular transcription
    if duration <= (chunk_duration_minutes * 60):
        return transcribe_audio_with_faster_whisper(audio_path, model_size, language, fast_mode)
    
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return "", []
    
    try:
        # Use smaller model for long files to save memory
        if duration > 1800:  # 30+ minutes
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
            
            # Create temporary chunk file
            chunk_path = f"{audio_path}.chunk_{chunk_idx}.wav"
            
            try:
                # Extract chunk using ffmpeg
                import subprocess
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start_time),
                    "-t", str(end_time - start_time),
                    "-i", audio_path,
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    chunk_path
                ]
                
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                
                # Transcribe chunk
                beam_size = 1 if fast_mode else 3  # Smaller beam for long files
                segments_iter, _info = model.transcribe(
                    chunk_path,
                    language=language,
                    vad_filter=False,
                    beam_size=beam_size,
                )
                
                # Adjust timestamps and collect results
                for seg in segments_iter:
                    adjusted_segment = {
                        "start": float(seg.start) + start_time,
                        "end": float(seg.end) + start_time,
                        "text": seg.text
                    }
                    all_segments.append(adjusted_segment)
                    
                    if seg.text:
                        all_text_parts.append(seg.text.strip())
                
            finally:
                # Clean up chunk file
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
        
        full_text = " ".join(all_text_parts).strip()
        return full_text, all_segments
        
    except Exception:
        # Fallback to regular transcription
        return transcribe_audio_with_faster_whisper(audio_path, model_size, language, fast_mode)

