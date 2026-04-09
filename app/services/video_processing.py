import os
import subprocess
import shutil

import requests


def _ensure_ffmpeg_exists() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not in PATH. Please install ffmpeg and retry.")


def download_video_file(video_url: str, destination_path: str, timeout: int = 1800) -> None:
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    with requests.get(video_url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(destination_path, "wb") as dest_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    dest_file.write(chunk)


def extract_audio_with_ffmpeg(input_video_path: str, output_audio_path: str, fast_mode: bool = False) -> None:
    _ensure_ffmpeg_exists()
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    
    if fast_mode:
        # Fast mode: lower sample rate for faster processing
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_video_path,
            # Explicitly select first audio stream (skip unknown codecs)
            "-map", "0:a:0",
            "-vn",  # No video
            "-acodec",
            "pcm_s16le",
            "-ar", "8000",  # Lower sample rate for faster processing
            "-ac", "1",  # Convert to mono (sufficient for emotion analysis)
            output_audio_path,
        ]
    else:
        # Original quality mode
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_video_path,
            # Explicitly select first audio stream (skip unknown codecs)
            "-map", "0:a:0",
            "-vn",  # No video
            "-acodec",
            "pcm_s16le",
            "-ar", "16000",  # Resample to 16kHz (standard for speech recognition)
            "-ac", "1",  # Convert to mono (sufficient for emotion analysis)
            output_audio_path,
        ]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {completed.stderr.decode(errors='ignore')}")


def has_video_stream(input_path: str) -> bool:
    """Check if a media file has a video stream."""
    _ensure_ffmpeg_exists()
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        input_path
    ]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return completed.returncode == 0 and "video" in completed.stdout.decode().lower()


def extract_frames_with_ffmpeg(
    input_video_path: str,
    frames_dir: str,
    fps: float = 1,
    filename_pattern: str = "frame_%04d.png",
    fast_mode: bool = False,
) -> None:
    _ensure_ffmpeg_exists()
    os.makedirs(frames_dir, exist_ok=True)
    
    # Check if the file has a video stream
    if not has_video_stream(input_video_path):
        raise RuntimeError(f"No video stream found in {input_video_path}. File appears to be audio-only.")
    
    output_pattern = os.path.join(frames_dir, filename_pattern)
    
    if fast_mode:
        # Fast mode: lower quality, smaller resolution for speed
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_video_path,
            # Explicitly select only video stream (skip audio/metadata)
            "-map", "0:v:0",
            # Fast processing: scale down, lower quality, fast preset
            "-vf",
            f"fps={fps},scale=320:240,format=yuv420p",
            # Use fast preset for speed over quality
            "-preset", "ultrafast",
            "-crf", "35",  # Lower quality for speed
            # Handle variable frame rates properly
            "-fps_mode", "vfr",
            # Ignore rotation metadata after applying it
            "-metadata:s:v", "rotate=0",
            output_pattern,
        ]
    else:
        # Original quality mode
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_video_path,
            # Explicitly select only video stream (skip audio/metadata)
            "-map", "0:v:0",
            # Auto-rotate based on metadata, then apply fps filter, then convert to standard 8-bit format
            "-vf",
            f"fps={fps},format=yuv420p",
            # Handle variable frame rates properly (important for fractional fps like 0.3)
            "-fps_mode", "vfr",  # Use -fps_mode instead of deprecated -vsync
            # Ignore rotation metadata after applying it (prevents double rotation)
            "-metadata:s:v", "rotate=0",
            output_pattern,
        ]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {completed.stderr.decode(errors='ignore')}")


def download_audio_file(audio_url: str, destination_path: str, timeout: int = 1800) -> None:
    """Download audio file from URL."""
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    with requests.get(audio_url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(destination_path, "wb") as dest_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    dest_file.write(chunk)


def convert_audio_to_wav(input_audio_path: str, output_audio_path: str, fast_mode: bool = False) -> None:
    """Convert any audio format to WAV using ffmpeg."""
    _ensure_ffmpeg_exists()
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    
    if fast_mode:
        # Fast mode: lower sample rate for faster processing
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_audio_path,
            "-acodec",
            "pcm_s16le",
            "-ar", "8000",   # Lower sample rate for faster processing
            "-ac", "1",      # Convert to mono (sufficient for emotion analysis)
            output_audio_path,
        ]
    else:
        # Original quality mode
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_audio_path,
            "-acodec",
            "pcm_s16le",
            "-ar", "16000",  # Resample to 16kHz (standard for speech recognition)
            "-ac", "1",      # Convert to mono (sufficient for emotion analysis)
            output_audio_path,
        ]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg audio conversion failed: {completed.stderr.decode(errors='ignore')}")


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    _ensure_ffmpeg_exists()
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode == 0:
        try:
            return float(completed.stdout.decode().strip())
        except ValueError:
            return 0.0
    return 0.0


def extract_audio_chunk(
    input_video_path: str, 
    output_audio_path: str, 
    start_time: float, 
    duration: float,
    fast_mode: bool = False
) -> None:
    """Extract a specific chunk of audio from video."""
    _ensure_ffmpeg_exists()
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    
    sample_rate = "8000" if fast_mode else "16000"
    
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_time),  # Start time
        "-t", str(duration),     # Duration
        "-i", input_video_path,
        "-map", "0:a:0",
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", sample_rate,
        "-ac", "1",
        output_audio_path,
    ]
    
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg audio chunk extraction failed: {completed.stderr.decode(errors='ignore')}")


def should_use_chunked_processing(video_path: str, chunk_size_minutes: int = 10) -> bool:
    """Determine if video should be processed in chunks based on duration."""
    duration = get_video_duration(video_path)
    return duration > (chunk_size_minutes * 60)  # Convert minutes to seconds

