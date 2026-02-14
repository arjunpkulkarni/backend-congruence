import os
import subprocess
import shutil
from typing import Optional

import requests


def _ensure_ffmpeg_exists() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not in PATH. Please install ffmpeg and retry.")


def download_video_file(video_url: str, destination_path: str, timeout: int = 60) -> None:
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    with requests.get(video_url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(destination_path, "wb") as dest_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    dest_file.write(chunk)


def extract_audio_with_ffmpeg(input_video_path: str, output_audio_path: str) -> None:
    _ensure_ffmpeg_exists()
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
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


def extract_frames_with_ffmpeg(
    input_video_path: str,
    frames_dir: str,
    fps: float = 1,
    filename_pattern: str = "frame_%04d.png",
) -> None:
    _ensure_ffmpeg_exists()
    os.makedirs(frames_dir, exist_ok=True)
    output_pattern = os.path.join(frames_dir, filename_pattern)
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
        "-vsync", "vfr",
        # Ignore rotation metadata after applying it (prevents double rotation)
        "-metadata:s:v", "rotate=0",
        output_pattern,
    ]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {completed.stderr.decode(errors='ignore')}")


