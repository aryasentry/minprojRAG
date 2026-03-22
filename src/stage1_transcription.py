"""
Stage 1: Audio Extraction & Transcription

- Extracts audio from video as WAV using ffmpeg
- Transcribes using faster-whisper with word-level timestamps
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from faster_whisper import WhisperModel

from .config import WHISPER_MODEL, logger




def extract_audio(video_path: str, output_path: str) -> bool:
    """
    Extract audio from video using ffmpeg.

    Args:
        video_path: Path to input video file
        output_path: Path to save extracted audio (WAV format)

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Extracting audio from {video_path}")

    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ffmpeg command to extract audio
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",              # No video
        "-acodec", "pcm_s16le",  # PCM 16-bit
        "-ar", "16000",     # 16kHz sample rate
        "-ac", "1",         # Mono
        "-y",               # Overwrite output
        output_path
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Audio extracted successfully to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg error: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("ffmpeg not found. Please install ffmpeg.")
        return False


# =============================================================================
# TRANSCRIPTION
# =============================================================================

def transcribe_audio(audio_path: str, output_path: str) -> Optional[Dict[str, Any]]:
    """
    Transcribe audio using faster-whisper with word-level timestamps.

    Args:
        audio_path: Path to audio file
        output_path: Path to save transcript JSON

    Returns:
        Transcript dictionary or None if failed
    """
    logger.info(f"Transcribing audio from {audio_path}")

    try:
        # Initialize Whisper model
        logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

        # Transcribe with word timestamps
        logger.info("Starting transcription...")
        segments, info = model.transcribe(
            audio_path,
            word_timestamps=True,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        # Build output structure
        transcript = {
            "video_id": Path(audio_path).stem,
            "segments": []
        }

        segment_id = 0
        for segment in segments:
            words_data = []

            # Extract word-level timestamps
            if segment.words:
                for word in segment.words:
                    words_data.append({
                        "word": word.word,
                        "start": round(word.start, 3),
                        "end": round(word.end, 3)
                    })

            segment_data = {
                "segment_id": segment_id,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text.strip(),
                "words": words_data
            }

            transcript["segments"].append(segment_data)
            segment_id += 1

            if segment_id % 10 == 0:
                logger.info(f"Processed {segment_id} segments...")

        # Save transcript
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)

        logger.info(f"Transcription complete: {len(transcript['segments'])} segments")
        logger.info(f"Transcript saved to {output_path}")

        return transcript

    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return None
