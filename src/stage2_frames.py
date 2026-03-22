"""
Stage 2: Frame Extraction

Extracts frames from video at a specified FPS using ffmpeg.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict

from .config import FRAME_FPS, logger


def extract_frames(video_path: str, output_dir: str, fps: int = FRAME_FPS) -> Dict[int, float]:
    """
    Extract frames from video at specified FPS using ffmpeg.

    Args:
        video_path: Path to input video
        output_dir: Directory to save frames (will create video-specific subdirectory)
        fps: Frames per second to extract

    Returns:
        Dictionary mapping frame index to timestamp
    """
    # Create video-specific subdirectory
    video_name = Path(video_path).stem
    video_frames_dir = os.path.join(output_dir, video_name)
    
    logger.info(f"Extracting frames at {fps} FPS to {video_frames_dir}")
    os.makedirs(video_frames_dir, exist_ok=True)

    # Frame filename pattern in video-specific folder
    frame_pattern = os.path.join(video_frames_dir, "frame_%04d.jpg")

    # ffmpeg command
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-qscale:v", "2",  # High quality
        frame_pattern
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Count extracted frames
        frame_files = sorted(Path(video_frames_dir).glob("frame_*.jpg"))
        num_frames = len(frame_files)

        # Build timestamp mapping: frame_index * (1/fps) = timestamp
        frame_timestamps = {}
        for frame_file in frame_files:
            # Extract frame index from filename
            frame_idx = int(frame_file.stem.split("_")[1])
            timestamp = frame_idx * (1 / fps)
            frame_timestamps[frame_idx] = timestamp

        logger.info(f"Extracted {num_frames} frames to {video_frames_dir}")
        return frame_timestamps

    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg error: {e.stderr}")
        return {}
    except FileNotFoundError:
        logger.error("ffmpeg not found. Please install ffmpeg.")
        return {}
