"""
Stage 3: Slide Change Detection

Detects slide transitions using SSIM (Structural Similarity Index).

Detection strategy (dual-comparison for higher recall):
  1. CONSECUTIVE comparison: current frame vs previous frame
     → catches sudden transitions (page flips, new slides)
  2. ANCHOR comparison: current frame vs first frame of current slide
     → catches gradual changes (text appearing line-by-line, animations
       building up) that don't trigger between consecutive frames

A slide change fires if EITHER comparison drops below its threshold,
subject to a minimum cooldown to suppress noise.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from imageio.v3 import imread
from skimage.metrics import structural_similarity
from tqdm import tqdm

from .config import FRAME_FPS, MIN_SLIDE_DURATION_SEC, SSIM_THRESHOLD, logger


def load_frame(frame_path: str, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Load and preprocess a frame for SSIM comparison.

    Args:
        frame_path: Path to frame image
        target_size: Size to resize frame to

    Returns:
        Grayscale numpy array
    """
    # Read image
    img = imread(frame_path)

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize for consistent comparison
    img = cv2.resize(img, target_size)

    return img


def detect_slides(
    frames_dir: str,
    threshold: float = SSIM_THRESHOLD,
    min_slide_duration: int = MIN_SLIDE_DURATION_SEC,
    anchor_threshold: float = None,
) -> List[Dict[str, Any]]:
    """
    Detect slide changes using dual SSIM comparison for high recall.

    Two comparisons run in parallel:
      - Consecutive: ssim(frame[i], frame[i-1]) < threshold
        → catches sudden jumps
      - Anchor: ssim(frame[i], anchor_frame) < anchor_threshold
        → catches gradual accumulation of changes since last slide started

    Args:
        frames_dir: Directory containing extracted frames
        threshold: SSIM threshold for consecutive comparison
        min_slide_duration: Minimum seconds between two slide changes
        anchor_threshold: SSIM threshold for anchor comparison.
            Defaults to threshold + 0.15 (more lenient because anchor
            drift is expected even on the same slide).

    Returns:
        List of slide metadata dictionaries
    """
    if anchor_threshold is None:
        # Anchor comparison uses a LOWER threshold (less sensitive)
        # because even the same slide drifts over time due to cursor,
        # highlights, compression. Only fire when the frame is truly
        # different from where the slide started.
        anchor_threshold = max(threshold - 0.23, 0.40)

    logger.info(f"Detecting slide changes (consecutive threshold: {threshold}, "
                f"anchor threshold: {anchor_threshold}, "
                f"min duration: {min_slide_duration}s)")

    # Get all frame files
    frame_files = sorted(Path(frames_dir).glob("frame_*.jpg"))

    if len(frame_files) < 2:
        logger.error("Not enough frames for slide detection")
        return []

    slides = []
    current_slide_start = 0
    slide_id = 0

    # Track the frame index of the last confirmed slide change
    last_change_frame_idx = 0

    prev_frame = None
    prev_frame_path = None

    # The "anchor" frame: first frame of the current slide segment
    anchor_frame = None

    for i, frame_file in enumerate(tqdm(frame_files, desc="Analyzing frames")):
        frame_idx = int(frame_file.stem.split("_")[1])
        timestamp = frame_idx * (1 / FRAME_FPS)

        try:
            curr_frame = load_frame(str(frame_file))

            # Set anchor for the very first frame
            if anchor_frame is None:
                anchor_frame = curr_frame

            if prev_frame is not None:
                # ---- Consecutive comparison ----
                ssim_consecutive = structural_similarity(prev_frame, curr_frame)

                # ---- Anchor comparison ----
                ssim_anchor = structural_similarity(anchor_frame, curr_frame)

                # Check cooldown
                frames_since_last_change = frame_idx - last_change_frame_idx
                seconds_since_last_change = frames_since_last_change / FRAME_FPS

                # Fire slide change if EITHER comparison triggers
                consecutive_triggered = ssim_consecutive < threshold
                anchor_triggered = ssim_anchor < anchor_threshold

                if (consecutive_triggered or anchor_triggered) and \
                   seconds_since_last_change >= min_slide_duration:

                    trigger_reason = []
                    if consecutive_triggered:
                        trigger_reason.append(f"consecutive={ssim_consecutive:.4f}<{threshold}")
                    if anchor_triggered:
                        trigger_reason.append(f"anchor={ssim_anchor:.4f}<{anchor_threshold}")

                    # End current slide
                    slide_data = {
                        "slide_id": slide_id,
                        "start_time": round(current_slide_start * (1 / FRAME_FPS), 3),
                        "end_time": round(timestamp, 3),
                        "last_frame_path": str(prev_frame_path),
                        "last_frame_index": i - 1,
                        "is_last_frame_before_change": True
                    }
                    slides.append(slide_data)

                    logger.debug(f"Slide change at frame {i}: {', '.join(trigger_reason)}, "
                                 f"gap={seconds_since_last_change:.1f}s")

                    # Start new slide
                    slide_id += 1
                    current_slide_start = frame_idx
                    last_change_frame_idx = frame_idx

                    # Reset anchor to current frame (start of new slide)
                    anchor_frame = curr_frame

            prev_frame = curr_frame
            prev_frame_path = frame_file

        except Exception as e:
            logger.warning(f"Error processing frame {frame_file}: {e}")
            continue

    # Add final slide
    if prev_frame is not None:
        final_timestamp = (len(frame_files) - 1) * (1 / FRAME_FPS)
        slide_data = {
            "slide_id": slide_id,
            "start_time": round(current_slide_start * (1 / FRAME_FPS), 3),
            "end_time": round(final_timestamp, 3),
            "last_frame_path": str(prev_frame_path),
            "last_frame_index": len(frame_files) - 1,
            "is_last_frame_before_change": True
        }
        slides.append(slide_data)

    logger.info(f"Detected {len(slides)} slides")
    return slides
