"""
Chunking Base Utilities

Common functions and the chunk factory used by all chunkers.
Every chunk in the system goes through `make_chunk()` so the
schema is guaranteed consistent.
"""

from typing import Any, Dict, List, Optional


def count_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Simple approximation: ~4 characters per token for English.
    This avoids pulling in a tokenizer dependency.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def make_chunk(
    *,
    chunk_id: str,
    source_type: str,
    modality: str,
    text: str,
    # Timing (for video)
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    # Legacy compatibility (kept from existing format)
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    # Identity
    source_file: str = "",
    video_id: str = "",
    slide_id: Optional[int] = None,
    page_number: Optional[int] = None,
    # Visual enrichment
    scene_description: Optional[str] = None,
    ocr_text: Optional[str] = None,
    frame_path: Optional[str] = None,
    is_last_frame_before_change: Optional[bool] = None,
    vision_model: Optional[str] = None,
    # Metrics
    token_count: Optional[int] = None,
    # Extras
    **extra_fields,
) -> Dict[str, Any]:
    """
    Create a standardized chunk dictionary.

    All chunkers use this factory so the output schema is guaranteed
    consistent across video-audio, video-visual, and PDF chunks.

    The schema keeps ALL existing fields (chunk_id, video_id, slide_id,
    modality, text, start_time, end_time, token_count, frame_path,
    is_last_frame_before_change) and adds the enriched fields
    (source_type, source_file, start_sec, end_sec, scene_description,
    ocr_text, page_number, vision_model).

    Returns:
        Chunk dictionary with all fields populated.
    """
    if token_count is None:
        token_count = count_tokens(text)

    # Sync start_sec / start_time (prefer start_sec if given)
    if start_sec is not None and start_time is None:
        start_time = start_sec
    if start_time is not None and start_sec is None:
        start_sec = start_time
    if end_sec is not None and end_time is None:
        end_time = end_sec
    if end_time is not None and end_sec is None:
        end_sec = end_time

    chunk = {
        # --- Core identity ---
        "chunk_id": chunk_id,
        "source_type": source_type,       # "video_transcript" | "video_visual" | "pdf_text"
        "source_file": source_file,
        "modality": modality,             # Legacy: "transcript" | "slide_caption" | "pdf"

        # --- Content ---
        "text": text,
        "token_count": token_count,

        # --- Timing (video) ---
        "start_sec": round(start_sec, 3) if start_sec is not None else None,
        "end_sec": round(end_sec, 3) if end_sec is not None else None,
        "start_time": round(start_time, 3) if start_time is not None else None,
        "end_time": round(end_time, 3) if end_time is not None else None,

        # --- Video identity ---
        "video_id": video_id or None,
        "slide_id": slide_id,

        # --- Visual enrichment ---
        "scene_description": scene_description,
        "ocr_text": ocr_text,
        "frame_path": frame_path,
        "is_last_frame_before_change": is_last_frame_before_change,
        "vision_model": vision_model,

        # --- PDF identity ---
        "page_number": page_number,
    }

    # Merge any extra fields
    chunk.update(extra_fields)

    return chunk


def format_timestamp(seconds: float) -> str:
    """
    Format seconds as HH:MM:SS for display.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string like "00:07:22"
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
