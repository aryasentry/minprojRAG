"""
Stage 6: Slide Caption Chunks

Creates separate chunks for slide descriptions (from the vision model).
"""

from typing import Any, Dict, List

from .config import logger


def create_slide_caption_chunks(
    slides: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Create caption chunks for each slide's vision-based description.

    Args:
        slides: List of slide metadata with descriptions

    Returns:
        List of slide caption chunks
    """
    logger.info("Creating slide caption chunks")

    caption_chunks = []
    video_id = "video"  # Default, will be updated from slides context

    for slide in slides:
        slide_id = slide["slide_id"]
        description = slide.get("description", "No description available")

        caption_chunk = {
            "chunk_id": f"video_slide_{slide_id:03d}",
            "video_id": video_id,
            "slide_id": slide_id,
            "modality": "slide_caption",
            "text": description,
            "start_time": slide["start_time"],
            "end_time": slide["end_time"],
            "frame_path": slide.get("last_frame_path", ""),
            "is_last_frame_before_change": slide.get("is_last_frame_before_change", True)
        }
        caption_chunks.append(caption_chunk)

    logger.info(f"Created {len(caption_chunks)} slide caption chunks")
    return caption_chunks
