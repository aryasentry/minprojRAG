"""
Stage 7: Output Assembly

Saves all pipeline outputs (transcript, slides, chunks) to JSON files.
"""

import json
import os
from typing import Any, Dict, List

from .config import logger


def save_outputs(
    transcript: Dict[str, Any],
    slides: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    output_dir: str
) -> bool:
    """
    Save all output files.

    Args:
        transcript: Transcript dictionary
        slides: List of slide metadata (enriched with OCR and descriptions)
        chunks: List of all chunks (combined format for backward compatibility)
        output_dir: Output directory

    Returns:
        True if successful
    """
    logger.info(f"Saving outputs to {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    try:
        # Save transcript
        transcript_path = os.path.join(output_dir, "transcript.json")
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {transcript_path}")

        # Save enriched slides (with OCR and descriptions)
        slides_path = os.path.join(output_dir, "slides.json")
        with open(slides_path, "w", encoding="utf-8") as f:
            json.dump(slides, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {slides_path} (enriched with OCR & descriptions)")

        # Save combined chunks (backward compatibility)
        chunks_path = os.path.join(output_dir, "chunks.json")
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {chunks_path} (combined format)")

        # Note: audio_chunks.json and video_chunks.json are saved by ChunkManager

        return True

    except Exception as e:
        logger.error(f"Error saving outputs: {e}")
        return False
