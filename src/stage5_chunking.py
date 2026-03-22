"""
Stage 5: Slide-Aware Transcript Chunking

Splits the transcript into token-limited chunks that respect slide boundaries.
"""

from typing import Any, Dict, List

from .config import MAX_TOKENS_PER_CHUNK, logger


def count_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Simple approximation: ~4 characters per token for English.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    # Simple heuristic: average 4 chars per token
    return max(1, len(text) // 4)


def chunk_transcript(
    transcript: Dict[str, Any],
    slides: List[Dict[str, Any]],
    max_tokens: int = MAX_TOKENS_PER_CHUNK
) -> List[Dict[str, Any]]:
    """
    Split transcript into chunks respecting slide boundaries.

    Args:
        transcript: Transcript dictionary
        slides: List of slide metadata
        max_tokens: Maximum tokens per chunk

    Returns:
        List of chunk dictionaries
    """
    logger.info(f"Chunking transcript (max {max_tokens} tokens per chunk)")

    chunks = []
    video_id = transcript.get("video_id", "video")
    chunk_counter = 0

    segments = transcript.get("segments", [])

    for slide in slides:
        slide_id = slide["slide_id"]
        slide_start = slide["start_time"]
        slide_end = slide["end_time"]

        logger.debug(f"Processing slide {slide_id}: {slide_start}s - {slide_end}s")

        # Collect segments within this slide's time range
        slide_segments = []
        for seg in segments:
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)

            # Check if segment overlaps with slide time range
            if seg_start <= slide_end and seg_end >= slide_start:
                slide_segments.append(seg)

        if not slide_segments:
            logger.debug(f"No transcript segments for slide {slide_id}")
            continue

        # Merge segments into chunks
        current_chunk_text = ""
        current_chunk_start = None
        current_chunk_end = None
        current_token_count = 0

        for seg in slide_segments:
            seg_text = seg.get("text", "").strip()
            if not seg_text:
                continue

            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            seg_tokens = count_tokens(seg_text)

            # Check if adding this segment exceeds token limit
            if current_token_count + seg_tokens > max_tokens and current_chunk_text:
                # Save current chunk
                chunk_data = {
                    "chunk_id": f"{video_id}_chunk_{chunk_counter:03d}",
                    "video_id": video_id,
                    "slide_id": slide_id,
                    "modality": "transcript",
                    "text": current_chunk_text.strip(),
                    "start_time": round(current_chunk_start, 3) if current_chunk_start else 0,
                    "end_time": round(current_chunk_end, 3) if current_chunk_end else 0,
                    "token_count": current_token_count
                }
                chunks.append(chunk_data)
                chunk_counter += 1

                # Start new chunk
                current_chunk_text = seg_text
                current_chunk_start = seg_start
                current_chunk_end = seg_end
                current_token_count = seg_tokens
            else:
                # Add to current chunk
                if current_chunk_text:
                    current_chunk_text += " " + seg_text
                else:
                    current_chunk_text = seg_text

                if current_chunk_start is None:
                    current_chunk_start = seg_start
                current_chunk_end = seg_end
                current_token_count += seg_tokens

        # Save final chunk for this slide
        if current_chunk_text:
            chunk_data = {
                "chunk_id": f"{video_id}_chunk_{chunk_counter:03d}",
                "video_id": video_id,
                "slide_id": slide_id,
                "modality": "transcript",
                "text": current_chunk_text.strip(),
                "start_time": round(current_chunk_start, 3) if current_chunk_start else 0,
                "end_time": round(current_chunk_end, 3) if current_chunk_end else 0,
                "token_count": current_token_count
            }
            chunks.append(chunk_data)
            chunk_counter += 1

    logger.info(f"Created {len(chunks)} transcript chunks")
    return chunks
