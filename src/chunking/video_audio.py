"""
Video Audio Chunker

Produces transcript chunks from Whisper output, aligned to slide boundaries.

Each chunk contains:
  - The spoken text within a slide's time range
  - Word-level timestamp metadata
  - Slide ID it belongs to
  - Token count

This is the enriched version of stage5_chunking.py,
using the unified make_chunk() factory.
"""

from pathlib import Path
from typing import Any, Dict, List

from ..config import MAX_TOKENS_PER_CHUNK, logger
from .base import count_tokens, make_chunk


def chunk_video_transcript(
    transcript: Dict[str, Any],
    slides: List[Dict[str, Any]],
    source_file: str = "",
    max_tokens: int = MAX_TOKENS_PER_CHUNK,
) -> List[Dict[str, Any]]:
    """
    Split transcript into chunks respecting slide boundaries.

    For each slide, collects overlapping transcript segments and
    merges them into chunks that stay under max_tokens.
    No chunk crosses a slide boundary.

    Args:
        transcript: Transcript dict with "video_id" and "segments"
        slides: List of slide metadata dicts
        source_file: Original video file name
        max_tokens: Maximum tokens per chunk

    Returns:
        List of chunk dicts (unified schema)
    """
    logger.info(f"[video_audio] Chunking transcript (max {max_tokens} tokens)")

    chunks = []
    # Get unique video_id from source_file
    if source_file:
        video_id = Path(source_file).stem  # Get filename without extension
    else:
        video_id = transcript.get("video_id", "video")
    chunk_counter = 0
    segments = transcript.get("segments", [])

    for slide in slides:
        slide_id = slide["slide_id"]
        slide_start = slide["start_time"]
        slide_end = slide["end_time"]

        # ── Collect segments overlapping this slide ──────────────────
        slide_segments = []
        for seg in segments:
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            if seg_start <= slide_end and seg_end >= slide_start:
                slide_segments.append(seg)

        if not slide_segments:
            continue

        # ── Merge segments into token-limited chunks ────────────────
        cur_text = ""
        cur_start = None
        cur_end = None
        cur_tokens = 0
        cur_words = []  # Accumulate word-level timestamps

        for seg in slide_segments:
            seg_text = seg.get("text", "").strip()
            if not seg_text:
                continue

            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            seg_tokens = count_tokens(seg_text)
            seg_words = seg.get("words", [])

            if cur_tokens + seg_tokens > max_tokens and cur_text:
                # ── Flush current chunk ──
                chunks.append(make_chunk(
                    chunk_id=f"{video_id}_transcript_{chunk_counter:03d}",
                    source_type="video_transcript",
                    source_file=source_file,
                    modality="transcript",
                    text=cur_text.strip(),
                    start_sec=cur_start,
                    end_sec=cur_end,
                    video_id=video_id,
                    slide_id=slide_id,
                    token_count=cur_tokens,
                    words=cur_words,  # Extra: word-level timestamps
                ))
                chunk_counter += 1

                # Start new chunk
                cur_text = seg_text
                cur_start = seg_start
                cur_end = seg_end
                cur_tokens = seg_tokens
                cur_words = list(seg_words)
            else:
                # Accumulate
                cur_text = (cur_text + " " + seg_text) if cur_text else seg_text
                if cur_start is None:
                    cur_start = seg_start
                cur_end = seg_end
                cur_tokens += seg_tokens
                cur_words.extend(seg_words)

        # ── Flush final chunk for this slide ──
        if cur_text:
            chunks.append(make_chunk(
                chunk_id=f"{video_id}_transcript_{chunk_counter:03d}",
                source_type="video_transcript",
                source_file=source_file,
                modality="transcript",
                text=cur_text.strip(),
                start_sec=cur_start,
                end_sec=cur_end,
                video_id=video_id,
                slide_id=slide_id,
                token_count=cur_tokens,
                words=cur_words,
            ))
            chunk_counter += 1

    logger.info(f"[video_audio] Created {len(chunks)} transcript chunks")
    return chunks
