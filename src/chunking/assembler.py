"""
Chunk Assembler

Merges chunks from all sources (video transcript, video visual, PDF)
into a unified chunks list with globally unique IDs and summary stats.
"""

import json
import os
from typing import Any, Dict, List, Optional

from ..config import logger
from .base import count_tokens


def assemble_chunks(
    *chunk_lists: List[Dict[str, Any]],
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Merge multiple chunk lists into a single unified output.

    Assigns globally sequential chunk IDs and computes summary statistics.
    Optionally writes the result to a JSON file.

    Args:
        *chunk_lists: Variable number of chunk lists to merge.
            Each list is typically the output of one chunker
            (video_audio, video_visual, pdf_text).
        output_path: Optional path to write the result as JSON.

    Returns:
        Dictionary with structure:
        {
            "total_chunks": int,
            "sources": {
                "video_transcript": int,
                "video_visual": int,
                "pdf_text": int,
            },
            "total_tokens": int,
            "chunks": [ ... ]
        }
    """
    # Flatten all lists
    all_chunks = []
    for chunk_list in chunk_lists:
        if chunk_list:
            all_chunks.extend(chunk_list)

    # Assign global sequential index
    for i, chunk in enumerate(all_chunks):
        chunk["global_index"] = i

    # Compute stats
    source_counts = {}
    total_tokens = 0
    for chunk in all_chunks:
        src = chunk.get("source_type", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
        total_tokens += chunk.get("token_count", 0)

    result = {
        "total_chunks": len(all_chunks),
        "sources": source_counts,
        "total_tokens": total_tokens,
        "chunks": all_chunks,
    }

    # Log summary
    logger.info(f"[assembler] Assembled {len(all_chunks)} chunks:")
    for src, count in source_counts.items():
        logger.info(f"  {src}: {count} chunks")
    logger.info(f"  Total tokens: ~{total_tokens}")

    # Write to file
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"[assembler] Written to {output_path}")

    return result


def load_chunks(path: str) -> Dict[str, Any]:
    """
    Load a previously assembled chunks file.

    Args:
        path: Path to chunks JSON file

    Returns:
        Parsed chunks dictionary
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_summary(assembled: Dict[str, Any]) -> None:
    """
    Print a human-readable summary of assembled chunks.
    """
    print(f"\n{'='*60}")
    print(f"CHUNK ASSEMBLY SUMMARY")
    print(f"{'='*60}")
    print(f"Total chunks: {assembled['total_chunks']}")
    print(f"Total tokens: ~{assembled['total_tokens']}")
    print(f"\nBy source:")
    for src, count in assembled.get("sources", {}).items():
        print(f"  {src:20s}: {count:4d} chunks")
    print()

    # Show a few examples from each source type
    chunks = assembled.get("chunks", [])
    seen_types = set()
    for chunk in chunks:
        src = chunk.get("source_type", "unknown")
        if src not in seen_types:
            seen_types.add(src)
            text_preview = chunk.get("text", "")[:120].replace("\n", " ")
            print(f"  Example [{src}]:")
            print(f"    ID:   {chunk.get('chunk_id')}")
            print(f"    Text: {text_preview}...")

            if chunk.get("start_sec") is not None:
                print(f"    Time: {chunk['start_sec']}s → {chunk.get('end_sec')}s")
            if chunk.get("page_number") is not None:
                print(f"    Page: {chunk['page_number']}")
            if chunk.get("scene_description"):
                desc_preview = chunk["scene_description"][:80].replace("\n", " ")
                print(f"    Scene: {desc_preview}...")
            if chunk.get("ocr_text"):
                ocr_preview = chunk["ocr_text"][:80].replace("\n", " ")
                print(f"    OCR:   {ocr_preview}...")
            print()

    print(f"{'='*60}")
