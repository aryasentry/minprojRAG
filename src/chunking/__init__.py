"""
Chunking Package

Modular chunking system for the multimodal RAG pipeline.
Handles three source types:
  - video_audio:  Transcript text aligned to slide boundaries
  - video_visual: Frame images → scene descriptions + OCR text
  - pdf_text:     PDF pages → text chunks with page tracking

Each chunker produces a list of standardized chunk dicts.
The assembler merges them into a single chunks.json.
"""

from .video_audio import chunk_video_transcript
from .video_visual import chunk_video_visual
from .pdf_text import chunk_pdf
from .assembler import assemble_chunks

__all__ = [
    "chunk_video_transcript",
    "chunk_video_visual",
    "chunk_pdf",
    "assemble_chunks",
]
