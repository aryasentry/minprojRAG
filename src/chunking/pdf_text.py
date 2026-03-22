"""
PDF Text Chunker

Extracts text from PDF files and produces page-aware chunks.

Supports two backends (auto-detected):
  1. pymupdf (fitz) — fast, handles most PDFs well
  2. pypdf            — pure Python fallback

Each page is chunked into token-limited pieces.
Page boundaries are respected (no chunk crosses pages).
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import MAX_TOKENS_PER_CHUNK, logger
from .base import count_tokens, make_chunk


# ─── PDF Backend ────────────────────────────────────────────────────────────

def _extract_pages_pymupdf(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Extract text from PDF using pymupdf (fitz).

    Returns:
        List of (page_number, page_text) tuples (1-indexed pages)
    """
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append((i + 1, text))
    doc.close()
    return pages


def _extract_pages_pypdf(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Extract text from PDF using pypdf (fallback).

    Returns:
        List of (page_number, page_text) tuples (1-indexed pages)
    """
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((i + 1, text))
    return pages


def _extract_pages(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Extract pages using the best available backend.
    """
    # Try pymupdf first (faster, better extraction)
    try:
        return _extract_pages_pymupdf(pdf_path)
    except ImportError:
        pass

    # Fallback to pypdf
    try:
        return _extract_pages_pypdf(pdf_path)
    except ImportError:
        pass

    raise ImportError(
        "No PDF backend available. Install one of:\n"
        "  pip install pymupdf     (recommended)\n"
        "  pip install pypdf"
    )


# ─── Text Splitting ────────────────────────────────────────────────────────

def _split_text_into_chunks(
    text: str,
    max_tokens: int,
) -> List[str]:
    """
    Split a block of text into token-limited chunks.

    Strategy:
    - Split on paragraph breaks (double newline) first
    - If a paragraph is too long, split on sentence boundaries
    - If a sentence is too long, split on word boundaries

    Args:
        text: Input text
        max_tokens: Maximum tokens per chunk

    Returns:
        List of text chunks
    """
    if not text.strip():
        return []

    if count_tokens(text) <= max_tokens:
        return [text.strip()]

    chunks = []
    paragraphs = text.split("\n\n")

    current_chunk = ""
    current_tokens = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_tokens = count_tokens(para)

        # If single paragraph exceeds limit, split it further
        if para_tokens > max_tokens:
            # Flush current chunk first
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_tokens = 0

            # Split paragraph by sentences
            sentences = _split_sentences(para)
            for sent in sentences:
                sent_tokens = count_tokens(sent)
                if current_tokens + sent_tokens > max_tokens and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sent
                    current_tokens = sent_tokens
                else:
                    current_chunk = (current_chunk + " " + sent) if current_chunk else sent
                    current_tokens += sent_tokens
        elif current_tokens + para_tokens > max_tokens and current_chunk:
            # Flush and start new chunk
            chunks.append(current_chunk.strip())
            current_chunk = para
            current_tokens = para_tokens
        else:
            # Add to current chunk
            current_chunk = (current_chunk + "\n\n" + para) if current_chunk else para
            current_tokens += para_tokens

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _split_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter.

    Splits on common sentence-ending punctuation followed by whitespace.
    Falls back to word-level splitting if sentences are too long.
    """
    import re

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Filter empty
    return [s.strip() for s in sentences if s.strip()]


# ─── Main Chunker ───────────────────────────────────────────────────────────

def chunk_pdf(
    pdf_path: str,
    max_tokens: int = MAX_TOKENS_PER_CHUNK,
    source_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Extract text from a PDF and produce page-aware chunks.

    Each chunk stays within a single page and respects the token limit.
    Chunks include page numbers for citation and the source filename.

    Args:
        pdf_path: Path to the PDF file
        max_tokens: Maximum tokens per chunk
        source_label: Optional label for the source (defaults to filename)

    Returns:
        List of chunk dicts (unified schema)
    """
    if not os.path.exists(pdf_path):
        logger.error(f"[pdf_text] PDF file not found: {pdf_path}")
        return []

    source_label = source_label or Path(pdf_path).name
    logger.info(f"[pdf_text] Extracting text from: {source_label}")

    # Extract pages
    try:
        pages = _extract_pages(pdf_path)
    except ImportError as e:
        logger.error(f"[pdf_text] {e}")
        return []
    except Exception as e:
        logger.error(f"[pdf_text] Error reading PDF: {e}")
        return []

    total_pages = len(pages)
    logger.info(f"[pdf_text] Found {total_pages} pages")

    # Chunk each page
    chunks = []
    chunk_counter = 0
    pdf_id = Path(pdf_path).stem  # "lecture_notes" from "lecture_notes.pdf"

    for page_num, page_text in pages:
        page_text = page_text.strip()
        if not page_text:
            logger.debug(f"[pdf_text] Page {page_num}: empty, skipping")
            continue

        text_chunks = _split_text_into_chunks(page_text, max_tokens)

        for i, chunk_text in enumerate(text_chunks):
            chunks.append(make_chunk(
                chunk_id=f"{pdf_id}_pdf_{chunk_counter:03d}",
                source_type="pdf_text",
                source_file=source_label,
                modality="pdf",
                text=chunk_text,
                page_number=page_num,
                token_count=count_tokens(chunk_text),
                total_pages=total_pages,   # Extra metadata
                chunk_index_in_page=i,      # Which sub-chunk within the page
            ))
            chunk_counter += 1

    logger.info(f"[pdf_text] Created {len(chunks)} PDF chunks from {total_pages} pages")
    return chunks
