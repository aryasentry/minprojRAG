"""
PDF Processing Script

Ingests a PDF file and produces chunks using the modular chunking system.
Result is saved to output/chunks.json (merging with existing chunks if present,
or creating new ones).
"""

import argparse
import json
import os
import sys
from pathlib import Path

from .chunking import chunk_pdf, assemble_chunks
from .config import OUTPUT_DIR, logger


def process_pdf(pdf_path: str, output_dir: str = OUTPUT_DIR) -> bool:
    """
    Process a PDF file into chunks.

    Args:
        pdf_path: Path to input PDF
        output_dir: Output directory

    Returns:
        True if successful
    """
    logger.info("=" * 60)
    logger.info("Starting PDF Processing")
    logger.info(f"Input: {pdf_path}")
    logger.info("=" * 60)

    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found: {pdf_path}")
        return False

    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate PDF chunks
    pdf_chunks = chunk_pdf(
        pdf_path, 
        source_label=Path(pdf_path).name
    )

    if not pdf_chunks:
        logger.error("No chunks extracted from PDF.")
        return False

    # 2. Load existing chunks if any (to merge)
    existing_chunks_path = os.path.join(output_dir, "chunks.json")
    other_chunks = []
    
    if os.path.exists(existing_chunks_path):
        try:
            with open(existing_chunks_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Filter out previous chunks from this same PDF to avoid duplicates
                # (simple check based on source_file)
                source_name = Path(pdf_path).name
                other_chunks = [
                    c for c in data.get("chunks", []) 
                    if c.get("source_file") != source_name
                ]
                logger.info(f"Loaded {len(other_chunks)} existing chunks to merge.")
        except Exception as e:
            logger.warning(f"Could not load existing chunks: {e}")

    # 3. Assemble
    assemble_chunks(
        other_chunks,
        pdf_chunks,
        output_path=existing_chunks_path
    )

    logger.info(f"Successfully processed PDF. Saved to {existing_chunks_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="PDF Ingestion Script")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")

    args = parser.parse_args()
    success = process_pdf(args.pdf_path, args.output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
