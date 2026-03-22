"""
Standalone Embedding Script

Embeds all chunks from chunks.json into separate FAISS indices by modality.
This script can be run after processing videos to create searchable vector stores.

Usage:
    python embed_chunks.py
    python embed_chunks.py --chunks-path output/custom_chunks.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.embedding import embed_all_chunks
from src.config import OUTPUT_DIR, EMBEDDING_MODEL, logger


def main():
    parser = argparse.ArgumentParser(description="Embed chunks into FAISS vector stores")
    parser.add_argument(
        "--chunks-path",
        type=str,
        default=os.path.join(OUTPUT_DIR, "chunks.json"),
        help="Path to chunks.json file (default: output/chunks.json)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-embedding even if indices exist"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Use incremental embedding (only embed new chunks)"
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("EMBEDDING CHUNKS WITH FAISS")
    logger.info("="*60)
    logger.info(f"Embedding Model: {EMBEDDING_MODEL}")
    
    if args.incremental:
        # Use ChunkManager for incremental embedding
        from src.embedding import embed_new_chunks
        
        logger.info("Mode: INCREMENTAL (only new chunks)")
        success = embed_new_chunks(force_rebuild=args.force)
        
    else:
        # Traditional: embed from combined chunks.json
        logger.info("Mode: FULL (from chunks.json)")
        
        # Validate input
        if not os.path.exists(args.chunks_path):
            logger.error(f"Chunks file not found: {args.chunks_path}")
            sys.exit(1)
        
        # Load and preview chunks
        with open(args.chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        chunks = data.get("chunks", [])
        sources = data.get("sources", {})
        
        logger.info(f"Input: {args.chunks_path}")
        logger.info(f"Total Chunks: {len(chunks)}")
        logger.info(f"Sources breakdown:")
        for source_type, count in sources.items():
            logger.info(f"  - {source_type}: {count}")
        logger.info("="*60)
        
        # Embed
        success = embed_all_chunks(args.chunks_path)
    
    if success:
        logger.info("✅ Embedding completed successfully!")
        logger.info(f"FAISS indices saved to: {os.path.join(OUTPUT_DIR, 'faiss_indices')}")
        logger.info("\nTo test retrieval, run:")
        logger.info("  python -m src.server")
        logger.info("  # Then query: http://localhost:8000/query")
    else:
        logger.error("❌ Embedding failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
