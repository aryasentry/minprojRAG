"""
Split Chunks Script

Splits the combined chunks.json into separate audio_chunks.json and video_chunks.json files.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.chunk_manager import ChunkManager
from src.config import OUTPUT_DIR, logger


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Split chunks.json into separate audio and video files")
    parser.add_argument(
        "--input",
        type=str,
        default="output/chunks.json",
        help="Path to combined chunks.json (default: output/chunks.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("SPLITTING CHUNKS INTO SEPARATE FILES")
    logger.info("="*60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output Dir: {args.output_dir}")
    
    # Create manager and split
    manager = ChunkManager(args.output_dir)
    audio_data, video_data = manager.split_combined_chunks(args.input)
    
    # Show stats
    logger.info("\n" + "="*60)
    logger.info("SPLIT COMPLETE")
    logger.info("="*60)
    logger.info(f"\n📄 Audio Chunks:")
    logger.info(f"   File: {args.output_dir}/audio_chunks.json")
    logger.info(f"   Total: {audio_data['total_chunks']}")
    logger.info(f"   Tokens: {audio_data['total_tokens']}")
    
    logger.info(f"\n📄 Video Chunks:")
    logger.info(f"   File: {args.output_dir}/video_chunks.json")
    logger.info(f"   Total: {video_data['total_chunks']}")
    logger.info(f"   Tokens: {video_data['total_tokens']}")
    
    logger.info("\n✅ Next steps:")
    logger.info("   1. Run: python embed_chunks.py --incremental")
    logger.info("   2. Or: python -m src.chunk_manager stats")
    logger.info("="*60)


if __name__ == "__main__":
    main()
