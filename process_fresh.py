"""
Fresh Video Processing Script

Processes a video from scratch, clearing all previous outputs.
Simulates uploading a new video through the admin interface.

Usage:
    python process_fresh.py path/to/video.mp4
    python process_fresh.py path/to/video.mp4 --skip-embedding
"""

import os
import sys
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import OUTPUT_DIR, FRAMES_DIR, logger
from src.pipeline import run_pipeline
from src.embedding import embed_new_chunks


def clear_outputs(output_dir: str):
    """
    Clear all previous outputs.
    
    Args:
        output_dir: Output directory to clear
    """
    logger.info("="*60)
    logger.info("CLEARING PREVIOUS OUTPUTS")
    logger.info("="*60)
    
    files_to_remove = [
        "chunks.json",
        "audio_chunks.json",
        "video_chunks.json",
        "chunk_tracking.json",
        "transcript.json",
        "slides.json",
        "test_slides.json",
        "test_assembled_chunks.json"
    ]
    
    dirs_to_remove = [
        "frames",
        "audio",
        "faiss_indices",
        "video_segments"
    ]
    
    # Remove files
    for filename in files_to_remove:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"  Removed: {filename}")
    
    # Remove directories
    for dirname in dirs_to_remove:
        dirpath = os.path.join(output_dir, dirname)
        if os.path.exists(dirpath):
            shutil.rmtree(dirpath)
            logger.info(f"  Removed: {dirname}/")
    
    logger.info("Previous outputs cleared")


def process_video_fresh(video_path: str, skip_embedding: bool = False) -> bool:
    """
    Process a video from scratch.
    
    Args:
        video_path: Path to video file
        skip_embedding: If True, skip embedding step
        
    Returns:
        True if successful
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False
    
    logger.info("="*60)
    logger.info("FRESH VIDEO PROCESSING")
    logger.info("="*60)
    logger.info(f"Video: {video_path}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info("="*60)
    
    # Step 1: Clear previous outputs
    clear_outputs(OUTPUT_DIR)
    
    # Step 2: Run full pipeline
    logger.info("\n" + "="*60)
    logger.info("STEP 1: RUNNING PIPELINE")
    logger.info("="*60)
    
    success = run_pipeline(video_path, OUTPUT_DIR)
    
    if not success:
        logger.error("Pipeline failed!")
        return False
    
    logger.info("Pipeline completed")
    
    # Step 3: Verify chunks were created
    audio_chunks_path = os.path.join(OUTPUT_DIR, "audio_chunks.json")
    video_chunks_path = os.path.join(OUTPUT_DIR, "video_chunks.json")
    
    if not os.path.exists(audio_chunks_path):
        logger.error(f"Audio chunks not found: {audio_chunks_path}")
        return False
    
    if not os.path.exists(video_chunks_path):
        logger.error(f"Video chunks not found: {video_chunks_path}")
        return False
    
    # Step 4: Embed chunks
    if not skip_embedding:
        logger.info("\n" + "="*60)
        logger.info("STEP 2: EMBEDDING CHUNKS")
        logger.info("="*60)
        
        embed_success = embed_new_chunks(force_rebuild=True)
        
        if not embed_success:
            logger.warning("Embedding failed, but pipeline completed")
        else:
            logger.info("Embedding completed")
    else:
        logger.info("\n  Skipping embedding (--skip-embedding)")
    
    # Step 5: Show summary
    import json
    
    with open(audio_chunks_path, "r", encoding="utf-8") as f:
        audio_data = json.load(f)
    
    with open(video_chunks_path, "r", encoding="utf-8") as f:
        video_data = json.load(f)
    
    logger.info("\n" + "="*60)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"\nResults:")
    logger.info(f"  Audio chunks: {audio_data['total_chunks']}")
    logger.info(f"  Video chunks: {video_data['total_chunks']}")
    logger.info(f"  Total tokens: {audio_data['total_tokens'] + video_data['total_tokens']}")
    
    logger.info(f"\nOutput files:")
    logger.info(f"  - {audio_chunks_path}")
    logger.info(f"  - {video_chunks_path}")
    logger.info(f"  - {OUTPUT_DIR}/transcript.json")
    logger.info(f"  - {OUTPUT_DIR}/slides.json")
    logger.info(f"  - {OUTPUT_DIR}/faiss_indices/")
    
    logger.info(f"\nNext steps:")
    if skip_embedding:
        logger.info(f"  1. Run: python embed_chunks.py --incremental")
        logger.info(f"  2. Start: python -m src.server")
    else:
        logger.info(f"  1. Start: python -m src.server")
        logger.info(f"  2. Open: http://localhost:3000")
    
    logger.info("="*60)
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process a video from scratch (clear all previous outputs)"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to video file"
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding step (faster, you can embed later)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    # Update output dir if specified
    if args.output_dir:
        import src.config
        src.config.OUTPUT_DIR = args.output_dir
    
    # Process video (use args.skip_embedding, output_dir from config)
    success = process_video_fresh(args.video_path, args.skip_embedding)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
