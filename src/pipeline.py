"""
Multimodal Video Preprocessing Pipeline

Orchestrator that runs all stages in sequence:
  Stage 1: Audio extraction + Whisper transcription
  Stage 2: Frame extraction (1 FPS)
  Stage 3: Slide change detection (SSIM)
  Stage 4: Vision-based slide descriptions (LLaVA via Ollama)
  Stage 5: Slide-aware transcript chunking
  Stage 6: Slide caption chunks
  Stage 7: Output assembly

Each stage is implemented in its own module under src/ for readability.
"""

import os
import sys
import time
import json
from pathlib import Path

from .config import FRAMES_DIR, OUTPUT_DIR, VISION_MODEL, logger
from .stage1_transcription import extract_audio, transcribe_audio
from .stage2_frames import extract_frames
from .stage3_slides import detect_slides
from .stage7_output import save_outputs

# New Modular Chunking System
from .chunking import (
    chunk_video_transcript,
    chunk_video_visual,
    assemble_chunks
)


def _update_progress(stage: str, status: str, message: str = "", output_dir: str = OUTPUT_DIR):
    """Write progress status to JSON file for admin panel polling."""
    progress_file = os.path.join(output_dir, "processing_status.json")
    data = {
        "stage": stage,
        "status": status,  # "running", "completed", "error"
        "message": message,
        "timestamp": time.time()
    }
    try:
        with open(progress_file, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.warning(f"Could not write progress: {e}")


def run_pipeline(video_path: str, output_dir: str = OUTPUT_DIR) -> bool:
    """
    Run the complete multimodal video preprocessing pipeline.

    Args:
        video_path: Path to input video
        output_dir: Output directory

    Returns:
        True if pipeline completed successfully
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Starting Multimodal Video Preprocessing Pipeline")
    logger.info(f"Input: {video_path}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    _update_progress("initialization", "running", "Starting pipeline", output_dir)

    # Validate input
    if not os.path.exists(video_path):
        logger.error(f"Input video not found: {video_path}")
        return False

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)

    # Create audio directory
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Stage 1: Extract Audio
    # -------------------------------------------------------------------------
    _update_progress("audio_extraction", "running", "Extracting audio from video", output_dir)
    audio_path = os.path.join(audio_dir, "audio.wav")
    if not extract_audio(video_path, audio_path):
        logger.error("Failed to extract audio. Aborting pipeline.")
        return False

    # -------------------------------------------------------------------------
    # Stage 2: Transcribe Audio (Whisper)
    # -------------------------------------------------------------------------
    _update_progress("transcription", "running", "Transcribing audio with Whisper", output_dir)
    transcript_path = os.path.join(output_dir, "transcript.json")
    transcript = transcribe_audio(audio_path, transcript_path)

    if transcript is None:
        logger.error("Failed to transcribe audio. Aborting pipeline.")
        return False

    # -------------------------------------------------------------------------
    # Stage 3: Extract Frames & Detect Slides
    # -------------------------------------------------------------------------
    _update_progress("frame_extraction", "running", "Extracting frames and detecting slides", output_dir)
    # Frames are stored in video-specific subdirectory
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_frames_dir = os.path.join(FRAMES_DIR, video_name)
    
    frame_timestamps = extract_frames(video_path, FRAMES_DIR)
    if not frame_timestamps:
        logger.error("Failed to extract frames. Aborting pipeline.")
        return False

    slides = detect_slides(video_frames_dir)
    if not slides:
        logger.warning("No slides detected. Assuming single slide.")
        # Single slide fallback
        max_time = max(frame_timestamps.values()) if frame_timestamps else 0
        slides = [{
            "slide_id": 0,
            "start_time": 0,
            "end_time": max_time,
            "last_frame_path": str(sorted(Path(FRAMES_DIR).glob("frame_*.jpg"))[-1]) if list(Path(FRAMES_DIR).glob("frame_*.jpg")) else "",
            "last_frame_index": len(frame_timestamps) - 1 if frame_timestamps else 0,
            "is_last_frame_before_change": True
        }]

    # -------------------------------------------------------------------------
    # Stage 4: Visual Processing & Chunking (LLaVA)
    # -------------------------------------------------------------------------
    _update_progress("enrichment", "running", f"Enriching {len(slides)} slides with OCR and vision AI", output_dir)
    logger.info("=" * 40)
    logger.info("Stage 4: Visual Processing (LLaVA Scene + OCR)")
    logger.info("=" * 40)
    logger.info(f"Ensure Ollama is running: ollama serve")
    
    # Generates visual chunks AND enriches 'slides' with descriptions/OCR in-place
    visual_chunks = chunk_video_visual(
        slides, 
        source_file=Path(video_path).name,
        run_enrichment=True
    )

    # -------------------------------------------------------------------------
    # Stage 5: Audio Transcript Chunking
    # -------------------------------------------------------------------------
    _update_progress("chunking", "running", "Creating audio and visual chunks", output_dir)
    logger.info("=" * 40)
    logger.info("Stage 5: Transcript Chunking")
    logger.info("=" * 40)
    
    transcript_chunks = chunk_video_transcript(
        transcript, 
        slides,
        source_file=Path(video_path).name
    )

    # -------------------------------------------------------------------------
    # Stage 6: Assembly & Save
    # -------------------------------------------------------------------------
    logger.info("=" * 40)
    logger.info("Stage 6: Assembly & Save")
    logger.info("=" * 40)

    # Save chunks using ChunkManager (separate audio and video files)
    from .chunk_manager import ChunkManager
    
    manager = ChunkManager(output_dir)
    
    # Add new chunks
    num_audio_added = manager.add_audio_chunks(transcript_chunks)
    num_video_added = manager.add_video_chunks(visual_chunks)
    
    # Also save combined format for backward compatibility
    assembly_result = assemble_chunks(
        transcript_chunks,
        visual_chunks,
        output_path=os.path.join(output_dir, "chunks.json")
    )

    all_chunks = assembly_result["chunks"]

    # Save intermediate outputs (transcript, slides) for debugging/reference
    # Note: 'slides' is now enriched with scene_description and ocr_text
    save_outputs(transcript, slides, all_chunks, output_dir)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    elapsed_time = time.time() - start_time

    _update_progress("completed", "completed", f"Pipeline completed in {elapsed_time/60:.1f} mins", output_dir)
    logger.info("=" * 60)
    logger.info("Pipeline Completed Successfully! ✅")
    logger.info(f"Total time: {elapsed_time/60:.2f} mins")
    logger.info(f"Slides: {len(slides)}")
    logger.info(f"Chunks Added: {num_audio_added} audio + {num_video_added} video")
    logger.info(f"Total Chunks: {len(transcript_chunks)} audio + {len(visual_chunks)} video")
    logger.info(f"Tokens: ~{assembly_result['total_tokens']}")
    logger.info("\nOutput Files:")
    logger.info(f"  - {output_dir}/audio_chunks.json")
    logger.info(f"  - {output_dir}/video_chunks.json")
    logger.info(f"  - {output_dir}/chunks.json (combined, backward compatible)")
    logger.info("\nNext: Run incremental embedding:")
    logger.info("  python embed_chunks.py --incremental")
    logger.info("=" * 60)

    return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multimodal Video Preprocessing Pipeline"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )

    args = parser.parse_args()

    success = run_pipeline(args.video_path, args.output_dir)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
