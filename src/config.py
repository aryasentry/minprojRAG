"""
Configuration constants for the Multimodal Video Preprocessing Pipeline.

All stages import their settings from here.
"""

import logging
import os
import sys

# =============================================================================
# SLIDE DETECTION
# =============================================================================

# SSIM threshold for CONSECUTIVE frame comparison.
# Lower = less sensitive → fewer slide changes (better for stable content)
# Range: 0.0-1.0 where 1.0 = identical
# Typical values: 0.65-0.75 for lecture videos
SSIM_THRESHOLD = 0.65

# Minimum seconds between slide changes (cooldown).
# Higher = fewer slides, better grouping of content
# Recommended: 8-15 seconds for lecture videos
MIN_SLIDE_DURATION_SEC = 10

# Frames per second to extract from video
# 1 FPS is standard for lecture videos (reduces processing time)
FRAME_FPS = 1

# Maximum slides to enrich with vision AI
# Prevents excessive processing time for long videos
# Set to None for no limit
MAX_SLIDES_TO_ENRICH = 50

# =============================================================================
# OCR SETTINGS
# =============================================================================

# Use Tesseract OCR via Docker (fast) instead of vision model for text extraction
# Set to False if Docker is not available or causing issues
USE_TESSERACT_OCR = True

# Timeout for Tesseract OCR in seconds
# Increase if Docker startup is slow on your system
TESSERACT_TIMEOUT = 30

# =============================================================================
# CHUNKING
# =============================================================================
MAX_TOKENS_PER_CHUNK = 400

# =============================================================================
# MODELS
# =============================================================================
WHISPER_MODEL = "small"
VISION_MODEL = "qwen3-vl:235b-cloud"
LLM_MODEL = "qwen2.5:1.5b"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

# =============================================================================
# OLLAMA
# =============================================================================
OLLAMA_HOST = "http://localhost:11434"

# =============================================================================
# DIRECTORIES
# =============================================================================
INPUT_DIR = "input"
OUTPUT_DIR = "output"
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging():
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log", mode="w")
        ]
    )
    return logging.getLogger("pipeline")


logger = setup_logging()
