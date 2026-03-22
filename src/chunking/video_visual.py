"""
Video Visual Chunker

Produces rich visual chunks from detected slide frames.

For each slide, the chunk contains:
  - scene_description : What's on the slide (from LLaVA)
  - ocr_text          : All readable text extracted from the frame (via LLaVA OCR prompt)
  - text              : Combined human-readable summary for embedding/search
  - frame_path        : Path to the actual frame image
  - start_sec / end_sec : When this slide was on screen

The OCR extraction uses LLaVA with a specialised "read all text" prompt,
so no extra dependencies (pytesseract, easyocr) are needed — just Ollama.
"""

import base64
import os
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm

from ..config import OLLAMA_HOST, VISION_MODEL, logger, USE_TESSERACT_OCR, TESSERACT_TIMEOUT

# Import MAX_SLIDES_TO_ENRICH with a fallback
try:
    from ..config import MAX_SLIDES_TO_ENRICH
except ImportError:
    MAX_SLIDES_TO_ENRICH = None
from .base import count_tokens, make_chunk


# ─── Prompts ────────────────────────────────────────────────────────────────

SCENE_DESCRIPTION_PROMPT = "Describe this slide in 1-2 sentences: main topic and key points."

# OCR moved to Tesseract for speed


# ─── Low-level API ──────────────────────────────────────────────────────────

def _tesseract_ocr(image_path: str) -> str:
    """
    Fast OCR using Tesseract via Docker.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Extracted text or empty string on failure
    """
    if not USE_TESSERACT_OCR:
        return ""
    
    try:
        # Convert to absolute path and normalize for Docker on Windows
        abs_path = os.path.abspath(image_path)
        
        # On Windows, convert C:\path\to\file to /c/path/to/file for Docker
        if os.name == 'nt':  # Windows
            # Replace backslashes with forward slashes
            docker_path = abs_path.replace('\\', '/')
            # Convert C: to /c
            if len(docker_path) > 1 and docker_path[1] == ':':
                docker_path = '/' + docker_path[0].lower() + docker_path[2:]
            volume_mount = f"{docker_path}:/tmp/img.jpg"
        else:
            volume_mount = f"{abs_path}:/tmp/img.jpg"
        
        # Docker command
        cmd = [
            "docker", "run", "--rm",
            "-v", volume_mount,
            "jitesoft/tesseract-ocr",
            "/tmp/img.jpg", "stdout"
        ]
        
        logger.debug(f"Running Tesseract OCR: {' '.join(cmd[:4])}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',  # Force UTF-8 encoding for Windows compatibility
            errors='replace',   # Replace invalid characters instead of crashing
            timeout=TESSERACT_TIMEOUT
        )
        
        if result.returncode == 0:
            # Exit code 0 = success
            text = result.stdout.strip() if result.stdout else ""
            if text:
                logger.debug(f"Tesseract extracted {len(text)} characters")
            else:
                logger.debug("Tesseract: no text found in image (blank slide or no text)")
            return text
        else:
            # Non-zero exit code = actual error
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            logger.warning(f"Tesseract OCR error (exit code {result.returncode}): {error_msg[:200]}")
            return ""
            
    except FileNotFoundError:
        logger.warning("Docker not found. Set USE_TESSERACT_OCR=False in config.py")
        return ""
    except subprocess.TimeoutExpired:
        logger.warning(f"Tesseract OCR timed out after {TESSERACT_TIMEOUT}s. Increase TESSERACT_TIMEOUT or set USE_TESSERACT_OCR=False")
        return ""
    except Exception as e:
        logger.warning(f"Tesseract OCR error: {e}")
        return ""


def _call_llava(image_path: str, prompt: str) -> Optional[str]:
    """
    Send an image + prompt to LLaVA via Ollama and return the response.

    Args:
        image_path: Path to image file
        prompt: Text prompt

    Returns:
        Model response text, or None on failure
    """
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()

        image_b64 = base64.b64encode(image_data).decode("utf-8")

        payload = {
            "model": VISION_MODEL,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }

        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=180,
        )

        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            logger.warning(f"Ollama API error {response.status_code}: {response.text[:200]}")
            return None

    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama. Ensure 'ollama serve' is running.")
        return None
    except requests.exceptions.Timeout:
        logger.warning("Ollama request timed out (180s)")
        return None
    except Exception as e:
        logger.error(f"LLaVA call error: {e}")
        return None


# ─── Per-slide enrichment ───────────────────────────────────────────────────

def _enrich_slide(slide: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich a single slide with scene_description and ocr_text.
    
    Fast approach:
      1. Tesseract OCR (Docker) → raw text extraction (fast)
      2. Vision model → 1-2 sentence description (brief)

    Mutates `slide` in-place and returns it.
    """
    frame_path = slide.get("last_frame_path", "")
    slide_id = slide.get("slide_id", "?")

    if not frame_path or not os.path.exists(frame_path):
        logger.warning(f"Slide {slide_id}: frame not found ({frame_path})")
        slide.setdefault("description", "Frame not available")
        slide.setdefault("scene_description", "Frame not available")
        slide.setdefault("ocr_text", "")
        return slide

    # 1. Fast OCR with Tesseract (Docker) - no API call needed
    if not slide.get("ocr_text"):
        if USE_TESSERACT_OCR:
            ocr = _tesseract_ocr(frame_path)
            slide["ocr_text"] = ocr or ""
            if not ocr:
                logger.debug(f"Slide {slide_id}: OCR returned no text (continuing without it)")
        else:
            slide["ocr_text"] = ""
            logger.debug(f"Slide {slide_id}: Tesseract OCR disabled")

    # 2. Brief scene description (vision model, 1-2 sentences only)
    if not slide.get("scene_description"):
        desc = _call_llava(frame_path, SCENE_DESCRIPTION_PROMPT)
        slide["scene_description"] = desc or "No description"
        slide["description"] = desc or slide.get("description", "No description")

    slide["vision_model"] = VISION_MODEL
    return slide


# ─── Main chunker ───────────────────────────────────────────────────────────

def chunk_video_visual(
    slides: List[Dict[str, Any]],
    source_file: str = "",
    run_enrichment: bool = True,
) -> List[Dict[str, Any]]:
    """
    Produce visual chunks from slide metadata.

    For each slide, creates a chunk with:
      - scene_description : detailed content description
      - ocr_text          : raw text extracted via OCR prompt
      - text              : combined searchable text (description + OCR)
      - frame_path        : path to the representative frame

    Args:
        slides: List of slide metadata (from stage3 detect_slides)
        source_file: Original video filename
        run_enrichment: If True, calls LLaVA for scene_description + OCR.
            Set to False if slides already have these fields populated
            (e.g., loaded from a previous run's slides.json).

    Returns:
        List of chunk dicts (unified schema)
    """
    logger.info(f"[video_visual] Creating visual chunks for {len(slides)} slides")

    if run_enrichment:
        # Limit slides to enrich if configured
        slides_to_enrich = slides
        if MAX_SLIDES_TO_ENRICH and len(slides) > MAX_SLIDES_TO_ENRICH:
            logger.warning(f"[video_visual] Too many slides ({len(slides)}), enriching only first {MAX_SLIDES_TO_ENRICH}")
            slides_to_enrich = slides[:MAX_SLIDES_TO_ENRICH]
        
        logger.info(f"[video_visual] Enriching {len(slides_to_enrich)} slides with LLaVA ({VISION_MODEL})...")
        for slide in tqdm(slides_to_enrich, desc="Enriching slides"):
            _enrich_slide(slide)

    chunks = []
    # Generate unique video_id from source_file (e.g., "terraformVideo" from "terraformVideo.mp4")
    if source_file:
        video_id = Path(source_file).stem  # Get filename without extension
    else:
        video_id = "video"

    for slide in slides:
        slide_id = slide["slide_id"]

        scene_desc = slide.get("scene_description", slide.get("description", ""))
        ocr_text = slide.get("ocr_text", "")

        # Build the combined text for embedding / search
        # This is what downstream RAG will actually search against
        text_parts = []
        if scene_desc and scene_desc != "Not clearly visible":
            text_parts.append(f"[Scene] {scene_desc}")
        if ocr_text:
            text_parts.append(f"[OCR Text] {ocr_text}")

        combined_text = "\n\n".join(text_parts) if text_parts else "No visual content"

        chunks.append(make_chunk(
            chunk_id=f"{video_id}_visual_{slide_id:03d}",
            source_type="video_visual",
            source_file=source_file,
            modality="slide_caption",
            text=combined_text,
            start_sec=slide.get("start_time"),
            end_sec=slide.get("end_time"),
            video_id=video_id,
            slide_id=slide_id,
            scene_description=scene_desc,
            ocr_text=ocr_text,
            frame_path=slide.get("last_frame_path", ""),
            is_last_frame_before_change=slide.get("is_last_frame_before_change", True),
            vision_model=VISION_MODEL,
        ))

    logger.info(f"[video_visual] Created {len(chunks)} visual chunks")
    return chunks
