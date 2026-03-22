"""
Stage 4: Vision Model Slide Description

Generates descriptions of detected slides using LLaVA via Ollama.

Key fix: The original code used image_data.hex() which produces hexadecimal,
but the Ollama API expects base64 encoding. This caused:
  "illegal base64 data at input byte XXXXX"
errors for every slide description attempt.
"""

import base64
import os
import time
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm

from .config import OLLAMA_HOST, VISION_MODEL, logger


# Prompt template for slide analysis
SLIDE_ANALYSIS_PROMPT = """
You are analyzing a lecture slide.

Describe:
- Main topic of the slide
- Key concepts mentioned
- Any equations (write them clearly using LaTeX format)
- Any code snippets
- Any diagrams or charts

Be concise but informative.
If something is not clearly visible, say "Not clearly visible."
"""


def describe_slide(image_path: str, prompt: str = SLIDE_ANALYSIS_PROMPT) -> Optional[str]:
    """
    Generate description for a slide using LLaVA vision model via Ollama.

    Args:
        image_path: Path to slide image
        prompt: Prompt template for analysis

    Returns:
        Generated description or None if failed
    """
    try:
        # Read and encode image as BASE64 (not hex!)
        with open(image_path, "rb") as f:
            image_data = f.read()

        # Ollama expects base64
        image_b64 = base64.b64encode(image_data).decode("utf-8")

        # Prepare request to Ollama
        url = f"{OLLAMA_HOST}/api/generate"

        payload = {
            "model": VISION_MODEL,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False
        }

        response = requests.post(url, json=payload, timeout=120)

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to Ollama. Ensure 'ollama serve' is running.")
        return None
    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out")
        return None
    except Exception as e:
        logger.error(f"Error describing slide: {e}")
        return None


def generate_slide_descriptions(
    slides: List[Dict[str, Any]],
    frames_dir: str
) -> List[Dict[str, Any]]:
    """
    Generate descriptions for all detected slides.

    Args:
        slides: List of slide metadata
        frames_dir: Directory containing frames

    Returns:
        Updated slides list with descriptions
    """
    logger.info(f"Generating descriptions for {len(slides)} slides using {VISION_MODEL}")

    for i, slide in enumerate(tqdm(slides, desc="Generating descriptions")):
        frame_path = slide.get("last_frame_path")

        if not frame_path or not os.path.exists(frame_path):
            logger.warning(f"Frame not found for slide {slide['slide_id']}")
            slide["description"] = "Frame not available"
            slide["vision_model"] = VISION_MODEL
            continue

        logger.debug(f"Analyzing slide {slide['slide_id']} ({os.path.basename(frame_path)})")

        description = describe_slide(frame_path)

        if description:
            slide["description"] = description
            slide["vision_model"] = VISION_MODEL
            logger.debug(f"Description generated for slide {slide['slide_id']}")
        else:
            slide["description"] = "Not clearly visible"
            slide["vision_model"] = VISION_MODEL
            logger.warning(f"Failed to generate description for slide {slide['slide_id']}")

        # Small delay to avoid overwhelming the API
        time.sleep(0.5)

    return slides
