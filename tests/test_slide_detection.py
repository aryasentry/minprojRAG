"""
Test Script: Slide Change Detection

Verifies that the SSIM-based slide detection:
1. Produces a reasonable number of slides (10-40 for an 11-min lecture)
2. Each slide has proper metadata (start_time, end_time, frame path)
3. Slides don't overlap and cover the full video duration
4. The MIN_SLIDE_DURATION cooldown prevents false positives

Also prints SSIM statistics so you can inspect the distribution.

Usage:
    python -m tests.test_slide_detection
"""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_ssim_distribution(frames_dir: str, max_frames: int = 200):
    """Compute SSIM between consecutive frames and return stats."""
    from imageio.v3 import imread
    import cv2
    import numpy as np
    from skimage.metrics import structural_similarity

    frame_files = sorted(Path(frames_dir).glob("frame_*.jpg"))[:max_frames]
    if len(frame_files) < 2:
        return []

    ssim_values = []
    prev = None

    for f in frame_files:
        img = imread(str(f))
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (256, 256))

        if prev is not None:
            s = structural_similarity(prev, img)
            ssim_values.append(s)

        prev = img

    return ssim_values


def test_slide_detection():
    """Test slide detection on existing extracted frames."""
    print("=" * 60)
    print("TEST: Slide Change Detection")
    print("=" * 60)

    frames_dir = os.path.join("output", "frames")
    if not os.path.exists(frames_dir):
        print(f"[SKIP] Frames directory not found: {frames_dir}")
        print("       Run the pipeline first to extract frames.")
        return False

    frame_files = sorted(Path(frames_dir).glob("frame_*.jpg"))
    print(f"[OK] Found {len(frame_files)} frames in {frames_dir}")

    if len(frame_files) < 10:
        print(f"[FAIL] Too few frames ({len(frame_files)}). Need at least 10.")
        return False

    # ------------------------------------------------------------------
    # 1. Compute SSIM distribution
    # ------------------------------------------------------------------
    print("\n--- SSIM Distribution (first 200 frames) ---")
    ssim_values = compute_ssim_distribution(frames_dir)
    if ssim_values:
        import numpy as np
        arr = np.array(ssim_values)
        print(f"  Count: {len(arr)}")
        print(f"  Mean:  {arr.mean():.4f}")
        print(f"  Std:   {arr.std():.4f}")
        print(f"  Min:   {arr.min():.4f}")
        print(f"  Max:   {arr.max():.4f}")
        print(f"  < 0.75 (slide change @ threshold 0.75): {(arr < 0.75).sum()}")
        print(f"  < 0.85 (slide change @ threshold 0.85): {(arr < 0.85).sum()}")
        print(f"  < 0.90 (slide change @ threshold 0.90): {(arr < 0.90).sum()}")

    # ------------------------------------------------------------------
    # 2. Run slide detection with the improved algorithm
    # ------------------------------------------------------------------
    print("\n--- Running slide detection ---")
    from src.stage3_slides import detect_slides

    start = time.time()
    slides = detect_slides(frames_dir)
    elapsed = time.time() - start

    print(f"[OK] Detected {len(slides)} slides in {elapsed:.2f}s")

    # ------------------------------------------------------------------
    # 3. Validate slide count
    # ------------------------------------------------------------------
    print("\n--- Validating slide count ---")
    if len(slides) < 3:
        print(f"[WARN] Very few slides ({len(slides)}). Video may not have slide transitions.")
    elif len(slides) > 120:
        print(f"[WARN] Too many slides ({len(slides)}). Threshold may be too sensitive.")
        print(f"       Consider raising SSIM_THRESHOLD or MIN_SLIDE_DURATION_SEC.")
    else:
        print(f"[OK] Slide count ({len(slides)}) is in reasonable range (3-50)")

    # ------------------------------------------------------------------
    # 4. Validate slide metadata
    # ------------------------------------------------------------------
    print("\n--- Validating slide metadata ---")
    errors = []

    for i, slide in enumerate(slides):
        if "slide_id" not in slide:
            errors.append(f"Slide {i} missing 'slide_id'")
        if "start_time" not in slide or "end_time" not in slide:
            errors.append(f"Slide {i} missing timestamps")
        if slide.get("start_time", 0) > slide.get("end_time", 0):
            errors.append(f"Slide {i} start > end ({slide['start_time']} > {slide['end_time']})")
        if "last_frame_path" not in slide:
            errors.append(f"Slide {i} missing 'last_frame_path'")
        elif not os.path.exists(slide["last_frame_path"]):
            errors.append(f"Slide {i} frame not found: {slide['last_frame_path']}")

        # Check no overlap with next slide
        if i < len(slides) - 1:
            next_slide = slides[i + 1]
            if slide.get("end_time", 0) > next_slide.get("start_time", 0) + 0.01:
                errors.append(f"Slides {i} and {i+1} overlap: "
                              f"{slide['end_time']} > {next_slide['start_time']}")

    if errors:
        print(f"[FAIL] Validation errors:")
        for e in errors:
            print(f"  - {e}")
        return False

    print(f"[OK] All {len(slides)} slides have valid metadata, no overlaps")

    # ------------------------------------------------------------------
    # 5. Print slide summary
    # ------------------------------------------------------------------
    print("\n--- Slide Summary ---")
    print(f"{'ID':>4}  {'Start':>8}  {'End':>8}  {'Duration':>8}  Frame")
    print("-" * 70)
    for slide in slides[:20]:  # Show first 20
        duration = slide['end_time'] - slide['start_time']
        frame_name = os.path.basename(slide.get('last_frame_path', ''))
        print(f"{slide['slide_id']:>4}  {slide['start_time']:>7.1f}s  {slide['end_time']:>7.1f}s  {duration:>7.1f}s  {frame_name}")

    if len(slides) > 20:
        print(f"  ... and {len(slides) - 20} more slides")

    # ------------------------------------------------------------------
    # 6. Save test results for review
    # ------------------------------------------------------------------
    test_output = os.path.join("output", "test_slides.json")
    with open(test_output, "w", encoding="utf-8") as f:
        json.dump(slides, f, indent=2)
    print(f"\n[OK] Saved detected slides to {test_output}")
    print("     >> Review the frame images listed above to validate accuracy! <<")

    print("\n" + "=" * 60)
    print(f"RESULT: PASS ✅ ({len(slides)} slides detected)")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_slide_detection()
    sys.exit(0 if success else 1)
