"""
Test Script: LLaVA Vision Description

Verifies that the LLaVA model via Ollama can:
1. Connect to the Ollama API
2. Accept a base64-encoded image
3. Return a meaningful description

Tests with a single frame first, then optionally tests 3 slides.

Usage:
    python -m tests.test_llava
"""

import base64
import json
import os
import sys
import time

import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


OLLAMA_HOST = "http://localhost:11434"
VISION_MODEL = "qwen3-vl:235b-cloud"


def test_ollama_connection():
    """Check Ollama is running."""
    print("--- Checking Ollama connection ---")
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            print(f"[OK] Ollama is running. Available models: {models}")
            return models
        else:
            print(f"[FAIL] Ollama returned status {resp.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        print(f"[FAIL] Cannot connect to Ollama at {OLLAMA_HOST}")
        print("       Start Ollama with: ollama serve")
        return None


def test_model_available(models):
    """Check if LLaVA model is available."""
    print(f"\n--- Checking {VISION_MODEL} availability ---")
    if models is None:
        print("[SKIP] Ollama not available")
        return False

    # Check if any variant of the model name matches
    found = any(VISION_MODEL.split(":")[0] in m for m in models)
    if found:
        print(f"[OK] {VISION_MODEL} is available")
        return True
    else:
        print(f"[FAIL] {VISION_MODEL} not found in available models")
        print(f"       Pull it with: ollama pull {VISION_MODEL}")
        return False


def test_base64_encoding():
    """Verify we're encoding images correctly as base64 (not hex)."""
    print("\n--- Testing base64 encoding ---")

    # Find a frame to test with
    from pathlib import Path
    frames_dir = os.path.join("output", "frames")
    frame_files = sorted(Path(frames_dir).glob("frame_*.jpg"))

    if not frame_files:
        print(f"[SKIP] No frames found in {frames_dir}")
        return None

    test_frame = str(frame_files[0])
    print(f"[OK] Using test frame: {test_frame}")

    # Read and encode
    with open(test_frame, "rb") as f:
        raw_data = f.read()

    # Correct: base64
    b64_encoded = base64.b64encode(raw_data).decode("utf-8")

    # Wrong: hex (what the original code was doing!)
    hex_encoded = raw_data.hex()

    print(f"  Raw size: {len(raw_data)} bytes")
    print(f"  Base64 size: {len(b64_encoded)} chars  (correct ✅)")
    print(f"  Hex size: {len(hex_encoded)} chars  (WRONG - was used before ❌)")
    print(f"  Base64 starts with: {b64_encoded[:40]}...")
    print(f"  Hex starts with: {hex_encoded[:40]}...")

    # Verify base64 is valid
    try:
        decoded = base64.b64decode(b64_encoded)
        assert decoded == raw_data
        print(f"[OK] Base64 round-trip verified")
    except Exception as e:
        print(f"[FAIL] Base64 round-trip failed: {e}")
        return None

    return test_frame, b64_encoded


def test_llava_describe(test_frame: str, image_b64: str):
    """Test LLaVA description generation."""
    print(f"\n--- Testing LLaVA description with {os.path.basename(test_frame)} ---")

    prompt = "Describe what you see in this image in 2-3 sentences."

    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False
    }

    url = f"{OLLAMA_HOST}/api/generate"

    try:
        start = time.time()
        resp = requests.post(url, json=payload, timeout=120)
        elapsed = time.time() - start

        if resp.status_code == 200:
            result = resp.json()
            description = result.get("response", "").strip()
            print(f"[OK] Got response in {elapsed:.1f}s")
            print(f"  Description: {description[:200]}")
            if len(description) < 10:
                print(f"[WARN] Description seems too short ({len(description)} chars)")
            return True
        else:
            print(f"[FAIL] API returned {resp.status_code}: {resp.text[:200]}")
            return False

    except requests.exceptions.Timeout:
        print(f"[FAIL] Request timed out after 120s")
        return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_describe_slide_function():
    """Test the fixed describe_slide function from stage4."""
    print("\n--- Testing describe_slide() function ---")
    try:
        from src.stage4_vision import describe_slide
        from pathlib import Path

        frames_dir = os.path.join("output", "frames")
        frame_files = sorted(Path(frames_dir).glob("frame_*.jpg"))

        if len(frame_files) < 3:
            print("[SKIP] Need at least 3 frames")
            return False

        # Test with 3 different frames
        test_frames = [frame_files[0], frame_files[len(frame_files)//2], frame_files[-1]]

        results = []
        for frame in test_frames:
            print(f"\n  Testing: {frame.name}...")
            start = time.time()
            desc = describe_slide(str(frame))
            elapsed = time.time() - start

            if desc:
                print(f"  [OK] {elapsed:.1f}s | {desc[:100]}...")
                results.append(True)
            else:
                print(f"  [FAIL] No description returned ({elapsed:.1f}s)")
                results.append(False)

        success_rate = sum(results) / len(results) * 100
        print(f"\n[{'OK' if all(results) else 'WARN'}] {success_rate:.0f}% success rate ({sum(results)}/{len(results)} frames)")
        return all(results)

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def test_llava():
    """Run all LLaVA tests."""
    print("=" * 60)
    print("TEST: LLaVA Vision Description")
    print("=" * 60)

    # 1. Check Ollama connection
    models = test_ollama_connection()
    if models is None:
        print("\n" + "=" * 60)
        print("RESULT: FAIL ❌ (Ollama not running)")
        print("=" * 60)
        return False

    # 2. Check model available
    if not test_model_available(models):
        print("\n" + "=" * 60)
        print(f"RESULT: FAIL ❌ ({VISION_MODEL} not available)")
        print("=" * 60)
        return False

    # 3. Test base64 encoding
    encoding_result = test_base64_encoding()
    if encoding_result is None:
        print("\n" + "=" * 60)
        print("RESULT: FAIL ❌ (No frames for testing)")
        print("=" * 60)
        return False

    test_frame, image_b64 = encoding_result

    # 4. Test raw API call
    api_ok = test_llava_describe(test_frame, image_b64)

    # 5. Test the fixed function
    func_ok = test_describe_slide_function()

    # Summary
    print("\n" + "=" * 60)
    if api_ok and func_ok:
        print("RESULT: PASS ✅")
    elif api_ok:
        print("RESULT: PARTIAL PASS ⚠️ (API works, function issues)")
    else:
        print("RESULT: FAIL ❌")
    print("=" * 60)

    return api_ok


if __name__ == "__main__":
    success = test_llava()
    sys.exit(0 if success else 1)
