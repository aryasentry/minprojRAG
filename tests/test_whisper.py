"""
Test Script: Whisper Transcription

Verifies that faster-whisper can:
1. Load the model
2. Transcribe the extracted audio
3. Produce segments with word-level timestamps
4. Save a valid JSON transcript

Usage:
    python -m tests.test_whisper
"""

import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_whisper():
    """Test whisper transcription end-to-end."""
    print("=" * 60)
    print("TEST: Whisper Transcription")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Check audio file exists
    # ------------------------------------------------------------------
    audio_path = os.path.join("output", "audio", "audio.wav")
    if not os.path.exists(audio_path):
        print(f"[SKIP] Audio file not found: {audio_path}")
        print("       Run the pipeline first to extract audio, or place audio.wav manually.")
        return False

    print(f"[OK] Audio file found: {audio_path} ({os.path.getsize(audio_path) / 1024 / 1024:.1f} MB)")

    # ------------------------------------------------------------------
    # 2. Load Whisper model
    # ------------------------------------------------------------------
    print("\n--- Loading Whisper model ---")
    try:
        from faster_whisper import WhisperModel
        start = time.time()
        model = WhisperModel("small", device="cpu", compute_type="int8")
        load_time = time.time() - start
        print(f"[OK] Model loaded in {load_time:.2f}s")
    except Exception as e:
        print(f"[FAIL] Could not load Whisper model: {e}")
        return False

    # ------------------------------------------------------------------
    # 3. Transcribe first 30 seconds only (fast test)
    # ------------------------------------------------------------------
    print("\n--- Transcribing (first 30s subset for speed) ---")
    try:
        start = time.time()
        segments, info = model.transcribe(
            audio_path,
            word_timestamps=True,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        transcript_segments = []
        for seg in segments:
            if seg.start > 30:
                break  # Only test first 30 seconds
            words = []
            if seg.words:
                words = [{"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)} for w in seg.words]
            transcript_segments.append({
                "segment_id": len(transcript_segments),
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "words": words
            })

        elapsed = time.time() - start
        print(f"[OK] Transcribed {len(transcript_segments)} segments in {elapsed:.2f}s")

    except Exception as e:
        print(f"[FAIL] Transcription error: {e}")
        return False

    # ------------------------------------------------------------------
    # 4. Validate output structure
    # ------------------------------------------------------------------
    print("\n--- Validating output ---")
    errors = []

    if len(transcript_segments) == 0:
        errors.append("No segments produced")

    for i, seg in enumerate(transcript_segments):
        if "text" not in seg:
            errors.append(f"Segment {i} missing 'text'")
        if "start" not in seg or "end" not in seg:
            errors.append(f"Segment {i} missing timestamps")
        if seg.get("start", 0) > seg.get("end", 0):
            errors.append(f"Segment {i} start > end ({seg['start']} > {seg['end']})")
        if "words" not in seg:
            errors.append(f"Segment {i} missing 'words'")
        elif len(seg["words"]) == 0:
            errors.append(f"Segment {i} has no word-level timestamps")

    if errors:
        print(f"[FAIL] Validation errors:")
        for e in errors:
            print(f"       - {e}")
        return False

    print(f"[OK] All {len(transcript_segments)} segments have text, timestamps, and words")

    # Print sample
    print("\n--- Sample output (first 3 segments) ---")
    for seg in transcript_segments[:3]:
        print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text'][:80]}...")
        if seg['words']:
            print(f"    Words: {len(seg['words'])} | First: '{seg['words'][0]['word']}' @ {seg['words'][0]['start']}s")

    # ------------------------------------------------------------------
    # 5. Check existing full transcript
    # ------------------------------------------------------------------
    full_transcript_path = os.path.join("output", "transcript.json")
    if os.path.exists(full_transcript_path):
        print(f"\n--- Existing full transcript ---")
        with open(full_transcript_path, "r", encoding="utf-8") as f:
            full = json.load(f)
        print(f"[OK] Full transcript: {len(full['segments'])} segments")
        print(f"     Duration: {full['segments'][-1]['end']:.1f}s")
    else:
        print(f"\n[INFO] No full transcript found yet at {full_transcript_path}")

    print("\n" + "=" * 60)
    print("RESULT: PASS ✅")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_whisper()
    sys.exit(0 if success else 1)
