"""
Test: Modular Chunking System

Validates:
  1. video_audio  — transcript chunks from Whisper output + slides
  2. video_visual — visual chunks from slide frames (without LLaVA calls)
  3. pdf_text     — PDF text extraction and chunking
  4. assembler    — merging all chunk types into unified output

Run: python -m tests.test_chunking
"""

import json
import os
import sys

# ── Helpers ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
TRANSCRIPT_PATH = os.path.join(OUTPUT_DIR, "transcript.json")
SLIDES_PATH = os.path.join(OUTPUT_DIR, "test_slides.json")


def check(condition: bool, msg: str):
    """Print OK / FAIL for a condition."""
    if condition:
        print(f"[OK] {msg}")
    else:
        print(f"[FAIL] {msg}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
#  Test 1: Video Audio Chunker
# ═══════════════════════════════════════════════════════════════════════════

def test_video_audio():
    print("\n" + "=" * 60)
    print("TEST 1: Video Audio Chunker")
    print("=" * 60)

    from src.chunking import chunk_video_transcript

    # Load data
    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    with open(SLIDES_PATH, "r", encoding="utf-8") as f:
        slides = json.load(f)

    check(len(transcript.get("segments", [])) > 0, f"Transcript has {len(transcript['segments'])} segments")
    check(len(slides) > 0, f"Loaded {len(slides)} slides")

    # Chunk
    chunks = chunk_video_transcript(transcript, slides, source_file="test_video.mp4")

    check(len(chunks) > 0, f"Produced {len(chunks)} transcript chunks")

    # Validate schema
    required_keys = [
        "chunk_id", "source_type", "source_file", "modality",
        "text", "token_count", "start_sec", "end_sec",
        "video_id", "slide_id",
    ]
    c = chunks[0]
    for key in required_keys:
        check(key in c, f"  Has field '{key}' = {repr(c[key])[:60]}")

    check(c["source_type"] == "video_transcript", "  source_type is 'video_transcript'")
    check(c["modality"] == "transcript", "  modality is 'transcript' (legacy)")
    check("words" in c, f"  Has 'words' field ({len(c.get('words', []))} words)")
    check(c["source_file"] == "test_video.mp4", "  source_file propagated")

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
#  Test 2: Video Visual Chunker (without LLaVA — just schema)
# ═══════════════════════════════════════════════════════════════════════════

def test_video_visual():
    print("\n" + "=" * 60)
    print("TEST 2: Video Visual Chunker (schema only, no LLaVA)")
    print("=" * 60)

    from src.chunking import chunk_video_visual

    with open(SLIDES_PATH, "r", encoding="utf-8") as f:
        slides = json.load(f)

    # Pre-populate fake enrichment data (skip LLaVA calls in test)
    for slide in slides[:5]:
        slide["scene_description"] = f"Test slide {slide['slide_id']} showing system design concepts"
        slide["ocr_text"] = f"Title: Concept {slide['slide_id']}\nBullet point 1\nBullet point 2"

    # Run without enrichment (use pre-populated data)
    chunks = chunk_video_visual(slides[:5], source_file="test_video.mp4", run_enrichment=False)

    check(len(chunks) == 5, f"Produced {len(chunks)} visual chunks (expected 5)")

    c = chunks[0]
    required_keys = [
        "chunk_id", "source_type", "modality",
        "text", "start_sec", "end_sec",
        "scene_description", "ocr_text", "frame_path",
    ]
    for key in required_keys:
        check(key in c, f"  Has field '{key}'")

    check(c["source_type"] == "video_visual", "  source_type is 'video_visual'")
    check("[Scene]" in c["text"], "  text contains [Scene] section")
    check("[OCR Text]" in c["text"], "  text contains [OCR Text] section")
    check(c["scene_description"] is not None, f"  scene_description: {c['scene_description'][:50]}...")
    check(c["ocr_text"] is not None, f"  ocr_text: {c['ocr_text'][:50]}...")

    print(f"\n  Sample visual chunk text:")
    print(f"  {'-'*50}")
    for line in c["text"].split("\n")[:5]:
        print(f"    {line}")
    print(f"  {'-'*50}")

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
#  Test 3: PDF Text Chunker
# ═══════════════════════════════════════════════════════════════════════════

def test_pdf():
    print("\n" + "=" * 60)
    print("TEST 3: PDF Text Chunker")
    print("=" * 60)

    from src.chunking import chunk_pdf

    # Check for PDF backend
    has_backend = False
    try:
        import fitz
        print(f"[OK] pymupdf (fitz) v{fitz.version[0]} available")
        has_backend = True
    except ImportError:
        print("[INFO] pymupdf not installed")

    if not has_backend:
        try:
            import pypdf
            print(f"[OK] pypdf available")
            has_backend = True
        except ImportError:
            print("[INFO] pypdf not installed")

    if not has_backend:
        print("[SKIP] No PDF backend available. Install: pip install pymupdf")
        return []

    # Create a test PDF with reportlab if possible, or a simple text-based one
    test_pdf_path = os.path.join(OUTPUT_DIR, "test_sample.pdf")

    try:
        _create_test_pdf(test_pdf_path)
        print(f"[OK] Created test PDF: {test_pdf_path}")
    except Exception as e:
        print(f"[SKIP] Could not create test PDF: {e}")
        return []

    # Chunk it
    chunks = chunk_pdf(test_pdf_path, max_tokens=200)

    check(len(chunks) > 0, f"Produced {len(chunks)} PDF chunks")

    c = chunks[0]
    required_keys = [
        "chunk_id", "source_type", "source_file", "modality",
        "text", "token_count", "page_number",
    ]
    for key in required_keys:
        check(key in c, f"  Has field '{key}' = {repr(c[key])[:60]}")

    check(c["source_type"] == "pdf_text", "  source_type is 'pdf_text'")
    check(c["modality"] == "pdf", "  modality is 'pdf'")
    check(c["page_number"] is not None, f"  page_number: {c['page_number']}")

    print(f"\n  Sample PDF chunk text:")
    print(f"  {'-'*50}")
    print(f"    {c['text'][:200]}...")
    print(f"  {'-'*50}")

    return chunks


def _create_test_pdf(output_path: str):
    """Create a simple test PDF using pymupdf."""
    import fitz

    doc = fitz.open()

    # Page 1
    page = doc.new_page()
    text = (
        "System Design Fundamentals\n\n"
        "This document covers the essential concepts of system design "
        "including load balancing, caching, database sharding, and "
        "microservices architecture.\n\n"
        "Load Balancing\n\n"
        "A load balancer distributes incoming network traffic across "
        "multiple servers. Common algorithms include round-robin, "
        "least connections, and IP hash. Load balancers can be "
        "hardware-based or software-based (e.g., Nginx, HAProxy).\n\n"
        "Benefits of load balancing:\n"
        "- Improved availability and reliability\n"
        "- Better resource utilization\n"
        "- Horizontal scalability\n"
        "- Session persistence options"
    )
    page.insert_textbox(fitz.Rect(50, 50, 550, 750), text, fontsize=11)

    # Page 2
    page = doc.new_page()
    text = (
        "Caching Strategies\n\n"
        "Caching stores frequently accessed data in fast storage "
        "to reduce latency and database load.\n\n"
        "Common caching patterns:\n"
        "1. Cache-Aside (Lazy Loading) - Application checks cache first\n"
        "2. Write-Through - Data written to cache and DB simultaneously\n"
        "3. Write-Behind - Data written to cache, async sync to DB\n"
        "4. Read-Through - Cache handles data fetching transparently\n\n"
        "Popular caching solutions:\n"
        "- Redis: In-memory data structure store\n"
        "- Memcached: Distributed memory caching\n"
        "- CDN Caching: Edge caching for static assets\n\n"
        "Database Sharding\n\n"
        "Sharding horizontally partitions data across multiple databases. "
        "Each shard holds a subset of the data, enabling horizontal scaling."
    )
    page.insert_textbox(fitz.Rect(50, 50, 550, 750), text, fontsize=11)

    # Page 3
    page = doc.new_page()
    text = (
        "REST API Design Principles\n\n"
        "RESTful APIs follow these key principles:\n\n"
        "1. Resource-Based URLs: /api/users/123\n"
        "2. HTTP Methods: GET, POST, PUT, DELETE\n"
        "3. Stateless: Each request contains all needed info\n"
        "4. JSON Responses: Standard data format\n\n"
        "Example API Endpoints:\n"
        "GET    /api/users          - List all users\n"
        "POST   /api/users          - Create a new user\n"
        "GET    /api/users/:id      - Get specific user\n"
        "PUT    /api/users/:id      - Update user\n"
        "DELETE /api/users/:id      - Delete user\n\n"
        "Status Codes:\n"
        "200 OK - Request succeeded\n"
        "201 Created - Resource created\n"
        "400 Bad Request - Invalid input\n"
        "404 Not Found - Resource doesn't exist\n"
        "500 Internal Server Error"
    )
    page.insert_textbox(fitz.Rect(50, 50, 550, 750), text, fontsize=11)

    doc.save(output_path)
    doc.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Test 4: Assembler
# ═══════════════════════════════════════════════════════════════════════════

def test_assembler(audio_chunks, visual_chunks, pdf_chunks):
    print("\n" + "=" * 60)
    print("TEST 4: Chunk Assembler")
    print("=" * 60)

    from src.chunking import assemble_chunks
    from src.chunking.assembler import print_summary

    output_path = os.path.join(OUTPUT_DIR, "test_assembled_chunks.json")

    result = assemble_chunks(
        audio_chunks,
        visual_chunks,
        pdf_chunks,
        output_path=output_path,
    )

    check(result["total_chunks"] > 0, f"Total chunks: {result['total_chunks']}")
    check("video_transcript" in result["sources"], "  Has video_transcript source")
    check("video_visual" in result["sources"], "  Has video_visual source")
    check(result["total_tokens"] > 0, f"  Total tokens: ~{result['total_tokens']}")

    # Check global_index
    for i, chunk in enumerate(result["chunks"]):
        assert chunk["global_index"] == i, f"global_index mismatch at {i}"
    check(True, "  All global_index values are sequential")

    # Check file was written
    check(os.path.exists(output_path), f"  Written to {output_path}")

    # Print summary
    print_summary(result)

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Modular Chunking System")
    print("=" * 60)

    audio_chunks = test_video_audio()
    visual_chunks = test_video_visual()
    pdf_chunks = test_pdf()
    test_assembler(audio_chunks, visual_chunks, pdf_chunks)

    print("\n" + "=" * 60)
    print("RESULT: ALL CHUNKING TESTS PASS ✅")
    print("=" * 60)
