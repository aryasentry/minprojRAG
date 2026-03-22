"""
System Verification Script

Checks all components of the RAG system to ensure everything is properly configured.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    INPUT_DIR, OUTPUT_DIR, FRAMES_DIR, OLLAMA_HOST,
    WHISPER_MODEL, VISION_MODEL, LLM_MODEL, EMBEDDING_MODEL,
    logger
)


def check_ollama():
    """Check if Ollama is running and models are available."""
    import requests
    
    print("\n🔍 Checking Ollama...")
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            print("  ✅ Ollama is running")
            
            required_models = [WHISPER_MODEL, VISION_MODEL, LLM_MODEL, EMBEDDING_MODEL]
            missing = []
            
            for model in required_models:
                # Check if model name is in the list (partial match)
                found = any(model.split(":")[0] in name for name in model_names)
                if found:
                    print(f"  ✅ Model found: {model}")
                else:
                    print(f"  ❌ Model missing: {model}")
                    missing.append(model)
            
            if missing:
                print("\n  📥 To install missing models, run:")
                for model in missing:
                    print(f"     ollama pull {model}")
                return False
            
            return True
        else:
            print("  ❌ Ollama is not responding properly")
            return False
    except requests.exceptions.ConnectionError:
        print("  ❌ Cannot connect to Ollama")
        print("     Make sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        print(f"  ❌ Error checking Ollama: {e}")
        return False


def check_ffmpeg():
    """Check if ffmpeg is installed."""
    print("\n🔍 Checking ffmpeg...")
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version_line = result.stdout.split("\n")[0]
            print(f"  ✅ ffmpeg is installed: {version_line}")
            return True
        else:
            print("  ❌ ffmpeg command failed")
            return False
    except FileNotFoundError:
        print("  ❌ ffmpeg not found")
        print("     Install ffmpeg: https://ffmpeg.org/download.html")
        return False
    except Exception as e:
        print(f"  ❌ Error checking ffmpeg: {e}")
        return False


def check_directories():
    """Check if required directories exist."""
    print("\n🔍 Checking directories...")
    
    dirs = [INPUT_DIR, OUTPUT_DIR, FRAMES_DIR]
    all_exist = True
    
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path} exists")
        else:
            print(f"  ⚠️  {dir_path} does not exist (will be created)")
            os.makedirs(dir_path, exist_ok=True)
            all_exist = False
    
    return True  # Always return True since we create them


def check_chunks():
    """Check if chunks.json exists and is valid."""
    print("\n🔍 Checking chunks...")
    
    chunks_path = os.path.join(OUTPUT_DIR, "chunks.json")
    if not os.path.exists(chunks_path):
        print(f"  ⚠️  No chunks found at {chunks_path}")
        print("     Run the pipeline first to process a video")
        return False
    
    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        total = data.get("total_chunks", 0)
        sources = data.get("sources", {})
        
        print(f"  ✅ Chunks file found: {total} total chunks")
        for source_type, count in sources.items():
            print(f"     - {source_type}: {count}")
        
        # Check if visual chunks have OCR and descriptions
        chunks = data.get("chunks", [])
        visual_chunks = [c for c in chunks if c.get("source_type") == "video_visual"]
        
        if visual_chunks:
            sample = visual_chunks[0]
            has_ocr = sample.get("ocr_text") is not None
            has_scene = sample.get("scene_description") is not None
            
            if has_ocr and has_scene:
                print("  ✅ Visual chunks have OCR and scene descriptions")
            else:
                print("  ⚠️  Visual chunks may be missing OCR or scene descriptions")
        
        return True
    except Exception as e:
        print(f"  ❌ Error reading chunks: {e}")
        return False


def check_embeddings():
    """Check if FAISS indices exist."""
    print("\n🔍 Checking embeddings...")
    
    from src.embedding import FAISS_INDEX_DIR, INDEX_TRANSCRIPT, INDEX_VISUAL, INDEX_PDF
    
    if not os.path.exists(FAISS_INDEX_DIR):
        print(f"  ⚠️  No FAISS indices found at {FAISS_INDEX_DIR}")
        print("     Run: python embed_chunks.py")
        return False
    
    indices = [INDEX_TRANSCRIPT, INDEX_VISUAL, INDEX_PDF]
    found_any = False
    
    for index_name in indices:
        index_path = os.path.join(FAISS_INDEX_DIR, index_name)
        if os.path.exists(index_path):
            print(f"  ✅ Index found: {index_name}")
            found_any = True
        else:
            print(f"  ⚠️  Index not found: {index_name}")
    
    if not found_any:
        print("\n  📥 To create embeddings, run:")
        print("     python embed_chunks.py")
        return False
    
    return True


def check_python_deps():
    """Check if required Python packages are installed."""
    print("\n🔍 Checking Python dependencies...")
    
    required = [
        "fastapi",
        "uvicorn",
        "faster_whisper",
        "cv2",
        "langchain",
        "faiss",
        "PIL",
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} not found")
            missing.append(package)
    
    if missing:
        print("\n  📥 To install missing packages, run:")
        print("     pip install -r requirements.txt")
        return False
    
    return True


def main():
    print("="*60)
    print("  MULTIMODAL RAG SYSTEM VERIFICATION")
    print("="*60)
    
    checks = [
        ("Python Dependencies", check_python_deps),
        ("Ollama", check_ollama),
        ("ffmpeg", check_ffmpeg),
        ("Directories", check_directories),
        ("Chunks", check_chunks),
        ("Embeddings", check_embeddings),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error during {name} check: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 All checks passed! System is ready.")
        print("\nTo start the system:")
        print("  1. Backend: python -m src.server")
        print("  2. Frontend: cd app && npm run dev")
        print("  3. Open: http://localhost:3000")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
