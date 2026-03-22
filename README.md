# Multimodal RAG Course Platform

A full-stack multimodal Retrieval-Augmented Generation (RAG) project that:
- processes course videos into transcript and visual chunks,
- embeds chunks into FAISS indexes,
- answers questions through a FastAPI backend,
- serves a Next.js frontend with chat + admin upload/processing.

## What You Need To Install

For anyone running this repo on a new machine, install these first:

1. Git
2. Python 3.10+ (recommended: 3.11)
3. Node.js 20+ (includes npm)
4. Ollama
5. FFmpeg
6. Docker Desktop (optional but recommended for OCR speed)

### Why each dependency is needed

- Git: clone/pull/push the repository.
- Python: backend API + video pipeline.
- Node.js: Next.js frontend app in app/.
- Ollama: local LLM + embedding inference.
- FFmpeg: audio extraction, frame extraction, and clip cutting.
- Docker Desktop: runs Tesseract OCR container (used when USE_TESSERACT_OCR = True in src/config.py).

## Quick Install Commands

### Windows (PowerShell)

- Install Python: https://www.python.org/downloads/
- Install Node.js: https://nodejs.org/
- Install Ollama: https://ollama.com/download
- Install Docker Desktop: https://www.docker.com/products/docker-desktop/

FFmpeg options:
- Option A (already in this repo): use the bundled binary path each session:

  $env:PATH = "$PWD\\ffmpeg\\ffmpeg-8.0.1-essentials_build\\bin;$env:PATH"

- Option B: install FFmpeg globally and add it to system PATH.

### macOS

brew install python node ffmpeg
brew install --cask ollama docker

### Ubuntu/Debian

sudo apt update
sudo apt install -y python3 python3-venv python3-pip nodejs npm ffmpeg docker.io

Install Ollama from:
https://ollama.com/download/linux

## Ollama Models To Pull

Start Ollama and pull models used by this repo:

ollama serve
ollama pull qwen2.5:1.5b
ollama pull qwen3-embedding:0.6b
ollama pull qwen3-vl:235b-cloud

Notes:
- Model names come from src/config.py.
- If you change model names in config, pull those exact names.

## Project Setup

From repo root:

1. Create and activate virtual environment

Windows:
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1

macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate

2. Install backend dependencies

pip install -r requirements.txt

3. Install frontend dependencies

cd app
npm install
cd ..

## Verify Your Environment

Run:

python verify_system.py

This checks:
- Python packages
- Ollama service + model availability
- FFmpeg availability
- required directories
- chunks and FAISS indexes

## First-Time Data Processing

Place at least one video in input/ (example: input/video.mp4), then run:

python process_fresh.py input/video.mp4

This will:
- clear old outputs,
- run full pipeline,
- generate audio/video chunks,
- build embeddings (unless --skip-embedding is used).

## Run The Application

Open 3 terminals from repo root.

Terminal 1 (Ollama):
ollama serve

Terminal 2 (Backend API):

Windows (with bundled ffmpeg):
$env:PATH = "$PWD\\ffmpeg\\ffmpeg-8.0.1-essentials_build\\bin;$env:PATH"
.\\.venv\\Scripts\\python.exe -m src.server

macOS/Linux:
python -m src.server

Terminal 3 (Frontend):
cd app
npm run dev

Open:
http://localhost:3000

Backend API default:
http://localhost:8000

## Frontend Environment Variable (Optional)

The frontend uses NEXT_PUBLIC_API_URL and defaults to http://localhost:8000.

Create app/.env.local if needed:

NEXT_PUBLIC_API_URL=http://localhost:8000

## Useful Commands

Backend health check:
GET http://localhost:8000/health

Manual embedding refresh:
python embed_chunks.py

Fresh test flow:
test_fresh_processing.bat

## Troubleshooting

1. ffmpeg not found
- Ensure ffmpeg is in PATH.
- On Windows, run the PATH export command shown above before starting backend.

2. Docker not found (OCR warnings)
- Install Docker Desktop, or
- set USE_TESSERACT_OCR = False in src/config.py.

3. Ollama model missing
- Run ollama pull for the missing model.
- Re-run python verify_system.py.

4. Frontend cannot reach backend
- Ensure backend runs on port 8000.
- Set app/.env.local with NEXT_PUBLIC_API_URL.

## Repository Layout (High-Level)

- src/: backend API + processing pipeline
- app/: Next.js frontend
- input/: source videos/PDFs
- output/: generated chunks, transcript, slides, indices
- tests/: Python tests
- UMLdiagrams/: architecture and design diagrams

## License

Add your license here (MIT, Apache-2.0, etc.).
