"""
RAG API Server

Exposes the retrieval system via REST API for the Next.js frontend.
Includes admin endpoints for video upload and processing.
"""

import os
import shutil
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import OUTPUT_DIR, INPUT_DIR, logger
from .retrieval import FusionRetriever
from .video_extractor import get_video_extractor


# ── Global State ────────────────────────────────────────────────────────────

retriever: Optional[FusionRetriever] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global retriever
    logger.info("Initializing FusionRetriever...")
    try:
        retriever = FusionRetriever()
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}")
    yield
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(title="Multimodal RAG API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve video files directly
video_dir = OUTPUT_DIR  # Or wherever videos are stored
# Actually, videos are in root usually. Let's serve PROJECT_ROOT
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/files", StaticFiles(directory=PROJECT_ROOT), name="files")


# ── Models ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class VideoSegment(BaseModel):
    video_id: str
    start_time: float
    end_time: float
    text_preview: str
    avg_score: float
    duration: float

class QueryResponse(BaseModel):
    answer: str
    video_segments: List[dict]
    pdf_segments: List[dict]


# ── Endpoints ───────────────────────────────────────────────────────────────

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from .config import LLM_MODEL, OLLAMA_HOST

# Initialize LLM
llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_HOST, temperature=0.3)

@app.post("/query", response_model=QueryResponse)
async def query_rag(req: QueryRequest):
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    results = retriever.query(req.query, top_k=req.top_k)
    
    videos = results.get("video_segments", [])
    pdfs = results.get("pdf_segments", [])
    
    # 1. Build Context
    context_parts = []
    
    if videos:
        context_parts.append("--- RELEVANT VIDEO SEGMENTS ---")
        for i, v in enumerate(videos[:3], 1):
            context_parts.append(f"Video {i} (Time: {v['start_time']:.1f}s - {v['end_time']:.1f}s): {v['text_preview']}")

    if pdfs:
        context_parts.append("--- RELEVANT PDF CONTENT ---")
        for i, p in enumerate(pdfs[:3], 1):
            context_parts.append(f"PDF Excerpt {i} (Page {p['page']}): {p['text']}")

    context_str = "\n\n".join(context_parts)
    
    if not context_str:
        return {
            "answer": "I couldn't find any relevant information in the course materials to answer your question.",
            "video_segments": [],
            "pdf_segments": []
        }

    # 2. Generate Answer
    system_prompt = (
        "You are an AI teaching assistant for a course. Use the provided context to answer the student's question. "
        "If the answer isn't in the context, say you don't know. "
        "Keep the answer concise, encouraging, and helpful. "
        "Reference the video segments or PDF pages if applicable."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({"context": context_str, "question": req.query})
        answer_text = response.content
    except Exception as e:
        logger.error(f"LLM Generation failed: {e}")
        answer_text = "Sorry, I encountered an error while generating the answer, but here are the relevant materials I found."

    return {
        "answer": answer_text,
        "video_segments": videos,
        "pdf_segments": pdfs
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "retriever": retriever is not None}


# ── Admin Endpoints ─────────────────────────────────────────────────────────

class ProcessVideoRequest(BaseModel):
    video_path: str

class VideoInfo(BaseModel):
    filename: str
    path: str
    size: int


@app.post("/admin/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for processing."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate file type
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Ensure input directory exists
    os.makedirs(INPUT_DIR, exist_ok=True)
    
    # Save file
    file_path = os.path.join(INPUT_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = os.path.getsize(file_path)
        logger.info(f"Video uploaded: {file.filename} ({file_size} bytes)")
        
        return {
            "message": "Video uploaded successfully",
            "filename": file.filename,
            "path": file_path,
            "size": file_size
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/admin/process-video")
async def process_video(req: ProcessVideoRequest):
    """Process an uploaded video through the pipeline."""
    if not os.path.exists(req.video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    try:
        from .pipeline import run_pipeline
        from .embedding import embed_new_chunks
        
        logger.info(f"Starting pipeline for: {req.video_path}")
        
        # Run pipeline (creates audio_chunks.json and video_chunks.json)
        success = run_pipeline(req.video_path, OUTPUT_DIR)
        if not success:
            raise HTTPException(status_code=500, detail="Pipeline failed")
        
        # Embed new chunks using incremental system
        logger.info("Embedding new chunks...")
        embed_success = embed_new_chunks(force_rebuild=False)
        if not embed_success:
            logger.warning("Embedding failed, but pipeline completed")
        
        # Reload retriever
        global retriever
        logger.info("Reloading retriever with new embeddings...")
        retriever = FusionRetriever()
        
        return {
            "message": "Video processed successfully",
            "pipeline": "completed",
            "embedding": "completed" if embed_success else "failed"
        }
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/admin/videos", response_model=List[VideoInfo])
async def list_videos():
    """List all videos in the input directory."""
    if not os.path.exists(INPUT_DIR):
        return []
    
    videos = []
    for filename in os.listdir(INPUT_DIR):
        filepath = os.path.join(INPUT_DIR, filename)
        if os.path.isfile(filepath):
            videos.append({
                "filename": filename,
                "path": filepath,
                "size": os.path.getsize(filepath)
            })
    
    return videos


@app.get("/admin/processing-status")
async def get_processing_status():
    """Get the current processing status for admin panel progress display."""
    import json
    status_file = os.path.join(OUTPUT_DIR, "processing_status.json")
    
    # If file doesn't exist, return idle status
    if not os.path.exists(status_file):
        return {
            "stage": "idle",
            "status": "idle",
            "message": "No processing in progress",
            "timestamp": 0
        }
    
    try:
        with open(status_file, "r") as f:
            status = json.load(f)
        return status
    except Exception as e:
        logger.error(f"Failed to read status file: {e}")
        return {
            "stage": "error",
            "status": "error",
            "message": f"Failed to read status: {str(e)}",
            "timestamp": 0
        }


# ── Video Extraction Endpoints ──────────────────────────────────────────────

class ExtractSegmentRequest(BaseModel):
    video_id: str
    source_file: str
    start_time: float
    end_time: float


@app.post("/extract-segment")
async def extract_video_segment(req: ExtractSegmentRequest):
    """Extract a specific time range from a video."""
    # Find video file
    video_path = None
    
    # Check in input directory
    input_path = os.path.join(INPUT_DIR, req.source_file)
    if os.path.exists(input_path):
        video_path = input_path
    else:
        # Check in project root
        root_path = os.path.join(PROJECT_ROOT, req.source_file)
        if os.path.exists(root_path):
            video_path = root_path
    
    if not video_path:
        raise HTTPException(status_code=404, detail=f"Video not found: {req.source_file}")
    
    try:
        extractor = get_video_extractor()
        segment_path = extractor.extract_segment(
            video_path,
            req.start_time,
            req.end_time
        )
        
        if not segment_path:
            raise HTTPException(status_code=500, detail="Failed to extract segment")
        
        # Return URL to access the segment
        segment_url = f"/files/output/video_segments/{os.path.basename(segment_path)}"
        
        return {
            "message": "Segment extracted successfully",
            "segment_url": segment_url,
            "start_time": req.start_time,
            "end_time": req.end_time,
            "duration": req.end_time - req.start_time
        }
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "ok", "retriever": retriever is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
