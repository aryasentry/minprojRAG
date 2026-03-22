# Multimodal Video Preprocessing Pipeline - TODO

## Project Overview
Build a pipeline that processes an 11-minute YouTube lecture video to extract:
- Audio transcriptions with word-level timestamps
- Slide changes detected via SSIM
- Vision-based slide descriptions
- Structured chunks for downstream RAG/embedding

---

## Setup & Installation

### 1. Create Virtual Environment with uv
- [ ] Install uv if not already installed: `pip install uv`
- [ ] Create virtual environment: `uv venv`
- [ ] Activate venv:
  - Windows: `.venv\Scripts\activate`
  - macOS/Linux: `source .venv/bin/activate`

### 2. Install Dependencies with uv
- [ ] Install core packages:
  ```bash
  uv pip install faster-whisper opencv-python scikit-image langchain langchain-community
  ```
- [ ] Install additional packages:
  ```bash
  uv pip install Pillow numpy requests tqdm
  ```
- [ ] Ensure ffmpeg is installed on system:
  - Windows: Download from ffmpeg.org or use `choco install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

### 3. Verify Ollama Setup
- [ ] Ensure Ollama is running: `ollama serve`
- [ ] Pull required models:
  ```bash
  ollama pull llava:7b
  ollama pull qwen2.5:1.5b
  ollama pull dimavz/whisper-tiny:latest
  ```

---

## Directory Structure

Create the following structure:

```
project/
│
├── input/
│   └── video.mp4              # Place input video here
│
├── output/
│   ├── transcript.json        # Audio transcription output
│   ├── slides.json            # Slide detection + descriptions
│   ├── chunks.json            # Final chunked output
│   └── frames/                # Extracted frames
│
├── src/
│   └── pipeline.py            # Main pipeline code
│
├── TODO.md                    # This file
└── pyproject.toml             # Project config (optional with uv)
```

---

## Implementation Steps

### STEP 1: Audio Transcription (~30 min)

**Goal:** Extract audio and transcribe with word-level timestamps.

#### Tasks:
- [ ] Create `extract_audio(video_path, output_path)` function
  - Use `subprocess` to call ffmpeg
  - Command: `ffmpeg -i video.mp4 -vn -acodec pcm_s16le output/audio.wav -ar 16000`
  - Handle errors if ffmpeg fails

- [ ] Create `transcribe_audio(audio_path, output_path)` function
  - Initialize `faster-whisper` with model size `small` or `base`
  - Set `word_timestamps=True`, `beam_size=5`
  - Process audio in segments
  - Build output structure:
    ```json
    {
      "video_id": "video01",
      "segments": [
        {
          "segment_id": 0,
          "start": 0.52,
          "end": 4.82,
          "text": "In this lecture we discuss gradient descent.",
          "words": [
            {"word": "In", "start": 0.52, "end": 0.63},
            ...
          ]
        }
      ]
    }
    ```

- [ ] Add logging for transcription progress
- [ ] Save output to `output/transcript.json`

---

### STEP 2: Frame Extraction (~15 min)

**Goal:** Extract frames at 1 FPS from video.

#### Tasks:
- [ ] Create `extract_frames(video_path, output_dir, fps=1)` function
  - Use ffmpeg via subprocess:
    ```bash
    ffmpeg -i video.mp4 -vf fps=1 output/frames/frame_%04d.jpg
    ```
  - Create output directory if it doesn't exist

- [ ] Create frame-to-timestamp mapping
  - Formula: `timestamp = frame_index * (1/fps)`
  - Store mapping in a dictionary for later use

- [ ] Add logging for extraction progress
- [ ] Handle case where frames directory already exists

---

### STEP 3: Slide Change Detection (~30 min)

**Goal:** Detect slide transitions using SSIM (Structural Similarity Index).

#### Tasks:
- [ ] Create `load_frame(frame_path)` helper function
  - Read image with OpenCV
  - Convert to grayscale
  - Resize to consistent size (e.g., 256x256) for faster comparison

- [ ] Create `detect_slides(frames_dir, threshold=0.85)` function
  - Iterate through consecutive frame pairs
  - Calculate SSIM score using `skimage.metrics.structural_similarity()`
  - If `ssim_score < threshold`:
    - Mark previous frame as slide boundary
    - Increment `slide_id`
    - Set `is_last_frame_before_change = True`

- [ ] Build slide metadata structure:
  ```json
  {
    "slide_id": 3,
    "start_time": 120.0,
    "end_time": 148.0,
    "last_frame_path": "frames/frame_0148.jpg",
    "is_last_frame_before_change": true
  }
  ```

- [ ] Make threshold configurable (constant at top of file)
- [ ] Save slides to `output/slides.json` (before vision descriptions)
- [ ] Add logging for slide detection progress

---

### STEP 4: Vision Model Description (~45 min)

**Goal:** Generate descriptions for each detected slide using LLaVA.

#### Tasks:
- [ ] Create `describe_slide(image_path, prompt)` function
  - Use Ollama's local API at `http://localhost:11434`
  - Endpoint: `POST /api/generate`
  - Model: `llava:7b`
  - Include image in request (base64 encode or path)

- [ ] Create prompt template:
  ```python
  PROMPT_TEMPLATE = """
  You are analyzing a lecture slide.

  Describe:
  - Main topic of the slide
  - Key concepts mentioned
  - Any equations (write them clearly)
  - Any code snippets
  - Any diagrams or charts

  Be concise but informative.
  If unsure, say "Not clearly visible."
  """
  ```

- [ ] Create `generate_slide_descriptions(slides, frames_dir)` function
  - Iterate through all slides
  - Call vision model for each slide's last frame
  - Store description in slide object:
    ```json
    {
      "slide_id": 3,
      ...
      "description": "...generated text...",
      "vision_model": "llava:7b"
    }
    ```

- [ ] Add retry logic for failed API calls
- [ ] Add progress logging
- [ ] Handle Ollama connection errors gracefully

---

### STEP 5: Slide-Aware Transcript Chunking (~45 min)

**Goal:** Split transcript into chunks respecting slide boundaries.

#### Tasks:
- [ ] Create `count_tokens(text)` helper function
  - Use simple whitespace-based estimation or `len(text) // 4`
  - Or use `langchain` token counter

- [ ] Create `chunk_transcript(transcript, slides, max_tokens=400)` function
  - For each slide:
    - Collect transcript segments within slide time range
    - Merge sentences until token limit reached
    - Create chunk with metadata

- [ ] Implement chunk structure:
  ```json
  {
    "chunk_id": "video01_chunk_07",
    "video_id": "video01",
    "slide_id": 3,
    "modality": "transcript",
    "text": "...",
    "start_time": 123.4,
    "end_time": 145.9,
    "token_count": 312
  }
  ```

- [ ] Ensure NO chunk crosses slide boundaries
- [ ] Preserve word-level timestamps within chunks
- [ ] Add logging for chunking progress

---

### STEP 6: Slide Caption Chunks (~15 min)

**Goal:** Create separate chunks for slide descriptions.

#### Tasks:
- [ ] Create `create_slide_caption_chunks(slides)` function
  - For each slide, create a caption chunk:
    ```json
    {
      "chunk_id": "video01_slide_03",
      "video_id": "video01",
      "slide_id": 3,
      "modality": "slide_caption",
      "text": "Slide description here...",
      "start_time": 120.0,
      "end_time": 148.0,
      "frame_path": "frames/frame_0148.jpg",
      "is_last_frame_before_change": true
    }
    ```

- [ ] Merge with transcript chunks in `chunks.json`

---

### STEP 7: Final Output Assembly (~15 min)

**Goal:** Save all output files with proper structure.

#### Tasks:
- [ ] Create `save_outputs(transcript, slides, chunks, output_dir)` function
- [ ] Save files:
  - `output/transcript.json` - Full transcription
  - `output/slides.json` - Slide metadata + descriptions
  - `output/chunks.json` - All chunks (transcript + slide captions)

- [ ] Add summary logging:
  - Number of segments
  - Number of slides detected
  - Number of chunks created
  - Total processing time

---

## Pipeline Orchestration

### Main Function (~30 min)

**Goal:** Create the main pipeline orchestrator.

#### Tasks:
- [ ] Create `run_pipeline(video_path, output_dir)` function
  - Step 1: Extract audio
  - Step 2: Transcribe audio
  - Step 3: Extract frames
  - Step 4: Detect slides
  - Step 5: Generate slide descriptions
  - Step 6: Chunk transcript
  - Step 7: Create slide caption chunks
  - Step 8: Save all outputs

- [ ] Add error handling at each stage
- [ ] Add timing/profiling for each step
- [ ] Create `if __name__ == "__main__":` entry point

---

## Testing & Validation

### Test Cases
- [ ] Test with sample 11-minute video
- [ ] Verify transcript.json has word timestamps
- [ ] Verify slides.json has descriptions
- [ ] Verify chunks.json has both modalities
- [ ] Verify no chunk crosses slide boundaries
- [ ] Verify token counts are under 400

### Edge Cases to Handle
- [ ] Video with no slides (constant frame)
- [ ] Video with rapid slide changes
- [ ] Audio with no speech
- [ ] Ollama API failures
- [ ] Missing ffmpeg
- [ ] Corrupted frames

---

## Code Quality Checklist

- [ ] All functions have docstrings
- [ ] Logging at INFO and ERROR levels
- [ ] Type hints on function signatures
- [ ] Configurable constants at top of file
- [ ] Error handling with meaningful messages
- [ ] No hardcoded paths (use parameters)
- [ ] Memory efficient (don't load all frames at once)

---

## Configuration Constants

Place at top of `pipeline.py`:

```python
# Slide detection
SSIM_THRESHOLD = 0.85
FRAME_FPS = 1

# Chunking
MAX_TOKENS_PER_CHUNK = 400

# Models
WHISPER_MODEL = "small"  # or "base"
VISION_MODEL = "llava:7b"
LLM_MODEL = "qwen2.5:1.5b"

# Ollama
OLLAMA_HOST = "http://localhost:11434"

# Directories
INPUT_DIR = "input"
OUTPUT_DIR = "output"
FRAMES_DIR = "output/frames"
```

---

## Estimated Total Time: ~4-5 hours

| Step | Time |
|------|------|
| Setup | 30 min |
| Step 1: Audio | 30 min |
| Step 2: Frames | 15 min |
| Step 3: Slides | 30 min |
| Step 4: Vision | 45 min |
| Step 5: Chunking | 45 min |
| Step 6: Captions | 15 min |
| Step 7: Output | 15 min |
| Orchestration | 30 min |
| Testing | 60 min |

---

## Notes

- **DO NOT implement:** embeddings, FAISS, RAG, database storage
- **Only processing pipeline** - output is JSON files ready for downstream use
- All processing must run locally with no cloud APIs
- Vision calls: expect 10-30 slides for 11-min video
- Total runtime target: under 10-15 minutes for full pipeline

---

## Quick Start Commands

```bash
# Create and activate venv
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
uv pip install faster-whisper opencv-python scikit-image langchain langchain-community Pillow numpy requests tqdm

# Run pipeline
python src/pipeline.py input/video.mp4
```
