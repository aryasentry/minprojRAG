# Video RAG Architecture

This document outlines the architecture and data flow for the Multimodal Video RAG system, utilizing the **Gemini Nanobanana** model suite for vision, reasoning, and generation.

## System Overview

The system processes video content through a multi-stage pipeline to extract, analyze, and index multimodal information (audio, visual slides, and text). This enables users to query video content using natural language and receive precise answers with timestamps and visual context.

## Data Flow Diagram

```mermaid
graph TD
    %% Styling
    classDef process fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef storage fill:#fff3e0,stroke:#ff6f00,stroke-width:2px;
    classDef model fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef user fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    subgraph "Ingestion Pipeline"
        Input([Input Video]) --> AudioExt[Audio Extraction]
        Input --> FrameExt[Frame Extraction (1 FPS)]
        
        AudioExt --> Audio[Audio WAV]
        FrameExt --> Frames[Frame Images]
        
        Audio --> Whisper[Whisper Model]:::model --> Transcript[Transcript JSON]
        Frames --> SSIM[Slide Detection (SSIM)]:::process --> UniqueSlides[Unique Slides]
        
        UniqueSlides --> Vision[Gemini Nanobanana (Vision)]:::model 
        Vision --> SlideDesc[Slide Descriptions + OCR]
        
        Transcript & SlideDesc --> Chunking[Smart Chunking]:::process
        Chunking --> Chunks[Unified Chunks]
    end

    subgraph "Indexing Layer"
        Chunks --> EmbedModel[Gemini Embedding Model]:::model
        EmbedModel --> VectorDB[(FAISS Vector Store)]:::storage
    end

    subgraph "Retrieval & Generation"
        User([User Query]):::user --> EmbedQuery[Query Embedding]
        EmbedQuery --> VectorDB
        
        VectorDB --> Retrieval[Multi-Index Retrieval]:::process
        Retrieval --> LateFusion[Late Fusion Merge]:::process
        LateFusion --> Context[Unified Context (Video + Text)]
        
        Context --> LLM[Gemini Nanobanana (LLM)]:::model
        LLM --> Answer([Final Answer]):::user
    end

    %% Styles
    linkStyle default stroke-width:2px,fill:none,stroke:#333;
```

## detailed Component Breakdown

### 1. Ingestion Pipeline
The ingestion process breaks down the video into analyzable components.
- **Audio Extraction**: Extracts audio track from video files using FFmpeg.
- **Whisper Transcription**: Converts audio speech to text with timestamps.
- **Frame Extraction**: Captures frames at 1 FPS to monitor visual content.
- **Slide Detection**: Uses SSIM (Structural Similarity Index) to detect when a unique slide appears, preventing redundant processing of static frames.
- **Visual Analysis**: The **Gemini Nanobanana (Vision)** model analyzes each unique slide to generate:
  - Concise Scene Descriptions
  - OCR Text Extraction
  
### 2. Smart Chunking
Aligns the transcript with the visual slides.
- **Transcript Chunking**: Breaks transcript into logical segments.
- **Visual Association**: Maps each transcript chunk to the corresponding slide visible at that time.
- **Result**: `TargetChunk` objects containing both text and visual context.

### 3. Indexing Layer
Stores the processed information for efficient retrieval.
- **Embedding**: Converts text and visual descriptions into vector embeddings using the configured embedding model.
- **Vector Store**: Uses FAISS to index these vectors, allowing for semantic similarity search.

### 4. Retrieval & Generation (RAG)
Handles user queries to provide answers.
- **Late Fusion Retrieval**: Queries independent indices (Transcript, Visual, PDF) and merges the results based on timestamps.
- **Gemini Nanobanana (LLM)**: The core reasoning engine. It takes the retrieved video segments and relevant text as context to generate an accurate, helpful response for the user.
