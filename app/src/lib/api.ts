// API Client for RAG Backend
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface VideoSegment {
  video_id: string;
  source_file: string;
  start_time: number;
  end_time: number;
  text_preview: string;
  avg_score: number;
  duration: number;
  chunk_ids?: string[];
  sources?: string[];
}

export interface PdfSegment {
  text: string;
  page: number;
  source: string;
  score: number;
}

export interface QueryResponse {
  answer: string;
  video_segments: VideoSegment[];
  pdf_segments: PdfSegment[];
}

export interface VideoInfo {
  filename: string;
  path: string;
  size: number;
}

export interface SegmentExtractionResponse {
  message: string;
  segment_url: string;
  start_time: number;
  end_time: number;
  duration: number;
}

export interface ProcessingStatus {
  stage: string;
  status: string;
  message: string;
  timestamp: number;
}

// Query RAG system
export async function queryRAG(query: string, top_k: number = 5): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE_URL}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k }),
  });
  
  if (!response.ok) {
    throw new Error(`Query failed: ${response.statusText}`);
  }
  
  return response.json();
}

// Admin: Upload video
export async function uploadVideo(file: File): Promise<{ filename: string; path: string; size: number }> {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${API_BASE_URL}/admin/upload-video`, {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Upload failed');
  }
  
  return response.json();
}

// Admin: Process video
export async function processVideo(videoPath: string): Promise<{ message: string; pipeline: string; embedding: string }> {
  const response = await fetch(`${API_BASE_URL}/admin/process-video`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video_path: videoPath }),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Processing failed');
  }
  
  return response.json();
}

// Admin: List videos
export async function listVideos(): Promise<VideoInfo[]> {
  const response = await fetch(`${API_BASE_URL}/admin/videos`);
  
  if (!response.ok) {
    throw new Error('Failed to list videos');
  }
  
  return response.json();
}

// Admin: Get processing status
export async function getProcessingStatus(): Promise<ProcessingStatus> {
  const response = await fetch(`${API_BASE_URL}/admin/processing-status`);
  
  if (!response.ok) {
    throw new Error('Failed to get processing status');
  }
  
  return response.json();
}

// Extract video segment
export async function extractSegment(
  videoId: string,
  sourceFile: string,
  startTime: number,
  endTime: number
): Promise<SegmentExtractionResponse> {
  const response = await fetch(`${API_BASE_URL}/extract-segment`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      video_id: videoId,
      source_file: sourceFile,
      start_time: startTime,
      end_time: endTime,
    }),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Extraction failed');
  }
  
  return response.json();
}

// Health check
export async function checkHealth(): Promise<{ status: string; retriever: boolean }> {
  const response = await fetch(`${API_BASE_URL}/health`);
  return response.json();
}
