'use client';

import { useState, useEffect } from 'react';
import { Upload, Video, CheckCircle, Loader, FileText, Clock, AlertCircle } from 'lucide-react';
import { uploadVideo, processVideo, listVideos, getProcessingStatus, type VideoInfo, type ProcessingStatus } from '@/lib/api';
import Link from 'next/link';

export default function AdminDashboard() {
  const [videos, setVideos] = useState<VideoInfo[]>([]);
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState<ProcessingStatus | null>(null);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // Load videos on mount
  useEffect(() => {
    loadVideos();
  }, []);

  // Poll processing status while processing
  useEffect(() => {
    if (!processing) {
      setProcessingStatus(null);
      return;
    }

    const pollInterval = setInterval(async () => {
      try {
        const status = await getProcessingStatus();
        setProcessingStatus(status);
        
        // Stop polling if completed, error, or idle
        if (status.status === 'completed' || status.status === 'error' || status.stage === 'completed' || status.stage === 'idle') {
          setProcessing(false);
          clearInterval(pollInterval);
          
          // Show completion message
          if (status.stage === 'completed') {
            setMessage({ type: 'success', text: 'Video processed successfully! Ready for queries.' });
          }
        }
      } catch (err) {
        console.error('Failed to get processing status:', err);
      }
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(pollInterval);
  }, [processing]);

  const loadVideos = async () => {
    try {
      const videoList = await listVideos();
      setVideos(videoList);
    } catch (err) {
      console.error('Failed to load videos:', err);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setMessage(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setMessage({ type: 'error', text: 'Please select a file first' });
      return;
    }

    setUploading(true);
    setMessage(null);

    try {
      const result = await uploadVideo(selectedFile);
      setMessage({ type: 'success', text: `Uploaded: ${result.filename}` });
      setSelectedFile(null);
      loadVideos();
    } catch (err) {
      setMessage({ type: 'error', text: `Upload failed: ${err}` });
    } finally {
      setUploading(false);
    }
  };

  const handleProcess = async (videoPath: string) => {
    if (!confirm('Process this video? This may take several minutes.')) {
      return;
    }

    setProcessing(true);
    setMessage(null);
    
    // Set initial processing status immediately
    setProcessingStatus({
      stage: 'initialization',
      status: 'running',
      message: 'Starting pipeline...',
      timestamp: Date.now()
    });

    try {
      const result = await processVideo(videoPath);
      setMessage({ type: 'success', text: 'Video processed successfully! Ready for queries.' });
      console.log('Processing result:', result);
    } catch (err) {
      setMessage({ type: 'error', text: `Processing failed: ${err}` });
      setProcessing(false);
      setProcessingStatus(null);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  };

  const getStageDisplay = (stage: string): string => {
    const stageMap: Record<string, string> = {
      'initialization': 'Initializing...',
      'audio_extraction': 'Extracting Audio',
      'transcription': 'Transcribing Audio',
      'frame_extraction': 'Extracting Frames & Detecting Slides',
      'enrichment': 'Enriching Slides (OCR + Vision)',
      'chunking': 'Creating Chunks',
      'completed': 'Completed!',
      'error': 'Error',
      'idle': 'Idle'
    };
    return stageMap[stage] || stage;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Admin Dashboard</h1>
            <p className="text-slate-600 mt-1">Upload and process course videos</p>
          </div>
          <Link
            href="/"
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            View Course →
          </Link>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {/* Upload Section */}
        <section className="bg-white rounded-xl shadow-md p-6 border border-slate-200">
          <div className="flex items-center gap-3 mb-6">
            <Upload className="text-blue-600" size={28} />
            <h2 className="text-2xl font-semibold text-slate-800">Upload New Video</h2>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Select video file (mp4, avi, mov, mkv, webm)
              </label>
              <input
                type="file"
                accept="video/*"
                onChange={handleFileSelect}
                className="block w-full text-sm text-slate-600 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer"
              />
              {selectedFile && (
                <p className="mt-2 text-sm text-slate-600">
                  Selected: <span className="font-medium">{selectedFile.name}</span> ({formatFileSize(selectedFile.size)})
                </p>
              )}
            </div>

            <button
              onClick={handleUpload}
              disabled={!selectedFile || uploading}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
            >
              {uploading ? (
                <>
                  <Loader className="animate-spin" size={20} />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload size={20} />
                  Upload Video
                </>
              )}
            </button>
          </div>

          {/* Message */}
          {message && (
            <div
              className={`mt-4 p-4 rounded-lg flex items-center gap-3 ${
                message.type === 'success' ? 'bg-green-50 text-green-800' : 'bg-red-50 text-red-800'
              }`}
            >
              {message.type === 'success' ? <CheckCircle size={20} /> : <AlertCircle size={20} />}
              <span>{message.text}</span>
            </div>
          )}

          {/* Processing Progress */}
          {(processing || processingStatus?.status === 'running') && (
            <div className="mt-4 p-4 rounded-lg bg-blue-50 border border-blue-200">
              <div className="flex items-center gap-3 mb-3">
                <Loader className="animate-spin text-blue-600" size={20} />
                <span className="font-semibold text-blue-900">Processing Video</span>
              </div>
              {processingStatus ? (
                <div className="space-y-2">
                  <div className="text-sm text-blue-800">
                    <span className="font-medium">Stage:</span> {getStageDisplay(processingStatus.stage)}
                  </div>
                  {processingStatus.message && (
                    <div className="text-sm text-blue-700">{processingStatus.message}</div>
                  )}
                  <div className="w-full bg-blue-200 rounded-full h-2 mt-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{
                        width: processingStatus.stage === 'completed' ? '100%' :
                               processingStatus.stage === 'enrichment' ? '80%' :
                               processingStatus.stage === 'chunking' ? '90%' :
                               processingStatus.stage === 'transcription' ? '30%' :
                               processingStatus.stage === 'frame_extraction' ? '50%' :
                               processingStatus.stage === 'audio_extraction' ? '20%' : '10%'
                      }}
                    />
                  </div>
                </div>
              ) : (
                <div className="text-sm text-blue-700">Starting processing...</div>
              )}
            </div>
          )}
        </section>

        {/* Video List */}
        <section className="bg-white rounded-xl shadow-md p-6 border border-slate-200">
          <div className="flex items-center gap-3 mb-6">
            <Video className="text-blue-600" size={28} />
            <h2 className="text-2xl font-semibold text-slate-800">Uploaded Videos</h2>
          </div>

          {videos.length === 0 ? (
            <div className="text-center py-12 text-slate-500">
              <FileText size={48} className="mx-auto mb-4 opacity-50" />
              <p>No videos uploaded yet</p>
            </div>
          ) : (
            <div className="space-y-3">
              {videos.map((video, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between p-4 border border-slate-200 rounded-lg hover:border-blue-300 transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <Video className="text-slate-400" size={24} />
                    <div>
                      <p className="font-medium text-slate-800">{video.filename}</p>
                      <p className="text-sm text-slate-500">{formatFileSize(video.size)}</p>
                    </div>
                  </div>
                  <button
                    onClick={() => handleProcess(video.path)}
                    disabled={processing}
                    className="px-4 py-2 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                  >
                    {processing ? (
                      <>
                        <Loader className="animate-spin" size={18} />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Clock size={18} />
                        Process Video
                      </>
                    )}
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
