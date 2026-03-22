'use client';

import { useState, useRef, useEffect } from 'react';
import Link from 'next/link';
import { 
  Play, 
  Pause, 
  MessageSquare, 
  Search, 
  BookOpen, 
  X, 
  Clock, 
  FileText,
  Send,
  User,
  Bot,
  Settings
} from 'lucide-react';
import { queryRAG, listVideos, type VideoSegment, type QueryResponse, type VideoInfo } from '@/lib/api';

export default function CourseDashboard() {
  // State
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<{ role: 'user' | 'assistant'; content: string; segments?: VideoSegment[] }[]>([
    { role: 'assistant', content: "Hi! I'm your AI course assistant. Ask me anything about the lecture!" }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [showPopup, setShowPopup] = useState(false);
  const [popupSegment, setPopupSegment] = useState<VideoSegment | null>(null);
  const [videos, setVideos] = useState<VideoInfo[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<VideoInfo | null>(null);

  // Refs
  const mainVideoRef = useRef<HTMLVideoElement>(null);
  const popupVideoRef = useRef<HTMLVideoElement>(null);

  // Load videos on mount
  useEffect(() => {
    loadVideos();
  }, []);

  const loadVideos = async () => {
    try {
      const videoList = await listVideos();
      setVideos(videoList);
      if (videoList.length > 0 && !selectedVideo) {
        setSelectedVideo(videoList[0]); // Auto-select first video
      }
    } catch (err) {
      console.error('Failed to load videos:', err);
    }
  };

  // Get current video source
  const VIDEO_SRC = selectedVideo 
    ? `http://localhost:8000/files/input/${selectedVideo.filename}` 
    : "";

  // Handle Query Submit
  const handleQuery = async () => {
    if (!query.trim()) return;

    const userMsg = query;
    setQuery('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setIsLoading(true);

    try {
      const data = await queryRAG(userMsg, 10);  // Increased from 5 to 10 for better retrieval
      
      setMessages(prev => [
        ...prev, 
        { 
          role: 'assistant', 
          content: data.answer,
          segments: data.video_segments 
        }
      ]);
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { role: 'assistant', content: "Sorry, I couldn't fetch the answer. Is the backend running?" }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Open Popup Player
  const playSegment = (segment: VideoSegment) => {
    setPopupSegment(segment);
    setShowPopup(true);
  };

  // Sync popup player time
  useEffect(() => {
    if (showPopup && popupVideoRef.current && popupSegment) {
      const video = popupVideoRef.current;
      video.currentTime = popupSegment.start_time;
      video.play();

      const handleTimeUpdate = () => {
        if (video.currentTime >= popupSegment.end_time) {
            video.pause();
        }
      };

      video.addEventListener('timeupdate', handleTimeUpdate);
      return () => video.removeEventListener('timeupdate', handleTimeUpdate);
    }
  }, [showPopup, popupSegment]);


  return (
    <div className="flex h-screen bg-gray-50 text-gray-900 font-sans overflow-hidden">
      
      {/* SIDEBAR */}
      <aside className="w-64 bg-white border-r border-gray-200 flex-col hidden md:flex">
        <div className="p-4 border-b border-gray-100">
          <h1 className="text-xl font-bold text-blue-600 flex items-center gap-2">
            <BookOpen className="w-6 h-6" />
            LearnAI
          </h1>
        </div>
        
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          <div className="px-3 py-2 text-xs font-semibold text-gray-400 uppercase tracking-wider">
            Lecture Content
          </div>
          
          {/* Video List */}
          {videos.length === 0 ? (
            <div className="px-3 py-4 text-sm text-gray-500 text-center">
              No videos uploaded yet
            </div>
          ) : (
            videos.map((video, i) => {
              const isSelected = selectedVideo?.filename === video.filename;
              const videoTitle = video.filename.replace(/\.(mp4|avi|mov|mkv|webm)$/i, '');
              
              return (
                <button 
                  key={i} 
                  onClick={() => setSelectedVideo(video)}
                  className={`w-full text-left px-3 py-2 rounded-md text-sm flex items-center gap-2 transition-colors ${
                    isSelected 
                      ? 'bg-blue-50 text-blue-600 font-medium' 
                      : 'text-gray-600 hover:bg-gray-50'
                  }`}
                >
                  <Play className={`w-4 h-4 ${isSelected ? 'text-blue-600' : 'text-gray-400'}`} />
                  <span className="truncate">{videoTitle}</span>
                </button>
              );
            })
          )}
        </div>

        <div className="p-4 border-t border-gray-100">
          <Link 
            href="/admin"
            className="w-full flex items-center gap-2 px-3 py-2 bg-gray-50 hover:bg-gray-100 text-gray-700 rounded-lg transition-colors text-sm"
          >
            <Settings className="w-4 h-4" />
            Admin Panel
          </Link>
        </div>
      </aside>

      {/* MAIN CONTENT Area */}
      <main className="flex-1 flex flex-col relative">
        <header className="h-16 bg-white border-b border-gray-200 flex items-center px-6 justify-between shadow-sm z-10">
          <h2 className="text-lg font-semibold text-gray-800">
            {selectedVideo ? selectedVideo.filename.replace(/\.(mp4|avi|mov|mkv|webm)$/i, '') : 'Select a video'}
          </h2>
          <div className="flex items-center gap-4">
             <span className="text-sm text-gray-500">{videos.length} video{videos.length !== 1 ? 's' : ''} available</span>
          </div>
        </header>

        <div className="flex-1 p-6 overflow-y-auto bg-gray-50 flex justify-center">
            <div className="max-w-4xl w-full bg-white rounded-xl shadow-lg overflow-hidden border border-gray-200 flex flex-col">
                {/* Main Video Player */}
                {selectedVideo ? (
                  <>
                    <div className="relative aspect-video bg-black">
                        <video 
                            key={VIDEO_SRC}
                            ref={mainVideoRef}
                            src={VIDEO_SRC}
                            className="w-full h-full object-contain"
                            controls
                        />
                    </div>
                    <div className="p-6">
                        <h3 className="text-2xl font-bold mb-2">{selectedVideo.filename.replace(/\.(mp4|avi|mov|mkv|webm)$/i, '')}</h3>
                        <p className="text-gray-600 leading-relaxed">
                            Watch this lecture and ask questions using the AI assistant on the right. 
                            The assistant can help you understand key concepts and point you to relevant sections.
                        </p>
                        <div className="mt-4 flex items-center gap-4 text-sm text-gray-500">
                          <span>Size: {(selectedVideo.size / (1024 * 1024)).toFixed(1)} MB</span>
                          <span>•</span>
                          <span>Format: {selectedVideo.filename.split('.').pop()?.toUpperCase()}</span>
                        </div>
                    </div>
                  </>
                ) : (
                  <div className="aspect-video bg-gray-100 flex items-center justify-center">
                    <div className="text-center p-8">
                      <BookOpen className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                      <p className="text-gray-500">No video selected</p>
                      <p className="text-sm text-gray-400 mt-2">Upload videos in the Admin Panel</p>
                    </div>
                  </div>
                )}
            </div>
        </div>
      </main>

      {/* CHATBOT PANEL */}
      <aside className="w-96 bg-white border-l border-gray-200 flex flex-col shadow-xl z-20">
        <div className="p-4 border-b border-gray-100 flex items-center justify-between bg-gradient-to-r from-blue-50 to-white">
            <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-600">
                    <Bot className="w-5 h-5" />
                </div>
                <div>
                    <h3 className="font-semibold text-sm">Course Coach</h3>
                    <p className="text-xs text-green-600 flex items-center gap-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-green-500"></span>
                        Online
                    </p>
                </div>
            </div>
            <button className="text-gray-400 hover:text-gray-600">
                <X className="w-4 h-4" />
            </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50/50">
            {messages.map((msg, idx) => (
                <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[85%] rounded-2xl px-4 py-3 shadow-sm ${
                        msg.role === 'user' 
                          ? 'bg-blue-600 text-white rounded-br-none' 
                          : 'bg-white border border-gray-100 text-gray-800 rounded-bl-none'
                    }`}>
                        <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                        
                        {/* Video Suggestions */}
                        {msg.segments && msg.segments.length > 0 && (
                            <div className="mt-3 space-y-2">
                                <p className="text-xs font-semibold opacity-70 mb-1">Relevant Video Clips:</p>
                                {msg.segments.slice(0, 3).map((seg, sIdx) => (
                                    <button 
                                        key={sIdx}
                                        onClick={() => playSegment(seg)}
                                        className="w-full text-left bg-blue-50 hover:bg-blue-100 border border-blue-100 rounded-lg p-2 transition-colors group"
                                    >
                                        <div className="flex items-center justify-between mb-1">
                                            <div className="flex items-center gap-1 text-xs font-semibold text-blue-700">
                                                <Play className="w-3 h-3 fill-current" />
                                                Video Clip
                                            </div>
                                            <span className="text-[10px] text-blue-500 bg-white px-1.5 py-0.5 rounded border border-blue-100">
                                                {seg.start_time.toFixed(0)}s - {seg.end_time.toFixed(0)}s
                                            </span>
                                        </div>
                                        <p className="text-xs text-gray-600 line-clamp-2 leading-snug group-hover:text-gray-900">
                                            {seg.text_preview}
                                        </p>
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            ))}
            {isLoading && (
                 <div className="flex justify-start">
                    <div className="bg-white border border-gray-100 rounded-2xl px-4 py-3 shadow-sm rounded-bl-none flex items-center gap-2">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200"></div>
                    </div>
                 </div>
            )}
        </div>

        <div className="p-4 bg-white border-t border-gray-100">
            <div className="relative">
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleQuery()}
                    placeholder="Ask about the lecture..."
                    className="w-full pl-4 pr-12 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 text-sm transition-all"
                />
                <button 
                    onClick={handleQuery}
                    disabled={!query.trim() || isLoading}
                    className="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                    <Send className="w-4 h-4" />
                </button>
            </div>
            <div className="mt-2 flex gap-2 overflow-x-auto pb-1">
                {['Explain load balancing', 'What is caching?', 'Define sharding'].map(suggestion => (
                    <button 
                        key={suggestion}
                        onClick={() => { setQuery(suggestion); }}
                        className="whitespace-nowrap px-3 py-1.5 bg-gray-50 hover:bg-gray-100 border border-gray-200 rounded-full text-xs text-gray-600 transition-colors"
                    >
                        {suggestion}
                    </button>
                ))}
            </div>
        </div>
      </aside>

      {/* POPUP VIDEO PLAYER */}
      {showPopup && popupSegment && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl overflow-hidden animate-in fade-in zoom-in duration-200">
                <div className="p-4 border-b border-gray-100 flex items-center justify-between">
                    <h3 className="font-bold text-gray-800 flex items-center gap-2">
                        <Play className="w-4 h-4 text-blue-600 fill-current" />
                        Relevant Segment
                    </h3>
                    <button 
                        onClick={() => setShowPopup(false)}
                        className="text-gray-400 hover:text-gray-600 transition-colors"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>
                
                <div className="relative aspect-video bg-black">
                    <video 
                        ref={popupVideoRef}
                        src={VIDEO_SRC}
                        className="w-full h-full"
                        controls
                        autoPlay
                    />
                    <div className="absolute top-4 right-4 bg-black/70 text-white text-xs px-2 py-1 rounded backdrop-blur">
                        Using Late Fusion Retrieval
                    </div>
                </div>
                
                <div className="p-4 bg-gray-50">
                    <p className="text-sm text-gray-600">
                        <span className="font-semibold text-gray-900">Transcript:</span> {" "}
                        {popupSegment.text_preview}
                    </p>
                    <div className="mt-3 flex items-center gap-4 text-xs text-gray-500">
                        <span className="flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            Duration: {popupSegment.duration.toFixed(1)}s
                        </span>
                        <span className="flex items-center gap-1">
                            <FileText className="w-3 h-3" />
                            Score: {popupSegment.avg_score.toFixed(2)}
                        </span>
                        <button 
                            onClick={() => {
                                if (mainVideoRef.current) {
                                    mainVideoRef.current.currentTime = popupSegment.start_time;
                                    mainVideoRef.current.play();
                                    setShowPopup(false);
                                }
                            }}
                            className="ml-auto text-blue-600 hover:text-blue-700 font-medium"
                        >
                            Jump to in Main Video →
                        </button>
                    </div>
                </div>
            </div>
        </div>
      )}

    </div>
  );
}
