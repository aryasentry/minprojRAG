"""
Video Extraction Utility

Extracts/crops video segments based on start and end timestamps.
Used for serving cropped video clips when RAG returns time-based segments.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional
from .config import logger


class VideoExtractor:
    """
    Handles video segment extraction using ffmpeg.
    """
    
    def __init__(self, output_dir: str = "output/video_segments"):
        """
        Initialize the video extractor.
        
        Args:
            output_dir: Directory to store extracted video segments
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """Check if ffmpeg is available."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("ffmpeg not found. Please install ffmpeg to use video extraction.")
            raise RuntimeError("ffmpeg not available")
    
    def extract_segment(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Extract a video segment from start_time to end_time.
        
        Args:
            video_path: Path to source video
            start_time: Start time in seconds
            end_time: End time in seconds
            output_filename: Optional custom filename (default: auto-generated)
            
        Returns:
            Path to extracted segment, or None if failed
        """
        if not os.path.exists(video_path):
            logger.error(f"Source video not found: {video_path}")
            return None
        
        duration = end_time - start_time
        if duration <= 0:
            logger.error(f"Invalid time range: {start_time} to {end_time}")
            return None
        
        # Generate output filename
        if output_filename is None:
            video_name = Path(video_path).stem
            output_filename = f"{video_name}_segment_{start_time:.1f}-{end_time:.1f}.mp4"
        
        output_path = self.output_dir / output_filename
        
        # Check if already extracted
        if output_path.exists():
            logger.info(f"Segment already exists: {output_path}")
            return str(output_path)
        
        logger.info(f"Extracting segment: {start_time}s - {end_time}s from {video_path}")
        
        # ffmpeg command for precise extraction
        # -ss: start time
        # -t: duration
        # -i: input file
        # -c copy: copy codec (fast, no re-encoding)
        # -avoid_negative_ts make_zero: handle timestamp issues
        cmd = [
            "ffmpeg",
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", video_path,
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            "-y",  # Overwrite output
            str(output_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0 and output_path.exists():
                logger.info(f"✅ Segment extracted successfully: {output_path}")
                return str(output_path)
            else:
                logger.error(f"ffmpeg failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("Video extraction timed out (60s)")
            return None
        except Exception as e:
            logger.error(f"Error extracting video segment: {e}")
            return None
    
    def extract_segment_hq(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Extract a video segment with re-encoding for better browser compatibility.
        Slower but ensures consistent playback.
        
        Args:
            video_path: Path to source video
            start_time: Start time in seconds
            end_time: End time in seconds
            output_filename: Optional custom filename
            
        Returns:
            Path to extracted segment, or None if failed
        """
        if not os.path.exists(video_path):
            logger.error(f"Source video not found: {video_path}")
            return None
        
        duration = end_time - start_time
        if duration <= 0:
            logger.error(f"Invalid time range: {start_time} to {end_time}")
            return None
        
        # Generate output filename
        if output_filename is None:
            video_name = Path(video_path).stem
            output_filename = f"{video_name}_segment_{start_time:.1f}-{end_time:.1f}_hq.mp4"
        
        output_path = self.output_dir / output_filename
        
        # Check if already exists
        if output_path.exists():
            logger.info(f"HQ segment already exists: {output_path}")
            return str(output_path)
        
        logger.info(f"Extracting HQ segment: {start_time}s - {end_time}s")
        
        # Re-encode with H.264 for better compatibility
        cmd = [
            "ffmpeg",
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", video_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",  # Optimize for web streaming
            "-y",
            str(output_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0 and output_path.exists():
                logger.info(f"✅ HQ segment extracted: {output_path}")
                return str(output_path)
            else:
                logger.error(f"ffmpeg failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("HQ extraction timed out (120s)")
            return None
        except Exception as e:
            logger.error(f"Error extracting HQ segment: {e}")
            return None
    
    def get_segment_url(self, segment_path: str, base_url: str = "/files") -> str:
        """
        Generate a URL for accessing an extracted segment.
        
        Args:
            segment_path: Path to segment file
            base_url: Base URL for file serving
            
        Returns:
            Full URL to segment
        """
        # Convert absolute path to relative from project root
        segment_path = Path(segment_path)
        return f"{base_url}/{segment_path.name}"


# Global instance
_extractor = None


def get_video_extractor() -> VideoExtractor:
    """Get or create the global video extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = VideoExtractor()
    return _extractor
