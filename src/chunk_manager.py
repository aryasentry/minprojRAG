"""
Chunk Manager

Handles separate audio and video chunk files with tracking for incremental embedding.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

from src.config import OUTPUT_DIR, logger


class ChunkManager:
    """
    Manages audio and video chunks separately with tracking.
    """
    
    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.audio_chunks_path = self.output_dir / "audio_chunks.json"
        self.video_chunks_path = self.output_dir / "video_chunks.json"
        self.tracking_path = self.output_dir / "chunk_tracking.json"
        
        self.audio_chunks = self._load_chunks(self.audio_chunks_path)
        self.video_chunks = self._load_chunks(self.video_chunks_path)
        self.tracking = self._load_tracking()
    
    def _load_chunks(self, path: Path) -> Dict:
        """Load chunks from file or return empty structure."""
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "total_chunks": 0,
            "last_updated": None,
            "chunks": []
        }
    
    def _load_tracking(self) -> Dict:
        """Load chunk tracking info."""
        if self.tracking_path.exists():
            with open(self.tracking_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "embedded_audio_chunks": [],
            "embedded_video_chunks": [],
            "last_embedding_date": None
        }
    
    def _save_chunks(self, chunks_data: Dict, path: Path):
        """Save chunks to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {chunks_data['total_chunks']} chunks to {path}")
    
    def _save_tracking(self):
        """Save tracking info."""
        with open(self.tracking_path, "w", encoding="utf-8") as f:
            json.dump(self.tracking, f, indent=2, ensure_ascii=False)
    
    def split_combined_chunks(self, combined_chunks_path: str):
        """
        Split a combined chunks.json into separate audio and video files.
        
        Args:
            combined_chunks_path: Path to the combined chunks.json file
        """
        logger.info(f"Splitting {combined_chunks_path} into separate files...")
        
        with open(combined_chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        all_chunks = data.get("chunks", [])
        
        # Separate by source_type
        audio_chunks_list = []
        video_chunks_list = []
        
        for chunk in all_chunks:
            source_type = chunk.get("source_type", "")
            
            if source_type == "video_transcript":
                # Rename chunk_id to audio_chunk_XXX
                old_id = chunk.get("chunk_id", "")
                if old_id.startswith("audio_transcript_"):
                    chunk_id_num = old_id.replace("audio_transcript_", "")
                    chunk["chunk_id"] = f"audio_chunk_{chunk_id_num}"
                audio_chunks_list.append(chunk)
                
            elif source_type == "video_visual":
                # Rename chunk_id to video_chunk_XXX
                old_id = chunk.get("chunk_id", "")
                if old_id.startswith("video_visual_"):
                    chunk_id_num = old_id.replace("video_visual_", "")
                    chunk["chunk_id"] = f"video_chunk_{chunk_id_num}"
                video_chunks_list.append(chunk)
        
        # Calculate token counts
        audio_tokens = sum(c.get("token_count", 0) for c in audio_chunks_list)
        video_tokens = sum(c.get("token_count", 0) for c in video_chunks_list)
        
        # Create audio chunks file
        audio_data = {
            "total_chunks": len(audio_chunks_list),
            "total_tokens": audio_tokens,
            "source_type": "audio_transcript",
            "last_updated": datetime.now().isoformat(),
            "chunks": audio_chunks_list
        }
        
        # Create video chunks file
        video_data = {
            "total_chunks": len(video_chunks_list),
            "total_tokens": video_tokens,
            "source_type": "video_visual",
            "last_updated": datetime.now().isoformat(),
            "chunks": video_chunks_list
        }
        
        # Save
        self._save_chunks(audio_data, self.audio_chunks_path)
        self._save_chunks(video_data, self.video_chunks_path)
        
        # Update in-memory
        self.audio_chunks = audio_data
        self.video_chunks = video_data
        
        logger.info(f"✅ Split complete:")
        logger.info(f"   Audio chunks: {len(audio_chunks_list)}")
        logger.info(f"   Video chunks: {len(video_chunks_list)}")
        
        return audio_data, video_data
    
    def add_audio_chunks(self, new_chunks: List[Dict]):
        """Add new audio chunks."""
        current_chunks = self.audio_chunks["chunks"]
        existing_ids = {c["chunk_id"] for c in current_chunks}
        
        # Filter out duplicates
        unique_new = [c for c in new_chunks if c["chunk_id"] not in existing_ids]
        
        if unique_new:
            current_chunks.extend(unique_new)
            self.audio_chunks["total_chunks"] = len(current_chunks)
            self.audio_chunks["total_tokens"] = sum(c.get("token_count", 0) for c in current_chunks)
            self.audio_chunks["last_updated"] = datetime.now().isoformat()
            
            self._save_chunks(self.audio_chunks, self.audio_chunks_path)
            logger.info(f"Added {len(unique_new)} new audio chunks")
        else:
            logger.info("No new audio chunks to add")
        
        return len(unique_new)
    
    def add_video_chunks(self, new_chunks: List[Dict]):
        """Add new video chunks."""
        current_chunks = self.video_chunks["chunks"]
        existing_ids = {c["chunk_id"] for c in current_chunks}
        
        # Filter out duplicates
        unique_new = [c for c in new_chunks if c["chunk_id"] not in existing_ids]
        
        if unique_new:
            current_chunks.extend(unique_new)
            self.video_chunks["total_chunks"] = len(current_chunks)
            self.video_chunks["total_tokens"] = sum(c.get("token_count", 0) for c in current_chunks)
            self.video_chunks["last_updated"] = datetime.now().isoformat()
            
            self._save_chunks(self.video_chunks, self.video_chunks_path)
            logger.info(f"Added {len(unique_new)} new video chunks")
        else:
            logger.info("No new video chunks to add")
        
        return len(unique_new)
    
    def get_unembedded_chunks(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Get chunks that haven't been embedded yet.
        
        Returns:
            Tuple of (unembedded_audio_chunks, unembedded_video_chunks)
        """
        embedded_audio_ids = set(self.tracking.get("embedded_audio_chunks", []))
        embedded_video_ids = set(self.tracking.get("embedded_video_chunks", []))
        
        # Filter unembedded
        unembedded_audio = [
            c for c in self.audio_chunks["chunks"]
            if c["chunk_id"] not in embedded_audio_ids
        ]
        
        unembedded_video = [
            c for c in self.video_chunks["chunks"]
            if c["chunk_id"] not in embedded_video_ids
        ]
        
        return unembedded_audio, unembedded_video
    
    def mark_chunks_embedded(self, audio_chunk_ids: List[str], video_chunk_ids: List[str]):
        """Mark chunks as embedded."""
        current_audio = set(self.tracking.get("embedded_audio_chunks", []))
        current_video = set(self.tracking.get("embedded_video_chunks", []))
        
        current_audio.update(audio_chunk_ids)
        current_video.update(video_chunk_ids)
        
        self.tracking["embedded_audio_chunks"] = sorted(list(current_audio))
        self.tracking["embedded_video_chunks"] = sorted(list(current_video))
        self.tracking["last_embedding_date"] = datetime.now().isoformat()
        
        self._save_tracking()
        
        logger.info(f"Marked {len(audio_chunk_ids)} audio + {len(video_chunk_ids)} video chunks as embedded")
    
    def get_all_chunks(self) -> List[Dict]:
        """Get all chunks (audio + video) combined."""
        return self.audio_chunks["chunks"] + self.video_chunks["chunks"]
    
    def get_stats(self) -> Dict:
        """Get statistics about chunks."""
        embedded_audio = len(self.tracking.get("embedded_audio_chunks", []))
        embedded_video = len(self.tracking.get("embedded_video_chunks", []))
        
        return {
            "audio_chunks": {
                "total": self.audio_chunks["total_chunks"],
                "embedded": embedded_audio,
                "unembedded": self.audio_chunks["total_chunks"] - embedded_audio
            },
            "video_chunks": {
                "total": self.video_chunks["total_chunks"],
                "embedded": embedded_video,
                "unembedded": self.video_chunks["total_chunks"] - embedded_video
            },
            "last_updated": {
                "audio": self.audio_chunks.get("last_updated"),
                "video": self.video_chunks.get("last_updated")
            },
            "last_embedding": self.tracking.get("last_embedding_date")
        }


def main():
    """CLI for chunk management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chunk Manager CLI")
    parser.add_argument("action", choices=["split", "stats"], help="Action to perform")
    parser.add_argument("--input", help="Input chunks.json file for split")
    
    args = parser.parse_args()
    
    manager = ChunkManager()
    
    if args.action == "split":
        if not args.input:
            print("Error: --input required for split action")
            return
        manager.split_combined_chunks(args.input)
    
    elif args.action == "stats":
        stats = manager.get_stats()
        print("\n" + "="*50)
        print("CHUNK STATISTICS")
        print("="*50)
        print(f"\nAudio Chunks:")
        print(f"  Total: {stats['audio_chunks']['total']}")
        print(f"  Embedded: {stats['audio_chunks']['embedded']}")
        print(f"  Unembedded: {stats['audio_chunks']['unembedded']}")
        print(f"\nVideo Chunks:")
        print(f"  Total: {stats['video_chunks']['total']}")
        print(f"  Embedded: {stats['video_chunks']['embedded']}")
        print(f"  Unembedded: {stats['video_chunks']['unembedded']}")
        print(f"\nLast Updated:")
        print(f"  Audio: {stats['last_updated']['audio']}")
        print(f"  Video: {stats['last_updated']['video']}")
        print(f"  Last Embedding: {stats['last_embedding']}")
        print("="*50)


if __name__ == "__main__":
    main()
