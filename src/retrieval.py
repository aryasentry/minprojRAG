"""
Retrieval Service

Implements Late Fusion strategy:
1. Queries separate vector stores (Transcript, Visual, PDF) independently.
2. Merges overlapping video chunks into cohesive playback segments.
3. Returns unified results with playback timestamps.
"""

import os
from typing import Any, Dict, List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import EMBEDDING_MODEL, OLLAMA_HOST, OUTPUT_DIR, logger
from .embedding import FAISS_INDEX_DIR, INDEX_PDF, INDEX_TRANSCRIPT, INDEX_VISUAL, get_embedding_model

# Constants
MERGE_WINDOW_SECONDS = 5.0  # Merge chunks if gap <= 5s
TOP_K_PER_INDEX = 5


class FusionRetriever:
    """
    Handles multi-index retrieval and late fusion merging.
    """
    def __init__(self):
        self.embeddings = get_embedding_model()
        self.stores = {}
        self._load_indices()

    def _load_indices(self):
        """Load FAISS indices if they exist."""
        for name in [INDEX_TRANSCRIPT, INDEX_VISUAL, INDEX_PDF]:
            path = os.path.join(FAISS_INDEX_DIR, name)
            if os.path.exists(path):
                try:
                    self.stores[name] = FAISS.load_local(
                        path, 
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    logger.info(f"[retrieval] Loaded index: {name}")
                except Exception as e:
                    logger.warning(f"[retrieval] Failed to load index {name}: {e}")
            else:
                logger.warning(f"[retrieval] Index not found: {name}")

    def query(self, query_text: str, top_k: int = TOP_K_PER_INDEX) -> Dict[str, Any]:
        """
        Perform search across all indices and fuse results.
        """
        results = {
            "video_segments": [],
            "pdf_segments": [],
            "raw_hits": []
        }

        # 1. Search each index
        hits = []
        for name, store in self.stores.items():
            docs = store.similarity_search_with_score(query_text, k=top_k)
            for doc, score in docs:
                hit = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                    "source_index": name
                }
                hits.append(hit)

        # 2. Separate by type
        video_hits = [h for h in hits if h["source_index"] in (INDEX_TRANSCRIPT, INDEX_VISUAL)]
        pdf_hits = [h for h in hits if h["source_index"] == INDEX_PDF]

        # 3. Fuse Video Hits (Time-based Merging)
        merged_video = self._merge_video_hits(video_hits)
        results["video_segments"] = merged_video

        # 4. Format PDF Hits
        results["pdf_segments"] = [
            {
                "text": h["content"],
                "page": h["metadata"].get("page_number"),
                "source": h["metadata"].get("source_file"),
                "score": h["score"]
            }
            for h in pdf_hits
        ]
        
        results["raw_hits"] = hits  # Debug info
        return results

    def _merge_video_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Smart Late Fusion Merge: Overlapping or close video chunks into playback segments.
        
        Algorithm:
          1. Group by video_id/source_file
          2. Sort by start_time
          3. Apply smart merging:
             - If chunks overlap or are within MERGE_WINDOW_SECONDS, extend to cover both
             - Take EARLIEST start_time and LATEST end_time (union of time ranges)
             - Combine content from all modalities (transcript + visual)
          4. This ensures we capture the full context when audio and visual results overlap
        """
        if not hits:
            return []

        # Filter out hits without timestamp
        valid_hits = [h for h in hits if h["metadata"].get("start_time") is not None]
        if not valid_hits:
            return []
        
        # Group by video
        by_video = {}
        for hit in valid_hits:
            video_id = hit["metadata"].get("video_id", "unknown")
            source_file = hit["metadata"].get("source_file", "")
            key = f"{video_id}_{source_file}"
            if key not in by_video:
                by_video[key] = []
            by_video[key].append(hit)
        
        all_segments = []
        
        for video_key, video_hits in by_video.items():
            # Sort by start time
            video_hits.sort(key=lambda x: x["metadata"]["start_time"])
            
            segments = []
            curr_segment = None
            
            for hit in video_hits:
                meta = hit["metadata"]
                start = meta["start_time"]
                end = meta["end_time"]
                
                if curr_segment is None:
                    # First segment for this video
                    curr_segment = {
                        "video_id": meta.get("video_id"),
                        "source_file": meta.get("source_file", ""),
                        "start_time": start,
                        "end_time": end,
                        "texts": [hit["content"]],
                        "score_sum": hit["score"],
                        "count": 1,
                        "sources": {hit["source_index"]},
                        "chunk_ids": [meta.get("chunk_id", "")],
                    }
                else:
                    # Check if this hit overlaps or is close to current segment
                    # Overlap: hit_start < segment_end
                    # Close: hit_start <= segment_end + MERGE_WINDOW
                    is_overlapping_or_close = start <= (curr_segment["end_time"] + MERGE_WINDOW_SECONDS)
                    
                    if is_overlapping_or_close:
                        # Merge: extend to cover both ranges (UNION)
                        curr_segment["start_time"] = min(curr_segment["start_time"], start)
                        curr_segment["end_time"] = max(curr_segment["end_time"], end)
                        curr_segment["texts"].append(hit["content"])
                        curr_segment["score_sum"] += hit["score"]
                        curr_segment["count"] += 1
                        curr_segment["sources"].add(hit["source_index"])
                        curr_segment["chunk_ids"].append(meta.get("chunk_id", ""))
                    else:
                        # No overlap - finalize current and start new
                        self._finalize_segment(curr_segment)
                        segments.append(curr_segment)
                        
                        curr_segment = {
                            "video_id": meta.get("video_id"),
                            "source_file": meta.get("source_file", ""),
                            "start_time": start,
                            "end_time": end,
                            "texts": [hit["content"]],
                            "score_sum": hit["score"],
                            "count": 1,
                            "sources": {hit["source_index"]},
                            "chunk_ids": [meta.get("chunk_id", "")],
                        }
            
            # Finalize last segment for this video
            if curr_segment:
                self._finalize_segment(curr_segment)
                segments.append(curr_segment)
            
            all_segments.extend(segments)
        
        # Sort all segments by relevance (avg_score)
        # FAISS with L2 distance: lower is better
        all_segments.sort(key=lambda x: x["avg_score"])
        
        return all_segments

    def _finalize_segment(self, segment: Dict[str, Any]):
        """Calculate final stats for a merged segment."""
        segment["avg_score"] = segment["score_sum"] / segment["count"]
        segment["text_preview"] = " ... ".join(segment["texts"])[:200]
        segment["duration"] = segment["end_time"] - segment["start_time"]
        segment["sources"] = list(segment["sources"])
