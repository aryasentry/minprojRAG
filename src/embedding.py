"""
Embedding Service

Creates and manages separate FAISS vector stores for different modalities
to enable Late Fusion retrieval strategies.

Uses:
  - Embeddings: qwen3-embedding:0.6b (via Ollama)
  - Vector Store: FAISS
"""

import json
import os
import shutil
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from .config import EMBEDDING_MODEL, OLLAMA_HOST, OUTPUT_DIR, logger


# Paths for vector stores
FAISS_INDEX_DIR = os.path.join(OUTPUT_DIR, "faiss_indices")
INDEX_TRANSCRIPT = "video_transcript"
INDEX_VISUAL = "video_visual"
INDEX_PDF = "pdf_text"


def get_embedding_model():
    """Get the configured Ollama embedding model."""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_HOST,
    )


def create_vector_db(chunks: List[Dict[str, Any]], index_name: str) -> None:
    """
    Create and save a FAISS index for a specific list of chunks.
    """
    if not chunks:
        logger.warning(f"[embedding] No chunks for index '{index_name}', skipping.")
        return

    logger.info(f"[embedding] Creating index '{index_name}' with {len(chunks)} chunks...")

    # Convert to LangChain Documents
    documents = []
    for chunk in chunks:
        # Combined text for embedding (already prepared in chunking stage)
        page_content = chunk.get("text", "")
        
        # Metadata
        metadata = {
            "chunk_id": chunk["chunk_id"],
            "source_type": chunk["source_type"],
            "source_file": chunk.get("source_file", ""),
            "start_time": chunk.get("start_time"),
            "end_time": chunk.get("end_time"),
            "page_number": chunk.get("page_number"),
            "video_id": chunk.get("video_id"),
        }
        
        documents.append(Document(page_content=page_content, metadata=metadata))

    # Embed and Index
    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save
    save_path = os.path.join(FAISS_INDEX_DIR, index_name)
    vectorstore.save_local(save_path)
    logger.info(f"[embedding] Saved index to {save_path}")


def embed_all_chunks(chunks_path: str = os.path.join(OUTPUT_DIR, "chunks.json")) -> bool:
    """
    Load chunks.json, split by modality, and create separate vector stores.
    """
    if not os.path.exists(chunks_path):
        logger.error(f"[embedding] Chunks file not found: {chunks_path}")
        return False

    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_chunks = data.get("chunks", [])
    logger.info(f"[embedding] Loaded {len(all_chunks)} chunks total.")

    # Split by source_type
    transcript_chunks = [c for c in all_chunks if c.get("source_type") == "video_transcript"]
    visual_chunks = [c for c in all_chunks if c.get("source_type") == "video_visual"]
    pdf_chunks = [c for c in all_chunks if c.get("source_type") == "pdf_text"]

    # Clear old indices
    if os.path.exists(FAISS_INDEX_DIR):
        shutil.rmtree(FAISS_INDEX_DIR)
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

    # Create indices
    create_vector_db(transcript_chunks, INDEX_TRANSCRIPT)
    create_vector_db(visual_chunks, INDEX_VISUAL)
    create_vector_db(pdf_chunks, INDEX_PDF)

    return True


def embed_new_chunks(force_rebuild: bool = False) -> bool:
    """
    Embed only new chunks using ChunkManager for tracking.
    
    Args:
        force_rebuild: If True, rebuild all indices from scratch
        
    Returns:
        True if successful
    """
    from .chunk_manager import ChunkManager
    
    manager = ChunkManager()
    stats = manager.get_stats()
    
    logger.info("="*60)
    logger.info("INCREMENTAL EMBEDDING")
    logger.info("="*60)
    logger.info(f"Audio chunks: {stats['audio_chunks']['total']} total, {stats['audio_chunks']['unembedded']} new")
    logger.info(f"Video chunks: {stats['video_chunks']['total']} total, {stats['video_chunks']['unembedded']} new")
    
    if force_rebuild:
        logger.info("Force rebuild enabled - embedding ALL chunks")
        audio_to_embed = manager.audio_chunks["chunks"]
        video_to_embed = manager.video_chunks["chunks"]
    else:
        # Get only unembedded chunks
        audio_to_embed, video_to_embed = manager.get_unembedded_chunks()
        
        if not audio_to_embed and not video_to_embed:
            logger.info("✅ No new chunks to embed!")
            return True
    
    logger.info(f"Embedding {len(audio_to_embed)} audio + {len(video_to_embed)} video chunks...")
    
    # Prepare FAISS directory
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    
    # If force rebuild or no existing indices, create from scratch
    if force_rebuild or not os.path.exists(os.path.join(FAISS_INDEX_DIR, INDEX_TRANSCRIPT)):
        logger.info("Creating audio transcript index from scratch...")
        create_vector_db(manager.audio_chunks["chunks"], INDEX_TRANSCRIPT)
        audio_embedded_ids = [c["chunk_id"] for c in manager.audio_chunks["chunks"]]
    else:
        # Append to existing index
        if audio_to_embed:
            logger.info(f"Appending {len(audio_to_embed)} chunks to audio index...")
            append_to_vector_db(audio_to_embed, INDEX_TRANSCRIPT)
            audio_embedded_ids = [c["chunk_id"] for c in audio_to_embed]
        else:
            audio_embedded_ids = []
    
    # Same for video
    if force_rebuild or not os.path.exists(os.path.join(FAISS_INDEX_DIR, INDEX_VISUAL)):
        logger.info("Creating video visual index from scratch...")
        create_vector_db(manager.video_chunks["chunks"], INDEX_VISUAL)
        video_embedded_ids = [c["chunk_id"] for c in manager.video_chunks["chunks"]]
    else:
        if video_to_embed:
            logger.info(f"Appending {len(video_to_embed)} chunks to video index...")
            append_to_vector_db(video_to_embed, INDEX_VISUAL)
            video_embedded_ids = [c["chunk_id"] for c in video_to_embed]
        else:
            video_embedded_ids = []
    
    # Mark as embedded
    manager.mark_chunks_embedded(audio_embedded_ids, video_embedded_ids)
    
    logger.info("✅ Incremental embedding complete!")
    return True


def append_to_vector_db(new_chunks: List[Dict[str, Any]], index_name: str) -> None:
    """
    Append new chunks to an existing FAISS index.
    """
    if not new_chunks:
        return
    
    index_path = os.path.join(FAISS_INDEX_DIR, index_name)
    
    if not os.path.exists(index_path):
        # Index doesn't exist, create it
        create_vector_db(new_chunks, index_name)
        return
    
    logger.info(f"[embedding] Appending {len(new_chunks)} chunks to '{index_name}'...")
    
    # Load existing index
    embeddings = get_embedding_model()
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Convert new chunks to documents
    documents = []
    for chunk in new_chunks:
        page_content = chunk.get("text", "")
        
        metadata = {
            "chunk_id": chunk["chunk_id"],
            "source_type": chunk["source_type"],
            "source_file": chunk.get("source_file", ""),
            "start_time": chunk.get("start_time"),
            "end_time": chunk.get("end_time"),
            "page_number": chunk.get("page_number"),
            "video_id": chunk.get("video_id"),
        }
        
        documents.append(Document(page_content=page_content, metadata=metadata))
    
    # Add to existing index
    vectorstore.add_documents(documents)
    
    # Save updated index
    vectorstore.save_local(index_path)
    logger.info(f"[embedding] Appended and saved to {index_path}")


if __name__ == "__main__":
    embed_all_chunks()
