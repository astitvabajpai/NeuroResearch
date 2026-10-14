"""
Vector Store — per-session FAISS index for RAG retrieval.

Graceful degradation:
- If faiss-cpu or langchain-huggingface are missing → silent no-op
- If running on HuggingFace Spaces → disabled (no persistent disk, memory limits)
- If embedding model fails to load → silent no-op
The embedding model is loaded once on first use (thread-safe singleton).
"""
from __future__ import annotations
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_CHUNK_SIZE    = 300
_CHUNK_OVERLAP = 60
_TOP_K         = 3

# Detect HuggingFace Spaces — RAG is disabled there due to memory/disk constraints
_ON_HF_SPACES = os.environ.get("SPACE_ID") is not None or os.environ.get("SPACES_ZERO_GPU") is not None

# Embedding singleton — loaded once, reused across all sessions
_embeddings      = None
_embeddings_lock = threading.Lock()
_embeddings_ok   = None   # None=untried, True=ok, False=failed


def _get_embeddings():
    """
    Load embedding model via langchain_huggingface.
    Returns None on HF Spaces or if dependencies are missing.
    Thread-safe singleton.
    """
    global _embeddings, _embeddings_ok

    if _ON_HF_SPACES:
        return None
    if _embeddings_ok is False:
        return None
    if _embeddings is not None:
        return _embeddings

    with _embeddings_lock:
        if _embeddings_ok is False:
            return None
        if _embeddings is not None:
            return _embeddings
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from src.config.settings import get_settings
            model = get_settings().EMBEDDING_MODEL
            logger.info("[VectorStore] Loading embedding model: %s", model)
            _embeddings = HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            _embeddings_ok = True
            logger.info("[VectorStore] Embedding model ready")
            return _embeddings
        except Exception as e:
            logger.warning("[VectorStore] Embeddings unavailable: %s — RAG disabled", e)
            _embeddings_ok = False
            return None


# ── Chunker ────────────────────────────────────────────────────────────────────

def _chunk_text(text: str, url: str) -> list[dict]:
    chunks = []
    start  = 0
    while start < len(text):
        end   = start + _CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({"text": chunk, "source": url})
        start += _CHUNK_SIZE - _CHUNK_OVERLAP
    return chunks


# ── Store registry ─────────────────────────────────────────────────────────────

_stores: dict = {}


def build_index(session_id: str, pages: list[dict]) -> bool:
    """
    Build a FAISS index for a session from {"url", "content"} dicts.
    Returns True if successful, False if RAG is unavailable.
    """
    if not pages:
        return False
    try:
        import faiss                                         # noqa: F401
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
    except ImportError:
        logger.warning("[VectorStore] faiss-cpu not installed — RAG disabled")
        return False

    embeddings = _get_embeddings()
    if embeddings is None:
        return False

    all_chunks = []
    for page in pages:
        all_chunks.extend(_chunk_text(page["content"], page["url"]))

    if not all_chunks:
        return False

    docs = [
        Document(page_content=c["text"], metadata={"source": c["source"]})
        for c in all_chunks
    ]

    try:
        index = FAISS.from_documents(docs, embeddings)
        _stores[session_id] = {"index": index}
        logger.info("[VectorStore] Index built for session %s: %d chunks from %d pages",
                    session_id, len(all_chunks), len(pages))
        return True
    except Exception as exc:
        logger.warning("[VectorStore] Index build failed: %s", exc)
        return False


def query(session_id: str, query_text: str, k: int = _TOP_K) -> list[dict]:
    """Retrieve top-k relevant chunks. Returns empty list if unavailable."""
    store = _stores.get(session_id)
    if not store:
        return []
    try:
        results = store["index"].similarity_search(query_text, k=k)
        return [
            {"text": doc.page_content, "source": doc.metadata.get("source", "")}
            for doc in results
        ]
    except Exception as exc:
        logger.warning("[VectorStore] Query failed: %s", exc)
        return []


def format_chunks_for_prompt(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "")
        text   = chunk.get("text", "").strip()
        parts.append(f"[Chunk {i}] Source: {source}\n{text}")
    return "\n\n".join(parts)


def clear_session(session_id: str) -> None:
    if session_id in _stores:
        del _stores[session_id]
        logger.debug("[VectorStore] Cleared session %s", session_id)


def is_available() -> bool:
    """Check if RAG can run in this environment."""
    if _ON_HF_SPACES:
        return False
    try:
        import faiss                                         # noqa: F401
        from langchain_community.vectorstores import FAISS  # noqa: F401
        from langchain_huggingface import HuggingFaceEmbeddings  # noqa: F401
        return True
    except ImportError:
        return False
