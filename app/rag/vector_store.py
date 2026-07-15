"""
app/rag/vector_store.py – Persistent ChromaDB vector store using Google embeddings.

Key differences from FAISS (used in similar projects):
  - Persistent on disk across restarts (no re-embedding needed).
  - Cosine similarity via ChromaDB built-in distance metric.
  - Deduplication via chunk_id metadata field.
  - Supports metadata filtering (e.g. is_financial=True for numeric queries).
"""

import logging
import time
from typing import List, Optional

# Free-tier Gemini embedding quota is 100 requests/min; stay under it with margin.
_EMBED_BATCH_SIZE = 90
_EMBED_BATCH_PAUSE_SECONDS = 65

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings

logger = logging.getLogger(__name__)

_vector_store: Optional[Chroma] = None


def _get_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        cfg = get_settings()
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=cfg.google_api_key,
        )
        client = chromadb.PersistentClient(
            path=cfg.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _vector_store = Chroma(
            client=client,
            collection_name=cfg.chroma_collection,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _vector_store


def is_populated() -> bool:
    """Return True if the vector store has at least one document."""
    try:
        return _get_store()._collection.count() > 0
    except Exception:
        return False


def ingest_documents(chunks: List[dict]) -> int:
    """
    Add chunks to ChromaDB.  Chunks already present (same chunk_id) are skipped.
    Returns the number of newly added chunks.
    """
    store = _get_store()

    # Collect existing IDs to skip duplicates
    try:
        existing = set(store._collection.get(include=[])["ids"])
    except Exception:
        existing = set()

    texts, metadatas, ids = [], [], []
    for chunk in chunks:
        cid = chunk["chunk_id"]
        if cid in existing:
            continue
        texts.append(chunk["text"])
        metadatas.append(
            {
                "doc_name": chunk["doc_name"],
                "quarter": chunk["quarter"],
                "source": chunk["source"],
                "is_financial": str(chunk["is_financial"]).lower(),  # Chroma requires strings
            }
        )
        ids.append(cid)

    if not texts:
        logger.info("No new chunks to ingest (all already present)")
        return 0

    total_batches = (len(texts) + _EMBED_BATCH_SIZE - 1) // _EMBED_BATCH_SIZE
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch_num = i // _EMBED_BATCH_SIZE + 1
        if batch_num > 1:
            logger.info(
                "Pausing %ds to stay under the embedding rate limit (batch %d/%d)",
                _EMBED_BATCH_PAUSE_SECONDS, batch_num, total_batches,
            )
            time.sleep(_EMBED_BATCH_PAUSE_SECONDS)
        store.add_texts(
            texts=texts[i : i + _EMBED_BATCH_SIZE],
            metadatas=metadatas[i : i + _EMBED_BATCH_SIZE],
            ids=ids[i : i + _EMBED_BATCH_SIZE],
        )
        logger.info("Ingested batch %d/%d (%d chunks)", batch_num, total_batches, min(_EMBED_BATCH_SIZE, len(texts) - i))

    logger.info("Ingested %d new chunks into ChromaDB", len(texts))
    return len(texts)


def similarity_search(
    query: str,
    k: int = 6,
    filter_financial: bool = False,
) -> List[dict]:
    """
    Semantic search over the vector store.

    Args:
        query:            Natural-language query string.
        k:                Number of results to return.
        filter_financial: If True, restrict results to chunks tagged is_financial=true.

    Returns:
        List of dicts with keys: text, doc_name, quarter, source, score.
    """
    store = _get_store()
    where = {"is_financial": "true"} if filter_financial else None

    try:
        results = store.similarity_search_with_score(query, k=k, filter=where)
    except Exception as exc:
        logger.warning("Filtered search failed (%s) – retrying without filter", exc)
        results = store.similarity_search_with_score(query, k=k)

    output = []
    for doc, score in results:
        output.append(
            {
                "text": doc.page_content,
                "doc_name": doc.metadata.get("doc_name", ""),
                "quarter": doc.metadata.get("quarter", ""),
                "source": doc.metadata.get("source", ""),
                "score": round(float(score), 4),
            }
        )
    return output
