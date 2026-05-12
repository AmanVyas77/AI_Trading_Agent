"""
ChromaDB vector store manager for Phase 4 RAG pipeline.

Maintains three persistent collections (one per source type) so queries can
be scoped by source, ticker, or date at retrieval time.

Collections:
    sec_filings       — 10-K / 10-Q / 8-K text chunks
    research_papers   — academic / practitioner PDF chunks
    obsidian_notes    — research_wiki markdown notes

Usage:
    from src.rag.store.chroma_store import ChromaStore
    store = ChromaStore()
    store.add_chunks(chunks)
    results = store.query("NVDA export control risk", n_results=5)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional, Any

from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    raise ImportError("chromadb not installed. Run: pip install chromadb")

from src.rag.ingest.chunker import Chunk
from src.rag.embed.embedder import Embedder

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parents[3]
CHROMA_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

# ── Collection routing ────────────────────────────────────────────────────────

SOURCE_TO_COLLECTION = {
    "sec_edgar":       "sec_filings",
    "research_paper":  "research_papers",
    "obsidian":        "obsidian_notes",
}

ALL_COLLECTIONS = list(set(SOURCE_TO_COLLECTION.values()))


# ── ChromaStore ───────────────────────────────────────────────────────────────

class ChromaStore:
    """
    Manages ChromaDB persistent collections.

    Parameters
    ----------
    db_dir   : path to the ChromaDB persist directory
    embedder : shared Embedder instance (creates one if not provided)
    """

    def __init__(
        self,
        db_dir: Optional[Path] = None,
        embedder: Optional[Embedder] = None,
    ):
        self.db_dir = db_dir or CHROMA_DB_DIR
        self.embedder = embedder or Embedder()

        self._client = chromadb.PersistentClient(
            path=str(self.db_dir),
        )

        # Pre-create all collections
        self._collections: Dict[str, Any] = {}
        for name in ALL_COLLECTIONS:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )

        logger.info(f"ChromaStore ready at {self.db_dir}")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 128) -> int:
        """
        Embed and store a list of Chunk objects.
        Returns the number of new documents added (deduped by content hash).
        """
        if not chunks:
            return 0

        # Group by collection
        groups: Dict[str, List[Chunk]] = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "obsidian")
            coll_name = SOURCE_TO_COLLECTION.get(source, "obsidian_notes")
            groups.setdefault(coll_name, []).append(chunk)

        total_added = 0
        for coll_name, group in groups.items():
            added = self._add_to_collection(coll_name, group, batch_size)
            total_added += added
            logger.debug(f"  {coll_name}: +{added} documents")

        return total_added

    def _add_to_collection(
        self,
        coll_name: str,
        chunks: List[Chunk],
        batch_size: int,
    ) -> int:
        collection = self._collections[coll_name]
        added = 0

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]

            ids        = [_chunk_id(c) for c in batch]
            texts      = [c.text for c in batch]
            metadatas  = [_serialise_metadata(c.metadata) for c in batch]

            # Deduplicate: skip IDs already in the collection
            existing = set(collection.get(ids=ids)["ids"])
            new_mask = [id_ not in existing for id_ in ids]

            if not any(new_mask):
                continue

            new_ids    = [id_ for id_, keep in zip(ids, new_mask) if keep]
            new_texts  = [t   for t,  keep in zip(texts, new_mask) if keep]
            new_metas  = [m   for m,  keep in zip(metadatas, new_mask) if keep]
            embeddings = self.embedder.embed(new_texts)

            collection.add(
                ids=new_ids,
                documents=new_texts,
                embeddings=embeddings,
                metadatas=new_metas,
            )
            added += len(new_ids)

        return added

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        n_results: int = 8,
        collections: Optional[List[str]] = None,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Semantic search across one or more collections.

        Parameters
        ----------
        query_text  : natural language question
        n_results   : top-k results per collection
        collections : list of collection names to search (default: all)
        where       : ChromaDB metadata filter dict
                      e.g. {"ticker": "NVDA"} or {"form": "10-K"}

        Returns
        -------
        List of result dicts sorted by distance (closest first), each with
        keys: text, metadata, distance, collection.
        """
        colls = collections or ALL_COLLECTIONS
        query_vec = self.embedder.embed_query(query_text)
        results: List[Dict] = []

        for coll_name in colls:
            if coll_name not in self._collections:
                continue
            collection = self._collections[coll_name]
            count = collection.count()
            if count == 0:
                continue

            try:
                res = collection.query(
                    query_embeddings=[query_vec],
                    n_results=min(n_results, count),
                    where=where,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception as exc:
                logger.warning(f"Query failed on {coll_name}: {exc}")
                continue

            for doc, meta, dist in zip(
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
            ):
                results.append({
                    "text":       doc,
                    "metadata":   meta,
                    "distance":   dist,
                    "collection": coll_name,
                })

        # Sort by distance (cosine distance: lower = more similar)
        results.sort(key=lambda x: x["distance"])
        return results[:n_results]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        """Return document counts per collection."""
        return {name: coll.count() for name, coll in self._collections.items()}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_id(chunk: Chunk) -> str:
    """Deterministic content-based ID so re-ingestion is idempotent."""
    content = chunk.text + json.dumps(chunk.metadata, sort_keys=True)
    return hashlib.sha1(content.encode()).hexdigest()


def _serialise_metadata(meta: Dict) -> Dict:
    """ChromaDB requires all metadata values to be str/int/float/bool."""
    clean = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, list):
            clean[k] = json.dumps(v)
        else:
            clean[k] = str(v)
    return clean
