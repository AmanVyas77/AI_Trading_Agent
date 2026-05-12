"""
Local sentence-transformers embedder for Phase 4 RAG pipeline.

Uses all-MiniLM-L6-v2 by default — 384-dimensional embeddings, fast on CPU,
good semantic quality for financial text.  Swap the model name in config/
settings.yaml (rag.embedding_model) for a larger model when needed.

Usage:
    from src.rag.embed.embedder import Embedder
    embedder = Embedder()
    vectors = embedder.embed(["NVDA gross margin compression...", "..."])
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")

import yaml

# ── Config ────────────────────────────────────────────────────────────────────

_SETTINGS_PATH = Path(__file__).resolve().parents[3] / "config" / "settings.yaml"

def _load_model_name() -> str:
    try:
        with open(_SETTINGS_PATH) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("rag", {}).get("embedding_model", "all-MiniLM-L6-v2")
    except Exception:
        return "all-MiniLM-L6-v2"


# ── Embedder ──────────────────────────────────────────────────────────────────

class Embedder:
    """
    Wraps SentenceTransformer with batching and a convenience method that
    returns plain Python lists (for ChromaDB compatibility).

    Parameters
    ----------
    model_name : HuggingFace model ID (default loaded from settings.yaml)
    batch_size : number of texts to encode per forward pass
    """

    def __init__(self, model_name: str = "", batch_size: int = 64):
        self.model_name = model_name or _load_model_name()
        self.batch_size = batch_size
        logger.info(f"Loading embedding model: {self.model_name}")
        self._model = SentenceTransformer(self.model_name)
        logger.success(f"Embedder ready — {self.model_name}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings.  Returns a list of float vectors (one per text),
        ready to pass directly to chromadb collection.add(embeddings=...).
        """
        if not texts:
            return []

        vectors: np.ndarray = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,   # cosine similarity == dot product
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        return self.embed([query])[0]

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()
