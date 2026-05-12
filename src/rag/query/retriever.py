"""
Retriever — semantic search with optional ticker/date/source filters.

Wraps ChromaStore.query() with convenience filters and result formatting
for the CLI and context builder.

Usage:
    from src.rag.query.retriever import Retriever
    r = Retriever(store)
    results = r.retrieve("NVDA export control risk", ticker="NVDA", n=6)
"""

from __future__ import annotations

from typing import List, Dict, Optional

from loguru import logger

from src.rag.store.chroma_store import ChromaStore, ALL_COLLECTIONS


class Retriever:
    """
    Semantic retriever with optional pre-filtering by ticker, form type,
    source category, or year range.

    Parameters
    ----------
    store : ChromaStore instance (shared with the rest of the pipeline)
    """

    def __init__(self, store: ChromaStore):
        self.store = store

    def retrieve(
        self,
        query: str,
        n: int = 8,
        ticker: Optional[str] = None,
        form: Optional[str] = None,
        source: Optional[str] = None,        # "sec_edgar" | "research_paper" | "obsidian"
        year_from: Optional[int] = None,
        collections: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Retrieve the top-n most relevant chunks for a query.

        If a metadata filter would produce zero results (e.g. ticker not
        present in the store), the filter is dropped and a broader search
        is retried automatically.

        Returns a list of result dicts with keys:
            text, metadata, distance, collection, citation
        """
        where = self._build_where(ticker=ticker, form=form, source=source)

        # Map source label to collection scope if specified
        if source:
            from src.rag.store.chroma_store import SOURCE_TO_COLLECTION
            coll_name = SOURCE_TO_COLLECTION.get(source)
            collections = [coll_name] if coll_name else None

        results = self.store.query(
            query_text=query,
            n_results=n,
            collections=collections,
            where=where if where else None,
        )

        # Fallback: if filter returns nothing, retry without it
        if not results and where:
            logger.debug("Metadata filter returned 0 results — retrying without filter.")
            results = self.store.query(
                query_text=query,
                n_results=n,
                collections=collections,
            )

        # Annotate each result with a human-readable citation
        for r in results:
            r["citation"] = self._format_citation(r["metadata"])

        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_where(
        ticker: Optional[str],
        form: Optional[str],
        source: Optional[str],
    ) -> Dict:
        """Build a ChromaDB $and filter from optional constraints."""
        clauses = []
        if ticker:
            clauses.append({"ticker": {"$eq": ticker.upper()}})
        if form:
            clauses.append({"form": {"$eq": form.upper()}})
        if source:
            clauses.append({"source": {"$eq": source}})

        if not clauses:
            return {}
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def _format_citation(meta: Dict) -> str:
        """Return a short human-readable source citation."""
        source = meta.get("source", "unknown")

        if source == "sec_edgar":
            ticker = meta.get("ticker", "?")
            form   = meta.get("form", "?")
            date   = meta.get("filing_date", "")
            section = meta.get("section", "")
            parts = [f"{ticker} {form}"]
            if date:
                parts.append(date[:7])   # YYYY-MM
            if section:
                parts.append(f"§{section}")
            return " | ".join(parts)

        elif source == "research_paper":
            stem     = meta.get("stem", meta.get("filename", "?"))
            category = meta.get("category", "")
            year     = meta.get("year", "")
            parts = [stem]
            if year and year != "unknown":
                parts.append(year)
            if category:
                parts.append(f"[{category}]")
            return " | ".join(parts)

        elif source == "obsidian":
            note  = meta.get("note_name", "?")
            sec   = meta.get("section", "")
            parts = [f"📝 {note}"]
            if sec and sec != "root":
                parts.append(sec)
            return " | ".join(parts)

        return str(meta.get("filepath", source))
