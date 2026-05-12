"""
Research paper PDF loader for Phase 4 RAG pipeline.

Scans the research_papers/ directory tree, extracts text from each PDF using
pdfplumber, and returns chunked Chunk objects ready for embedding.

Folder conventions (used to auto-tag the 'category' metadata field):
    research_papers/fundamentals/       → category = "fundamentals"
    research_papers/ai_finance/         → category = "ai_finance"
    research_papers/prompt_engineering/ → category = "prompt_engineering"
    research_papers/governance/         → category = "governance"

Usage:
    from src.rag.ingest.pdf_loader import PDFLoader
    loader = PDFLoader()
    chunks = loader.load_all()
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from loguru import logger

try:
    import pdfplumber
except ImportError:
    raise ImportError("pdfplumber not installed. Run: pip install pdfplumber")

from src.rag.ingest.chunker import ResearchPaperChunker, Chunk

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT       = Path(__file__).resolve().parents[3]
RESEARCH_PAPERS_DIR = PROJECT_ROOT / "research_papers"


# ── Loader ────────────────────────────────────────────────────────────────────

class PDFLoader:
    """
    Loads all PDFs under research_papers/ and returns flat list of Chunks.

    Parameters
    ----------
    papers_dir  : root folder to scan (default: research_papers/)
    chunk_size  : target words per chunk
    min_chars   : minimum extracted character count to bother chunking a page
    """

    def __init__(
        self,
        papers_dir: Optional[Path] = None,
        chunk_size: int = 800,
        min_chars: int = 200,
    ):
        self.papers_dir = papers_dir or RESEARCH_PAPERS_DIR
        self.chunker = ResearchPaperChunker(chunk_size=chunk_size)
        self.min_chars = min_chars

    # ── Public API ────────────────────────────────────────────────────────────

    def load_all(self) -> List[Chunk]:
        """Scan the papers directory and chunk every PDF found."""
        pdf_paths = sorted(self.papers_dir.rglob("*.pdf"))

        if not pdf_paths:
            logger.warning(
                f"No PDFs found under {self.papers_dir}. "
                "Drop PDF files there and re-run ingestion."
            )
            return []

        logger.info(f"Found {len(pdf_paths)} PDF(s) to process.")
        all_chunks: List[Chunk] = []

        for pdf_path in pdf_paths:
            try:
                chunks = self.load_pdf(pdf_path)
                all_chunks.extend(chunks)
                logger.debug(f"  {pdf_path.name}: {len(chunks)} chunks")
            except Exception as exc:
                logger.warning(f"  Failed to load {pdf_path.name}: {exc}")

        logger.success(f"PDFLoader: {len(all_chunks)} total chunks from {len(pdf_paths)} files.")
        return all_chunks

    def load_pdf(self, pdf_path: Path) -> List[Chunk]:
        """Extract text from a single PDF and chunk it."""
        text = self._extract_text(pdf_path)
        if len(text) < self.min_chars:
            logger.debug(f"  Skipping {pdf_path.name} — too little text ({len(text)} chars)")
            return []

        # Infer category from parent folder name
        category = pdf_path.parent.name if pdf_path.parent != self.papers_dir else "general"

        # Try to extract year from filename (e.g. "kim_2024_financial_llm.pdf")
        year_match = re.search(r"(19|20)\d{2}", pdf_path.stem)
        year = year_match.group(0) if year_match else "unknown"

        metadata = {
            "source":    "research_paper",
            "filename":  pdf_path.name,
            "stem":      pdf_path.stem,
            "category":  category,
            "year":      year,
            "filepath":  str(pdf_path),
        }

        return self.chunker.chunk(text, metadata)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_text(self, pdf_path: Path) -> str:
        """
        Use pdfplumber to extract text page by page.
        Joins pages with double newline to preserve paragraph boundaries.
        """
        pages: List[str] = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text and len(page_text.strip()) >= 50:
                    # Light cleanup: remove header/footer noise (short lines at top/bottom)
                    lines = page_text.split("\n")
                    body_lines = [l for l in lines if len(l.strip()) > 20 or l.strip() == ""]
                    pages.append("\n".join(body_lines))

        full_text = "\n\n".join(pages)

        # Normalise hyphenation from PDF line breaks (e.g. "fi-\nnancial" → "financial")
        full_text = re.sub(r"(\w)-\n(\w)", r"\1\2", full_text)

        return full_text.strip()
