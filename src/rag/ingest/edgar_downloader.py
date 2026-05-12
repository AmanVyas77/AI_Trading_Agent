"""
EDGAR filing downloader for Phase 4 RAG pipeline.

Downloads 10-K, 10-Q, and 8-K filings for every ticker in the universe,
parses the raw HTML/text into clean strings, chunks them, and returns
chunk lists ready for embedding.

Usage (standalone):
    python -m src.rag.ingest.edgar_downloader --forms 10-K 8-K --limit 3

Usage (as module):
    from src.rag.ingest.edgar_downloader import EDGARDownloader
    downloader = EDGARDownloader()
    chunks = downloader.download_and_chunk("NVDA", forms=["10-K"], limit=2)
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

from loguru import logger

# sec-edgar-downloader is already in requirements
try:
    from sec_edgar_downloader import Downloader
except ImportError:
    raise ImportError("sec-edgar-downloader not installed. Run: pip install sec-edgar-downloader")

try:
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError("beautifulsoup4 not installed. Run: pip install beautifulsoup4")

from src.rag.ingest.chunker import SECFilingChunker, Chunk

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UNIVERSE_CSV  = PROJECT_ROOT / "data" / "universe" / "universe.csv"
CIK_MAP_PATH  = PROJECT_ROOT / "data" / "universe" / "cik_map.json"
EDGAR_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "edgar_filings"
EDGAR_RAW_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_universe() -> List[str]:
    """Return the list of tickers from universe.csv."""
    import pandas as pd
    df = pd.read_csv(UNIVERSE_CSV)
    return df["ticker"].str.upper().tolist()


def _clean_html(html: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    soup = BeautifulSoup(html, "lxml")
    # Remove script / style noise
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Main downloader class ─────────────────────────────────────────────────────

class EDGARDownloader:
    """
    Wraps sec-edgar-downloader + HTML cleaning + SEC-aware chunking.

    Parameters
    ----------
    forms   : which form types to fetch (default: 10-K and 8-K)
    limit   : max filings per ticker per form type (default: 3 → ~3 years of 10-Ks)
    email   : SEC EDGAR requires a valid email in the User-Agent header
    """

    DEFAULT_FORMS = ["10-K", "8-K"]
    RATE_LIMIT_SECS = 0.5   # SEC fair-use: max ~10 req/s; we stay conservative

    def __init__(
        self,
        forms: Optional[List[str]] = None,
        limit: int = 3,
        email: str = "amanvyas1505@gmail.com",
        chunk_size: int = 1_000,
        overlap: int = 200,
    ):
        self.forms = forms or self.DEFAULT_FORMS
        self.limit = limit
        self.email = email
        self.chunker = SECFilingChunker(chunk_size=chunk_size, overlap=overlap)
        self._dl = Downloader("QuantResearch", email, EDGAR_RAW_DIR)

    # ── Public API ────────────────────────────────────────────────────────────

    def download_and_chunk(
        self,
        ticker: str,
        forms: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Chunk]:
        """Download filings for one ticker and return a flat list of Chunk objects."""
        forms = forms or self.forms
        limit = limit or self.limit
        all_chunks: List[Chunk] = []

        for form in forms:
            try:
                chunks = self._process_form(ticker.upper(), form, limit)
                all_chunks.extend(chunks)
                logger.debug(f"{ticker} {form}: {len(chunks)} chunks")
            except Exception as exc:
                logger.warning(f"Failed {ticker} {form}: {exc}")

        return all_chunks

    def download_universe(
        self,
        forms: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, List[Chunk]]:
        """Download filings for every ticker in the universe."""
        tickers = _load_universe()
        results: Dict[str, List[Chunk]] = {}

        logger.info(f"Downloading {self.forms} for {len(tickers)} tickers (limit={self.limit} each)…")

        for i, ticker in enumerate(tickers):
            logger.info(f"[{i+1}/{len(tickers)}] {ticker}")
            results[ticker] = self.download_and_chunk(ticker, forms=forms, limit=limit)
            time.sleep(self.RATE_LIMIT_SECS)

        total_chunks = sum(len(v) for v in results.values())
        logger.success(f"Done. {total_chunks:,} total chunks from {len(results)} tickers.")
        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _process_form(self, ticker: str, form: str, limit: int) -> List[Chunk]:
        """Download (or use cached), parse, and chunk filings for one form type."""
        # sec-edgar-downloader stores files under EDGAR_RAW_DIR/sec-edgar-filings/{ticker}/{form}/
        filing_dir = EDGAR_RAW_DIR / "sec-edgar-filings" / ticker / form

        if not filing_dir.exists() or not any(filing_dir.iterdir()):
            logger.debug(f"  Downloading {form} for {ticker}…")
            try:
                self._dl.get(form, ticker, limit=limit)
            except Exception as e:
                logger.warning(f"  EDGAR download error for {ticker} {form}: {e}")
                return []
            time.sleep(self.RATE_LIMIT_SECS)

        chunks: List[Chunk] = []
        filing_folders = sorted(filing_dir.iterdir()) if filing_dir.exists() else []

        for filing_folder in filing_folders[:limit]:
            if not filing_folder.is_dir():
                continue
            text = self._parse_filing_folder(filing_folder)
            if not text:
                continue

            # Extract filing date from folder name (format: YYYYMMDD or YYYY-MM-DD)
            date_str = self._extract_date(filing_folder.name)

            metadata = {
                "source":      "sec_edgar",
                "ticker":      ticker,
                "form":        form,
                "filing_date": date_str,
                "folder":      str(filing_folder),
            }
            chunks.extend(self.chunker.chunk(text, metadata))

        return chunks

    def _parse_filing_folder(self, folder: Path) -> str:
        """Find the primary document in a filing folder and extract clean text."""
        # Prefer .htm/.html files; fall back to .txt
        candidates = (
            list(folder.glob("*.htm"))
            + list(folder.glob("*.html"))
            + list(folder.glob("*.txt"))
        )

        # Skip index files
        candidates = [f for f in candidates if "index" not in f.name.lower()]

        if not candidates:
            return ""

        # Take the largest file (usually the main filing body)
        primary = max(candidates, key=lambda f: f.stat().st_size)

        try:
            raw = primary.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"  Could not read {primary}: {e}")
            return ""

        # Clean HTML if needed
        if primary.suffix.lower() in {".htm", ".html"}:
            return _clean_html(raw)
        return re.sub(r"\s+", " ", raw).strip()

    @staticmethod
    def _extract_date(folder_name: str) -> str:
        """Try to extract a YYYY-MM-DD date from a folder name."""
        m = re.search(r"(\d{4})-?(\d{2})-?(\d{2})", folder_name)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        return folder_name


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and chunk EDGAR filings.")
    parser.add_argument("--tickers", nargs="*", help="Specific tickers (default: full universe)")
    parser.add_argument("--forms", nargs="*", default=["10-K", "8-K"])
    parser.add_argument("--limit", type=int, default=3, help="Filings per ticker per form")
    args = parser.parse_args()

    dl = EDGARDownloader(forms=args.forms, limit=args.limit)

    if args.tickers:
        for t in args.tickers:
            chunks = dl.download_and_chunk(t)
            print(f"{t}: {len(chunks)} chunks")
    else:
        results = dl.download_universe()
        for ticker, chunks in results.items():
            print(f"{ticker}: {len(chunks)} chunks")
