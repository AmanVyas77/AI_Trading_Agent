"""
FinBERT Sentiment Pipeline
==========================
Downloads 8-K filings (earnings releases) from SEC EDGAR, scores them with
ProsusAI/finbert, and stores numeric sentiment scores in SQLite for use as a
fundamental factor.

Workflow
--------
1. Resolve tickers → CIKs via SEC company_tickers.json
2. Download 8-K filing documents from EDGAR (2015–2024)
3. Strip HTML/boilerplate, chunk to 512 tokens
4. Score each chunk with FinBERT; aggregate to a single -1.0 → +1.0 score
5. Upsert results to `sentiment_scores` table
6. `load_sentiment()` returns daily-frequency DataFrame for downstream joins

Storage: SQLite table `sentiment_scores`
  - ticker TEXT, filing_date TEXT, period_of_report TEXT,
    finbert_score REAL, num_chunks INT, source TEXT
  - PRIMARY KEY (ticker, filing_date)

Usage:
  python -m src.strategies.fundamental.sentiment_pipeline
  python -m src.strategies.fundamental.sentiment_pipeline --tickers AAPL MSFT
  python -m src.strategies.fundamental.sentiment_pipeline --cpu-only
"""

from __future__ import annotations

import os
import re
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[3]

with open(ROOT / "config" / "settings.yaml") as f:
    CFG = yaml.safe_load(f)

SENT_CFG = CFG["fundamental_factors"]["finbert_sentiment"]
DB_PATH = ROOT / CFG["data"]["paths"]["db"]
DB_URL = f"sqlite:///{DB_PATH}"

SEC_HEADERS = {
    "User-Agent": os.getenv("SEC_USER_AGENT", "QuantResearch research@example.com"),
    "Accept-Encoding": "gzip, deflate",
}

MAX_TOKENS = SENT_CFG.get("max_tokens", 512)
FILING_TYPES = SENT_CFG.get("filing_types", ["8-K"])
MODEL_NAME = SENT_CFG.get("model", "ProsusAI/finbert")

# Date range for filing downloads
DATE_START = "2015-01-01"
DATE_END = "2024-12-31"

# SEC rate-limit: ≤10 req/s → 0.15s between calls
SEC_SLEEP = 0.15

# ── CIK Resolution ───────────────────────────────────────────────────────────

def _get_cik_map() -> dict[str, str]:
    """Load or download the full SEC ticker→CIK mapping."""
    import json

    cache_path = ROOT / "data" / "universe" / "cik_map.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    mapping = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in data.values()}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(mapping, f)

    logger.info(f"CIK map downloaded: {len(mapping)} tickers")
    return mapping


# ── SEC EDGAR 8-K Filing Retrieval ────────────────────────────────────────────

def _fetch_submissions(cik: str) -> dict:
    """Fetch the full submissions JSON for a CIK."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _collect_8k_filings(
    submissions: dict,
    date_start: str = DATE_START,
    date_end: str = DATE_END,
) -> list[dict]:
    """
    Extract 8-K filing metadata from the submissions JSON.

    Returns a list of dicts with keys:
      accession_number, filing_date, primary_document, period_of_report
    """
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    periods = recent.get("reportDate", [])

    filings = []
    for i, form in enumerate(forms):
        if form not in FILING_TYPES:
            continue

        filing_date = dates[i] if i < len(dates) else ""
        if filing_date < date_start or filing_date > date_end:
            continue

        accession = accessions[i] if i < len(accessions) else ""
        primary_doc = primary_docs[i] if i < len(primary_docs) else ""
        period = periods[i] if i < len(periods) else filing_date

        if not accession or not primary_doc:
            continue

        filings.append({
            "accession_number": accession,
            "filing_date": filing_date,
            "primary_document": primary_doc,
            "period_of_report": period,
        })

    return filings


def _download_filing_text(cik: str, accession: str, primary_doc: str) -> Optional[str]:
    """
    Download the primary document text for a single filing.

    URL format:
      https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{primary_doc}
    """
    acc_clean = accession.replace("-", "")
    cik_clean = cik.lstrip("0") or "0"
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_clean}/{acc_clean}/{primary_doc}"
    )

    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.debug(f"Failed to download {url}: {e}")
        return None


# ── Text Preprocessing ───────────────────────────────────────────────────────

# Patterns for boilerplate removal
_BOILERPLATE_PATTERNS = [
    re.compile(r"(?i)forward[- ]looking\s+statement.*?(?:\n\n|\Z)", re.DOTALL),
    re.compile(r"(?i)safe\s+harbor.*?(?:\n\n|\Z)", re.DOTALL),
    re.compile(r"(?i)this\s+press\s+release\s+contains\s+forward.*?(?:\n\n|\Z)", re.DOTALL),
    re.compile(r"(?i)(?:about|contact)\s+(?:the\s+)?company.*?(?:\n\n|\Z)", re.DOTALL),
    re.compile(r"(?i)investor\s+(?:relations|contact).*?(?:\n\n|\Z)", re.DOTALL),
    re.compile(r"(?i)###\s*$", re.MULTILINE),
    re.compile(r"(?i)press\s+release\s*$", re.MULTILINE),
]

# Common legal disclaimer markers
_LEGAL_MARKERS = [
    "This press release contains forward-looking",
    "Safe Harbor Statement",
    "Non-GAAP Financial Measures",
    "Cautionary Statement",
    "Private Securities Litigation Reform Act",
]


def clean_filing_text(raw_html: str) -> str:
    """
    Extract meaningful text from an 8-K HTML/text filing.

    Steps:
      1. Parse HTML and extract text
      2. Remove boilerplate headers and legal disclaimers
      3. Collapse whitespace
      4. Return cleaned text
    """
    # Parse HTML
    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove script, style, and header tags entirely
    for tag in soup(["script", "style", "head", "meta", "link"]):
        tag.decompose()

    # Extract text
    text_content = soup.get_text(separator="\n")

    # Remove boilerplate patterns
    for pattern in _BOILERPLATE_PATTERNS:
        text_content = pattern.sub("", text_content)

    # Remove lines that are purely legal / disclaimer markers
    lines = text_content.split("\n")
    cleaned_lines = []
    skip_rest = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue

        # Once we hit a major legal disclaimer section, skip the rest
        if any(marker.lower() in stripped.lower() for marker in _LEGAL_MARKERS):
            skip_rest = True

        if skip_rest:
            continue

        cleaned_lines.append(stripped)

    text_content = "\n".join(cleaned_lines)

    # Collapse runs of whitespace / blank lines
    text_content = re.sub(r"\n{3,}", "\n\n", text_content)
    text_content = re.sub(r"[ \t]+", " ", text_content)

    return text_content.strip()


def chunk_text(text_content: str, max_tokens: int = MAX_TOKENS) -> list[str]:
    """
    Split text into chunks of approximately `max_tokens` tokens.

    Uses a simple whitespace tokeniser (≈ word count) as a fast proxy for
    subword tokens. FinBERT's WordPiece tokeniser may expand some words,
    so we use a conservative 0.85 factor.
    """
    words = text_content.split()
    effective_max = int(max_tokens * 0.85)   # conservative limit

    if not words:
        return []

    chunks = []
    for i in range(0, len(words), effective_max):
        chunk = " ".join(words[i : i + effective_max])
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks


# ── FinBERT Model ─────────────────────────────────────────────────────────────

class FinBERTScorer:
    """
    Lazy-loading wrapper around ProsusAI/finbert for sentiment scoring.

    Scores range from -1.0 (bearish) to +1.0 (bullish):
      positive → +1.0 × confidence
      negative → -1.0 × confidence
      neutral  →  0.0
    """

    _LABEL_MAP = {
        "positive": 1.0,
        "negative": -1.0,
        "neutral": 0.0,
    }

    def __init__(self, model_name: str = MODEL_NAME, cpu_only: bool = False):
        self.model_name = model_name
        self.cpu_only = cpu_only
        self._pipeline = None

    def _load(self):
        """Lazy-load the HuggingFace pipeline on first use."""
        if self._pipeline is not None:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

        logger.info(f"Loading FinBERT model: {self.model_name}")

        # Device selection
        if self.cpu_only:
            device = -1   # CPU in HF pipeline convention
            logger.info("Forcing CPU-only inference (--cpu-only)")
        elif torch.cuda.is_available():
            device = 0
            logger.info("Using CUDA GPU for inference")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS can be flaky for some transformer ops — allow but warn
            device = "mps"
            logger.info("Using Apple MPS for inference (use --cpu-only if unstable)")
        else:
            device = -1
            logger.info("No GPU detected; using CPU for inference")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        # For MPS, move model manually; pipeline device param works for CUDA/CPU
        if device == "mps":
            model = model.to("mps")
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device="mps",
                truncation=True,
                max_length=MAX_TOKENS,
            )
        else:
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=device,
                truncation=True,
                max_length=MAX_TOKENS,
            )

        logger.info("FinBERT model loaded successfully")

    def score_text(self, text_content: str) -> float:
        """
        Score a single text string. Returns a float in [-1.0, +1.0].
        """
        self._load()

        if not text_content or not text_content.strip():
            return 0.0

        result = self._pipeline(text_content[:10000])[0]   # safety truncation
        label = result["label"].lower()
        confidence = result["score"]

        weight = self._LABEL_MAP.get(label, 0.0)
        return weight * confidence

    def score_chunks(self, chunks: list[str]) -> tuple[float, int]:
        """
        Score a list of text chunks and return (weighted_avg_score, num_chunks).

        Each chunk gets a score in [-1.0, +1.0]. The final score is the
        simple average across all chunks.
        """
        self._load()

        if not chunks:
            return 0.0, 0

        scores = []
        for chunk in chunks:
            try:
                s = self.score_text(chunk)
                scores.append(s)
            except Exception as e:
                logger.debug(f"Chunk scoring error: {e}")
                continue

        if not scores:
            return 0.0, 0

        avg_score = sum(scores) / len(scores)
        # Clamp to [-1.0, +1.0]
        avg_score = max(-1.0, min(1.0, avg_score))
        return avg_score, len(scores)


# ── Database ──────────────────────────────────────────────────────────────────

def _get_engine():
    """Create or return the SQLAlchemy engine with table creation."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(DB_URL, echo=False)
    _create_tables(engine)
    return engine


def _create_tables(engine):
    """Create the sentiment_scores table if it doesn't exist."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sentiment_scores (
                ticker           TEXT    NOT NULL,
                filing_date      TEXT    NOT NULL,
                period_of_report TEXT,
                finbert_score    REAL,
                num_chunks       INTEGER,
                source           TEXT,
                PRIMARY KEY (ticker, filing_date)
            )
        """))


def _upsert_score(
    engine,
    ticker: str,
    filing_date: str,
    period_of_report: str,
    finbert_score: float,
    num_chunks: int,
    source: str = "sec_8k",
) -> None:
    """Insert or replace a single sentiment score row."""
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT OR REPLACE INTO sentiment_scores
                    (ticker, filing_date, period_of_report,
                     finbert_score, num_chunks, source)
                VALUES
                    (:ticker, :filing_date, :period_of_report,
                     :finbert_score, :num_chunks, :source)
            """),
            {
                "ticker": ticker,
                "filing_date": filing_date,
                "period_of_report": period_of_report,
                "finbert_score": finbert_score,
                "num_chunks": num_chunks,
                "source": source,
            },
        )


# ── Load helper ───────────────────────────────────────────────────────────────

def load_sentiment(
    tickers: Optional[list[str]] = None,
    engine=None,
) -> pd.DataFrame:
    """
    Load sentiment scores from the DB and return a DataFrame indexed by
    (ticker, date) with a `finbert_score` column, forward-filled to daily
    frequency for easy joining with price data.

    Parameters
    ----------
    tickers : list of ticker strings, or None for all
    engine  : optional SQLAlchemy engine (created if None)

    Returns
    -------
    pd.DataFrame with MultiIndex (ticker, date) and column finbert_score
    """
    if engine is None:
        engine = _get_engine()

    where = ""
    params: dict = {}
    if tickers:
        placeholders = ",".join(f":t{i}" for i in range(len(tickers)))
        where = f"WHERE ticker IN ({placeholders})"
        params = {f"t{i}": t for i, t in enumerate(tickers)}

    sql = f"""
        SELECT ticker, filing_date, finbert_score
        FROM sentiment_scores
        {where}
        ORDER BY ticker, filing_date
    """

    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, params=params)

    if df.empty:
        return pd.DataFrame(
            columns=["finbert_score"],
            index=pd.MultiIndex.from_tuples([], names=["ticker", "date"]),
        )

    df["filing_date"] = pd.to_datetime(df["filing_date"])

    # Forward-fill to daily frequency per ticker
    daily_frames = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.set_index("filing_date").sort_index()
        # Reindex to daily frequency from first filing to today
        idx = pd.date_range(
            start=grp.index.min(),
            end=pd.Timestamp.today().normalize(),
            freq="D",
        )
        daily = grp[["finbert_score"]].reindex(idx).ffill()
        daily["ticker"] = ticker
        daily.index.name = "date"
        daily_frames.append(daily.reset_index())

    result = pd.concat(daily_frames, ignore_index=True)
    result = result.set_index(["ticker", "date"]).sort_index()
    return result


# ── Pipeline Orchestrator ─────────────────────────────────────────────────────

def run_sentiment_pipeline(
    tickers: Optional[list[str]] = None,
    cpu_only: bool = False,
) -> None:
    """
    Full FinBERT sentiment pipeline:
      1. Resolve tickers → CIKs
      2. For each ticker, download 8-K filings from EDGAR
      3. Clean text, chunk, score with FinBERT
      4. Upsert scores to SQLite
    """
    # Load universe if no tickers specified
    if tickers is None:
        uni_path = ROOT / "data" / "universe" / "universe.csv"
        if uni_path.exists():
            uni = pd.read_csv(uni_path)
            tickers = uni["ticker"].tolist()
            logger.info(f"Loaded {len(tickers)} tickers from universe.csv")
        else:
            logger.error(
                f"Universe file not found: {uni_path}  "
                "Run screener first or pass --tickers explicitly."
            )
            return

    engine = _get_engine()
    cik_map = _get_cik_map()
    scorer = FinBERTScorer(model_name=MODEL_NAME, cpu_only=cpu_only)

    total_scored = 0
    total_filings = 0

    for idx, ticker in enumerate(tickers):
        ticker_upper = ticker.upper()
        cik = cik_map.get(ticker_upper)
        if not cik:
            logger.warning(f"[{ticker_upper}] CIK not found — skipping")
            continue

        logger.info(
            f"[{idx + 1}/{len(tickers)}] Processing {ticker_upper} (CIK {cik})"
        )

        # 1. Fetch submissions
        try:
            submissions = _fetch_submissions(cik)
        except Exception as e:
            logger.warning(f"[{ticker_upper}] Failed to fetch submissions: {e}")
            time.sleep(SEC_SLEEP)
            continue
        time.sleep(SEC_SLEEP)

        # 2. Collect 8-K filings
        filings = _collect_8k_filings(submissions)
        if not filings:
            logger.info(f"[{ticker_upper}] No 8-K filings found in date range")
            continue

        logger.info(f"[{ticker_upper}] Found {len(filings)} 8-K filings")
        total_filings += len(filings)

        # 3. Download, clean, score each filing
        for filing in filings:
            raw_text = _download_filing_text(
                cik, filing["accession_number"], filing["primary_document"]
            )
            time.sleep(SEC_SLEEP)

            if not raw_text:
                logger.debug(
                    f"[{ticker_upper}] Could not download filing "
                    f"{filing['filing_date']}"
                )
                continue

            # Clean and chunk
            cleaned = clean_filing_text(raw_text)
            if len(cleaned) < 50:
                logger.debug(
                    f"[{ticker_upper}] Filing {filing['filing_date']} "
                    f"too short after cleaning ({len(cleaned)} chars) — skipping"
                )
                continue

            chunks = chunk_text(cleaned, max_tokens=MAX_TOKENS)
            if not chunks:
                continue

            # Score
            finbert_score, num_chunks = scorer.score_chunks(chunks)

            # Upsert
            _upsert_score(
                engine=engine,
                ticker=ticker_upper,
                filing_date=filing["filing_date"],
                period_of_report=filing["period_of_report"],
                finbert_score=round(finbert_score, 6),
                num_chunks=num_chunks,
                source="sec_8k",
            )

            total_scored += 1
            logger.debug(
                f"[{ticker_upper}] {filing['filing_date']}  "
                f"score={finbert_score:+.4f}  chunks={num_chunks}"
            )

        logger.info(
            f"[{ticker_upper}] Scoring complete — "
            f"{total_scored} filings scored so far"
        )

    logger.info(
        f"Pipeline complete: {total_scored} filings scored across "
        f"{len(tickers)} tickers ({total_filings} 8-K filings found total)"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="FinBERT Sentiment Pipeline — score 8-K filings from SEC EDGAR",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Tickers to process (default: full universe from universe.csv)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        default=False,
        help="Force CPU inference (skip MPS / CUDA detection)",
    )
    args = parser.parse_args()

    run_sentiment_pipeline(tickers=args.tickers, cpu_only=args.cpu_only)


if __name__ == "__main__":
    main()
