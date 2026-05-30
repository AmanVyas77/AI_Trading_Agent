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


def _collect_filings(
    submissions: dict,
    form_types: list[str],
    date_start: str = DATE_START,
    date_end: str = DATE_END,
) -> list[dict]:
    """
    Extract filing metadata for the requested form_types from the SEC
    submissions JSON. Form-agnostic — pass form_types=["8-K"] for FinBERT or
    form_types=["10-K"] for Loughran-McDonald LM scoring.

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
        if form not in form_types:
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

        # 2. Collect 8-K filings (FILING_TYPES from settings.yaml; defaults to ["8-K"])
        filings = _collect_filings(submissions, form_types=FILING_TYPES)
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


# ─────────────────────────────────────────────────────────────────────────────
# Loughran-McDonald (LM) Sentiment — 10-K annual filings
# Paper: Loughran & McDonald (2011), "When Is a Liability Not a Liability?"
# FinBERT (8-K) and LM (10-K) are complementary: different filings, different
# methodologies, different cadences — therefore separate storage tables.
# ─────────────────────────────────────────────────────────────────────────────

# Per-request gap honouring SEC EDGAR's ~10 req/s rate limit (0.10s minimum).
LM_SEC_SLEEP = 0.11


# ── 10-K Filing Collection ───────────────────────────────────────────────────

def _create_edgar_10k_filings_table(engine) -> None:
    """Create the raw 10-K text store if it doesn't exist."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS edgar_10k_filings (
                ticker            TEXT NOT NULL,
                filing_date       TEXT NOT NULL,
                fiscal_year_end   TEXT,
                accession_number  TEXT NOT NULL,
                filing_text       TEXT,
                PRIMARY KEY (ticker, accession_number)
            )
        """))


def _upsert_10k_filing(
    engine,
    ticker: str,
    filing_date: str,
    fiscal_year_end: str,
    accession_number: str,
    filing_text: str,
) -> None:
    """Insert or replace a cleaned 10-K filing into edgar_10k_filings."""
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT OR REPLACE INTO edgar_10k_filings
                    (ticker, filing_date, fiscal_year_end, accession_number, filing_text)
                VALUES
                    (:ticker, :filing_date, :fiscal_year_end, :accession_number, :filing_text)
            """),
            {
                "ticker": ticker,
                "filing_date": filing_date,
                "fiscal_year_end": fiscal_year_end,
                "accession_number": accession_number,
                "filing_text": filing_text,
            },
        )


def run_lm_collection_pipeline(
    tickers: list[str],
    start_year: int,
    end_year: int,
    engine,
    cfg: dict,
) -> None:
    """
    Download 10-K filings from SEC EDGAR for the given tickers and year range,
    strip HTML, drop short filings, and upsert the cleaned text into the
    edgar_10k_filings table. Scoring is a separate pass — see
    run_lm_scoring_pipeline().

    Parameters
    ----------
    tickers     : list of ticker strings
    start_year  : inclusive starting calendar year (e.g. 2015)
    end_year    : inclusive ending calendar year (e.g. 2024)
    engine      : SQLAlchemy engine
    cfg         : the fundamental_factors config dict (expects an
                  "lm_sentiment" sub-block with "min_words")
    """
    lm_cfg = cfg.get("lm_sentiment", {})
    min_words = lm_cfg.get("min_words", 500)
    date_start = f"{start_year}-01-01"
    date_end = f"{end_year}-12-31"

    _create_edgar_10k_filings_table(engine)
    cik_map = _get_cik_map()

    total_collected = 0
    total_skipped = 0
    total_failed = 0

    for idx, ticker in enumerate(tickers):
        ticker_upper = ticker.upper()
        cik = cik_map.get(ticker_upper)
        if not cik:
            logger.warning(f"[{ticker_upper}] CIK not found — skipping")
            continue

        logger.info(
            f"[{idx + 1}/{len(tickers)}] {ticker_upper} (CIK {cik}) "
            f"— collecting 10-K filings {start_year}-{end_year}"
        )

        # 1. Fetch the submissions index
        try:
            submissions = _fetch_submissions(cik)
        except Exception as e:
            logger.warning(f"[{ticker_upper}] Failed to fetch submissions: {e}")
            time.sleep(LM_SEC_SLEEP)
            continue
        time.sleep(LM_SEC_SLEEP)

        # 2. Collect 10-K filing metadata for the requested window
        filings = _collect_filings(
            submissions, form_types=["10-K"],
            date_start=date_start, date_end=date_end,
        )
        if not filings:
            logger.info(
                f"[{ticker_upper}] No 10-K filings found in {start_year}-{end_year}"
            )
            continue

        logger.info(f"[{ticker_upper}] Found {len(filings)} 10-K filings")

        # 3. Download text, clean HTML, skip if too short, upsert
        for filing in filings:
            raw_text = _download_filing_text(
                cik, filing["accession_number"], filing["primary_document"]
            )
            time.sleep(LM_SEC_SLEEP)

            if not raw_text:
                logger.debug(
                    f"[{ticker_upper}] Could not download filing "
                    f"{filing['filing_date']}"
                )
                total_failed += 1
                continue

            cleaned = clean_filing_text(raw_text)
            word_count = len(cleaned.split())
            if word_count < min_words:
                logger.debug(
                    f"[{ticker_upper}] Filing {filing['filing_date']} too short "
                    f"({word_count} words < {min_words}) — skipping"
                )
                total_skipped += 1
                continue

            _upsert_10k_filing(
                engine=engine,
                ticker=ticker_upper,
                filing_date=filing["filing_date"],
                fiscal_year_end=filing["period_of_report"],
                accession_number=filing["accession_number"],
                filing_text=cleaned,
            )
            total_collected += 1
            logger.debug(
                f"[{ticker_upper}] {filing['filing_date']} stored "
                f"({word_count} words)"
            )

    logger.info(
        f"10-K collection complete: {total_collected} stored, "
        f"{total_skipped} skipped (too short), {total_failed} download failures"
    )


# ── LM Scoring (Loughran-McDonald 2011 dictionary) ───────────────────────────

def lm_sentiment(
    text: str,
    word_set_negative: set,
    word_set_positive: set,
    word_set_uncertainty: set,
    word_set_litigious: set,
) -> dict:
    """
    Apply the Loughran-McDonald master dictionary to a single piece of text.

    Tokenisation:
        re.findall(r"[a-zA-Z']+", text.upper())
    Uppercase-only matching matches the LM dictionary convention.

    Returns
    -------
    {
        "lm_net_score":      (positive_count - negative_count) / max(total_words, 1),
        "negative_count":    int,
        "positive_count":    int,
        "uncertainty_count": int,
        "litigious_count":   int,
        "total_words":       int,
    }
    """
    tokens = re.findall(r"[a-zA-Z']+", text.upper())
    total_words = len(tokens)

    negative_count = sum(1 for t in tokens if t in word_set_negative)
    positive_count = sum(1 for t in tokens if t in word_set_positive)
    uncertainty_count = sum(1 for t in tokens if t in word_set_uncertainty)
    litigious_count = sum(1 for t in tokens if t in word_set_litigious)

    lm_net_score = (positive_count - negative_count) / max(total_words, 1)

    return {
        "lm_net_score": lm_net_score,
        "negative_count": negative_count,
        "positive_count": positive_count,
        "uncertainty_count": uncertainty_count,
        "litigious_count": litigious_count,
        "total_words": total_words,
    }


def load_lm_word_sets(word_list_path: str) -> tuple[set, set, set, set]:
    """
    Load the Loughran-McDonald master dictionary CSV.

    Expected columns: 'Word', 'Negative', 'Positive', 'Uncertainty', 'Litigious'.
    A word belongs to a category when the corresponding column value > 0
    (the LM dictionary stores annual frequencies).

    Returns (negative_set, positive_set, uncertainty_set, litigious_set), each
    a set of UPPERCASE strings.  Graceful degradation:
      - If the CSV is missing, try the pysentiment2 package as a fallback.
      - If both routes fail, log a warning and return four empty sets — LM
        scoring will then produce zeros, and run_lm_scoring_pipeline will
        abort before writing rows.
    """
    path = Path(word_list_path)
    if path.exists():
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.warning(f"Failed to read LM CSV {word_list_path}: {e}")
            return set(), set(), set(), set()

        if "Word" not in df.columns:
            logger.warning(
                f"LM CSV missing required 'Word' column; got "
                f"{df.columns.tolist()[:6]}"
            )
            return set(), set(), set(), set()

        def _category_set(col: str) -> set:
            if col not in df.columns:
                logger.warning(f"LM CSV missing '{col}' column")
                return set()
            mask = pd.to_numeric(df[col], errors="coerce").fillna(0) > 0
            return set(df.loc[mask, "Word"].astype(str).str.upper())

        neg = _category_set("Negative")
        pos = _category_set("Positive")
        unc = _category_set("Uncertainty")
        lit = _category_set("Litigious")

        logger.info(
            f"Loaded LM dictionary from {word_list_path} "
            f"(neg={len(neg)}, pos={len(pos)}, unc={len(unc)}, lit={len(lit)})"
        )
        return neg, pos, unc, lit

    # File missing — try pysentiment2 fallback
    logger.warning(
        f"LM word list not found at {word_list_path}; "
        "trying pysentiment2 fallback"
    )
    try:
        import pysentiment2 as ps
        lm = ps.LM()
        # pysentiment2 stores the dictionary as a dict-of-sets keyed by category.
        neg = {w.upper() for w in lm.dict.get("Negative", [])}
        pos = {w.upper() for w in lm.dict.get("Positive", [])}
        unc = {w.upper() for w in lm.dict.get("Uncertainty", [])}
        lit = {w.upper() for w in lm.dict.get("Litigious", [])}
        logger.info(
            f"Loaded LM dictionary via pysentiment2 "
            f"(neg={len(neg)}, pos={len(pos)}, unc={len(unc)}, lit={len(lit)})"
        )
        return neg, pos, unc, lit
    except ImportError:
        logger.warning(
            "pysentiment2 not installed; LM scoring will produce zeros "
            "until either the LM CSV is placed at the expected path or "
            "pysentiment2 is installed (pip install pysentiment2)"
        )
        return set(), set(), set(), set()
    except Exception as e:
        logger.warning(f"pysentiment2 fallback failed: {e}; returning empty sets")
        return set(), set(), set(), set()


def _create_lm_sentiment_scores_table(engine) -> None:
    """Create the LM scores output table if it doesn't exist."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS lm_sentiment_scores (
                ticker             TEXT    NOT NULL,
                filing_date        TEXT    NOT NULL,
                fiscal_year_end    TEXT,
                negative_count     INTEGER,
                positive_count     INTEGER,
                uncertainty_count  INTEGER,
                litigious_count    INTEGER,
                total_words        INTEGER,
                lm_net_score       REAL,
                PRIMARY KEY (ticker, filing_date)
            )
        """))


def _upsert_lm_score(
    engine,
    ticker: str,
    filing_date: str,
    fiscal_year_end: str,
    scores: dict,
) -> None:
    """Insert or replace a single LM sentiment score row."""
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT OR REPLACE INTO lm_sentiment_scores
                    (ticker, filing_date, fiscal_year_end,
                     negative_count, positive_count, uncertainty_count,
                     litigious_count, total_words, lm_net_score)
                VALUES
                    (:ticker, :filing_date, :fiscal_year_end,
                     :negative_count, :positive_count, :uncertainty_count,
                     :litigious_count, :total_words, :lm_net_score)
            """),
            {
                "ticker": ticker,
                "filing_date": filing_date,
                "fiscal_year_end": fiscal_year_end,
                "negative_count": int(scores["negative_count"]),
                "positive_count": int(scores["positive_count"]),
                "uncertainty_count": int(scores["uncertainty_count"]),
                "litigious_count": int(scores["litigious_count"]),
                "total_words": int(scores["total_words"]),
                "lm_net_score": float(scores["lm_net_score"]),
            },
        )


def run_lm_scoring_pipeline(engine, cfg: dict) -> None:
    """
    Score every 10-K filing in edgar_10k_filings that has not already been
    written to lm_sentiment_scores, using the Loughran-McDonald master
    dictionary loaded from the path under cfg["lm_sentiment"]["word_list_path"].

    Parameters
    ----------
    engine : SQLAlchemy engine
    cfg    : the fundamental_factors config dict (expects an "lm_sentiment"
             sub-block with "word_list_path")
    """
    lm_cfg = cfg.get("lm_sentiment", {})
    word_list_path = lm_cfg.get(
        "word_list_path", "data/raw/lm_master_dictionary.csv"
    )

    # Resolve relative paths against the project root
    word_list_full = Path(word_list_path)
    if not word_list_full.is_absolute():
        word_list_full = ROOT / word_list_full

    neg, pos, unc, lit = load_lm_word_sets(str(word_list_full))
    if not (neg or pos or unc or lit):
        logger.warning(
            "All LM word sets are empty — scoring would produce zeros. "
            "Aborting before writing rows; resolve the dictionary first."
        )
        return

    _create_lm_sentiment_scores_table(engine)

    # Find 10-K filings in edgar_10k_filings that are NOT yet in lm_sentiment_scores.
    # Join on (ticker, filing_date) because that's the PK on the scores table.
    sql = """
        SELECT f.ticker, f.filing_date, f.fiscal_year_end, f.filing_text
        FROM edgar_10k_filings f
        LEFT JOIN lm_sentiment_scores s
            ON f.ticker = s.ticker AND f.filing_date = s.filing_date
        WHERE s.ticker IS NULL
        ORDER BY f.ticker, f.filing_date
    """
    with engine.connect() as conn:
        unprocessed = pd.read_sql_query(text(sql), conn)

    if unprocessed.empty:
        logger.info(
            "No unscored 10-K filings in edgar_10k_filings — nothing to do"
        )
        return

    logger.info(
        f"Scoring {len(unprocessed)} 10-K filings with LM dictionary"
    )

    total_scored = 0
    total_failed = 0
    for _, row in unprocessed.iterrows():
        try:
            scores = lm_sentiment(
                row["filing_text"] or "", neg, pos, unc, lit
            )
            _upsert_lm_score(
                engine=engine,
                ticker=row["ticker"],
                filing_date=row["filing_date"],
                fiscal_year_end=row["fiscal_year_end"],
                scores=scores,
            )
            total_scored += 1
            logger.debug(
                f"[{row['ticker']}] {row['filing_date']} "
                f"lm_net_score={scores['lm_net_score']:+.5f} "
                f"(neg={scores['negative_count']}, "
                f"pos={scores['positive_count']}, "
                f"total={scores['total_words']})"
            )
        except Exception as e:
            total_failed += 1
            logger.warning(
                f"[{row['ticker']}] {row['filing_date']} LM scoring failed: {e}"
            )

    logger.info(
        f"LM scoring complete: {total_scored} scored, {total_failed} failed"
    )


# ── LM Score Loader (for xbrl_features.build_feature_matrix integration) ────

def load_lm_scores(
    tickers: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    engine=None,
) -> pd.DataFrame:
    """
    Load LM sentiment scores aligned to quarter-end dates so they can be
    merged into xbrl_features.build_feature_matrix() the same way the
    FinBERT scores are.

    fiscal_year_end is rounded to its containing calendar-quarter-end using
        pd.to_datetime(fiscal_year_end).dt.to_period("Q").dt.end_time.dt.normalize()
    so e.g. a 2023-09-30 fiscal year-end maps to 2023-09-30 (Q3) and a
    2024-01-31 fiscal year-end maps to 2024-03-31 (Q1).

    Returns
    -------
    pd.DataFrame with columns [ticker, quarter_end, lm_sentiment_score]
    (lm_net_score from the DB is renamed to lm_sentiment_score for
    consistency with the FEATURE_COLS naming convention).

    Empty DataFrame returned (with the right columns) if the table does
    not exist yet, so callers can merge unconditionally.
    """
    if engine is None:
        engine = _get_engine()

    # Verify the table exists before querying — table may not have been
    # created yet if no LM scoring run has happened.
    try:
        with engine.connect() as conn:
            present = conn.execute(text(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='lm_sentiment_scores'"
            )).fetchall()
        if not present:
            logger.info(
                "lm_sentiment_scores table does not exist yet — "
                "returning empty frame"
            )
            return pd.DataFrame(
                columns=["ticker", "quarter_end", "lm_sentiment_score"]
            )
    except Exception as e:
        logger.warning(f"Could not check lm_sentiment_scores existence: {e}")
        return pd.DataFrame(
            columns=["ticker", "quarter_end", "lm_sentiment_score"]
        )

    where_clauses: list[str] = []
    params: dict = {}
    if tickers:
        placeholders = ",".join(f":t{i}" for i in range(len(tickers)))
        where_clauses.append(f"ticker IN ({placeholders})")
        params.update({f"t{i}": t for i, t in enumerate(tickers)})
    if start:
        where_clauses.append("filing_date >= :start")
        params["start"] = start
    if end:
        where_clauses.append("filing_date <= :end")
        params["end"] = end
    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sql = f"""
        SELECT ticker, filing_date, fiscal_year_end, lm_net_score
        FROM lm_sentiment_scores
        {where}
        ORDER BY ticker, filing_date
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, params=params)

    if df.empty:
        return pd.DataFrame(
            columns=["ticker", "quarter_end", "lm_sentiment_score"]
        )

    # Round fiscal_year_end to its containing quarter-end.
    df["quarter_end"] = (
        pd.to_datetime(df["fiscal_year_end"])
          .dt.to_period("Q").dt.end_time.dt.normalize()
    )
    df = df.rename(columns={"lm_net_score": "lm_sentiment_score"})
    return df[["ticker", "quarter_end", "lm_sentiment_score"]].reset_index(drop=True)


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
