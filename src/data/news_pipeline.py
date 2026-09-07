"""
Sprint 9 News Ingestion Pipeline
================================
Unified store for per-ticker news across two sources:
  - FNSPID (Zihan1004/FNSPID · Stock_news/All_external.csv) — historical bulk
  - Alpha Vantage NEWS_SENTIMENT                             — 2022-01 → today

Schema (SQLite, quant_research.db)
----------------------------------
news_articles(ticker, published_at, title, snippet, site, source,
              relevance_score, article_hash PK)
news_ingest_log(source, ticker, period, completed_at,
                requests_used, items_fetched, PK(source,ticker,period))

Feature-path ruling (Prompt 1 review, R1)
-----------------------------------------
AV rows carry per-ticker `relevance_score` in the raw table for provenance,
but the feature aggregation is symmetric — plain per-ticker article means
across BOTH sources, no relevance weighting.  Weighted variant = future work.

GOOG/GOOGL ruling (R2)
----------------------
AV is fetched for GOOG only. Aggregate-time mirroring to GOOGL happens in
Prompt 3 (feature_matrix side).  FNSPID rows arrive under whichever symbol
the source uses; not mirrored at ingest.

CLI
---
  python -m src.data.news_pipeline fnspid           # one-shot bulk stream
  python -m src.data.news_pipeline av-backfill      # daily launchd job
  python -m src.data.news_pipeline status           # coverage per ticker
"""

from __future__ import annotations

import argparse
import csv
import hashlib

# FNSPID article bodies exceed Python's 128 KiB default csv field cap
csv.field_size_limit(2**31 - 1)
import io
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

# ── paths / env ──────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "quant_research.db"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

AV_BACKFILL_STATE = LOG_DIR / "av_backfill_state.jsonl"


def _load_env() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("news_pipeline")


# ── constants ────────────────────────────────────────────────────────────────

FNSPID_URL_ALL = (
    "https://huggingface.co/datasets/Zihan1004/FNSPID/"
    "resolve/main/Stock_news/All_external.csv"
)
FNSPID_URL_NASDAQ = (
    "https://huggingface.co/datasets/Zihan1004/FNSPID/"
    "resolve/main/Stock_news/nasdaq_exteral_data.csv"
)
FNSPID_URL = FNSPID_URL_ALL  # default; overridable via CLI
FNSPID_START = "2015-01-01"
FNSPID_END = "2023-12-31"

AV_KEY_ENV = "ALPHAVANTAGE_API_KEY"
AV_URL = "https://www.alphavantage.co/query"
AV_START_YEAR = 2022
AV_SLEEP_S = 1.5           # ruling R3
AV_DAILY_BUDGET = 20       # leave ~5 of free 25 as headroom
AV_ITEM_CAP = 1000         # documented max
AV_SPLIT_THRESHOLD = 950   # feed at/above this ⇒ window truncated, must paginate

# Transient-network retry (see 2026-08-11 postmortem). 31 windows across the
# backfill were permanently marked "completed" after a single DNS / reset /
# read-timeout failure, silently losing the whole window. Retry first, and
# never log a window complete unless it actually succeeded.
AV_MAX_ATTEMPTS = 3
AV_RETRY_BACKOFF_S = 4.0

# time_from/time_to have minute granularity, so a truncated page resumes at the
# last item's minute. If a single minute ever holds ≥ AV_SPLIT_THRESHOLD items
# the cursor cannot advance on its own — step it forward by this much instead.
AV_CURSOR_MIN_STEP = timedelta(minutes=1)

# Window lifecycle states recorded in news_ingest_log.status:
#   ok      — fetched to completion, no truncation outstanding
#   empty   — API answered successfully with zero items (genuine no-coverage)
#   partial — truncated and ran out of budget mid-window; resume next run
#   error   — fetch failed / throttled; nothing durable was learned, retry
AV_DONE_STATES = ("ok", "empty")


# ── schema ───────────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS news_articles(
    ticker           TEXT NOT NULL,
    published_at     TEXT NOT NULL,   -- ISO UTC "YYYY-MM-DD HH:MM:SS"
    title            TEXT,
    snippet          TEXT,
    site             TEXT,            -- publisher / source_domain
    source           TEXT NOT NULL,   -- 'fnspid' | 'alphavantage'
    relevance_score  REAL,            -- AV only (per R1: stored, not used in features)
    article_hash     TEXT PRIMARY KEY
);
CREATE INDEX IF NOT EXISTS idx_news_ticker_date ON news_articles(ticker, published_at);
CREATE INDEX IF NOT EXISTS idx_news_source      ON news_articles(source);

CREATE TABLE IF NOT EXISTS news_ingest_log(
    source          TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    period          TEXT NOT NULL,   -- 'FNSPID_BULK' or 'YYYY-MM..YYYY-MM'
    completed_at    TEXT NOT NULL,
    requests_used   INTEGER DEFAULT 0,
    items_fetched   INTEGER DEFAULT 0,  -- rows inserted for THIS window only
    PRIMARY KEY(source, ticker, period)
);
"""

# Added 2026-08-11. Without a status the log could not distinguish "fetched
# clean" from "fetch blew up but we wrote a row anyway", which is exactly how
# the NVDA 2026 blackout stayed invisible for 5 days.
MIGRATIONS = [
    ("news_ingest_log", "status", "ALTER TABLE news_ingest_log ADD COLUMN status TEXT"),
    ("news_ingest_log", "last_error", "ALTER TABLE news_ingest_log ADD COLUMN last_error TEXT"),
]


def _migrate(con: sqlite3.Connection) -> None:
    for table, column, ddl in MIGRATIONS:
        cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            con.execute(ddl)
    con.commit()


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.executescript(DDL)
    _migrate(con)
    return con


def _hash(ticker: str, published_at: str, title: str) -> str:
    payload = f"{ticker}|{published_at}|{title or ''}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_universe() -> list[str]:
    """Candidate tickers from `prices`, minus benchmark series.

    `prices` also stores benchmarks (SPY) so the SPY-relative verdict can read
    frozen DB rows instead of calling yfinance at run time. Those rows are not
    candidates: without this filter the AV backfill would spend its daily quota
    fetching benchmark news and write benchmark rows into `news_articles`.
    """
    from src.utils.benchmarks import strip_benchmarks

    con = sqlite3.connect(DB_PATH)
    try:
        rows = [r[0] for r in con.execute(
            "SELECT DISTINCT ticker FROM prices ORDER BY ticker"
        )]
    finally:
        con.close()
    return strip_benchmarks(rows)


# ── FNSPID (Phase A) ─────────────────────────────────────────────────────────

def _fnspid_normalize_date(raw: str) -> Optional[str]:
    """'2020-06-05 06:30:54 UTC' → '2020-06-05 06:30:54' (or None on garbage)."""
    if not raw:
        return None
    s = raw.strip()
    # tolerate trailing " UTC" and other suffixes
    s = s.replace(" UTC", "").replace("+00:00", "").strip()
    # accept both space and 'T' separators
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return None


def stream_fnspid(url: str = FNSPID_URL,
                  universe: Optional[list[str]] = None,
                  start: str = FNSPID_START,
                  end: str = FNSPID_END,
                  batch: int = 5000,
                  period_label: str = "FNSPID_BULK") -> dict:
    """
    Stream a FNSPID CSV via curl → csv.DictReader → filtered SQLite writes.
    Never materializes the raw CSV on disk.

    period_label is written to news_ingest_log so repeat calls for the
    other file (nasdaq_exteral_data.csv) are idempotent independently.
    """
    universe = universe or load_universe()
    uset = {t.upper() for t in universe}
    con = _connect()

    # Check idempotency — if this specific bulk file already ingested, skip.
    existing = con.execute(
        "SELECT COUNT(*) FROM news_ingest_log "
        "WHERE source='fnspid' AND period=?",
        (period_label,),
    ).fetchone()[0]
    if existing:
        log.warning(f"{period_label} already logged — skipping. "
                    "DELETE the log row to force re-ingest.")
        con.close()
        return {"skipped": True}

    log.info(f"streaming FNSPID from {url}")
    log.info(f"filter: {len(uset)} tickers, date window {start} → {end}")

    curl = subprocess.Popen(
        ["curl", "-sSL", "--fail", "--connect-timeout", "60",
         "--retry", "3", "--retry-delay", "5", url],
        stdout=subprocess.PIPE, bufsize=8 * 1024 * 1024,
    )
    # buffered text wrapper for csv reader
    stream = io.TextIOWrapper(curl.stdout, encoding="utf-8",
                              errors="replace", newline="")

    reader = csv.DictReader(stream)
    expected = {"Date", "Article_title", "Stock_symbol", "Url", "Publisher"}
    if not expected.issubset(set(reader.fieldnames or [])):
        curl.kill()
        con.close()
        raise SystemExit(
            f"FNSPID header mismatch. got: {reader.fieldnames}  need: {expected}"
        )

    rows_seen = 0
    rows_kept = 0
    rows_dedup = 0
    by_year: dict[str, int] = {}
    by_ticker: dict[str, int] = {}
    buffer: list[tuple] = []
    start_t = time.time()

    def flush(buf):
        nonlocal rows_dedup
        if not buf:
            return
        cur = con.executemany(
            "INSERT OR IGNORE INTO news_articles "
            "(ticker, published_at, title, snippet, site, source, "
            " relevance_score, article_hash) "
            "VALUES (?, ?, ?, ?, ?, 'fnspid', NULL, ?)",
            buf,
        )
        con.commit()
        # rowcount is the number of *changes* — dedup = attempted − changes
        rows_dedup += len(buf) - cur.rowcount

    try:
        for row in reader:
            rows_seen += 1
            if rows_seen % 500_000 == 0:
                elapsed = time.time() - start_t
                log.info(f"  seen={rows_seen:,} kept={rows_kept:,} "
                         f"rate={rows_seen/elapsed:,.0f}/s")

            sym = (row.get("Stock_symbol") or "").strip().upper()
            if sym not in uset:
                continue
            dt_norm = _fnspid_normalize_date(row.get("Date", ""))
            if dt_norm is None:
                continue
            if not (start <= dt_norm[:10] <= end):
                continue

            title = (row.get("Article_title") or "").strip()
            snippet = ((row.get("Article") or "").strip()[:2000]
                       or (row.get("Lsa_summary") or "").strip()[:2000]
                       or None)
            site = (row.get("Publisher") or "").strip() or None
            h = _hash(sym, dt_norm, title)

            buffer.append((sym, dt_norm, title or None, snippet, site, h))
            rows_kept += 1
            by_year[dt_norm[:4]] = by_year.get(dt_norm[:4], 0) + 1
            by_ticker[sym] = by_ticker.get(sym, 0) + 1

            if len(buffer) >= batch:
                flush(buffer)
                buffer.clear()

        flush(buffer)

        con.execute(
            "INSERT OR REPLACE INTO news_ingest_log "
            "(source, ticker, period, completed_at, "
            " requests_used, items_fetched) "
            "VALUES ('fnspid', 'ALL', ?, ?, 0, ?)",
            (period_label,
             datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
             rows_kept),
        )
        con.commit()
    finally:
        try:
            stream.close()
        except Exception:
            pass
        curl.wait(timeout=10)
        con.close()

    elapsed = time.time() - start_t
    return {
        "rows_seen": rows_seen,
        "rows_kept": rows_kept,
        "rows_dedup": rows_dedup,
        "by_year": dict(sorted(by_year.items())),
        "by_ticker": dict(sorted(by_ticker.items())),
        "elapsed_s": round(elapsed, 1),
    }


# ── Alpha Vantage (Phase B) ──────────────────────────────────────────────────

def _av_key() -> str:
    k = os.environ.get(AV_KEY_ENV, "").strip()
    if not k or len(k) < 5:
        raise SystemExit(f"{AV_KEY_ENV} missing from environment / .env")
    return k


def _av_ticker_set() -> list[str]:
    """AV fetch list: 54 minus GOOGL (per R2)."""
    return [t for t in load_universe() if t != "GOOGL"]


def _av_year_windows(today: Optional[datetime] = None) -> list[tuple[str, str]]:
    """Yield (time_from, time_to) year-slices from 2022-01-01 through today."""
    today = today or datetime.now(timezone.utc)
    out = []
    for yr in range(AV_START_YEAR, today.year + 1):
        tf = f"{yr}0101T0000"
        # cap the last window at today (to avoid empty look-aheads)
        end_dt = datetime(yr, 12, 31, 23, 59, tzinfo=timezone.utc)
        if end_dt > today:
            end_dt = today
        tt = end_dt.strftime("%Y%m%dT%H%M")
        out.append((tf, tt))
    return out


def _av_period_key(tf: str, tt: str) -> str:
    """
    '20220101T0000','20221231T2359' → '2022-01..2022-12'.

    The key is derived from the *year* only, never from the truncated `tt` of a
    still-running year.  Before 2026-08-11 the current-year key tracked today's
    month ('2026-01..2026-07' → '2026-01..2026-08' on Aug 1), so every rollover
    minted a brand-new "pending" window and orphaned the old log row.
    """
    yr = tf[:4]
    return f"{yr}-01..{yr}-12"


def _av_transient(exc: Exception) -> bool:
    """True for the connection-level failures that wrecked the backfill."""
    import http.client
    import socket
    return isinstance(exc, (urllib.error.URLError, socket.timeout, TimeoutError,
                            http.client.IncompleteRead,
                            http.client.RemoteDisconnected,
                            ConnectionResetError, ConnectionError))


def _av_fetch(ticker: str, tf: str, tt: str, key: str,
              limit: int = AV_ITEM_CAP,
              attempts: int = AV_MAX_ATTEMPTS) -> tuple[dict, bool]:
    """
    Return (parsed_json, throttled_bool).

    Retries transient network errors with linear backoff. A retry re-issues the
    request, so callers must count every attempt against the API budget.
    """
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": tf,
        "time_to": tt,
        "sort": "EARLIEST",
        "limit": str(limit),
        "apikey": key,
    }
    url = f"{AV_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "sprint9-av"})

    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                body = r.read().decode()
            d = json.loads(body)
            throttled = ("Information" in d) or ("Note" in d)
            return d, throttled
        except Exception as e:                      # noqa: BLE001 — re-raised below
            last_exc = e
            if attempt == attempts or not _av_transient(e):
                break
            wait = AV_RETRY_BACKOFF_S * attempt
            log.warning(f"    {ticker} {tf}: transient ({type(e).__name__}: {e}) "
                        f"— retry {attempt}/{attempts - 1} in {wait:.0f}s")
            time.sleep(wait)
    raise last_exc  # type: ignore[misc]


def _av_persist(con: sqlite3.Connection, ticker: str,
                feed: list[dict]) -> tuple[int, Optional[str]]:
    """
    Insert per-item rows keyed on hash.
    Returns (rows_actually_inserted, max_time_published_seen).

    Unparseable timestamps used to `continue` in silence; they are now counted
    and logged, so a format change upstream shows up instead of looking like a
    quiet coverage gap.
    """
    rows = []
    dropped = 0
    max_tp: Optional[str] = None
    for it in feed:
        tp = it.get("time_published", "")
        # AV format: 20240215T093012 → 2024-02-15 09:30:12
        try:
            dt = datetime.strptime(tp, "%Y%m%dT%H%M%S")
            published = dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            dropped += 1
            continue
        if max_tp is None or tp > max_tp:
            max_tp = tp
        title = (it.get("title") or "").strip()
        snippet = (it.get("summary") or "").strip()[:2000] or None
        site = (it.get("source_domain") or it.get("source") or "").strip() or None
        # locate this ticker's relevance in the ticker_sentiment array
        rel = None
        for ts in it.get("ticker_sentiment", []):
            if (ts.get("ticker") or "").upper() == ticker.upper():
                try:
                    rel = float(ts.get("relevance_score"))
                except (TypeError, ValueError):
                    rel = None
                break
        h = _hash(ticker, published, title)
        rows.append((ticker, published, title or None, snippet, site,
                     "alphavantage", rel, h))
    if dropped:
        log.warning(f"    {ticker}: dropped {dropped}/{len(feed)} items with "
                    f"unparseable time_published")
    if not rows:
        return 0, max_tp
    cur = con.executemany(
        "INSERT OR IGNORE INTO news_articles "
        "(ticker, published_at, title, snippet, site, source, "
        " relevance_score, article_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.commit()
    return cur.rowcount, max_tp


def _av_resume_cursor(con: sqlite3.Connection, ticker: str,
                      tf: str, tt: str) -> str:
    """
    Where to (re)start fetching a window.

    Because the feed is sorted EARLIEST-first, whatever we already hold for this
    ticker/window is a contiguous prefix — so resume at the newest row we have
    rather than re-downloading it. This is what makes repairing a truncated
    window cost only the requests for the *missing* tail.
    """
    lo = datetime.strptime(tf, "%Y%m%dT%H%M")
    hi = datetime.strptime(tt, "%Y%m%dT%H%M")
    row = con.execute(
        "SELECT MAX(published_at) FROM news_articles "
        "WHERE ticker=? AND source='alphavantage' AND published_at BETWEEN ? AND ?",
        (ticker, lo.strftime("%Y-%m-%d %H:%M:%S"), hi.strftime("%Y-%m-%d %H:%M:%S")),
    ).fetchone()
    if not row or not row[0]:
        return tf
    have = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
    return max(lo, have).strftime("%Y%m%dT%H%M")


def _av_collect(con: sqlite3.Connection, ticker: str, tf: str, tt: str,
                key: str, budget_left: int) -> dict:
    """
    Fetch one ticker/window to completion, paginating through AV's 1000-item cap.

    The endpoint caps every call at AV_ITEM_CAP items sorted EARLIEST-first, so a
    full feed means "there is more after the last item", not "that is all of it".
    Previously the code merely logged a WARNING and moved on, which is why ~40
    windows stop dead in January. Here a full page advances time_from to the last
    item's minute and fetches again until a short page arrives.

    Returns {status, inserted, requests, pages, last_seen, error}.
    """
    cursor = _av_resume_cursor(con, ticker, tf, tt)
    if cursor > tt:
        return {"status": "ok", "inserted": 0, "requests": 0, "pages": 0,
                "last_seen": cursor, "error": None}

    inserted = requests = pages = 0
    status = "ok"
    error = None
    capped = False
    last_seen = cursor

    while requests < budget_left:
        time.sleep(AV_SLEEP_S)   # R3
        try:
            data, throttled = _av_fetch(ticker, cursor, tt, key)
        except Exception as e:                      # noqa: BLE001
            # Every attempt already burned budget inside _av_fetch.
            requests += AV_MAX_ATTEMPTS
            error = f"{type(e).__name__}: {e}"
            log.error(f"  {ticker} {cursor}..{tt}: fetch failed — {error}")
            status = "partial" if pages else "error"
            break
        requests += 1
        pages += 1

        if throttled:
            msg = str(data.get("Information") or data.get("Note") or "throttled")
            log.warning(f"  {ticker} {cursor}: throttled — {msg[:120]}")
            error = msg[:200]
            if "requests per day" in msg or "rate limit" in msg.lower():
                # Quota gone for the day — the caller must stop, whether or not
                # this particular window got some rows in first.
                capped = True
                status = "partial" if inserted else "capped"
            else:
                time.sleep(AV_SLEEP_S * 3)
                status = "partial" if inserted else "error"
            break

        feed = data.get("feed", []) or []
        n_ins, max_tp = _av_persist(con, ticker, feed)
        inserted += n_ins
        if max_tp:
            last_seen = max_tp
        log.info(f"  {ticker} {cursor}..{tt}: page={pages} items={len(feed)} "
                 f"inserted={n_ins}")

        if len(feed) < AV_SPLIT_THRESHOLD:
            # "empty" = the API genuinely has nothing for this ticker/window, as
            # opposed to a failure. Only claim it when we asked for the whole
            # window from its start and got nothing back.
            status = "empty" if (pages == 1 and not feed and cursor == tf) else "ok"
            break

        # Truncated page → resume at the newest item we just stored.
        if not max_tp:
            status = "error"
            error = "truncated page with no parseable timestamp"
            log.error(f"  {ticker} {cursor}: {error}")
            break
        nxt = datetime.strptime(max_tp, "%Y%m%dT%H%M%S").strftime("%Y%m%dT%H%M")
        if nxt <= cursor:
            # ≥950 items inside one minute — force progress rather than spin.
            nxt = (datetime.strptime(cursor, "%Y%m%dT%H%M")
                   + AV_CURSOR_MIN_STEP).strftime("%Y%m%dT%H%M")
            log.warning(f"  {ticker}: cursor stalled at {cursor}, stepping to {nxt}")
        cursor = nxt
        if cursor > tt:
            status = "ok"
            break
    else:
        # Loop exited on budget, not on a short page → tail still outstanding.
        status = "partial"

    return {"status": status, "inserted": inserted, "requests": requests,
            "pages": pages, "last_seen": last_seen, "error": error,
            "capped": capped}


def _av_log_window(con: sqlite3.Connection, ticker: str, period: str,
                   status: str, requests: int, inserted: int,
                   error: Optional[str]) -> None:
    """
    Record the outcome of ONE ticker/window.

    `items_fetched` is this window's own insert count. It used to be the run-
    cumulative `added` total, which made NVDA's dead 2026 window report 6835
    items — identical to the previous window — and hid the failure completely.
    """
    con.execute(
        "INSERT OR REPLACE INTO news_ingest_log "
        "(source, ticker, period, completed_at, requests_used, items_fetched, "
        " status, last_error) VALUES ('alphavantage', ?, ?, ?, ?, ?, ?, ?)",
        (ticker, period, _now_iso(), requests, inserted, status, error),
    )
    con.commit()


def _av_pending(con: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """
    Return (ticker, time_from, time_to) windows still needing work.

    A window is "done" only when its log row says so. Anything absent, errored,
    or partially fetched comes back around — the old version treated the mere
    *existence* of a log row as success, so a window whose fetch raised was
    marked complete forever.

    Ordering is worst-first, by how far the window's coverage lags its end.
    The daily free-tier quota (25) is smaller than the universe (53 tickers), so
    a fixed alphabetical walk would refresh AAPL every night and never reach the
    back of the alphabet. Sorting by staleness makes the queue self-rotating:
    whatever went longest without an update is served next.
    """
    state = {(t, p): s for t, p, s in con.execute(
        "SELECT ticker, period, COALESCE(status,'ok') FROM news_ingest_log "
        "WHERE source='alphavantage'"
    )}
    this_year = datetime.now(timezone.utc).year
    scored = []
    for tk in _av_ticker_set():
        for tf, tt in _av_year_windows():
            per = _av_period_key(tf, tt)
            st = state.get((tk, per))
            # The current year is never "finished" — new articles land daily.
            if st in AV_DONE_STATES and int(tf[:4]) != this_year:
                continue
            cursor = _av_resume_cursor(con, tk, tf, tt)
            if st in AV_DONE_STATES and int(tf[:4]) == this_year:
                # refresh only if we are behind the window end
                if cursor >= tt:
                    continue
            lag = (datetime.strptime(tt, "%Y%m%dT%H%M")
                   - datetime.strptime(cursor, "%Y%m%dT%H%M")).days
            # never-fetched and errored windows outrank a merely stale refresh
            rank = 0 if st not in AV_DONE_STATES else 1
            scored.append((rank, -lag, tk, tf, tt))
    scored.sort()
    return [(tk, tf, tt) for _, _, tk, tf, tt in scored]


def av_backfill(budget: int = AV_DAILY_BUDGET,
                only: Optional[list[str]] = None) -> dict:
    """
    Consume up to `budget` requests, then exit 0.  Resumable via news_ingest_log.
    If the API returns the daily-cap message, record and exit cleanly.

    `only` restricts the run to specific tickers — the free tier allows 25
    requests/day, so a large repair has to be aimed at the worst gaps rather
    than walking the universe alphabetically.
    """
    key = _av_key()
    con = _connect()
    pending = _av_pending(con)
    if only:
        want = {t.upper() for t in only}
        pending = [p for p in pending if p[0] in want]

    if not pending:
        log.info("BACKFILL COMPLETE — all tickers × years fetched clean.")
        _append_state({"run_at": _now_iso(),
                       "requests_used": 0,
                       "note": "BACKFILL COMPLETE"})
        con.close()
        return {"complete": True}

    log.info(f"pending={len(pending)} windows across "
             f"{len({p[0] for p in pending})} tickers; budget={budget}")

    used = 0
    added = 0
    cap_hit = False
    last_cursor = None
    outcomes: dict[str, int] = {}

    for ticker, tf, tt in pending:
        if used >= budget:
            break

        res = _av_collect(con, ticker, tf, tt, key, budget - used)
        used += res["requests"]
        added += res["inserted"]
        if res["pages"]:
            last_cursor = (ticker, tf, tt)

        status = res["status"]
        if status != "capped":
            # 'capped' means nothing durable was learned for this window — leave
            # it absent rather than recording a hollow "completed" row.
            outcomes[status] = outcomes.get(status, 0) + 1
            _av_log_window(con, ticker, _av_period_key(tf, tt), status,
                           res["requests"], res["inserted"], res["error"])
            log.info(f"  {ticker} {_av_period_key(tf, tt)}: status={status} "
                     f"inserted={res['inserted']} requests={res['requests']}")

        if res["capped"]:
            # Every remaining request today would just re-fetch this message.
            cap_hit = True
            break

    con.close()

    summary = {
        "run_at": _now_iso(),
        "requests_used": used,
        "items_added": added,
        "cap_hit_early": cap_hit,
        "pending_before": len(pending),
        "outcomes": outcomes,
        "last_cursor": last_cursor,
    }
    _append_state(summary)
    log.info(f"exit summary: {summary}")
    return summary


# _maybe_split() removed 2026-08-11. Its hardcoded MEGA list only ever split
# full *calendar* years into fixed halves — it never fired on the current-year
# window (tt is not 1231), and a half that still returned 1000 items was merely
# warned about. _av_collect()'s cursor pagination subsumes it: it adapts to any
# density, on any window, with no ticker list to maintain.


# ── status ───────────────────────────────────────────────────────────────────

def status() -> str:
    con = _connect()
    tickers = _av_ticker_set()
    windows = _av_year_windows()
    total_windows = len(windows)

    # av coverage per ticker
    counts: dict[str, int] = {}
    for tk, per in con.execute(
        "SELECT ticker, period FROM news_ingest_log WHERE source='alphavantage'"
    ):
        counts[tk] = counts.get(tk, 0) + 1

    # article counts per source per ticker
    art_fnspid = dict(con.execute(
        "SELECT ticker, COUNT(*) FROM news_articles "
        "WHERE source='fnspid' GROUP BY ticker"
    ))
    art_av = dict(con.execute(
        "SELECT ticker, COUNT(*) FROM news_articles "
        "WHERE source='alphavantage' GROUP BY ticker"
    ))
    fnspid_periods = [r[0] for r in con.execute(
        "SELECT period FROM news_ingest_log WHERE source='fnspid'"
    )]
    fnspid_done = ("FNSPID_ALL" in fnspid_periods
                   and "FNSPID_NASDAQ" in fnspid_periods)
    con.close()

    lines = []
    lines.append("=" * 72)
    lines.append(f"SPRINT 9 NEWS INGEST STATUS — {_now_iso()}")
    lines.append("=" * 72)
    lines.append(f"FNSPID bulk ingest: {'DONE' if fnspid_done else 'PARTIAL'} "
                 f"(periods logged: {fnspid_periods or '[]'})")
    lines.append(f"AV windows per ticker: {total_windows} "
                 f"({windows[0][0][:4]}..{windows[-1][0][:4]})")
    lines.append(f"AV tickers in scope: {len(tickers)} (GOOGL excluded per R2)")
    complete = sum(1 for t in tickers if counts.get(t, 0) >= total_windows)
    lines.append(f"AV tickers COMPLETE: {complete}/{len(tickers)}  "
                 f"({100*complete/len(tickers):.0f}%)")
    lines.append("")
    lines.append(f"{'ticker':<8}{'av_win':>8}{'av_rows':>10}{'fn_rows':>10}  status")
    lines.append("-" * 60)
    for tk in tickers:
        w = counts.get(tk, 0)
        s = "DONE" if w >= total_windows else f"({w}/{total_windows})"
        lines.append(f"{tk:<8}{w:>8}{art_av.get(tk,0):>10}"
                     f"{art_fnspid.get(tk,0):>10}  {s}")
    lines.append("")
    lines.append(f"total news_articles rows: "
                 f"fnspid={sum(art_fnspid.values()):,}  "
                 f"av={sum(art_av.values()):,}")
    return "\n".join(lines)


# ── audit / reconcile ────────────────────────────────────────────────────────

def _av_audit_rows(con: sqlite3.Connection) -> list[dict]:
    """
    Per logged window: rows actually in news_articles, and whether the window
    looks truncated (data stops well before the window end).

    This reads the *articles*, not the log, precisely because the log was the
    thing that lied.
    """
    today = datetime.now(timezone.utc).date()
    out = []
    for tk, per, comp, items, st in con.execute(
        "SELECT ticker, period, completed_at, items_fetched, COALESCE(status,'?') "
        "FROM news_ingest_log WHERE source='alphavantage' ORDER BY ticker, period"
    ):
        yr = int(per[:4])
        start = datetime(yr, 1, 1).date()
        end = min(datetime(yr, 12, 31).date(), today)
        n, mx = con.execute(
            "SELECT COUNT(*), MAX(published_at) FROM news_articles "
            "WHERE ticker=? AND source='alphavantage' AND date(published_at) "
            "BETWEEN ? AND ?", (tk, start.isoformat(), end.isoformat())
        ).fetchone()
        last = mx[:10] if mx else None
        gap = ((end - datetime.strptime(last, "%Y-%m-%d").date()).days
               if last else (end - start).days)
        out.append({"ticker": tk, "period": per, "status": st, "rows": n,
                    "last": last, "win_end": end.isoformat(), "gap_days": gap,
                    "logged_items": items, "completed_at": comp})
    return out


def av_audit() -> str:
    con = _connect()
    rows = _av_audit_rows(con)
    con.close()

    zero = [r for r in rows if r["rows"] == 0]
    trunc = [r for r in rows if r["rows"] >= AV_SPLIT_THRESHOLD - 50
             and r["gap_days"] > 14]
    lines = ["=" * 78,
             f"AV WINDOW AUDIT — {_now_iso()}",
             "=" * 78,
             f"logged windows : {len(rows)}",
             f"  zero rows    : {len(zero)}",
             f"  truncated    : {len(trunc)}  "
             f"(≥{AV_SPLIT_THRESHOLD - 50} rows AND >14d short of window end)",
             f"  look clean   : {len(rows) - len(zero) - len(trunc)}",
             ""]
    if zero:
        lines.append("ZERO-ROW WINDOWS")
        lines.append(f"  {'tk':<6}{'period':<20}{'status':<10}{'logged_items':>13}")
        for r in zero:
            lines.append(f"  {r['ticker']:<6}{r['period']:<20}"
                         f"{r['status']:<10}{r['logged_items']:>13}")
        lines.append("")
    if trunc:
        lines.append("TRUNCATED WINDOWS (by days of coverage lost)")
        lines.append(f"  {'tk':<6}{'period':<20}{'rows':>6}{'last_have':>13}"
                     f"{'win_end':>13}{'gap_d':>7}")
        for r in sorted(trunc, key=lambda x: -x["gap_days"]):
            lines.append(f"  {r['ticker']:<6}{r['period']:<20}{r['rows']:>6}"
                         f"{str(r['last']):>13}{r['win_end']:>13}{r['gap_days']:>7}")
    return "\n".join(lines)


def av_reconcile(apply: bool = False) -> str:
    """
    Back-fill the `status` column for log rows written before it existed, and
    fold the drifted current-year period keys ('2026-01..2026-07') into the
    stable '2026-01..2026-12' form.

    Rows are classified from article evidence: no rows ⇒ 'error' (retry once,
    after which the fetcher records 'empty' if AV really has no coverage);
    truncated ⇒ 'partial'; otherwise 'ok'.
    """
    con = _connect()
    changes: list[str] = []

    # 1. collapse drifted keys onto the stable year key
    drifted = list(con.execute(
        "SELECT ticker, period FROM news_ingest_log WHERE source='alphavantage' "
        "AND period NOT LIKE '%-01..%-12'"))
    for tk, per in drifted:
        stable = f"{per[:4]}-01..{per[:4]}-12"
        changes.append(f"  rekey {tk:<6} {per} → {stable}")
        if apply:
            con.execute("DELETE FROM news_ingest_log WHERE source='alphavantage' "
                        "AND ticker=? AND period=?", (tk, per))
    if apply:
        con.commit()

    # 2. classify every remaining row from what is actually in news_articles
    for r in _av_audit_rows(con):
        if r["period"] not in (f"{r['period'][:4]}-01..{r['period'][:4]}-12",):
            continue
        if r["rows"] == 0:
            new = "error"
        elif r["rows"] >= AV_SPLIT_THRESHOLD - 50 and r["gap_days"] > 14:
            new = "partial"
        else:
            new = "ok"
        if r["status"] != new:
            changes.append(f"  status {r['ticker']:<6} {r['period']} "
                           f"{r['status']} → {new}")
            if apply:
                con.execute(
                    "UPDATE news_ingest_log SET status=? WHERE source='alphavantage' "
                    "AND ticker=? AND period=?", (new, r["ticker"], r["period"]))
    if apply:
        con.commit()
    con.close()

    head = (f"av-reconcile — {len(changes)} change(s) "
            f"{'APPLIED' if apply else 'DRY RUN (pass --apply)'}")
    return "\n".join([head] + changes)


# ── verification helpers ─────────────────────────────────────────────────────

def verify() -> str:
    con = _connect()
    out = []
    # year × source histogram
    out.append("year × source histogram")
    for src, yr, n in con.execute(
        "SELECT source, substr(published_at,1,4) AS y, COUNT(*) "
        "FROM news_articles GROUP BY source, y ORDER BY y, source"
    ):
        out.append(f"  {yr}  {src:<14} {n:>8,}")
    # date-range sanity
    row = con.execute(
        "SELECT MIN(published_at), MAX(published_at), COUNT(*) FROM news_articles"
    ).fetchone()
    out.append(f"overall: min={row[0]}  max={row[1]}  rows={row[2]:,}")
    # dedup evidence — hash is PK so duplicates would have failed
    dup = con.execute(
        "SELECT article_hash, COUNT(*) c FROM news_articles "
        "GROUP BY article_hash HAVING c > 1 LIMIT 1"
    ).fetchone()
    out.append(f"duplicate hashes: {'NONE (PK enforced)' if dup is None else dup}")
    # 5 random rows per source
    for src in ("fnspid", "alphavantage"):
        out.append(f"random 5 from {src}:")
        for r in con.execute(
            "SELECT ticker, published_at, substr(title,1,70) "
            "FROM news_articles WHERE source=? ORDER BY random() LIMIT 5",
            (src,),
        ):
            out.append(f"  {r[0]:<6} {r[1]}  {r[2]}")
    con.close()
    return "\n".join(out)


# ── utils ────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _append_state(payload: dict) -> None:
    with AV_BACKFILL_STATE.open("a") as f:
        f.write(json.dumps(payload) + "\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_fn = sub.add_parser("fnspid", help="one-shot FNSPID bulk stream")
    p_fn.add_argument("--source", choices=("all", "nasdaq"), default="all",
                      help="which FNSPID CSV to stream")
    p_av = sub.add_parser("av-backfill", help="consume today's AV budget")
    p_av.add_argument("--budget", type=int, default=AV_DAILY_BUDGET)
    p_av.add_argument("--only", help="restrict to these tickers (comma-separated)")
    sub.add_parser("status", help="print human-readable coverage")
    sub.add_parser("verify", help="run integrity checks")
    sub.add_parser("av-audit", help="report zero-row / truncated AV windows")
    p_rc = sub.add_parser("av-reconcile",
                          help="set window status from article evidence")
    p_rc.add_argument("--apply", action="store_true",
                      help="write changes (default is a dry run)")
    args = ap.parse_args()

    if args.cmd == "fnspid":
        if args.source == "nasdaq":
            r = stream_fnspid(url=FNSPID_URL_NASDAQ,
                              period_label="FNSPID_NASDAQ")
        else:
            r = stream_fnspid(url=FNSPID_URL_ALL,
                              period_label="FNSPID_ALL")
        print(json.dumps(r, indent=2, default=str))
    elif args.cmd == "av-backfill":
        only = ([t.strip().upper() for t in args.only.split(",")]
                if args.only else None)
        av_backfill(budget=args.budget, only=only)
    elif args.cmd == "status":
        print(status())
    elif args.cmd == "verify":
        print(verify())
    elif args.cmd == "av-audit":
        print(av_audit())
    elif args.cmd == "av-reconcile":
        print(av_reconcile(apply=args.apply))


if __name__ == "__main__":
    _cli()
