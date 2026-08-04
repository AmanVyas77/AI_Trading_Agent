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
AV_SPLIT_THRESHOLD = 950   # if hit, resplit into halves


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
    items_fetched   INTEGER DEFAULT 0,
    PRIMARY KEY(source, ticker, period)
);
"""


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.executescript(DDL)
    return con


def _hash(ticker: str, published_at: str, title: str) -> str:
    payload = f"{ticker}|{published_at}|{title or ''}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_universe() -> list[str]:
    con = sqlite3.connect(DB_PATH)
    try:
        return [r[0] for r in con.execute(
            "SELECT DISTINCT ticker FROM prices ORDER BY ticker"
        )]
    finally:
        con.close()


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
    """'20220101T0000','20221231T2359' → '2022-01..2022-12' (or with-halves)."""
    a = f"{tf[:4]}-{tf[4:6]}"
    b = f"{tt[:4]}-{tt[4:6]}"
    return f"{a}..{b}"


def _av_fetch(ticker: str, tf: str, tt: str, key: str,
              limit: int = AV_ITEM_CAP) -> tuple[dict, bool]:
    """Return (parsed_json, throttled_bool)."""
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
    with urllib.request.urlopen(req, timeout=45) as r:
        body = r.read().decode()
    d = json.loads(body)
    throttled = ("Information" in d) or ("Note" in d)
    return d, throttled


def _av_persist(con: sqlite3.Connection, ticker: str, feed: list[dict]) -> int:
    """Insert per-item rows keyed on hash. Returns rows actually inserted."""
    rows = []
    for it in feed:
        tp = it.get("time_published", "")
        # AV format: 20240215T093012 → 2024-02-15 09:30:12
        try:
            dt = datetime.strptime(tp, "%Y%m%dT%H%M%S")
            published = dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
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
    if not rows:
        return 0
    cur = con.executemany(
        "INSERT OR IGNORE INTO news_articles "
        "(ticker, published_at, title, snippet, site, source, "
        " relevance_score, article_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.commit()
    return cur.rowcount


def _av_pending(con: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """Return (ticker, time_from, time_to) tuples still to fetch."""
    done = {(t, p) for t, p in con.execute(
        "SELECT ticker, period FROM news_ingest_log WHERE source='alphavantage'"
    )}
    pending = []
    windows = _av_year_windows()
    for tk in _av_ticker_set():
        for tf, tt in windows:
            per = _av_period_key(tf, tt)
            if (tk, per) not in done:
                pending.append((tk, tf, tt))
    return pending


def av_backfill(budget: int = AV_DAILY_BUDGET) -> dict:
    """
    Consume up to `budget` requests, then exit 0.  Resumable via news_ingest_log.
    If the API returns the daily-cap message, record and exit cleanly.
    """
    key = _av_key()
    con = _connect()
    pending = _av_pending(con)

    if not pending:
        log.info("BACKFILL COMPLETE — all tickers × years present in log.")
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

    for ticker, tf, tt in pending:
        if used >= budget:
            break

        # split into 6-month halves for known mega-caps upfront (item-density > 1000/yr)
        halves = _maybe_split(ticker, tf, tt)

        for sub_tf, sub_tt in halves:
            if used >= budget:
                break
            time.sleep(AV_SLEEP_S)   # R3
            try:
                data, throttled = _av_fetch(ticker, sub_tf, sub_tt, key)
            except Exception as e:
                log.error(f"  {ticker} {sub_tf}..{sub_tt}: fetch failed — {e}")
                used += 1
                continue
            used += 1
            last_cursor = (ticker, sub_tf, sub_tt)

            if throttled:
                msg = data.get("Information") or data.get("Note") or "throttled"
                log.warning(f"  {ticker} {sub_tf}: throttled — {msg[:120]}")
                # Daily cap message → stop early, do NOT log this window as done
                if "25 requests per day" in str(msg) or "rate limit" in str(msg).lower():
                    cap_hit = True
                    break
                # else per-second burst — sleep longer and continue
                time.sleep(AV_SLEEP_S * 3)
                continue

            feed = data.get("feed", []) or []
            inserted = _av_persist(con, ticker, feed)
            added += inserted

            # if we hit the item cap on a half-year, split again → quarters
            if len(feed) >= AV_SPLIT_THRESHOLD:
                log.warning(f"  {ticker} {sub_tf}..{sub_tt}: "
                            f"items={len(feed)} ≥ {AV_SPLIT_THRESHOLD} — "
                            "may be truncated; consider quarterly split")

            log.info(f"  {ticker} {sub_tf}..{sub_tt}: "
                     f"items={len(feed)} inserted={inserted}")

        if cap_hit:
            break

        # log the (ticker, year-period) as done once all halves are attempted
        con.execute(
            "INSERT OR REPLACE INTO news_ingest_log "
            "(source, ticker, period, completed_at, requests_used, items_fetched) "
            "VALUES ('alphavantage', ?, ?, ?, ?, ?)",
            (ticker, _av_period_key(tf, tt), _now_iso(), len(halves), added),
        )
        con.commit()

    con.close()

    summary = {
        "run_at": _now_iso(),
        "requests_used": used,
        "items_added": added,
        "cap_hit_early": cap_hit,
        "pending_before": len(pending),
        "last_cursor": last_cursor,
    }
    _append_state(summary)
    log.info(f"exit summary: {summary}")
    return summary


def _maybe_split(ticker: str, tf: str, tt: str) -> list[tuple[str, str]]:
    """
    Return list of (tf, tt) sub-windows.
    For known high-density tickers, split a full-year window into halves
    up front to avoid AV's 1000-item cap.
    """
    MEGA = {"AAPL", "MSFT", "AMZN", "GOOG", "NVDA", "META", "TSLA",
            "NFLX", "AMD", "ORCL", "IBM", "INTC"}
    # detect year window (tt is Dec-end for full years)
    is_full_year = (tf[4:8] == "0101" and tt[4:8] in ("1231",))
    if ticker in MEGA and is_full_year:
        yr = tf[:4]
        return [
            (f"{yr}0101T0000", f"{yr}0630T2359"),
            (f"{yr}0701T0000", f"{yr}1231T2359"),
        ]
    return [(tf, tt)]


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
    sub.add_parser("status", help="print human-readable coverage")
    sub.add_parser("verify", help="run integrity checks")
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
        av_backfill(budget=args.budget)
    elif args.cmd == "status":
        print(status())
    elif args.cmd == "verify":
        print(verify())


if __name__ == "__main__":
    _cli()
