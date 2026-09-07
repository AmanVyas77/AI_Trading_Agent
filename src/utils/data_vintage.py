"""Price-vintage fingerprinting.

Why this exists
---------------
`data/quant_research.db` is not append-only. The Alpha Vantage backfill
(`scripts/av_backfill_wrapper.sh`, state in `logs/av_backfill_state.jsonl`)
rewrites *historical* `adj_close` rows on an ongoing basis. That means a
backtest result is only meaningful alongside the price vintage it was computed
on: re-running the identical Sprint 7 code today reproduces 1.0909, not the
published 1.0160, purely because the underlying prices changed.

Every backtest that writes a result should record the fingerprint returned by
:func:`price_vintage` so a future reader can tell whether a mismatch is a code
change or a data change.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


def price_vintage(db_path: str | Path, start: str, end: str,
                  table: str = "prices") -> dict:
    """Fingerprint the (date, ticker, adj_close) rows in [start, end].

    Returns a dict with a sha256 over the exact rows a backtest would read,
    plus row/ticker counts and the date range actually present — enough to
    distinguish "same data" from "same query, different data".
    """
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute(
            f"SELECT date, ticker, adj_close FROM {table} "
            "WHERE date >= ? AND date <= ? ORDER BY date, ticker",
            (start, end),
        )
        h = hashlib.sha256()
        n_rows = 0
        tickers: set[str] = set()
        min_d: str | None = None
        max_d: str | None = None
        for date, ticker, adj_close in cur:
            # repr() of the float keeps full precision; a revised price changes
            # the digest even when the row count is unchanged.
            h.update(f"{date}|{ticker}|{adj_close!r}\n".encode())
            n_rows += 1
            tickers.add(ticker)
            if min_d is None:
                min_d = date
            max_d = date

    return {
        "sha256": h.hexdigest(),
        "n_rows": n_rows,
        "n_tickers": len(tickers),
        "date_min": min_d,
        "date_max": max_d,
        "window": [start, end],
        "table": table,
    }


def news_vintage(db_path: str | Path,
                 table: str = "news_articles") -> dict:
    """Fingerprint the news corpus in `news_articles`.

    Why this is separate from :func:`price_vintage`
    -----------------------------------------------
    The nightly Alpha Vantage backfill (`av_backfill`) is not append-only in
    practice: it walks year-windows per ticker and re-fetches, so both the row
    count and the per-year distribution shift underneath any comparison that
    reads the corpus. `price_vintage` will not notice — it only digests
    `prices`. A sentiment-path result computed on Monday's corpus is not
    comparable to the same code run on Tuesday's, and today nothing detects it.

    What it digests
    ---------------
    * total row count
    * MIN / MAX ``published_at``
    * per-``source`` per-year row counts (the backfill's working unit, so this
      is where a re-fetch shows up first)

    The sha256 is taken over the canonicalised (source, year, count) grid plus
    the row count and the published_at bounds — deliberately NOT over article
    bodies, so the digest stays cheap on a 440k-row table and is stable against
    the snippet-truncation differences between the two ingest sources.

    Not wired into the verdict path. This is the detector a later session calls
    to establish that the corpus did or did not move between two runs.

    Returns
    -------
    dict with sha256, n_rows, published_min/max, and the per-source-per-year
    grid as a sorted list of [source, year, count].
    """
    with sqlite3.connect(str(db_path)) as conn:
        n_rows = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        pub_min, pub_max = conn.execute(
            f"SELECT MIN(published_at), MAX(published_at) FROM {table}"
        ).fetchone()
        grid = conn.execute(
            f"SELECT source, substr(published_at, 1, 4) AS yr, COUNT(*) "
            f"FROM {table} GROUP BY source, yr ORDER BY source, yr"
        ).fetchall()
        n_sources = conn.execute(
            f"SELECT COUNT(DISTINCT source) FROM {table}").fetchone()[0]
        n_tickers = conn.execute(
            f"SELECT COUNT(DISTINCT ticker) FROM {table}").fetchone()[0]

    h = hashlib.sha256()
    h.update(f"rows={n_rows}\n".encode())
    h.update(f"published_min={pub_min}\npublished_max={pub_max}\n".encode())
    for source, yr, count in grid:
        h.update(f"{source}|{yr}|{count}\n".encode())

    return {
        "sha256": h.hexdigest(),
        "n_rows": n_rows,
        "n_sources": n_sources,
        "n_tickers": n_tickers,
        "published_min": pub_min,
        "published_max": pub_max,
        "source_year_counts": [[s, y, c] for s, y, c in grid],
        "table": table,
    }


def format_news_vintage(v: dict) -> str:
    """One-line human-readable form for report headers."""
    return (f"news_vintage sha256={v['sha256'][:16]}… "
            f"rows={v['n_rows']} sources={v['n_sources']} "
            f"tickers={v['n_tickers']} "
            f"span={v['published_min']}→{v['published_max']}")


def format_vintage(v: dict) -> str:
    """One-line human-readable form for report headers."""
    return (f"price_vintage sha256={v['sha256'][:16]}… "
            f"rows={v['n_rows']} tickers={v['n_tickers']} "
            f"span={v['date_min']}→{v['date_max']}")
