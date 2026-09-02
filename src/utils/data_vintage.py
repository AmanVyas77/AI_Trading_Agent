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


def format_vintage(v: dict) -> str:
    """One-line human-readable form for report headers."""
    return (f"price_vintage sha256={v['sha256'][:16]}… "
            f"rows={v['n_rows']} tickers={v['n_tickers']} "
            f"span={v['date_min']}→{v['date_max']}")
