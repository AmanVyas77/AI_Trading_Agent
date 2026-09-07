"""Benchmark-ticker registry.

Why this exists
---------------
`prices` holds benchmark series (SPY) alongside the 54 candidate tickers, so
that a SPY-relative verdict rests on frozen DB rows rather than a live
yfinance call. Benchmarks are NOT universe members: they must never enter a
cross-sectional z-score, a training label, an equal-weight proxy, or the
Alpha Vantage news fetch list.

Several call sites historically derived "the universe" from
`SELECT DISTINCT ticker FROM prices`, which was correct only while `prices`
and `universe.csv` held the same 54 symbols. This module is the single place
that says which tickers in `prices` are benchmarks, so those call sites can
subtract them.

The canonical candidate list remains `data/universe/universe.csv`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def benchmark_tickers() -> frozenset[str]:
    """Tickers present in `prices` that are benchmarks, not candidates."""
    with open(ROOT / "config" / "settings.yaml") as f:
        cfg = yaml.safe_load(f)
    bm = cfg.get("benchmarks", {}) or {}
    explicit = bm.get("tickers") or []
    return frozenset(t.upper() for t in explicit)


def strip_benchmarks(tickers) -> list[str]:
    """Drop benchmark symbols from an iterable of tickers, preserving order."""
    bad = benchmark_tickers()
    return [t for t in tickers if str(t).upper() not in bad]


@lru_cache(maxsize=1)
def universe_tickers() -> tuple[str, ...]:
    """The canonical candidate universe, read from universe.csv."""
    import csv as _csv

    path = ROOT / "data" / "universe" / "universe.csv"
    with open(path, newline="") as f:
        return tuple(r["ticker"] for r in _csv.DictReader(f))
