"""
Quant Data Pipeline
===================
Downloads and stores:
  1. Daily OHLCV price/volume data (yfinance) for all universe tickers
  2. Macro time-series from FRED

Persists everything to SQLite (data/quant_research.db) via SQLAlchemy.
Tables:
  - prices          : daily OHLCV, adjusted, per ticker
  - macro_series    : FRED series values by date

Usage:
  python -m src.data.quant_pipeline               # full run
  python -m src.data.quant_pipeline --tickers AAPL MSFT   # specific tickers
  python -m src.data.quant_pipeline --macro-only           # just FRED
"""

from __future__ import annotations

import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
import yaml
from dotenv import load_dotenv
from fredapi import Fred
from sqlalchemy import create_engine, text

load_dotenv()
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config" / "settings.yaml") as f:
    CFG = yaml.safe_load(f)

TIMELINE = CFG["timeline"]
FRED_SERIES = CFG["fred"]["series"]
DB_PATH = ROOT / CFG["data"]["paths"]["db"]
DB_URL = f"sqlite:///{DB_PATH}"


# ── Database helpers ──────────────────────────────────────────────────────────

def get_engine():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(DB_URL, echo=False)
    _create_tables(engine)
    return engine


def _create_tables(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS prices (
                ticker      TEXT    NOT NULL,
                date        TEXT    NOT NULL,
                open        REAL,
                high        REAL,
                low         REAL,
                close       REAL,
                volume      REAL,
                adj_close   REAL,
                PRIMARY KEY (ticker, date)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS macro_series (
                series_id   TEXT    NOT NULL,
                series_name TEXT,
                date        TEXT    NOT NULL,
                value       REAL,
                PRIMARY KEY (series_id, date)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at      TEXT,
                run_type    TEXT,
                tickers     TEXT,
                rows_written INTEGER
            )
        """))


# ── Price Data ────────────────────────────────────────────────────────────────

def fetch_prices(
    tickers: list[str],
    start: str = TIMELINE["train_start"],
    end: str = None,
    engine=None,
) -> pd.DataFrame:
    """
    Download daily OHLCV for given tickers from yfinance and upsert to DB.

    Parameters
    ----------
    tickers : list of ticker symbols
    start   : start date string YYYY-MM-DD
    end     : end date string YYYY-MM-DD (defaults to today)
    engine  : SQLAlchemy engine (created if None)

    Returns
    -------
    DataFrame with all price rows written
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    if engine is None:
        engine = get_engine()

    logger.info(f"Downloading prices for {len(tickers)} tickers  [{start} → {end}]")

    all_rows = []
    for i, ticker in enumerate(tickers):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if raw.empty:
                logger.warning(f"  [{ticker}] no data returned")
                continue

            # yfinance returns MultiIndex columns when auto_adjust=True
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            raw = raw.rename(columns=str.lower)
            raw["ticker"] = ticker
            raw["date"] = raw.index.strftime("%Y-%m-%d")
            raw["adj_close"] = raw.get("close", pd.Series(dtype=float))

            rows = raw[["ticker", "date", "open", "high", "low", "close", "volume", "adj_close"]]
            _upsert_prices(rows, engine)
            all_rows.append(rows)

            if (i + 1) % 10 == 0:
                logger.info(f"  {i+1}/{len(tickers)} tickers done")
            time.sleep(0.2)

        except Exception as e:
            logger.error(f"  [{ticker}] price fetch failed: {e}")

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        logger.info(f"Prices written: {len(combined):,} rows across {len(tickers)} tickers")
        return combined
    return pd.DataFrame()


def _upsert_prices(df: pd.DataFrame, engine) -> None:
    """INSERT OR REPLACE price rows."""
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT OR REPLACE INTO prices
                    (ticker, date, open, high, low, close, volume, adj_close)
                VALUES
                    (:ticker, :date, :open, :high, :low, :close, :volume, :adj_close)
            """), row.to_dict())


def load_prices(
    tickers: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    engine=None,
) -> pd.DataFrame:
    """Load prices from DB into a wide DataFrame (date × ticker → adj_close)."""
    if engine is None:
        engine = get_engine()

    conditions = []
    params: dict = {}
    if tickers:
        placeholders = ",".join(f":t{i}" for i in range(len(tickers)))
        conditions.append(f"ticker IN ({placeholders})")
        params.update({f"t{i}": t for i, t in enumerate(tickers)})
    if start:
        conditions.append("date >= :start")
        params["start"] = start
    if end:
        conditions.append("date <= :end")
        params["end"] = end

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"SELECT ticker, date, adj_close FROM prices {where} ORDER BY date"

    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, params=params)

    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="ticker", values="adj_close")
    return pivot


# ── Benchmark Data ────────────────────────────────────────────────────────────

BENCHMARK_TICKER = CFG.get("benchmarks", {}).get("phase3", "SPY")


def load_benchmark(
    start: str,
    end: str,
    ticker: str = BENCHMARK_TICKER,
    engine=None,
    allow_download: bool = True,
) -> pd.Series:
    """Load a benchmark adj_close series from the frozen `prices` table.

    The benchmark belongs in the DB for the same reason the candidates do: a
    pre-registered SPY-relative verdict must be reproducible from a frozen
    vintage, not from whatever yfinance serves on the day it is re-run. Rows
    are ingested by `fetch_prices` through the identical code path as the
    universe, and are covered by `scripts/freeze_vintage.py`.

    Falls back to a live yfinance download only if the DB cannot satisfy the
    request, and says so loudly — a fallback means the result is no longer
    reproducible from the vintage.

    Parameters
    ----------
    start, end     : inclusive date bounds, YYYY-MM-DD
    ticker         : benchmark symbol (default from settings.yaml benchmarks.phase3)
    engine         : SQLAlchemy engine (created if None)
    allow_download : if False, raise instead of falling back to the network

    Returns
    -------
    pd.Series of adj_close indexed by DatetimeIndex, named `ticker`.
    """
    if engine is None:
        engine = get_engine()

    with engine.connect() as conn:
        df = pd.read_sql_query(
            text(
                "SELECT date, adj_close FROM prices "
                "WHERE ticker = :t AND date >= :start AND date <= :end "
                "ORDER BY date"
            ),
            conn,
            params={"t": ticker, "start": start, "end": end},
        )

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        series = df.set_index("date")["adj_close"].rename(ticker)
        logger.info(
            f"Benchmark [{ticker}] loaded from DB `prices`: {len(series)} rows "
            f"[{series.index.min().date()} → {series.index.max().date()}]"
        )
        return series

    reason = f"no `{ticker}` rows in `prices` for [{start} → {end}]"
    if not allow_download:
        raise RuntimeError(
            f"Benchmark [{ticker}] unavailable from DB ({reason}) and "
            "allow_download=False — refusing to fall back to the network."
        )

    logger.warning(
        "=" * 72 + "\n"
        f"FALLBACK: benchmark [{ticker}] NOT served from the DB.\n"
        f"  reason        : {reason}\n"
        f"  fallback used : live yfinance download (yf.download)\n"
        f"  consequence   : this series is NOT part of any frozen vintage and\n"
        f"                  will not reproduce. Ingest it with\n"
        f"                  `.venv/bin/python -m src.data.quant_pipeline "
        f"--tickers {ticker} --price-only`\n" + "=" * 72
    )
    raw = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False
    )
    if raw.empty:
        raise RuntimeError(f"Benchmark [{ticker}] fallback download returned no data")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw["Close"].rename(ticker)


# ── Macro Data ────────────────────────────────────────────────────────────────

def fetch_macro(
    series: list[dict] = FRED_SERIES,
    start: str = TIMELINE["train_start"],
    end: str = None,
    engine=None,
) -> pd.DataFrame:
    """
    Download FRED macro series and upsert to DB.

    Parameters
    ----------
    series : list of {id, name} dicts from config
    start  : start date
    end    : end date (defaults to today)
    engine : SQLAlchemy engine

    Returns
    -------
    DataFrame (date × series_name)
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError("FRED_API_KEY not set in .env")

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    if engine is None:
        engine = get_engine()

    fred = Fred(api_key=api_key)
    dfs = []

    for s in series:
        sid, sname = s["id"], s["name"]
        try:
            data = fred.get_series(sid, observation_start=start, observation_end=end)
            df = data.rename(sname).reset_index()
            df.columns = ["date", "value"]
            df["series_id"] = sid
            df["series_name"] = sname
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
            _upsert_macro(df, engine)
            dfs.append(df.set_index("date")["value"].rename(sname))
            logger.info(f"  FRED [{sid}] {sname}: {len(df)} observations")
        except Exception as e:
            logger.error(f"  FRED [{sid}] fetch failed: {e}")

    if dfs:
        combined = pd.concat(dfs, axis=1)
        combined.index = pd.to_datetime(combined.index)
        return combined.sort_index()
    return pd.DataFrame()


def _upsert_macro(df: pd.DataFrame, engine) -> None:
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT OR REPLACE INTO macro_series
                    (series_id, series_name, date, value)
                VALUES
                    (:series_id, :series_name, :date, :value)
            """), row.to_dict())


def load_macro(
    series_names: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    engine=None,
) -> pd.DataFrame:
    """Load macro data from DB, forward-filled to daily frequency."""
    if engine is None:
        engine = get_engine()

    conditions = []
    params: dict = {}
    if series_names:
        placeholders = ",".join(f":s{i}" for i in range(len(series_names)))
        conditions.append(f"series_name IN ({placeholders})")
        params.update({f"s{i}": s for i, s in enumerate(series_names)})
    if start:
        conditions.append("date >= :start")
        params["start"] = start
    if end:
        conditions.append("date <= :end")
        params["end"] = end

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"SELECT series_name, date, value FROM macro_series {where} ORDER BY date"

    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, params=params)

    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="series_name", values="value")
    # Forward-fill to handle weekends/holidays (FRED is not daily for all series)
    pivot = pivot.resample("D").last().ffill()
    return pivot


# ── Pipeline Orchestrator ─────────────────────────────────────────────────────

def run_pipeline(
    tickers: Optional[list[str]] = None,
    price_only: bool = False,
    macro_only: bool = False,
) -> None:
    """
    Run the full quant data pipeline.

    Loads the universe from data/universe/universe.csv if no tickers provided.
    """
    from src.universe.screener import load_universe

    engine = get_engine()

    if not macro_only:
        if tickers is None:
            uni_path = ROOT / "data" / "universe" / "universe.csv"
            if uni_path.exists():
                uni = load_universe(uni_path)
                tickers = uni["ticker"].tolist()
                logger.info(f"Loaded {len(tickers)} tickers from universe file")
            else:
                raise FileNotFoundError(
                    f"Universe file not found at {uni_path}. "
                    "Run src/universe/screener.py first."
                )
        fetch_prices(tickers, engine=engine)

    if not price_only:
        fetch_macro(engine=engine)

    # Log run
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO pipeline_runs (run_at, run_type, tickers)
            VALUES (:run_at, :run_type, :tickers)
        """), {
            "run_at": datetime.now().isoformat(),
            "run_type": "macro_only" if macro_only else ("price_only" if price_only else "full"),
            "tickers": ",".join(tickers or []),
        })

    logger.info("Pipeline run complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Quant data pipeline")
    parser.add_argument("--tickers", nargs="+", help="Override universe with specific tickers")
    parser.add_argument("--price-only", action="store_true", help="Skip FRED macro")
    parser.add_argument("--macro-only", action="store_true", help="Skip price data")
    args = parser.parse_args()

    run_pipeline(
        tickers=args.tickers,
        price_only=args.price_only,
        macro_only=args.macro_only,
    )
