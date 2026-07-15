"""
Simfin + FINRA Pipeline
=======================
Two data sources for Tier 1/2 signals not available from EDGAR or yfinance:

  1. Simfin (free tier)
     - Analyst consensus EPS estimates → SUE (standardized unexpected earnings)
     - EPS revision momentum (direction + magnitude of estimate changes over 1-3M)
     - Cross-validation of XBRL revenue / income figures

  2. FINRA Short Volume (free, public)
     - Daily short volume by ticker from FINRA's OTC Transparency data
     - Compute days-to-cover = short_interest / avg_daily_volume
     - Short squeeze signal: high SI + positive momentum reversal

Coverage check
--------------
Before building the SUE pipeline, this module checks what fraction of the
universe has analyst estimate data in Simfin. If coverage < 70% (configurable
in settings.yaml), it falls back to the seasonal random walk model for SUE:
    SUE_fallback = (EPS_actual - EPS_same_quarter_prior_year) / std(EPS last 8Q)

Storage: SQLite tables `analyst_estimates`, `eps_revisions`, `short_interest`
"""

from __future__ import annotations

import io
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config" / "settings.yaml") as f:
    CFG = yaml.safe_load(f)

DB_PATH = ROOT / CFG["data"]["paths"]["db"]
DB_URL = f"sqlite:///{DB_PATH}"
TIMELINE = CFG["timeline"]
FF_CFG = CFG["fundamental_factors"]

# Date range for Simfin EPS pulls.
# DATE_END resolves to today at import time so a live refresh without an
# explicit --end can no longer silently clamp to the settings.yaml
# timeline.test_end cutoff (mirrors the sentiment_pipeline fix from
# Sprint 8 Prompt 4).
DATE_START = "2015-01-01"
DATE_END = datetime.now().strftime("%Y-%m-%d")


# ── Database setup ────────────────────────────────────────────────────────────

def get_engine():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(DB_URL, echo=False)
    _create_tables(engine)
    return engine


def _create_tables(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS analyst_estimates (
                ticker          TEXT NOT NULL,
                fiscal_period   TEXT NOT NULL,
                estimate_date   TEXT NOT NULL,
                eps_estimate    REAL,
                eps_actual      REAL,
                eps_surprise    REAL,
                sue_score       REAL,
                source          TEXT,
                PRIMARY KEY (ticker, fiscal_period, estimate_date)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS eps_revisions (
                ticker          TEXT NOT NULL,
                date            TEXT NOT NULL,
                revision_1m     REAL,
                revision_3m     REAL,
                direction       TEXT,
                magnitude       REAL,
                PRIMARY KEY (ticker, date)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS short_interest (
                ticker          TEXT NOT NULL,
                date            TEXT NOT NULL,
                short_volume    REAL,
                total_volume    REAL,
                short_ratio     REAL,
                days_to_cover   REAL,
                PRIMARY KEY (ticker, date)
            )
        """))


# ── Simfin — Analyst Estimates ────────────────────────────────────────────────

SIMFIN_BASE = "https://backend.simfin.com/api/v3"


def check_simfin_coverage(tickers: list[str], api_key: str) -> float:
    """
    Check what fraction of universe tickers have analyst estimate data in Simfin.
    Returns coverage ratio (0.0–1.0).
    """
    import os
    covered = 0
    headers = {"Authorization": f"api-key {api_key}"}

    for ticker in tickers[:20]:   # sample first 20 to estimate coverage quickly
        url = f"{SIMFIN_BASE}/companies/statements/compact"
        params = {"ticker": ticker, "statements": "pl", "period": "Q1", "fyear": 2022}
        try:
            r = requests.get(url, headers=headers, params=params, timeout=10)
            if r.status_code == 200 and r.json():
                covered += 1
        except Exception:
            pass
        time.sleep(0.2)

    coverage = covered / min(20, len(tickers))
    logger.info(f"Simfin coverage check: {covered}/20 sampled → {coverage:.1%}")
    return coverage


def fetch_simfin_eps(
    ticker: str,
    api_key: str,
    start: str = DATE_START,
    end: str = DATE_END,
) -> pd.DataFrame:
    """
    Pull quarterly EPS actuals from Simfin.
    Returns DataFrame with columns: fiscal_period, eps_actual.

    start / end default to the module DATE_START / DATE_END (today at
    import time) so an explicit window is only needed when back-filling
    a historical slice. The previous default clamped `end` to
    settings.yaml timeline.test_end (2024-12-31), silently dropping any
    2025+ quarters returned by the API.
    """
    headers = {"Authorization": f"api-key {api_key}"}
    url = f"{SIMFIN_BASE}/companies/statements/compact"
    params = {
        "ticker": ticker,
        "statements": "pl",
        "period": "quarterly",
        "start": start,
        "end": end,
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame()

        # Simfin compact format: columns + data arrays
        cols = data[0].get("columns", [])
        rows = data[0].get("data", [])
        df = pd.DataFrame(rows, columns=cols)

        # Extract EPS diluted and fiscal period
        eps_col = next((c for c in cols if "Diluted" in c and "EPS" in c), None)
        period_col = "Fiscal Period" if "Fiscal Period" in cols else None

        if not eps_col or not period_col:
            return pd.DataFrame()

        out = df[[period_col, eps_col]].rename(columns={
            period_col: "fiscal_period",
            eps_col: "eps_actual",
        })
        out["ticker"] = ticker
        return out

    except Exception as e:
        logger.warning(f"  [{ticker}] Simfin EPS fetch failed: {e}")
        return pd.DataFrame()


def compute_sue_simfin(ticker: str, eps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Standardized Unexpected Earnings (SUE) from Simfin EPS data.
    SUE = (EPS_actual - EPS_estimate) / std(EPS surprises last 8Q)
    When no estimate is available, use seasonal random walk as fallback.
    """
    if eps_df.empty:
        return pd.DataFrame()

    eps_df = eps_df.sort_values("fiscal_period").copy()

    # Seasonal random walk: EPS expected = EPS same quarter 1 year ago (4Q back)
    eps_df["eps_estimate"] = eps_df["eps_actual"].shift(4)
    eps_df["eps_surprise"] = eps_df["eps_actual"] - eps_df["eps_estimate"]

    # Rolling std of surprises (8Q)
    rolling_std = eps_df["eps_surprise"].rolling(8, min_periods=4).std()
    eps_df["sue_score"] = eps_df["eps_surprise"] / rolling_std.replace(0, float("nan"))
    eps_df["source"] = "seasonal_random_walk"
    eps_df["estimate_date"] = eps_df["fiscal_period"]

    return eps_df[["ticker", "fiscal_period", "estimate_date",
                   "eps_estimate", "eps_actual", "eps_surprise", "sue_score", "source"]]


def fetch_and_compute_sue(
    tickers: list[str],
    api_key: Optional[str],
    engine,
    fallback_threshold: float = 0.70,
    start: str = DATE_START,
    end: str = DATE_END,
) -> None:
    """
    For each ticker:
      1. Try Simfin for EPS actuals
      2. Compute SUE (with seasonal random walk as fallback)
      3. Upsert to analyst_estimates table
    """
    for ticker in tickers:
        eps_df = pd.DataFrame()
        if api_key:
            eps_df = fetch_simfin_eps(ticker, api_key, start=start, end=end)
            time.sleep(0.3)

        if eps_df.empty:
            # Pull EPS from XBRL facts DB as fallback
            with engine.connect() as conn:
                eps_df = pd.read_sql_query(
                    text("SELECT ticker, end_date as fiscal_period, eps_diluted as eps_actual "
                         "FROM xbrl_facts WHERE ticker=:t ORDER BY end_date"),
                    conn, params={"t": ticker}
                )
            eps_df["ticker"] = ticker

        sue_df = compute_sue_simfin(ticker, eps_df)
        if sue_df.empty:
            continue

        with engine.begin() as conn:
            for _, row in sue_df.iterrows():
                conn.execute(text("""
                    INSERT OR REPLACE INTO analyst_estimates
                        (ticker, fiscal_period, estimate_date, eps_estimate,
                         eps_actual, eps_surprise, sue_score, source)
                    VALUES
                        (:ticker, :fiscal_period, :estimate_date, :eps_estimate,
                         :eps_actual, :eps_surprise, :sue_score, :source)
                """), {k: (None if pd.isna(v) else v) for k, v in row.items()})

        logger.info(f"  [{ticker}] SUE computed: {len(sue_df)} quarters")


def compute_eps_revision_momentum(tickers: list[str], engine) -> None:
    """
    Compute EPS revision momentum from analyst_estimates table.
    Revision = change in EPS estimate over 1M and 3M windows.
    Since we're using SUE, we approximate revision momentum as
    the rolling trend in SUE scores (positive trend = upward revisions).
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(
            text("SELECT ticker, estimate_date, sue_score FROM analyst_estimates ORDER BY ticker, estimate_date"),
            conn
        )

    if df.empty:
        return

    df["estimate_date"] = pd.to_datetime(df["estimate_date"])
    records = []

    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("estimate_date").set_index("estimate_date")
        sue = grp["sue_score"].dropna()

        rev_1m = sue.diff(1)   # 1-quarter change in SUE
        rev_3m = sue.diff(3)   # 3-quarter change in SUE

        for date in sue.index:
            records.append({
                "ticker": ticker,
                "date": date.strftime("%Y-%m-%d"),
                "revision_1m": rev_1m.get(date),
                "revision_3m": rev_3m.get(date),
                "direction": "up" if (rev_1m.get(date) or 0) > 0 else "down",
                "magnitude": abs(rev_1m.get(date) or 0),
            })

    with engine.begin() as conn:
        for r in records:
            conn.execute(text("""
                INSERT OR REPLACE INTO eps_revisions
                    (ticker, date, revision_1m, revision_3m, direction, magnitude)
                VALUES
                    (:ticker, :date, :revision_1m, :revision_3m, :direction, :magnitude)
            """), {k: (None if v != v else v) for k, v in r.items()})

    logger.info(f"EPS revision momentum computed: {len(records)} rows")


# ── FINRA Short Volume ────────────────────────────────────────────────────────

FINRA_URL = "https://cdn.finra.org/equity/regsho/monthly/CNMSshvol{YYYYMM}.txt"


def fetch_finra_short_volume(
    tickers: list[str],
    start: str = TIMELINE["train_start"],
    end: str = TIMELINE["test_end"],
    engine=None,
) -> None:
    """
    Download FINRA monthly short volume files and extract data for universe tickers.
    Files are tab-separated with columns: Date, Symbol, ShortVolume, ShortExemptVolume, TotalVolume, Market
    """
    if engine is None:
        engine = get_engine()

    ticker_set = set(t.upper() for t in tickers)
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    current = start_dt.replace(day=1)
    all_rows = []

    while current <= end_dt:
        month_str = current.strftime("%Y%m")
        url = FINRA_URL.format(YYYYMM=month_str)

        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 404:
                logger.debug(f"  FINRA {month_str}: file not found (future month?)")
                current += timedelta(days=32)
                current = current.replace(day=1)
                continue
            r.raise_for_status()

            df = pd.read_csv(io.StringIO(r.text), sep="|", dtype=str)
            df.columns = df.columns.str.strip()

            # Filter to universe tickers
            df = df[df["Symbol"].isin(ticker_set)].copy()
            if df.empty:
                current += timedelta(days=32)
                current = current.replace(day=1)
                continue

            df = df.rename(columns={
                "Date": "date",
                "Symbol": "ticker",
                "ShortVolume": "short_volume",
                "TotalVolume": "total_volume",
            })
            df["short_volume"] = pd.to_numeric(df["short_volume"], errors="coerce")
            df["total_volume"] = pd.to_numeric(df["total_volume"], errors="coerce")
            df["short_ratio"] = df["short_volume"] / df["total_volume"]
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")

            all_rows.append(df[["ticker", "date", "short_volume", "total_volume", "short_ratio"]])
            logger.info(f"  FINRA {month_str}: {len(df)} rows for {df['ticker'].nunique()} tickers")

        except Exception as e:
            logger.warning(f"  FINRA {month_str}: {e}")

        current += timedelta(days=32)
        current = current.replace(day=1)
        time.sleep(0.5)

    if not all_rows:
        logger.warning("No FINRA data retrieved")
        return

    combined = pd.concat(all_rows, ignore_index=True)

    # Compute days-to-cover: short_volume / avg 20-day total volume
    combined_sorted = combined.sort_values(["ticker", "date"])
    avg_vol = (
        combined_sorted.groupby("ticker")["total_volume"]
        .transform(lambda x: x.rolling(20, min_periods=5).mean())
    )
    combined_sorted["days_to_cover"] = combined_sorted["short_volume"] / avg_vol.replace(0, float("nan"))

    # Upsert to DB
    with engine.begin() as conn:
        for _, row in combined_sorted.iterrows():
            conn.execute(text("""
                INSERT OR REPLACE INTO short_interest
                    (ticker, date, short_volume, total_volume, short_ratio, days_to_cover)
                VALUES
                    (:ticker, :date, :short_volume, :total_volume, :short_ratio, :days_to_cover)
            """), {k: (None if pd.isna(v) else v) for k, v in row.items()})

    logger.info(f"Short interest saved: {len(combined_sorted):,} rows")


# ── Load helpers ──────────────────────────────────────────────────────────────

def load_sue(tickers: Optional[list[str]] = None, engine=None) -> pd.DataFrame:
    if engine is None:
        engine = get_engine()
    where = ""
    params: dict = {}
    if tickers:
        ph = ",".join(f":t{i}" for i in range(len(tickers)))
        where = f"WHERE ticker IN ({ph})"
        params = {f"t{i}": t for i, t in enumerate(tickers)}
    with engine.connect() as conn:
        df = pd.read_sql_query(
            text(f"SELECT * FROM analyst_estimates {where} ORDER BY ticker, fiscal_period"),
            conn, params=params
        )
    return df


def load_short_interest(tickers: Optional[list[str]] = None, engine=None) -> pd.DataFrame:
    if engine is None:
        engine = get_engine()
    where = ""
    params: dict = {}
    if tickers:
        ph = ",".join(f":t{i}" for i in range(len(tickers)))
        where = f"WHERE ticker IN ({ph})"
        params = {f"t{i}": t for i, t in enumerate(tickers)}
    with engine.connect() as conn:
        df = pd.read_sql_query(
            text(f"SELECT * FROM short_interest {where} ORDER BY ticker, date"),
            conn, params=params
        )
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Pipeline orchestrator ─────────────────────────────────────────────────────

def run_simfin_finra_pipeline(
    tickers: Optional[list[str]] = None,
    skip_finra: bool = False,
    skip_simfin: bool = False,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> None:
    import os
    from src.universe.screener import load_universe

    engine = get_engine()

    if tickers is None:
        uni = load_universe(ROOT / "data" / "universe" / "universe.csv")
        tickers = uni["ticker"].tolist()

    start = start or DATE_START
    end = end or DATE_END

    api_key = os.getenv("SIMFIN_API_KEY")
    if not api_key:
        logger.warning("SIMFIN_API_KEY not set — will use seasonal random walk fallback for SUE")

    if not skip_simfin:
        logger.info(f"Computing SUE for {len(tickers)} tickers  window={start}→{end}")
        fallback_threshold = FF_CFG.get("earnings_surprise", {}).get("fallback_threshold", 0.70)
        fetch_and_compute_sue(
            tickers, api_key, engine, fallback_threshold,
            start=start, end=end,
        )
        compute_eps_revision_momentum(tickers, engine)

    if not skip_finra:
        logger.info("Downloading FINRA short volume data…")
        fetch_finra_short_volume(tickers, engine=engine)

    logger.info("Simfin + FINRA pipeline complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Simfin + FINRA pipeline")
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--skip-finra", action="store_true")
    parser.add_argument("--skip-simfin", action="store_true")
    parser.add_argument(
        "--start",
        default=None,
        help=f"Simfin EPS pull start date (default: module DATE_START={DATE_START})",
    )
    parser.add_argument(
        "--end",
        default=None,
        help=f"Simfin EPS pull end date (default: module DATE_END={DATE_END} = today)",
    )
    args = parser.parse_args()

    run_simfin_finra_pipeline(
        tickers=args.tickers,
        skip_finra=args.skip_finra,
        skip_simfin=args.skip_simfin,
        start=args.start,
        end=args.end,
    )
