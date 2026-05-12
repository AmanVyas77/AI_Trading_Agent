"""
Fundamental Factor Score Exporter
==================================
Loads raw fundamental features from SQLite (xbrl_derived, analyst_estimates,
eps_revisions), aligns to quarter-end dates, applies cross-sectional z-scoring,
and saves a tidy long-format parquet for the ensemble combiner.

Features Exported
-----------------
  gross_profitability   — GP / Total Assets (Novy-Marx quality proxy)
  fcf_yield             — FCF / Enterprise Value (from xbrl_derived)
  revenue_acceleration  — second derivative of YoY revenue growth
  deferred_revenue_yoy  — YoY change in deferred revenue
  rd_intensity          — R&D / Revenue (higher = positive for tech)
  sue_score             — Standardized Unexpected Earnings
  eps_revision_1m       — 1-month EPS revision momentum
  eps_revision_3m       — 3-month EPS revision momentum

Output
------
  data/processed/fundamental_factor_scores.parquet
  Columns: [quarter_end, ticker, gross_profitability, fcf_yield,
            revenue_acceleration, deferred_revenue_yoy, rd_intensity,
            sue_score, eps_revision_1m, eps_revision_3m]

CLI
---
  python -m src.strategies.ensemble.factor_export_fundamental
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect

load_dotenv()
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
with open(ROOT / "config" / "settings.yaml") as f:
    CFG = yaml.safe_load(f)

TIMELINE = CFG["timeline"]
DB_PATH = ROOT / CFG["data"]["paths"]["db"]
DB_URL = f"sqlite:///{DB_PATH}"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature columns in the output
FACTOR_COLS = [
    "gross_profitability",
    "fcf_yield",
    "revenue_acceleration",
    "deferred_revenue_yoy",
    "rd_intensity",
    "sue_score",
    "eps_revision_1m",
    "eps_revision_3m",
]

ZSCORE_CAP = 3.0
MIN_OBS_FOR_ZSCORE = 5     # need ≥5 tickers to compute a meaningful z-score
MAX_FFILL_QUARTERS = 1     # forward-fill at most 1 quarter for missing values
HIGH_NAN_THRESHOLD = 0.50  # warn if >50% NaN for a feature in a quarter


# ── DB Access ─────────────────────────────────────────────────────────────────

def _get_engine(engine=None):
    if engine is not None:
        return engine
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(DB_URL, echo=False)


def _table_exists(engine, name: str) -> bool:
    return name in inspect(engine).get_table_names()


def _ticker_clause(tickers: Optional[list[str]]) -> tuple[str, dict]:
    """Build a SQL IN-clause + params for optional ticker filter."""
    if not tickers:
        return "", {}
    ph = ",".join(f":t{i}" for i in range(len(tickers)))
    return f"ticker IN ({ph})", {f"t{i}": t for i, t in enumerate(tickers)}


# ── Data Loaders ──────────────────────────────────────────────────────────────

def _load_xbrl_derived(tickers: Optional[list[str]], engine) -> pd.DataFrame:
    """
    Load xbrl_derived → (ticker, end_date, gross_profitability, fcf_yield,
    revenue_acceleration, deferred_revenue_yoy, rd_intensity).
    """
    if not _table_exists(engine, "xbrl_derived"):
        logger.warning("xbrl_derived table not found")
        return pd.DataFrame()

    clause, params = _ticker_clause(tickers)
    where = f"WHERE {clause}" if clause else ""
    sql = f"""
        SELECT ticker, end_date,
               gross_profitability, fcf_yield,
               revenue_acceleration, deferred_revenue_yoy, rd_intensity
        FROM xbrl_derived
        {where}
        ORDER BY ticker, end_date
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, params=params)

    if not df.empty:
        df["end_date"] = pd.to_datetime(df["end_date"])
    return df


def _load_sue(tickers: Optional[list[str]], engine) -> pd.DataFrame:
    """Load SUE scores from analyst_estimates table."""
    if not _table_exists(engine, "analyst_estimates"):
        logger.warning("analyst_estimates table not found")
        return pd.DataFrame()

    clause, params = _ticker_clause(tickers)
    where = f"WHERE {clause}" if clause else ""
    sql = f"""
        SELECT ticker, fiscal_period, sue_score
        FROM analyst_estimates
        {where}
        ORDER BY ticker, fiscal_period
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, params=params)

    if not df.empty:
        df["fiscal_period"] = pd.to_datetime(df["fiscal_period"])
        # Drop rows with null sue_score
        df = df.dropna(subset=["sue_score"])
    return df


def _load_eps_revisions(tickers: Optional[list[str]], engine) -> pd.DataFrame:
    """Load EPS revision momentum from eps_revisions table."""
    if not _table_exists(engine, "eps_revisions"):
        logger.warning("eps_revisions table not found")
        return pd.DataFrame()

    clause, params = _ticker_clause(tickers)
    where = f"WHERE {clause}" if clause else ""
    sql = f"""
        SELECT ticker, date, revision_1m, revision_3m
        FROM eps_revisions
        {where}
        ORDER BY ticker, date
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, params=params)

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


# ── Quarter-End Alignment ─────────────────────────────────────────────────────

def _to_quarterly(
    df: pd.DataFrame,
    date_col: str,
    value_cols: list[str],
) -> pd.DataFrame:
    """
    Resample a (ticker, date, value…) DataFrame to quarter-end frequency.
    Takes the last observation per ticker per quarter.
    Returns DataFrame with columns: [ticker, quarter_end, *value_cols].
    """
    if df.empty:
        return pd.DataFrame(columns=["ticker", "quarter_end"] + value_cols)

    df = df.copy()
    df["quarter_end"] = df[date_col].dt.to_period("Q").dt.to_timestamp("Q")
    agg = df.groupby(["ticker", "quarter_end"])[value_cols].last().reset_index()
    return agg


# ── Cross-Sectional Z-Scoring ────────────────────────────────────────────────

def _zscore_cross_section(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Z-score each column cross-sectionally (across tickers) per quarter_end.
    Caps values to [-ZSCORE_CAP, ZSCORE_CAP].
    Requires >= MIN_OBS_FOR_ZSCORE non-null observations; otherwise NaN.
    """
    result = df.copy()

    for col in cols:
        if col not in result.columns:
            result[col] = np.nan
            continue

        def _zscore_group(s: pd.Series) -> pd.Series:
            valid = s.dropna()
            if len(valid) < MIN_OBS_FOR_ZSCORE:
                return pd.Series(np.nan, index=s.index)
            mu, sigma = valid.mean(), valid.std()
            if sigma == 0 or np.isnan(sigma):
                return pd.Series(0.0, index=s.index)
            z = (s - mu) / sigma
            return z.clip(-ZSCORE_CAP, ZSCORE_CAP)

        result[col] = result.groupby("quarter_end")[col].transform(_zscore_group)

    return result


# ── Coverage Diagnostics ─────────────────────────────────────────────────────

def _log_coverage(df: pd.DataFrame) -> None:
    """Log per-feature, per-quarter coverage stats and warn on >50% NaN."""
    if df.empty:
        return

    for qe, grp in df.groupby("quarter_end"):
        n = len(grp)
        warnings = {}
        for col in FACTOR_COLS:
            if col not in grp.columns:
                continue
            nan_rate = grp[col].isna().sum() / n
            if nan_rate > HIGH_NAN_THRESHOLD:
                warnings[col] = f"{nan_rate:.0%}"
        if warnings:
            logger.warning(
                f"  {pd.Timestamp(qe).date()}  "
                f"{n} tickers | high NaN (>50%): {warnings}"
            )


# ── Public API ────────────────────────────────────────────────────────────────

def build_fundamental_scores(
    tickers: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    engine=None,
) -> pd.DataFrame:
    """
    Build z-scored fundamental factor scores at quarter-end frequency.

    Parameters
    ----------
    tickers : list of ticker symbols; defaults to all tickers in xbrl_derived
    start   : start date (YYYY-MM-DD); defaults to timeline.train_start
    end     : end date (YYYY-MM-DD); defaults to timeline.test_end
    engine  : SQLAlchemy engine; created from settings.yaml if not provided

    Returns
    -------
    pd.DataFrame with columns:
        [quarter_end, ticker, gross_profitability, fcf_yield,
         revenue_acceleration, deferred_revenue_yoy, rd_intensity,
         sue_score, eps_revision_1m, eps_revision_3m]
    """
    engine = _get_engine(engine)
    start = start or TIMELINE["train_start"]
    end = end or TIMELINE["test_end"]

    logger.info(f"Building fundamental factor scores [{start} → {end}]")

    # ── 1. Load raw data ──────────────────────────────────────────────
    logger.info("Loading xbrl_derived…")
    derived = _load_xbrl_derived(tickers, engine)

    logger.info("Loading analyst_estimates (SUE)…")
    sue = _load_sue(tickers, engine)

    logger.info("Loading eps_revisions…")
    revs = _load_eps_revisions(tickers, engine)

    # ── 2. Align each source to quarter-end dates ─────────────────────
    xbrl_cols = ["gross_profitability", "fcf_yield", "revenue_acceleration",
                 "deferred_revenue_yoy", "rd_intensity"]
    qe_derived = _to_quarterly(derived, "end_date", xbrl_cols)

    qe_sue = _to_quarterly(sue, "fiscal_period", ["sue_score"])

    qe_rev = _to_quarterly(revs, "date", ["revision_1m", "revision_3m"])
    if not qe_rev.empty:
        qe_rev = qe_rev.rename(columns={
            "revision_1m": "eps_revision_1m",
            "revision_3m": "eps_revision_3m",
        })

    # ── 3. Build scaffold of (ticker, quarter_end) combinations ───────
    quarter_ends = pd.date_range(start=start, end=end, freq="QE")

    all_tickers: set[str] = set()
    for frame in [qe_derived, qe_sue, qe_rev]:
        if isinstance(frame, pd.DataFrame) and not frame.empty and "ticker" in frame.columns:
            all_tickers.update(frame["ticker"].unique())
    if tickers:
        all_tickers = all_tickers.intersection(set(tickers)) or set(tickers)

    if not all_tickers:
        logger.warning("No tickers found in any source table")
        return pd.DataFrame(columns=["quarter_end", "ticker"] + FACTOR_COLS)

    scaffold = pd.MultiIndex.from_product(
        [sorted(all_tickers), quarter_ends], names=["ticker", "quarter_end"]
    ).to_frame(index=False)

    logger.info(f"  Scaffold: {len(all_tickers)} tickers × {len(quarter_ends)} quarters = {len(scaffold)} rows")

    # ── 4. Merge all sources onto scaffold ────────────────────────────
    def _merge(base, right, on=("ticker", "quarter_end")):
        if right is None or (isinstance(right, pd.DataFrame) and right.empty):
            return base
        return base.merge(right, on=list(on), how="left")

    matrix = scaffold.copy()
    matrix = _merge(matrix, qe_derived)
    matrix = _merge(matrix, qe_sue)
    matrix = _merge(matrix, qe_rev)

    # ── 5. Forward-fill missing values (at most 1 quarter per ticker) ─
    matrix = matrix.sort_values(["ticker", "quarter_end"])
    available_factor_cols = [c for c in FACTOR_COLS if c in matrix.columns]
    matrix[available_factor_cols] = (
        matrix.groupby("ticker")[available_factor_cols]
        .apply(lambda g: g.ffill(limit=MAX_FFILL_QUARTERS))
        .reset_index(drop=True)
    )

    # ── 6. Filter to requested date range ─────────────────────────────
    matrix = matrix[
        (matrix["quarter_end"] >= pd.Timestamp(start))
        & (matrix["quarter_end"] <= pd.Timestamp(end))
    ]

    # ── 7. Log coverage diagnostics ───────────────────────────────────
    logger.info("Coverage diagnostics:")
    _log_coverage(matrix)

    # ── 8. Cross-sectional z-score per quarter ────────────────────────
    logger.info("Applying cross-sectional z-scoring…")
    matrix = _zscore_cross_section(matrix, FACTOR_COLS)

    # ── 9. Ensure all factor columns exist (fill missing with NaN) ────
    for col in FACTOR_COLS:
        if col not in matrix.columns:
            matrix[col] = np.nan

    # ── 10. Drop rows where ALL factors are NaN ──────────────────────
    matrix = matrix.dropna(subset=FACTOR_COLS, how="all")

    # ── 11. Final column ordering and sort ────────────────────────────
    output_cols = ["quarter_end", "ticker"] + FACTOR_COLS
    result = matrix[output_cols].sort_values(["quarter_end", "ticker"]).reset_index(drop=True)

    logger.info(
        f"Output: {result.shape[0]} rows, "
        f"{result['ticker'].nunique()} tickers, "
        f"{result['quarter_end'].nunique()} quarters"
    )

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    import argparse
    parser = argparse.ArgumentParser(
        description="Export fundamental factor scores to parquet",
    )
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Override universe tickers")
    parser.add_argument("--start", default=None,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None,
                        help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    df = build_fundamental_scores(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
    )

    if df.empty:
        logger.warning("No data produced — parquet NOT saved")
        return

    # Save parquet
    out_path = OUTPUT_DIR / "fundamental_factor_scores.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info(f"Saved → {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")

    # Print summary
    print(f"\n{'═' * 64}")
    print(f"  Fundamental Factor Scores Export")
    print(f"{'═' * 64}")
    print(f"  Shape:      {df.shape}")
    print(f"  Tickers:    {df['ticker'].nunique()}")
    print(f"  Quarters:   {df['quarter_end'].nunique()}")
    print(f"  Date range: {df['quarter_end'].min()} → {df['quarter_end'].max()}")
    print(f"{'═' * 64}")

    # Coverage stats per feature
    print(f"\n  Per-feature coverage:")
    for col in FACTOR_COLS:
        if col in df.columns:
            n_valid = df[col].notna().sum()
            pct = n_valid / len(df) * 100
            print(f"    {col:<28s}  {n_valid:>6d} / {len(df)}  ({pct:5.1f}%)")

    # Factor statistics
    print(f"\n  Factor statistics:")
    print(df[FACTOR_COLS].describe().round(4).to_string())

    print(f"\n  Head:")
    print(df.head(10).to_string(index=False))
    print()


if __name__ == "__main__":
    main()
