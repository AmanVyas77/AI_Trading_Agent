"""
Target Variable + Walk-Forward Split Builder
=============================================
Computes 1-month forward returns for each ticker, compares against XLK
benchmark, and produces a binary outperformance label. Provides a strict
walk-forward fold generator for time-series–aware ML training.

Target
------
  label = 1 if ticker's 1-month forward return > XLK's 1-month forward return
  label = 0 otherwise

Walk-Forward Folds
------------------
  - Minimum training window: 36 months
  - Each fold: train on all data up to month T, predict month T+1
  - Yields (train_df, test_df) for each month from the 37th onward

Output
------
  data/processed/ensemble_labeled.parquet

CLI
---
  python -m src.strategies.ensemble.target_builder
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
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
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

BENCHMARK = "XLK"
MIN_TRAIN_MONTHS = 36
# Retired 2026-09-09 (Prompt 3b): XLK now comes from the frozen `prices`
# table, not this gitignored cache. Kept only so a stale cache on disk is
# recognisable as dead, never read.
_RETIRED_XLK_CACHE_PATH = ROOT / "data" / "raw" / "xlk_monthly.csv"


# ── DB helper ─────────────────────────────────────────────────────────────────

def _get_engine(engine=None):
    if engine is not None:
        return engine
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(DB_URL, echo=False)


# ── 1. Load universe prices ──────────────────────────────────────────────────

def _load_monthly_prices(engine, start: Optional[str] = None,
                         end: Optional[str] = None) -> pd.DataFrame:
    """
    Load adj_close from SQLite, pivot to wide (date × ticker),
    resample to month-end.
    """
    if not inspect(engine).get_table_names().__contains__("prices"):
        logger.error("prices table not found")
        return pd.DataFrame()

    sql = """
        SELECT ticker, date, adj_close
        FROM prices
        WHERE date >= :start AND date <= :end
        ORDER BY date
    """
    with engine.connect() as conn:
        # start/end default to the settings.yaml timeline, so calling with no
        # arguments stays bit-identical to the pre-Prompt-3b behaviour.
        # TIMELINE["test_end"] itself is NOT mutated — it is read by the live
        # matrix build and changing it would ripple into the production path.
        df = pd.read_sql_query(
            text(sql), conn,
            params={"start": start or TIMELINE["train_start"],
                    "end": end or TIMELINE["test_end"]},
        )

    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="ticker", values="adj_close")

    # `prices` also holds benchmark series (SPY); _build_labels iterates every
    # column, so leaving them in would emit benchmark rows into the label set.
    from src.utils.benchmarks import strip_benchmarks

    pivot = pivot[strip_benchmarks(pivot.columns)]

    # Resample daily → month-end (last trading day of each month)
    monthly = pivot.resample("ME").last()
    return monthly


# ── 2. Load / download XLK benchmark ─────────────────────────────────────────

def _load_xlk_monthly(
    monthly_dates: pd.DatetimeIndex,
    engine=None,
    allow_download: bool = False,
) -> pd.Series:
    """XLK month-end adj_close, served from the frozen `prices` table.

    Why this changed (Prompt 3b, 2026-09-09)
    ----------------------------------------
    This function used to fetch XLK from yfinance with ``end="2025-01-15"``
    hard-coded and cache it to ``data/raw/xlk_monthly.csv`` — a file that is
    gitignored and appears in no frozen vintage. Since ``label = 1 if a
    ticker's forward return beats XLK's``, that put THE DEPENDENT VARIABLE of
    the project on an unfrozen network fetch: re-running old code could
    silently produce different labels. XLK now lives in `prices`, ingested
    through the same path as the 54 candidates, and is covered by
    scripts/freeze_vintage.py.

    ``allow_download`` defaults to False, so a missing benchmark raises
    instead of quietly reaching for the network. Passing True routes through
    quant_pipeline.load_benchmark, which downloads only behind a loud logged
    warning naming the fallback.
    """
    from src.data.quant_pipeline import load_benchmark

    start = monthly_dates.min().to_period("M").to_timestamp().strftime("%Y-%m-%d")
    end = monthly_dates.max().strftime("%Y-%m-%d")

    daily = load_benchmark(start, end, ticker=BENCHMARK, engine=engine,
                           allow_download=allow_download)
    monthly = daily.resample("ME").last()
    covered = monthly.reindex(monthly_dates).notna().sum()
    logger.info(
        f"  {BENCHMARK} from `prices`: {len(daily)} daily rows → "
        f"{covered}/{len(monthly_dates)} month-ends covered"
    )
    return monthly.reindex(monthly_dates).rename(BENCHMARK)


# ── 3. Compute forward returns ───────────────────────────────────────────────

def _compute_forward_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 1-month forward return for each ticker:
      fwd_ret_t = price_{t+1} / price_t - 1

    Returns a DataFrame with the same shape as input, shifted so that
    row at date t contains the return from t to t+1.
    """
    fwd = monthly_prices.shift(-1) / monthly_prices - 1
    return fwd


# ── 4. Build labels ──────────────────────────────────────────────────────────

def _build_labels(
    ticker_fwd: pd.DataFrame,
    xlk_fwd: pd.Series,
) -> pd.DataFrame:
    """
    Binary label: 1 if ticker's forward return > XLK forward return, else 0.
    Returns a long-format DataFrame: [date, ticker, fwd_return, xlk_return, label].
    """
    # Align XLK to ticker dates
    xlk_aligned = xlk_fwd.reindex(ticker_fwd.index)

    # Excess return = ticker - XLK
    excess = ticker_fwd.sub(xlk_aligned, axis=0)
    labels = (excess > 0).astype(int)

    # Melt to long format
    records = []
    for date in ticker_fwd.index:
        for ticker in ticker_fwd.columns:
            fwd_val = ticker_fwd.loc[date, ticker]
            xlk_val = xlk_aligned.get(date, np.nan)
            lab = labels.loc[date, ticker]

            if pd.isna(fwd_val) or pd.isna(xlk_val):
                continue

            records.append({
                "date": date,
                "ticker": ticker,
                "fwd_return": round(float(fwd_val), 6),
                "xlk_return": round(float(xlk_val), 6),
                "label": int(lab),
            })

    return pd.DataFrame(records)


# ── Public API ────────────────────────────────────────────────────────────────

def build_labeled_dataset(engine=None, start: Optional[str] = None,
                          end: Optional[str] = None) -> pd.DataFrame:
    """
    Build the labeled dataset for the ensemble ML combiner.

    Steps:
      1. Load monthly prices from SQLite
      2. Compute 1-month forward returns per ticker
      3. Download XLK benchmark, compute its forward return
      4. Label: 1 if ticker beats XLK next month, else 0
      5. Merge labels onto the feature matrix
      6. Drop rows with NaN labels (last month)
      7. Save to parquet

    Returns
    -------
    pd.DataFrame — the feature matrix with appended columns:
        fwd_return, xlk_return, label
    """
    engine = _get_engine(engine)

    # ── Load feature matrix ───────────────────────────────────────────
    feat_path = PROCESSED / "ensemble_feature_matrix.parquet"
    if not feat_path.exists():
        logger.error(f"Feature matrix not found: {feat_path}")
        logger.error("Run: python -m src.strategies.ensemble.feature_matrix")
        return pd.DataFrame()

    features = pd.read_parquet(feat_path)
    features["date"] = pd.to_datetime(features["date"])
    logger.info(f"Feature matrix: {features.shape[0]} rows, {features['ticker'].nunique()} tickers")

    # ── Load monthly prices ───────────────────────────────────────────
    logger.info("Loading monthly prices from DB…")
    monthly_prices = _load_monthly_prices(engine, start=start, end=end)
    if monthly_prices.empty:
        logger.error("No price data available")
        return pd.DataFrame()
    logger.info(f"  Monthly prices: {monthly_prices.shape[0]} months × {monthly_prices.shape[1]} tickers")

    # ── XLK benchmark ────────────────────────────────────────────────
    logger.info("Loading XLK benchmark…")
    xlk_monthly = _load_xlk_monthly(monthly_prices.index)
    if xlk_monthly.empty:
        logger.error("Failed to load XLK data")
        return pd.DataFrame()
    logger.info(f"  XLK: {xlk_monthly.dropna().shape[0]} months of data")

    # ── Forward returns ───────────────────────────────────────────────
    logger.info("Computing 1-month forward returns…")
    ticker_fwd = _compute_forward_returns(monthly_prices)
    xlk_fwd = xlk_monthly.pct_change().shift(-1)
    # Recompute XLK forward return same way as tickers
    xlk_prices_monthly = xlk_monthly.dropna()
    xlk_fwd = xlk_prices_monthly.shift(-1) / xlk_prices_monthly - 1

    # ── Build labels ──────────────────────────────────────────────────
    logger.info("Building outperformance labels…")
    labels_df = _build_labels(ticker_fwd, xlk_fwd)
    logger.info(f"  Labels: {labels_df.shape[0]} rows")

    # ── Merge onto feature matrix ─────────────────────────────────────
    logger.info("Merging labels onto feature matrix…")
    labeled = features.merge(
        labels_df[["date", "ticker", "fwd_return", "xlk_return", "label"]],
        on=["date", "ticker"],
        how="left",
    )

    # ── Drop rows with NaN label (last month has no forward return) ───
    rows_before = len(labeled)
    labeled = labeled.dropna(subset=["label"]).copy()
    labeled["label"] = labeled["label"].astype(int)
    rows_dropped = rows_before - len(labeled)
    logger.info(f"  Dropped {rows_dropped} rows with NaN label (last month / missing)")

    # ── Strict temporal ordering ──────────────────────────────────────
    labeled = labeled.sort_values(["date", "ticker"]).reset_index(drop=True)

    # ── Diagnostics ───────────────────────────────────────────────────
    n_rows = len(labeled)
    n_tickers = labeled["ticker"].nunique()
    n_months = labeled["date"].nunique()
    label_balance = labeled["label"].mean() * 100

    logger.info(f"Labeled dataset: {n_rows} rows, {n_tickers} tickers, {n_months} months")
    logger.info(f"  Label balance: {label_balance:.1f}% outperform XLK (class=1)")
    logger.info(f"  Date range: {labeled['date'].min().date()} → {labeled['date'].max().date()}")

    return labeled


# ── Walk-Forward Fold Generator ───────────────────────────────────────────────

def walk_forward_folds(
    df: pd.DataFrame,
    min_train_months: int = MIN_TRAIN_MONTHS,
    purge_months: int = 0,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate walk-forward train/test splits (anchored expanding window).

    Each fold uses all data up to (and including) ``train_end`` as training,
    and the month at index ``i`` as the test set.  With ``purge_months > 0``,
    an embargo period is held out between ``train_end`` and the test month
    to prevent the early-stopping eval set from peeking at labels whose
    1-month forward return overlaps the test period.

    Parameters
    ----------
    df               : labeled dataset (must have 'date' column, sorted by date)
    min_train_months : minimum number of months in the training window
    purge_months     : number of months to embargo between train_end and
                       test_month.  Default 0 preserves existing behavior.
                       AlgoXpert WFA spec recommends purge_months=3 (1 quarter).

    Yields
    ------
    (train_df, test_df) for each prediction month
    """
    df = df.sort_values("date")
    unique_months = sorted(df["date"].unique())

    if len(unique_months) <= min_train_months + purge_months:
        logger.warning(
            f"Only {len(unique_months)} months available, "
            f"need >{min_train_months + purge_months} "
            f"(min_train_months={min_train_months} + purge_months={purge_months}) "
            f"for walk-forward. No folds generated."
        )
        return

    n_folds = len(unique_months) - min_train_months - purge_months
    logger.info(
        f"Walk-forward: {len(unique_months)} months total, "
        f"{min_train_months} month burn-in + {purge_months} month purge → "
        f"{n_folds} folds"
    )

    for i in range(min_train_months + purge_months, len(unique_months)):
        train_end = unique_months[i - 1 - purge_months]
        test_month = unique_months[i]

        train_df = df[df["date"] <= train_end]
        test_df = df[df["date"] == test_month]

        if test_df.empty:
            continue

        yield train_df, test_df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    import argparse
    ap = argparse.ArgumentParser(description="Build the ensemble labeled dataset")
    ap.add_argument("--start", default=None,
                    help="price window start (default: timeline.train_start)")
    ap.add_argument("--end", default=None,
                    help="price window end (default: timeline.test_end). "
                         "Prompt 3b: pass 2026-08-31 for the pre-registered "
                         "Sprint 9B span; TIMELINE is deliberately NOT mutated.")
    args = ap.parse_args()

    df = build_labeled_dataset(start=args.start, end=args.end)

    if df.empty:
        logger.warning("Empty result — parquet NOT saved")
        return

    # Save
    out_path = PROCESSED / "ensemble_labeled.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info(f"Saved → {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")

    # Print summary
    print(f"\n{'═' * 68}")
    print(f"  Ensemble Labeled Dataset")
    print(f"{'═' * 68}")
    print(f"  Shape:         {df.shape}")
    print(f"  Tickers:       {df['ticker'].nunique()}")
    print(f"  Months:        {df['date'].nunique()}")
    print(f"  Date range:    {df['date'].min()} → {df['date'].max()}")
    print(f"  Label balance: {df['label'].mean()*100:.1f}% outperform (class=1)")
    print(f"{'═' * 68}")

    # Forward return stats
    print(f"\n  Forward return stats:")
    print(f"    Ticker mean:  {df['fwd_return'].mean()*100:+.2f}%")
    print(f"    XLK mean:     {df['xlk_return'].mean()*100:+.2f}%")
    print(f"    Excess mean:  {(df['fwd_return'] - df['xlk_return']).mean()*100:+.2f}%")

    # Walk-forward fold summary
    print(f"\n  Walk-forward folds:")
    fold_count = 0
    for train, test in walk_forward_folds(df):
        fold_count += 1
    print(f"    Total folds: {fold_count}")
    if fold_count > 0:
        # Show first and last fold
        folds = list(walk_forward_folds(df))
        first_train, first_test = folds[0]
        last_train, last_test = folds[-1]
        print(f"    First fold:  train {first_train['date'].min().date()} → "
              f"{first_train['date'].max().date()} "
              f"({first_train['date'].nunique()} months), "
              f"test {first_test['date'].iloc[0].date()} "
              f"({len(first_test)} rows)")
        print(f"    Last fold:   train {last_train['date'].min().date()} → "
              f"{last_train['date'].max().date()} "
              f"({last_train['date'].nunique()} months), "
              f"test {last_test['date'].iloc[0].date()} "
              f"({len(last_test)} rows)")

    # Head
    print(f"\n  Head (with label):")
    pd.set_option("display.max_columns", 25)
    pd.set_option("display.width", 220)
    show_cols = ["date", "ticker", "momentum_1m", "fwd_return", "xlk_return", "label"]
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].head(10).to_string(index=False))
    print()


if __name__ == "__main__":
    main()
