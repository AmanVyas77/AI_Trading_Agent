"""
Combined Feature Matrix Builder
=================================
Merges quant factor scores (monthly), fundamental factor scores (quarterly →
forward-filled to monthly), and macro regime indicators into a single tidy
DataFrame for the ensemble ML combiner.

Input Files
-----------
  data/processed/quant_factor_scores.parquet        (monthly)
  data/processed/fundamental_factor_scores.parquet  (quarterly)
  SQLite: macro_series table                        (daily → resampled monthly)

Output
------
  data/processed/ensemble_feature_matrix.parquet
  Columns: [date, ticker,
            momentum_1m, momentum_3m, momentum_6m, momentum_12m,
            volume_zscore, inv_vol,
            gross_profitability, fcf_yield, revenue_acceleration,
            deferred_revenue_yoy, rd_intensity,
            sue_score, eps_revision_1m, eps_revision_3m,
            piotroski_f, qmj_safety, qmj_payout,
            finbert_score, lm_sentiment_score,
            vix, yield_spread_10y2y, fed_funds_rate, cpi]

CLI
---
  python -m src.strategies.ensemble.feature_matrix
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
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── Column definitions ───────────────────────────────────────────────────────

QUANT_COLS = [
    "momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m",
    "volume_zscore", "inv_vol",
    "timesfm_pred_return_1m",
]

FUND_COLS = [
    "gross_profitability", "fcf_yield", "revenue_acceleration",
    "deferred_revenue_yoy", "rd_intensity",
    "sue_score", "eps_revision_1m", "eps_revision_3m",
    "piotroski_f", "qmj_safety", "qmj_payout",
    "finbert_score", "lm_sentiment_score",
]

MACRO_COLS = ["vix", "yield_spread_10y2y", "fed_funds_rate", "cpi"]

# Sprint 9B R3/R3-A. Never NaN by construction: a ticker-month with no
# articles falls back to that month's cross-sectional mean via the shrinkage
# branch, so adding this column can only LOWER a row's NaN fraction and can
# never change which rows survive the NAN_DROP_THRESHOLD filter below.
NEWS_COLS = ["news_sentiment"]

# All factor columns (quant + fundamental + news, *excluding* macro for NaN-drop calc)
FACTOR_COLS = QUANT_COLS + FUND_COLS + NEWS_COLS

# Final output column order
OUTPUT_COLS = ["date", "ticker"] + QUANT_COLS + FUND_COLS + NEWS_COLS + MACRO_COLS

# Macro series that get z-scored over a rolling window
MACRO_ZSCORE_SERIES = {"vix", "cpi"}
MACRO_ZSCORE_WINDOW = 36  # months

# NaN threshold: drop rows where >50% of factor columns are NaN
NAN_DROP_THRESHOLD = 0.50


# ── DB helper ─────────────────────────────────────────────────────────────────

def _get_engine(engine=None):
    if engine is not None:
        return engine
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(DB_URL, echo=False)


# ── 1. Load quant scores ─────────────────────────────────────────────────────

def _load_quant_scores() -> pd.DataFrame:
    """Load monthly quant factor scores from parquet."""
    path = PROCESSED / "quant_factor_scores.parquet"
    if not path.exists():
        logger.error(f"Quant scores not found: {path}")
        return pd.DataFrame(columns=["date", "ticker"] + QUANT_COLS)

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"  Quant scores: {df.shape[0]} rows, {df['ticker'].nunique()} tickers")
    return df


def _load_news_scores(start: Optional[str] = None,
                      end: Optional[str] = None) -> pd.DataFrame:
    """Load the monthly news-sentiment factor (Sprint 9B R3/R3-A).

    Computed live from `news_sentiment_scores` rather than read from parquet,
    so the point-in-time filters and the partial-month drop apply to whatever
    window the caller asked for. AV-only per REV 2; see factor_export_news.
    """
    from src.strategies.ensemble.factor_export_news import load_news_sentiment

    df = load_news_sentiment(start or TIMELINE["train_start"],
                             end or TIMELINE["test_end"])
    if df.empty:
        logger.warning("News scores empty")
        return pd.DataFrame(columns=["date", "ticker"] + NEWS_COLS)
    logger.info(f"  News scores: {df.shape[0]} rows, {df['ticker'].nunique()} tickers")
    return df


# ── 2. Load & forward-fill fundamental scores ────────────────────────────────

def _load_fundamental_scores(end: Optional[str] = None) -> pd.DataFrame:
    """
    Load quarterly fundamental factor scores and forward-fill to monthly.

    Each quarter-end value propagates forward until the next quarter-end
    value appears for that ticker.

    `end` extends the monthly scaffold past the last quarter-end so the
    intra-quarter months of the current quarter are carried forward too.
    Without it the scaffold stops at the last quarter-end and those months
    come back all-NaN, then get dropped by the NaN threshold in
    build_feature_matrix().
    """
    path = PROCESSED / "fundamental_factor_scores.parquet"
    if not path.exists():
        logger.error(f"Fundamental scores not found: {path}")
        return pd.DataFrame(columns=["date", "ticker"] + FUND_COLS)

    df = pd.read_parquet(path)
    df["quarter_end"] = pd.to_datetime(df["quarter_end"])
    logger.info(
        f"  Fundamental scores (raw): {df.shape[0]} rows, "
        f"{df['ticker'].nunique()} tickers, "
        f"{df['quarter_end'].nunique()} quarters"
    )

    # Build a monthly date scaffold covering the full range
    scaffold_end = df["quarter_end"].max()
    if end is not None:
        scaffold_end = max(scaffold_end, pd.Timestamp(end))
    all_months = pd.date_range(
        start=df["quarter_end"].min(),
        end=scaffold_end,
        freq="ME",
    )
    all_tickers = sorted(df["ticker"].unique())

    # Create scaffold: every (ticker, month-end) combination
    scaffold = pd.MultiIndex.from_product(
        [all_tickers, all_months], names=["ticker", "date"]
    ).to_frame(index=False)

    # Merge quarterly data onto scaffold (quarter-end dates align with ME)
    merged = scaffold.merge(
        df.rename(columns={"quarter_end": "date"}),
        on=["ticker", "date"],
        how="left",
    )

    # Forward-fill within each ticker (quarterly → monthly)
    merged = merged.sort_values(["ticker", "date"])
    merged[FUND_COLS] = (
        merged.groupby("ticker")[FUND_COLS]
        .apply(lambda g: g.ffill())
        .reset_index(drop=True)
    )

    logger.info(
        f"  Fundamental scores (monthly): {merged.shape[0]} rows after forward-fill"
    )
    return merged


# ── 3. Load & process macro series ────────────────────────────────────────────

def _load_macro(engine) -> pd.DataFrame:
    """
    Load macro series from SQLite, resample to month-end, and normalize:
      - VIX, CPI: z-score over rolling 36-month window
      - yield_spread_10y2y, fed_funds_rate: raw values (already interpretable)

    Returns a DataFrame indexed by date with columns = MACRO_COLS.
    """
    if not inspect(engine).get_table_names().__contains__("macro_series"):
        logger.warning("macro_series table not found — returning empty macro")
        return pd.DataFrame(columns=["date"] + MACRO_COLS)

    target = MACRO_COLS
    ph = ",".join(f":s{i}" for i in range(len(target)))
    params = {f"s{i}": s for i, s in enumerate(target)}

    sql = f"""
        SELECT series_name, date, value
        FROM macro_series
        WHERE series_name IN ({ph})
        ORDER BY date
    """
    with engine.connect() as conn:
        raw = pd.read_sql_query(text(sql), conn, params=params)

    if raw.empty:
        logger.warning("No macro data found")
        return pd.DataFrame(columns=["date"] + MACRO_COLS)

    raw["date"] = pd.to_datetime(raw["date"])

    # Pivot to wide: date × series_name
    wide = raw.pivot_table(index="date", columns="series_name", values="value", aggfunc="last")

    # Resample daily → month-end (last observation)
    monthly = wide.resample("ME").last()
    monthly = monthly.ffill()  # fill gaps in macro reporting

    # Z-score VIX and CPI over rolling 36-month window
    for col in MACRO_ZSCORE_SERIES:
        if col in monthly.columns:
            roll_mean = monthly[col].rolling(MACRO_ZSCORE_WINDOW, min_periods=12).mean()
            roll_std = monthly[col].rolling(MACRO_ZSCORE_WINDOW, min_periods=12).std()
            monthly[col] = (monthly[col] - roll_mean) / roll_std.replace(0, np.nan)

    # Keep only target columns (in case extras exist)
    present = [c for c in MACRO_COLS if c in monthly.columns]
    monthly = monthly[present]

    monthly = monthly.reset_index().rename(columns={"index": "date"})
    # Ensure the column is named "date"
    if monthly.columns[0] != "date":
        monthly = monthly.rename(columns={monthly.columns[0]: "date"})

    logger.info(f"  Macro series: {monthly.shape[0]} months, columns={present}")
    return monthly


# ── 4. Merge everything ──────────────────────────────────────────────────────

def _merge_all(
    quant: pd.DataFrame,
    fund: pd.DataFrame,
    macro: pd.DataFrame,
    news: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge quant + fundamental + news + macro on [date, ticker].
    Macro is broadcast to all tickers (same value per date).
    """
    # Start with quant as the base (monthly frequency, all tickers)
    merged = quant.copy()

    # Merge fundamental scores
    if not fund.empty:
        merged = merged.merge(fund, on=["date", "ticker"], how="left")
        logger.info(f"  After quant+fund merge: {merged.shape[0]} rows")

    # Merge news sentiment (Sprint 9B R3/R3-A)
    if news is not None and not news.empty:
        merged = merged.merge(news, on=["date", "ticker"], how="left")
        logger.info(f"  After +news merge: {merged.shape[0]} rows")

    # Merge macro (date-level join, same values for all tickers)
    if not macro.empty:
        merged = merged.merge(macro, on="date", how="left")
        logger.info(f"  After macro merge: {merged.shape[0]} rows")

    return merged


# ── Public API ────────────────────────────────────────────────────────────────

def build_feature_matrix(
    start: Optional[str] = None,
    end: Optional[str] = None,
    engine=None,
) -> pd.DataFrame:
    """
    Build the combined ensemble feature matrix.

    Merges quant scores (monthly), fundamental scores (quarterly →
    forward-filled to monthly), and macro indicators into a single
    tidy DataFrame.

    Parameters
    ----------
    start  : start date (YYYY-MM-DD); defaults to timeline.train_start
    end    : end date (YYYY-MM-DD); defaults to timeline.test_end
    engine : SQLAlchemy engine; created from settings.yaml if not provided

    Returns
    -------
    pd.DataFrame with columns = OUTPUT_COLS
    """
    engine = _get_engine(engine)
    start = start or TIMELINE["train_start"]
    end = end or TIMELINE["test_end"]

    logger.info(f"Building ensemble feature matrix [{start} → {end}]")

    # ── Load components ───────────────────────────────────────────────
    quant = _load_quant_scores()
    fund = _load_fundamental_scores(end)
    macro = _load_macro(engine)
    news = _load_news_scores(start, end)

    # ── Merge ─────────────────────────────────────────────────────────
    logger.info("Merging quant + fundamental + news + macro…")
    matrix = _merge_all(quant, fund, macro, news)

    # ── Filter to date range ──────────────────────────────────────────
    matrix = matrix[
        (matrix["date"] >= pd.Timestamp(start))
        & (matrix["date"] <= pd.Timestamp(end))
    ]

    # ── Drop rows where >50% of factor columns are NaN ────────────────
    n_factor = len(FACTOR_COLS)
    available_factors = [c for c in FACTOR_COLS if c in matrix.columns]
    nan_count = matrix[available_factors].isna().sum(axis=1)
    nan_pct = nan_count / len(available_factors)
    rows_before = len(matrix)
    matrix = matrix[nan_pct <= NAN_DROP_THRESHOLD].copy()
    rows_dropped = rows_before - len(matrix)
    if rows_dropped > 0:
        logger.info(f"  Dropped {rows_dropped} rows with >{NAN_DROP_THRESHOLD:.0%} NaN factors")

    # ── Ensure all output columns exist ───────────────────────────────
    for col in OUTPUT_COLS:
        if col not in matrix.columns:
            matrix[col] = np.nan

    # ── Final ordering and sort ───────────────────────────────────────
    result = matrix[OUTPUT_COLS].sort_values(["date", "ticker"]).reset_index(drop=True)

    # ── Diagnostics ───────────────────────────────────────────────────
    n_rows = len(result)
    n_tickers = result["ticker"].nunique()
    n_months = result["date"].nunique()

    logger.info(f"Final matrix: {n_rows} rows, {n_tickers} tickers, {n_months} months")

    if n_rows > 0:
        nan_pct_per_col = result[QUANT_COLS + FUND_COLS + NEWS_COLS + MACRO_COLS].isna().mean() * 100
        logger.info("  NaN % per column:")
        for col, pct in nan_pct_per_col.items():
            status = "⚠" if pct > 30 else "✓"
            logger.info(f"    {status} {col:<28s}  {pct:5.1f}%")

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
        description="Build combined ensemble feature matrix",
    )
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    df = build_feature_matrix(start=args.start, end=args.end)

    if df.empty:
        logger.warning("Empty result — parquet NOT saved")
        return

    # Save
    out_path = PROCESSED / "ensemble_feature_matrix.parquet"
    # Drop columns that are 100% NaN — no signal, just noise for XGBoost
    df = df.dropna(axis=1, how="all")
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info(f"Saved → {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")

    # Print summary
    print(f"\n{'═' * 68}")
    print(f"  Ensemble Feature Matrix")
    print(f"{'═' * 68}")
    print(f"  Shape:      {df.shape}")
    print(f"  Tickers:    {df['ticker'].nunique()}")
    print(f"  Months:     {df['date'].nunique()}")
    print(f"  Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"{'═' * 68}")

    # NaN summary
    all_factors = QUANT_COLS + FUND_COLS + NEWS_COLS + MACRO_COLS
    print(f"\n  NaN % per column:")
    for col in all_factors:
        if col in df.columns:
            pct = df[col].isna().mean() * 100
            bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
            print(f"    {col:<28s}  {bar}  {pct:5.1f}%")

    # Descriptive stats — only include columns that exist after dropping 100% NaN
    all_factors = [c for c in all_factors if c in df.columns]
    print(f"\n  Factor statistics:")
    print(df[all_factors].describe().round(4).to_string())

    print(f"\n  Head:")
    pd.set_option("display.max_columns", 22)
    pd.set_option("display.width", 200)
    print(df.head(8).to_string(index=False))
    print()


if __name__ == "__main__":
    main()
