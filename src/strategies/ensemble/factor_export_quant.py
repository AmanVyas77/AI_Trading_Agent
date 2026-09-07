"""
Quant Factor Score Exporter
============================
Computes per-ticker per-day factor scores from the SQLite prices table,
resamples to month-end frequency, and saves a tidy long-format parquet.

Factors
-------
  momentum_1m            : 21-day return (skip 5), cross-sectional z-score, clipped [-3,3]
  momentum_3m            : 63-day return (skip 5), cross-sectional z-score, clipped [-3,3]
  momentum_6m            : 126-day return (skip 5), cross-sectional z-score, clipped [-3,3]
  momentum_12m           : 252-day return (skip 5), cross-sectional z-score, clipped [-3,3]
  volume_zscore          : 20-day rolling volume z-score, cross-sectional z-score, clipped [-3,3]
  inv_vol                : inverse 21-day realized vol, cross-sectional z-score, clipped [-3,3]
  timesfm_pred_return_1m : TimesFM 1-month median return forecast (raw return, not z-scored)

Output
------
  data/processed/quant_factor_scores.parquet
  Columns: [date, ticker, momentum_1m, momentum_3m, momentum_6m,
            momentum_12m, volume_zscore, inv_vol, timesfm_pred_return_1m]

CLI
---
  python -m src.strategies.ensemble.factor_export_quant
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

from src.strategies.quant.timesfm_factor import (
    compute_timesfm_predictions,
)

load_dotenv()
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
with open(ROOT / "config" / "settings.yaml") as f:
    CFG = yaml.safe_load(f)

QF_CFG = CFG["quant_factors"]
TIMELINE = CFG["timeline"]
DB_PATH = ROOT / CFG["data"]["paths"]["db"]
DB_URL = f"sqlite:///{DB_PATH}"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Factor parameters from settings.yaml
LOOKBACKS = QF_CFG["momentum"]["lookbacks_days"]   # [21, 63, 126, 252]
SKIP_DAYS = QF_CFG["momentum"]["skip_days"]        # 5
VOL_WINDOW = QF_CFG["volume"]["rolling_window"]    # 20
IVOL_WINDOW = QF_CFG["volatility"]["window"]       # 21
ZSCORE_CAP = 3.0

# Map each lookback to its label
_LOOKBACK_LABELS = {21: "momentum_1m", 63: "momentum_3m",
                    126: "momentum_6m", 252: "momentum_12m"}


# ── DB Access ─────────────────────────────────────────────────────────────────

def _get_engine():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(DB_URL, echo=False)


def _load_prices_wide(
    tickers: Optional[list[str]],
    start: str,
    end: str,
    engine,
) -> pd.DataFrame:
    """Load adj_close prices into a wide DataFrame (date × ticker)."""
    conditions = ["date >= :start", "date <= :end"]
    params: dict = {"start": start, "end": end}

    if tickers:
        ph = ",".join(f":t{i}" for i in range(len(tickers)))
        conditions.append(f"ticker IN ({ph})")
        params.update({f"t{i}": t for i, t in enumerate(tickers)})

    where = "WHERE " + " AND ".join(conditions)
    sql = f"SELECT ticker, date, adj_close FROM prices {where} ORDER BY date"

    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, params=params)

    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="ticker", values="adj_close")
    return pivot


def _load_volumes_wide(
    tickers: Optional[list[str]],
    start: str,
    end: str,
    engine,
) -> pd.DataFrame:
    """Load daily volume into a wide DataFrame (date × ticker)."""
    conditions = ["date >= :start", "date <= :end"]
    params: dict = {"start": start, "end": end}

    if tickers:
        ph = ",".join(f":t{i}" for i in range(len(tickers)))
        conditions.append(f"ticker IN ({ph})")
        params.update({f"t{i}": t for i, t in enumerate(tickers)})

    where = "WHERE " + " AND ".join(conditions)
    sql = f"SELECT ticker, date, volume FROM prices {where} ORDER BY date"

    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, params=params)

    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="ticker", values="volume")
    return pivot


# ── Cross-sectional z-score helper ────────────────────────────────────────────

def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional z-score each row (across tickers for a given date).
    Returns values clipped to [-ZSCORE_CAP, ZSCORE_CAP].
    """
    mu = df.mean(axis=1)
    sigma = df.std(axis=1).replace(0, np.nan)
    z = df.sub(mu, axis=0).div(sigma, axis=0)
    return z.clip(-ZSCORE_CAP, ZSCORE_CAP)


# ── Factor Computations ──────────────────────────────────────────────────────

def _compute_momentum_factors(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Compute 4 momentum factors: 1M, 3M, 6M, 12M.
    Each = return from (lb+skip) days ago to (skip) days ago, then
    cross-sectional z-score, clipped to [-3, 3].
    """
    factors = {}
    for lb in LOOKBACKS:
        label = _LOOKBACK_LABELS[lb]
        ret = prices.shift(SKIP_DAYS) / prices.shift(lb + SKIP_DAYS) - 1
        factors[label] = _cs_zscore(ret)
    return factors


def _compute_volume_zscore(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
) -> pd.DataFrame:
    """
    20-day rolling volume z-score, then cross-sectional z-score, clipped.
    If volume data is unavailable, returns a zero-filled DataFrame.
    """
    if volumes.empty or volumes.shape[0] < VOL_WINDOW:
        logger.warning("Volume data unavailable — filling volume_zscore with 0.0")
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # Align volumes to prices
    volumes = volumes.reindex(index=prices.index, columns=prices.columns)

    vol_mean = volumes.rolling(VOL_WINDOW).mean()
    vol_std = volumes.rolling(VOL_WINDOW).std().replace(0, np.nan)
    vol_z = (volumes - vol_mean) / vol_std

    return _cs_zscore(vol_z)


def _compute_inv_vol(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Inverse 21-day realized volatility, cross-sectional z-scored, clipped.
    Lower-vol stocks score higher.
    """
    log_ret = np.log(prices / prices.shift(1))
    rvol = log_ret.rolling(IVOL_WINDOW).std() * np.sqrt(252)
    inv_vol = 1.0 / rvol.replace(0, np.nan)
    return _cs_zscore(inv_vol)


# ── Wide → Long Conversion ────────────────────────────────────────────────────

def _wide_to_long(factor_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple wide (date × ticker) factor DataFrames into a single
    tidy long DataFrame with columns:
      [date, ticker, momentum_1m, momentum_3m, …, volume_zscore, inv_vol]
    """
    pieces = []
    for name, df in factor_dfs.items():
        melted = df.reset_index().melt(
            id_vars="date", var_name="ticker", value_name=name,
        )
        pieces.append(melted.set_index(["date", "ticker"]))

    combined = pieces[0]
    for p in pieces[1:]:
        combined = combined.join(p, how="outer")

    return combined.reset_index()


# ── Public API ────────────────────────────────────────────────────────────────

def build_quant_scores(
    tickers: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    engine=None,
) -> pd.DataFrame:
    """
    Compute all quant factor scores, resample to month-end, and return a
    tidy long-format DataFrame.

    Parameters
    ----------
    tickers : list of ticker symbols; defaults to all tickers in the DB
    start   : start date string (YYYY-MM-DD); defaults to timeline.train_start
    end     : end date string (YYYY-MM-DD); defaults to timeline.test_end
    engine  : SQLAlchemy engine; created from settings.yaml if not provided

    Returns
    -------
    pd.DataFrame with columns:
        [date, ticker, momentum_1m, momentum_3m, momentum_6m,
         momentum_12m, volume_zscore, inv_vol]
    """
    if engine is None:
        engine = _get_engine()
    if start is None:
        start = TIMELINE["train_start"]
    if end is None:
        end = TIMELINE["test_end"]
    if tickers is None:
        # Previously this fell through as None → "every ticker in `prices`".
        # `prices` now also holds benchmark series (SPY), and every factor here
        # is cross-sectionally z-scored, so an extra column would shift the
        # score of every candidate. Pin to the canonical universe instead.
        from src.utils.benchmarks import universe_tickers

        tickers = list(universe_tickers())
        logger.info(f"  tickers defaulted to universe.csv ({len(tickers)})")

    # ── 1. Load data ──────────────────────────────────────────────────
    logger.info(f"Loading prices [{start} → {end}]…")
    prices = _load_prices_wide(tickers, start, end, engine)

    if prices.empty:
        logger.error("No price data — returning empty DataFrame")
        return pd.DataFrame(columns=[
            "date", "ticker", "momentum_1m", "momentum_3m",
            "momentum_6m", "momentum_12m", "volume_zscore", "inv_vol",
            "timesfm_pred_return_1m",
        ])

    # Drop tickers with < 252 trading days of history
    min_obs = max(LOOKBACKS) + SKIP_DAYS + 1  # need full 12M lookback
    prices = prices.loc[:, prices.count() >= min_obs]
    logger.info(f"  Tickers with sufficient history: {prices.shape[1]}")

    logger.info("Loading volume data…")
    volumes = _load_volumes_wide(tickers, start, end, engine)

    # ── 2. Compute factor scores (daily, wide) ───────────────────────
    logger.info("Computing momentum factors (4 windows)…")
    factor_dfs = _compute_momentum_factors(prices)

    logger.info("Computing volume z-score…")
    factor_dfs["volume_zscore"] = _compute_volume_zscore(prices, volumes)

    logger.info("Computing inverse volatility…")
    factor_dfs["inv_vol"] = _compute_inv_vol(prices)

    # ── 3. Resample to month-end ─────────────────────────────────────
    logger.info("Resampling to month-end frequency…")
    monthly_factors = {}
    for name, df in factor_dfs.items():
        monthly_factors[name] = df.resample("ME").last()

    # ── 3.5 TimesFM 1-month return forecast (per month-end) ──────────
    logger.info("Computing TimesFM 1-month return forecasts…")
    month_ends = next(iter(monthly_factors.values())).index
    try:
        timesfm_df = compute_timesfm_predictions(prices, month_ends)
    except Exception as e:
        logger.error(
            "TimesFM factor failed (%s) — column will be all NaN", e,
        )
        timesfm_df = pd.DataFrame(
            columns=["ticker", "date", "timesfm_pred_return_1m"],
        )

    # ── 4. Convert to tidy long format ───────────────────────────────
    logger.info("Converting to long format…")
    result = _wide_to_long(monthly_factors)

    # Merge TimesFM forecasts (already long-format on [date, ticker])
    if not timesfm_df.empty:
        result = result.merge(
            timesfm_df, on=["date", "ticker"], how="left",
        )
    else:
        result["timesfm_pred_return_1m"] = np.nan

    # Drop rows where ALL factor columns are NaN
    factor_cols = [c for c in result.columns if c not in ("date", "ticker")]
    result = result.dropna(subset=factor_cols, how="all")

    # Sort for deterministic output
    result = result.sort_values(["date", "ticker"]).reset_index(drop=True)

    logger.info(
        f"Output: {result.shape[0]} rows, "
        f"{result['ticker'].nunique()} tickers, "
        f"{result['date'].nunique()} months"
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
        description="Export quant factor scores to parquet",
    )
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Override universe tickers")
    parser.add_argument("--start", default=None,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None,
                        help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    df = build_quant_scores(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
    )

    if df.empty:
        logger.warning("No data produced — parquet NOT saved")
        return

    # Save parquet
    out_path = OUTPUT_DIR / "quant_factor_scores.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info(f"Saved → {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")

    # Print summary
    print(f"\n{'═' * 60}")
    print(f"  Quant Factor Scores Export")
    print(f"{'═' * 60}")
    print(f"  Shape:    {df.shape}")
    print(f"  Tickers:  {df['ticker'].nunique()}")
    print(f"  Months:   {df['date'].nunique()}")
    print(f"  Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"{'═' * 60}")
    print(f"\n  Columns: {list(df.columns)}")
    print(f"\n  Factor statistics:")
    factor_cols = [c for c in df.columns if c not in ("date", "ticker")]
    print(df[factor_cols].describe().round(4).to_string())
    print(f"\n  Head:")
    print(df.head(10).to_string(index=False))
    print()


if __name__ == "__main__":
    main()
