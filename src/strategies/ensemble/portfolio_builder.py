"""
Ensemble Portfolio Constructor
================================
Converts ensemble outperformance scores into a tradeable daily weight matrix.
Selects top-N tickers each month (above score threshold), assigns equal weights,
and forward-fills to daily frequency for backtesting.

Portfolio Rules
---------------
  - top_n: 20
  - weighting: equal-weight (1/N)
  - rebalance: monthly (on each date in ensemble_scores)
  - min_score_threshold: 0.52 (model confidence > 52%)
  - long_only: true

Output
------
  data/processed/ensemble_weights.parquet
  Wide format: daily date index × ticker columns, values = portfolio weight

CLI
---
  python -m src.strategies.ensemble.portfolio_builder
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

# Portfolio construction parameters
TOP_N = 20
MIN_SCORE = 0.52
REBALANCE_FREQ = "monthly"


# ── DB helper ─────────────────────────────────────────────────────────────────

def _get_engine(engine=None):
    if engine is not None:
        return engine
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(DB_URL, echo=False)


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_scores(path: Path = None) -> pd.DataFrame:
    if path is None:
        path = PROCESSED / "ensemble_scores.parquet"
    if not path.exists():
        logger.error(f"Scores not found: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_prices_wide(engine) -> pd.DataFrame:
    """Load daily adj_close from SQLite → wide (date × ticker)."""
    if "prices" not in inspect(engine).get_table_names():
        logger.error("prices table not found")
        return pd.DataFrame()

    sql = """
        SELECT ticker, date, adj_close
        FROM prices
        WHERE date >= :start AND date <= :end
        ORDER BY date
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(
            text(sql), conn,
            params={"start": TIMELINE["train_start"], "end": TIMELINE["test_end"]},
        )

    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="ticker", values="adj_close")


# ── Portfolio construction ────────────────────────────────────────────────────

def _select_monthly_holdings(
    scores_df: pd.DataFrame,
    prices: pd.DataFrame,
    top_n: int = TOP_N,
    min_score: float = MIN_SCORE,
) -> dict[pd.Timestamp, dict[str, float]]:
    """
    For each rebalance date, select tickers and assign equal weights.

    Returns dict: {rebalance_date: {ticker: weight, ...}}
    """
    holdings = {}
    rebalance_dates = sorted(scores_df["date"].unique())

    for date in rebalance_dates:
        month_scores = scores_df[scores_df["date"] == date].copy()

        # Step (a): filter by score threshold
        above_threshold = month_scores[month_scores["ensemble_score"] > min_score]
        n_above = len(above_threshold)

        # Step (b): select top-N by score
        selected = above_threshold.nlargest(min(top_n, len(above_threshold)), "ensemble_score")

        # Step (c): intersect with tickers that have price data
        if not prices.empty:
            # Find the nearest available price date (within 5 business days)
            available_dates = prices.index
            closest_idx = available_dates.searchsorted(date)
            if closest_idx >= len(available_dates):
                closest_idx = len(available_dates) - 1
            price_date = available_dates[closest_idx]

            # Check which tickers have prices on that date
            valid_tickers = prices.loc[price_date].dropna().index.tolist()
            before_filter = len(selected)
            selected = selected[selected["ticker"].isin(valid_tickers)]
            n_dropped = before_filter - len(selected)
        else:
            n_dropped = 0

        # Step (d): assign equal weight
        n_selected = len(selected)
        if n_selected > 0:
            weight = 1.0 / n_selected
            ticker_weights = {row["ticker"]: weight for _, row in selected.iterrows()}
        else:
            ticker_weights = {}

        holdings[date] = ticker_weights

        # Step (e): log
        top5 = selected.head(5)
        top5_str = ", ".join(
            f"{r['ticker']}({r['ensemble_score']:.3f})"
            for _, r in top5.iterrows()
        )

        log_level = logging.INFO if n_selected > 0 else logging.WARNING
        msg = (
            f"  {pd.Timestamp(date).date()} │ "
            f"above {min_score}: {n_above:2d} │ "
            f"selected: {n_selected:2d}/{top_n} │ "
            f"dropped (no px): {n_dropped}"
        )
        if n_selected > 0:
            msg += f" │ top5: {top5_str}"
        logger.log(log_level, msg)

    return holdings


def _build_daily_weights(
    holdings: dict[pd.Timestamp, dict[str, float]],
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert monthly holdings to daily weight matrix by forward-filling.
    Weights persist until the next rebalance date.
    """
    all_tickers = set()
    for tw in holdings.values():
        all_tickers.update(tw.keys())
    all_tickers = sorted(all_tickers)

    if not all_tickers:
        logger.warning("No tickers selected across any month")
        return pd.DataFrame()

    # Build monthly weight matrix at rebalance dates
    rebalance_dates = sorted(holdings.keys())
    monthly_weights = pd.DataFrame(
        0.0, index=rebalance_dates, columns=all_tickers
    )
    for date, tw in holdings.items():
        for ticker, weight in tw.items():
            monthly_weights.loc[date, ticker] = weight

    # Reindex to daily using the price index, forward-fill
    first_rebal = rebalance_dates[0]
    last_rebal = rebalance_dates[-1]

    # Get daily trading dates from prices within our range
    daily_index = prices.index[
        (prices.index >= first_rebal) & (prices.index <= prices.index[-1])
    ]

    daily_weights = monthly_weights.reindex(daily_index).ffill().fillna(0.0)

    logger.info(
        f"Daily weight matrix: {daily_weights.shape[0]} days × "
        f"{daily_weights.shape[1]} tickers"
    )

    return daily_weights


# ── Public API ────────────────────────────────────────────────────────────────

def build_portfolio_weights(
    scores_df: Optional[pd.DataFrame] = None,
    prices: Optional[pd.DataFrame] = None,
    engine=None,
) -> pd.DataFrame:
    """
    Build daily portfolio weight matrix from ensemble scores.

    Parameters
    ----------
    scores_df : ensemble scores; loads from parquet if None
    prices    : wide (date × ticker) prices; loads from DB if None
    engine    : SQLAlchemy engine; created from settings.yaml if None

    Returns
    -------
    pd.DataFrame — daily weight matrix (date index × ticker columns)
    """
    engine = _get_engine(engine)

    if scores_df is None:
        scores_df = _load_scores()
    if scores_df.empty:
        logger.error("No scores available")
        return pd.DataFrame()

    if prices is None:
        logger.info("Loading daily prices from DB…")
        prices = _load_prices_wide(engine)
    if prices.empty:
        logger.error("No price data available")
        return pd.DataFrame()

    logger.info(
        f"Building portfolio weights: {scores_df['date'].nunique()} months, "
        f"{scores_df['ticker'].nunique()} tickers"
    )
    logger.info(f"  Score threshold: {MIN_SCORE}, top_n: {TOP_N}")

    # Select holdings per month
    holdings = _select_monthly_holdings(scores_df, prices)

    # Summary stats
    n_months_with_holdings = sum(1 for tw in holdings.values() if tw)
    avg_holdings = np.mean([len(tw) for tw in holdings.values() if tw]) if n_months_with_holdings > 0 else 0

    logger.info(
        f"  Months with holdings: {n_months_with_holdings}/{len(holdings)}, "
        f"avg holdings per month: {avg_holdings:.1f}"
    )

    # Build daily weights
    daily_weights = _build_daily_weights(holdings, prices)

    return daily_weights


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    weights = build_portfolio_weights()

    if weights.empty:
        logger.warning("No weights generated — parquet NOT saved")
        return

    # Save
    out_path = PROCESSED / "ensemble_weights.parquet"
    weights.to_parquet(out_path, engine="pyarrow")
    logger.info(f"Saved → {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")

    # Summary
    print(f"\n{'═' * 72}")
    print(f"  Ensemble Portfolio Weights")
    print(f"{'═' * 72}")
    print(f"  Shape:        {weights.shape[0]} days × {weights.shape[1]} tickers")
    print(f"  Date range:   {weights.index.min().date()} → {weights.index.max().date()}")
    print(f"  Unique tickers ever held: {(weights > 0).any().sum()}")
    print(f"  Score threshold: {MIN_SCORE}")
    print(f"  Max positions:   {TOP_N}")
    print(f"{'═' * 72}")

    # Monthly rebalance summary
    monthly_count = weights.resample("ME").first().gt(0).sum(axis=1)
    print(f"\n  Positions per rebalance month:")
    print(f"    Mean: {monthly_count.mean():.1f}")
    print(f"    Min:  {monthly_count.min()}")
    print(f"    Max:  {monthly_count.max()}")

    # Concentration
    print(f"\n  Weight concentration (daily):")
    daily_nonzero = weights.gt(0).sum(axis=1)
    print(f"    Avg active positions: {daily_nonzero.mean():.1f}")
    print(f"    Days with 0 positions: {(daily_nonzero == 0).sum()}")

    # Top tickers by frequency of inclusion
    inclusion = weights.gt(0).sum()
    top_tickers = inclusion.nlargest(10)
    print(f"\n  Top 10 most frequently held tickers:")
    total_days = len(weights)
    for ticker, days in top_tickers.items():
        bar = "█" * int(days / total_days * 30)
        print(f"    {ticker:<6s}  {bar}  {days:4d} days ({days/total_days*100:.0f}%)")

    print()


if __name__ == "__main__":
    main()
