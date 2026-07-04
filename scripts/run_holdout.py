"""
Sprint 7 Prompt 3 Part C: true-holdout run wrapper.

Scores 2025-01→2026-06, builds monthly portfolio weights (with the
historical regime gate applied automatically inside build_portfolio_weights),
and runs a vectorbt backtest over that window plus a SPY benchmark.

The clamped internal loaders in portfolio_builder / backtest are bypassed
by INJECTION (scores_df=, prices=) rather than editing those modules.

Output: backtests/results/holdout_2025_26_equity.csv (columns:
strategy, benchmark). Prompt 4 owns the verdict computation — this
script writes equity only.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import vectorbt as vbt
import yaml
import yfinance as yf
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

with open(ROOT / "config" / "settings.yaml") as f:
    CFG = yaml.safe_load(f)

BT_CFG = CFG["backtest"]
DB_PATH = ROOT / CFG["data"]["paths"]["db"]
DB_URL = f"sqlite:///{DB_PATH}"

RESULTS_DIR = ROOT / "backtests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_START = "2025-01-01"
HOLDOUT_END = "2026-06-30"

logger = logging.getLogger(__name__)


def _load_prices_range(start: str, end: str) -> pd.DataFrame:
    """Wide (date × ticker) adj_close from the DB — no clamped defaults."""
    engine = create_engine(DB_URL, echo=False)
    with engine.connect() as conn:
        df = pd.read_sql_query(
            text(
                "SELECT ticker, date, adj_close FROM prices "
                "WHERE date >= :start AND date <= :end ORDER BY date"
            ),
            conn,
            params={"start": start, "end": end},
        )
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="ticker", values="adj_close")


def _download_spy(start: str, end: str) -> pd.Series:
    logger.info("Downloading SPY benchmark via yfinance …")
    raw = yf.download(
        "SPY",
        start=start,
        end=(pd.Timestamp(end) + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw["Close"].rename("SPY")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from src.live.scorer import score_months
    from src.strategies.ensemble.portfolio_builder import build_portfolio_weights

    # ── 1. Score months with the frozen model ──────────────────────────
    logger.info("Scoring holdout window %s → %s", HOLDOUT_START, HOLDOUT_END)
    scores_df = score_months(HOLDOUT_START, HOLDOUT_END)

    # ── 2. Load daily prices for the holdout window ───────────────────
    prices = _load_prices_range(HOLDOUT_START, HOLDOUT_END)
    logger.info(
        "Loaded prices: %s → %s, %d days × %d tickers",
        prices.index.min().date(), prices.index.max().date(),
        prices.shape[0], prices.shape[1],
    )

    # ── 3. Build monthly portfolio weights (regime gate applied
    #      automatically inside build_portfolio_weights) ────────────────
    weights_df = build_portfolio_weights(scores_df=scores_df, prices=prices)
    if weights_df.empty:
        logger.error("build_portfolio_weights returned empty — aborting")
        return

    logger.info(
        "Weights: %s → %s, %d days × %d tickers",
        weights_df.index.min().date(), weights_df.index.max().date(),
        weights_df.shape[0], weights_df.shape[1],
    )

    # ── 4. VectorBT backtest over the holdout window only ─────────────
    common_dates = prices.index.intersection(weights_df.index)
    common_tickers = [t for t in weights_df.columns if t in prices.columns]
    px = prices.loc[common_dates, common_tickers]
    wt = weights_df.loc[common_dates, common_tickers].fillna(0.0)
    logger.info(
        "Aligned backtest window: %s → %s, %d days × %d tickers",
        common_dates.min().date(), common_dates.max().date(),
        len(common_dates), len(common_tickers),
    )

    strategy_pf = vbt.Portfolio.from_orders(
        close=px,
        size=wt,
        size_type="targetpercent",
        init_cash=BT_CFG["initial_capital"],
        fees=BT_CFG["commission_pct"],
        slippage=BT_CFG["slippage_pct"],
        freq=BT_CFG["freq"],
        group_by=True,
        cash_sharing=True,
    )

    spy = _download_spy(HOLDOUT_START, HOLDOUT_END)
    spy = spy.reindex(common_dates).ffill().dropna()
    benchmark_pf = vbt.Portfolio.from_holding(
        close=spy,
        init_cash=BT_CFG["initial_capital"],
        freq=BT_CFG["freq"],
    )

    strat_equity = strategy_pf.value()
    bench_equity = benchmark_pf.value()
    equity_df = pd.DataFrame({
        "strategy": strat_equity,
        "benchmark": bench_equity.reindex(strat_equity.index),
    })
    equity_df.index.name = "date"

    out_path = RESULTS_DIR / "holdout_2025_26_equity.csv"
    equity_df.to_csv(out_path)
    logger.info("Wrote equity CSV → %s (rows=%d)", out_path, len(equity_df))

    # ── Plumbing summary (NO Sharpe/CAGR/DD — that's Prompt 4) ────────
    print("\n" + "═" * 64)
    print("  Holdout dry-run — plumbing summary (no verdict metrics)")
    print("═" * 64)
    print(f"  Score months:    {scores_df['date'].nunique()}")
    print(f"  Score rows:      {len(scores_df)}")
    print(f"  Score min:       {scores_df['ensemble_score'].min():.4f}")
    print(f"  Score mean:      {scores_df['ensemble_score'].mean():.4f}")
    print(f"  Score max:       {scores_df['ensemble_score'].max():.4f}")
    breadth = (
        scores_df.assign(above=lambda d: d["ensemble_score"] >= 0.52)
                 .groupby("date")["above"].sum()
                 .rename("n_above_0.52")
    )
    print(f"  Monthly breadth ≥0.52: min={breadth.min()}  "
          f"median={int(breadth.median())}  max={breadth.max()}")
    print(f"  Equity rows:     {len(equity_df)}")
    print(f"  Equity dates:    {equity_df.index.min().date()} → "
          f"{equity_df.index.max().date()}")


if __name__ == "__main__":
    main()
