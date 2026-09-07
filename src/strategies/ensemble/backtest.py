"""
Ensemble VectorBT Backtest + Three-Way Comparison
====================================================
Runs the ensemble strategy through VectorBT using the daily weight matrix,
compares against the Phase 1 quant strategy, Phase 2 fundamental strategy,
and SPY buy-and-hold.

Output
------
  backtests/results/ensemble_sprint3_train_equity.csv
  backtests/results/ensemble_sprint3_test_equity.csv
  backtests/results/ensemble_sprint3_stats.csv

CLI
---
  python -m src.strategies.ensemble.backtest
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt
import yaml
import yfinance as yf
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect

load_dotenv()
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
with open(ROOT / "config" / "settings.yaml") as f:
    CFG = yaml.safe_load(f)

BT_CFG = CFG["backtest"]
TIMELINE = CFG["timeline"]
DB_PATH = ROOT / CFG["data"]["paths"]["db"]
DB_URL = f"sqlite:///{DB_PATH}"
PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "backtests" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loaders ──────────────────────────────────────────────────────────────

def _get_engine(engine=None):
    if engine is not None:
        return engine
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(DB_URL, echo=False)


def _load_prices(engine) -> pd.DataFrame:
    """Load adj_close from SQLite → wide (date × ticker)."""
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
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="ticker", values="adj_close")


def _load_weights() -> pd.DataFrame:
    path = PROCESSED / "ensemble_weights.parquet"
    if not path.exists():
        logger.error(f"Weights not found: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


def _download_spy(start, end) -> pd.Series:
    """SPY benchmark, served from the frozen `prices` table.

    Previously an unconditional yfinance call, which left the benchmark leg of
    every SPY-relative result network-dependent and outside the vintage that
    `scripts/freeze_vintage.py` snapshots. `load_benchmark` reads the DB and
    only falls back to the network with a loud warning naming the fallback.
    """
    from src.data.quant_pipeline import load_benchmark

    return load_benchmark(start, end, ticker="SPY")


# ── Prior strategy equity loaders ─────────────────────────────────────────────

def _load_equity_csv(path: Path) -> pd.Series:
    """Load an equity curve CSV, return the strategy column as a Series."""
    if not path.exists():
        logger.warning(f"Equity file not found: {path}")
        return pd.Series(dtype=float)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "strategy" in df.columns:
        return df["strategy"]
    return df.iloc[:, 0]


def _compute_metrics(equity: pd.Series) -> dict:
    """Compute standard backtest metrics from an equity curve."""
    if equity.empty or len(equity) < 2:
        return {k: np.nan for k in [
            "Total Return", "CAGR", "Sharpe Ratio",
            "Sortino Ratio", "Max Drawdown", "Calmar Ratio"
        ]}

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    daily_ret = equity.pct_change().dropna()
    if daily_ret.std() == 0:
        sharpe = 0.0
        sortino = 0.0
    else:
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
        downside = daily_ret[daily_ret < 0].std()
        sortino = daily_ret.mean() / downside * np.sqrt(252) if downside > 0 else 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def run_backtest(weights_df: pd.DataFrame = None, engine=None) -> dict:
    """
    Run VectorBT backtest for the ensemble strategy.

    Returns dict with keys:
      strategy_pf  : VectorBT Portfolio (ensemble)
      benchmark_pf : VectorBT Portfolio (SPY)
      equity       : pd.DataFrame with strategy + benchmark equity curves
      metrics      : dict of computed metrics
    """
    engine = _get_engine(engine)

    # Load data
    if weights_df is None:
        weights_df = _load_weights()
    if weights_df.empty:
        logger.error("No weights available")
        return {}

    logger.info("Loading prices from DB…")
    prices = _load_prices(engine)

    # Align weights and prices to the same date range and tickers
    common_dates = prices.index.intersection(weights_df.index)
    common_tickers = [t for t in weights_df.columns if t in prices.columns]

    if not len(common_dates) or not len(common_tickers):
        logger.error("No overlapping dates/tickers between weights and prices")
        return {}

    px = prices.loc[common_dates, common_tickers]
    wt = weights_df.loc[common_dates, common_tickers].fillna(0.0)

    logger.info(
        f"Aligned: {len(common_dates)} days, {len(common_tickers)} tickers "
        f"[{common_dates.min().date()} → {common_dates.max().date()}]"
    )

    # ── VectorBT ensemble strategy ────────────────────────────────────
    logger.info("Running VectorBT backtest (ensemble)…")
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

    # ── SPY benchmark ─────────────────────────────────────────────────
    spy_prices = _download_spy(
        common_dates.min().strftime("%Y-%m-%d"),
        (common_dates.max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
    )
    spy_prices = spy_prices.reindex(common_dates).ffill().dropna()

    benchmark_pf = vbt.Portfolio.from_holding(
        close=spy_prices,
        init_cash=BT_CFG["initial_capital"],
        freq=BT_CFG["freq"],
    )

    # ── Equity curves ─────────────────────────────────────────────────
    strat_equity = strategy_pf.value()
    bench_equity = benchmark_pf.value()

    equity_df = pd.DataFrame({
        "strategy": strat_equity,
        "benchmark": bench_equity.reindex(strat_equity.index),
    })
    equity_df.index.name = "date"

    # ── Split train / test ────────────────────────────────────────────
    train_end = pd.Timestamp(TIMELINE["train_end"])
    test_start = pd.Timestamp(TIMELINE["test_start"])

    train_equity = equity_df[equity_df.index <= train_end]
    test_equity = equity_df[equity_df.index >= test_start]

    # Save equity CSVs
    train_path = RESULTS_DIR / "ensemble_sprint3_train_equity.csv"
    test_path = RESULTS_DIR / "ensemble_sprint3_test_equity.csv"
    train_equity.to_csv(train_path)
    test_equity.to_csv(test_path)
    logger.info(f"Saved train equity → {train_path}")
    logger.info(f"Saved test equity → {test_path}")

    # Save stats CSV
    strat_stats = strategy_pf.stats().to_frame("strategy")
    bench_stats = benchmark_pf.stats().to_frame("benchmark")
    stats_df = pd.concat([strat_stats, bench_stats], axis=1)
    stats_path = RESULTS_DIR / "ensemble_sprint3_stats.csv"
    stats_df.to_csv(stats_path)
    logger.info(f"Saved stats → {stats_path}")

    # Compute our own metrics for consistency
    metrics = {
        "full": _compute_metrics(equity_df["strategy"]),
        "train": _compute_metrics(train_equity["strategy"]),
        "test": _compute_metrics(test_equity["strategy"]),
        "spy_full": _compute_metrics(equity_df["benchmark"]),
    }

    return {
        "strategy_pf": strategy_pf,
        "benchmark_pf": benchmark_pf,
        "equity": equity_df,
        "train_equity": train_equity,
        "test_equity": test_equity,
        "metrics": metrics,
    }


# ── Three-way comparison ─────────────────────────────────────────────────────

def _build_comparison_table(ensemble_metrics: dict) -> pd.DataFrame:
    """
    Build a three-way + SPY comparison table from saved equity CSVs.
    """
    # Load quant equity
    quant_full = _load_equity_csv(RESULTS_DIR / "quant_sprint1_full_equity.csv")
    quant_metrics = _compute_metrics(quant_full) if not quant_full.empty else {}

    # Load fundamental equity — combine train + test
    fund_train = _load_equity_csv(RESULTS_DIR / "fundamental_sprint2_train_equity.csv")
    fund_test = _load_equity_csv(RESULTS_DIR / "fundamental_sprint2_test_equity.csv")
    if not fund_train.empty and not fund_test.empty:
        # Scale test to continue from train end value
        scale = fund_train.iloc[-1] / fund_test.iloc[0] if fund_test.iloc[0] != 0 else 1.0
        fund_test_scaled = fund_test * scale
        fund_full = pd.concat([fund_train, fund_test_scaled.iloc[1:]])
        fund_metrics = _compute_metrics(fund_full)
    elif not fund_train.empty:
        fund_metrics = _compute_metrics(fund_train)
    else:
        fund_metrics = {}

    # Ensemble full-period metrics
    ens_metrics = ensemble_metrics.get("full", {})
    spy_metrics = ensemble_metrics.get("spy_full", {})

    # Build comparison table
    metric_names = [
        "Total Return", "CAGR", "Sharpe Ratio",
        "Sortino Ratio", "Max Drawdown", "Calmar Ratio",
    ]

    rows = []
    for m in metric_names:
        row = {
            "Metric": m,
            "Quant": quant_metrics.get(m, np.nan),
            "Fundamental": fund_metrics.get(m, np.nan),
            "Ensemble": ens_metrics.get(m, np.nan),
            "SPY B&H": spy_metrics.get(m, np.nan),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def _format_metric(val, metric_name: str) -> str:
    """Format a metric value for display."""
    if pd.isna(val):
        return "    —"
    if metric_name in ("Total Return", "CAGR", "Max Drawdown"):
        return f"{val * 100:+8.2f}%"
    else:
        return f"{val:8.3f}"


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    results = run_backtest()

    if not results:
        logger.error("Backtest failed")
        return

    equity = results["equity"]
    metrics = results["metrics"]

    # ── Print ensemble summary ────────────────────────────────────────
    print(f"\n{'═' * 76}")
    print(f"  Ensemble Strategy Backtest (Phase 3)")
    print(f"{'═' * 76}")
    print(f"  Period:  {equity.index.min().date()} → {equity.index.max().date()}")
    print(f"  Initial: ${BT_CFG['initial_capital']:,.0f}")
    print(f"  Final:   ${equity['strategy'].iloc[-1]:,.0f}")
    print(f"{'═' * 76}")

    # ── Train / Test split metrics ────────────────────────────────────
    for period, label in [("train", "TRAIN (in-sample)"), ("test", "TEST (out-of-sample)")]:
        m = metrics.get(period, {})
        if m:
            print(f"\n  {label}:")
            for k, v in m.items():
                print(f"    {k:<20s} {_format_metric(v, k)}")

    # ── Three-way comparison table ────────────────────────────────────
    comp = _build_comparison_table(metrics)

    header = f"  {'Metric':<20s} {'Quant':>12s} {'Fundamental':>14s} {'Ensemble':>12s} {'SPY B&H':>12s}"
    sep = "  " + "─" * 72

    print(f"\n{'═' * 76}")
    print(f"  THREE-WAY STRATEGY COMPARISON")
    print(f"{'═' * 76}")
    print(header)
    print(sep)

    for _, row in comp.iterrows():
        m = row["Metric"]
        q = _format_metric(row["Quant"], m)
        f = _format_metric(row["Fundamental"], m)
        e = _format_metric(row["Ensemble"], m)
        s = _format_metric(row["SPY B&H"], m)
        print(f"  {m:<20s} {q:>12s} {f:>14s} {e:>12s} {s:>12s}")

    print(sep)

    # Highlight winner
    for _, row in comp.iterrows():
        m = row["Metric"]
        vals = {"Quant": row["Quant"], "Fundamental": row["Fundamental"],
                "Ensemble": row["Ensemble"]}
        vals = {k: v for k, v in vals.items() if not pd.isna(v)}
        if not vals:
            continue
        if m == "Max Drawdown":
            winner = max(vals, key=vals.get)  # least negative
        else:
            winner = max(vals, key=vals.get)
        print(f"    ★ {m}: {winner} wins")

    print(f"\n{'═' * 76}")
    print()


if __name__ == "__main__":
    main()
