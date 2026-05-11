"""
Quant Factors Strategy
======================
Computes cross-sectional momentum + macro-conditioned signals for the tech
universe, then backtests against buy-and-hold SPY using VectorBT.

Signals
-------
  1. Price momentum   : 1M, 3M, 6M, 12M returns (skip last 5 days)
  2. Volume trend     : 20-day volume Z-score
  3. Volatility adj.  : inverse realized vol weighting
  4. Macro regime     : risk-on/risk-off filter based on VIX + yield spread

Portfolio construction
----------------------
  - Monthly rebalance, long-only
  - Hold top-N stocks by composite score
  - Equal-weight within selected basket
  - 10bps commission + 5bps slippage per VectorBT sim

Benchmarks
----------
  - XLK (Phase 1 primary)
  - SPY buy-and-hold (time-in-market comparison)

Output
------
  - backtests/results/quant_sprint1.pkl   (VectorBT portfolio objects)
  - backtests/reports/quant_sprint1.html  (VectorBT tearsheet)
  - Console: key metrics table
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import vectorbt as vbt
import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
with open(ROOT / "config" / "settings.yaml") as f:
    CFG = yaml.safe_load(f)

QF_CFG = CFG["quant_factors"]
BT_CFG = CFG["backtest"]
TIMELINE = CFG["timeline"]
RESULTS_DIR = ROOT / "backtests" / "results"
REPORTS_DIR = ROOT / "backtests" / "reports"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Factor Computation ────────────────────────────────────────────────────────

def compute_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional momentum for each lookback window, skipping last N days.
    Returns a DataFrame (date × ticker) of z-scored momentum scores.
    """
    skip = QF_CFG["momentum"]["skip_days"]
    lookbacks = QF_CFG["momentum"]["lookbacks_days"]
    scores = []

    for lb in lookbacks:
        # return from lb days ago to skip days ago
        ret = prices.shift(skip) / prices.shift(lb + skip) - 1
        # cross-sectional z-score
        z = ret.sub(ret.mean(axis=1), axis=0).div(ret.std(axis=1).replace(0, np.nan), axis=0)
        scores.append(z)

    # equal-weight across lookback windows
    mom_score = pd.concat(scores).groupby(level=0).mean()
    return mom_score


def compute_volume_trend(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """
    Volume Z-score: how much above/below 20-day avg is today's volume?
    Proxy for institutional interest / breakout confirmation.
    """
    window = QF_CFG["volume"]["rolling_window"]
    vol_mean = volumes.rolling(window).mean()
    vol_std = volumes.rolling(window).std().replace(0, np.nan)
    vol_z = (volumes - vol_mean) / vol_std
    # cross-sectional z-score
    vol_z_cs = vol_z.sub(vol_z.mean(axis=1), axis=0).div(vol_z.std(axis=1).replace(0, np.nan), axis=0)
    return vol_z_cs


def compute_volatility_score(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Inverse realized volatility (21-day) — lower vol stocks score higher.
    Used as a tilt/weight modifier, not a standalone signal.
    """
    window = QF_CFG["volatility"]["window"]
    log_ret = np.log(prices / prices.shift(1))
    rvol = log_ret.rolling(window).std() * np.sqrt(252)
    inv_vol = 1.0 / rvol.replace(0, np.nan)
    # cross-sectional z-score
    inv_vol_z = inv_vol.sub(inv_vol.mean(axis=1), axis=0).div(
        inv_vol.std(axis=1).replace(0, np.nan), axis=0
    )
    return inv_vol_z


def macro_regime_filter(macro: pd.DataFrame, prices_index: pd.Index) -> pd.Series:
    """
    Binary risk-on / risk-off flag based on:
      - VIX < 25  (low fear)
      - 10Y-2Y yield spread > 0  (non-inverted)

    Returns a boolean Series aligned to prices_index.
    True = risk-on (deploy capital), False = risk-off (stay in cash).
    """
    regime = pd.Series(True, index=prices_index)

    if "vix" in macro.columns:
        vix_aligned = macro["vix"].reindex(prices_index, method="ffill")
        regime &= vix_aligned < 25.0

    if "yield_spread_10y2y" in macro.columns:
        spread_aligned = macro["yield_spread_10y2y"].reindex(prices_index, method="ffill")
        regime &= spread_aligned > 0.0

    logger.info(
        f"Macro regime filter: {regime.sum()} risk-on days / {len(regime)} total "
        f"({regime.mean()*100:.1f}%)"
    )
    return regime


def composite_score(
    prices: pd.DataFrame,
    volumes: Optional[pd.DataFrame] = None,
    macro: Optional[pd.DataFrame] = None,
    weights: dict = None,
) -> pd.DataFrame:
    """
    Combine factor scores into a single composite rank signal.

    Default weights: momentum=0.5, volume=0.3, inv_vol=0.2
    """
    if weights is None:
        weights = {"momentum": 0.5, "volume": 0.3, "inv_vol": 0.2}

    mom = compute_momentum(prices)
    score = weights["momentum"] * mom

    if volumes is not None:
        vol_score = compute_volume_trend(prices, volumes)
        score = score + weights["volume"] * vol_score.reindex(score.index)

    inv_vol = compute_volatility_score(prices)
    score = score + weights["inv_vol"] * inv_vol.reindex(score.index)

    return score


# ── Portfolio Construction ────────────────────────────────────────────────────

def build_monthly_signals(
    scores: pd.DataFrame,
    macro_regime: Optional[pd.Series] = None,
    top_n: int = None,
) -> pd.DataFrame:
    """
    Convert daily factor scores to monthly rebalance signals (0/1 entry flags).

    On each monthly rebalance date:
      - If macro is risk-off, hold cash (all zeros)
      - Otherwise, select top-N stocks by composite score, equal-weight

    Returns a boolean DataFrame (date × ticker) of positions (True = hold).
    """
    if top_n is None:
        top_n = QF_CFG["top_n"]

    # Resample to month-end rebalance dates
    monthly_scores = scores.resample("ME").last()
    positions = pd.DataFrame(False, index=monthly_scores.index, columns=monthly_scores.columns)

    for date in monthly_scores.index:
        row = monthly_scores.loc[date].dropna()
        if row.empty:
            continue

        # Macro gate
        if macro_regime is not None:
            regime_on_date = macro_regime.reindex([date], method="ffill").iloc[0]
            if not regime_on_date:
                logger.debug(f"  {date.date()} — risk-off, staying in cash")
                continue

        # Select top-N
        selected = row.nlargest(min(top_n, len(row))).index
        positions.loc[date, selected] = True

    # Forward-fill to daily (hold until next rebalance)
    daily_positions = positions.reindex(scores.index, method="ffill").fillna(False)
    return daily_positions


# ── VectorBT Backtest ─────────────────────────────────────────────────────────

def run_backtest(
    prices: pd.DataFrame,
    positions: pd.DataFrame,
    name: str = "Quant Sprint 1",
    benchmark_ticker: str = "SPY",
) -> dict:
    """
    Run VectorBT backtest for strategy and SPY buy-and-hold.

    Returns dict with 'strategy' and 'benchmark' Portfolio objects.
    """
    # Align
    common_dates = prices.index.intersection(positions.index)
    px = prices.loc[common_dates]
    pos = positions.loc[common_dates].reindex(columns=px.columns).fillna(False)

    # Equal-weight within selected: allocate 1/N of capital to each True position
    n_selected = pos.sum(axis=1).replace(0, np.nan)
    weights_matrix = pos.div(n_selected, axis=0).fillna(0.0)

    logger.info(f"Running VectorBT backtest: {len(common_dates)} trading days")

    # Strategy portfolio
    strategy_pf = vbt.Portfolio.from_orders(
        close=px,
        size=weights_matrix,
        size_type="targetpercent",
        init_cash=BT_CFG["initial_capital"],
        fees=BT_CFG["commission_pct"],
        slippage=BT_CFG["slippage_pct"],
        freq=BT_CFG["freq"],
        group_by=True,
        cash_sharing=True,
    )

    # SPY benchmark
    spy_prices = _fetch_benchmark(benchmark_ticker, px.index[0], px.index[-1])
    spy_pf = vbt.Portfolio.from_holding(
        close=spy_prices,
        init_cash=BT_CFG["initial_capital"],
        freq=BT_CFG["freq"],
    )

    return {"strategy": strategy_pf, "benchmark": spy_pf, "spy_prices": spy_prices}


def _fetch_benchmark(ticker: str, start, end) -> pd.Series:
    """Download benchmark prices aligned to the strategy's date range."""
    import yfinance as yf
    raw = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw["Close"].rename(ticker)


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_metrics(results: dict) -> None:
    """Print a side-by-side metrics comparison table."""
    strat = results["strategy"]
    bench = results["benchmark"]

    metrics = {
        "Total Return": [
            f"{strat.total_return() * 100:.2f}%",
            f"{bench.total_return() * 100:.2f}%",
        ],
        "Annual Return (CAGR)": [
            f"{strat.annualized_return() * 100:.2f}%",
            f"{bench.annualized_return() * 100:.2f}%",
        ],
        "Sharpe Ratio": [
            f"{strat.sharpe_ratio():.3f}",
            f"{bench.sharpe_ratio():.3f}",
        ],
        "Max Drawdown": [
            f"{strat.max_drawdown() * 100:.2f}%",
            f"{bench.max_drawdown() * 100:.2f}%",
        ],
        "Sortino Ratio": [
            f"{strat.sortino_ratio():.3f}",
            f"{bench.sortino_ratio():.3f}",
        ],
        "Calmar Ratio": [
            f"{strat.calmar_ratio():.3f}",
            f"{bench.calmar_ratio():.3f}",
        ],
    }

    header = f"{'Metric':<25} {'Strategy':>15} {'SPY B&H':>15}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for metric, (s_val, b_val) in metrics.items():
        print(f"{metric:<25} {s_val:>15} {b_val:>15}")
    print("=" * len(header))


def save_results(results: dict, tag: str = "quant_sprint1") -> None:
    """Save portfolio objects and HTML tearsheet."""
    import pickle

    pkl_path = RESULTS_DIR / f"{tag}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Saved portfolio objects → {pkl_path}")

    html_path = REPORTS_DIR / f"{tag}.html"
    results["strategy"].stats().to_frame("Strategy").to_html(str(html_path))
    logger.info(f"Saved report → {html_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_quant_strategy(
    tickers: Optional[list[str]] = None,
    train_only: bool = False,
    save: bool = True,
) -> dict:
    """
    Full pipeline:
      1. Load prices + macro from DB
      2. Compute composite factor scores
      3. Build monthly signal
      4. Run VectorBT backtest vs SPY
      5. Print metrics + save results

    Parameters
    ----------
    tickers    : override universe tickers
    train_only : if True, backtest only on 2015-2022 train period
    save       : if True, persist results to disk
    """
    from src.data.quant_pipeline import load_prices, load_macro, get_engine
    from src.universe.screener import load_universe

    engine = get_engine()

    # Load tickers
    if tickers is None:
        uni_path = ROOT / "data" / "universe" / "universe.csv"
        uni = load_universe(uni_path)
        tickers = uni["ticker"].tolist()

    # Date range
    end_date = TIMELINE["train_end"] if train_only else TIMELINE["test_end"]
    start_date = TIMELINE["train_start"]

    # Load data
    logger.info(f"Loading prices for {len(tickers)} tickers  [{start_date} → {end_date}]")
    prices = load_prices(tickers=tickers, start=start_date, end=end_date, engine=engine)

    logger.info("Loading macro series…")
    macro = load_macro(start=start_date, end=end_date, engine=engine)

    # Drop tickers with insufficient history (< 252 trading days)
    min_obs = 252
    prices = prices.loc[:, prices.count() >= min_obs]
    logger.info(f"Tickers with sufficient history: {prices.shape[1]}")

    # Compute composite score (volumes not yet in DB — use prices as proxy)
    logger.info("Computing factor scores…")
    scores = composite_score(prices, macro=macro)

    # Macro regime
    regime = macro_regime_filter(macro, prices.index) if not macro.empty else None

    # Build signals
    logger.info("Building monthly rebalance signals…")
    positions = build_monthly_signals(scores, macro_regime=regime)

    # Backtest
    logger.info("Running VectorBT backtest…")
    results = run_backtest(prices, positions)

    # Report
    print_metrics(results)

    if save:
        save_results(results, tag="quant_sprint1" + ("_train" if train_only else "_full"))

    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Run quant factors strategy + backtest")
    parser.add_argument("--tickers", nargs="+", help="Override universe")
    parser.add_argument("--train-only", action="store_true",
                        help="Backtest on 2015-2022 train period only")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to disk")
    args = parser.parse_args()

    run_quant_strategy(
        tickers=args.tickers,
        train_only=args.train_only,
        save=not args.no_save,
    )
