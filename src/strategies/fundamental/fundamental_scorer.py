"""
Fundamental Scorer & Independent Backtest
==========================================
Combines z-scored fundamental features from xbrl_features.py into a weighted
composite score, builds a quarterly-rebalanced long-only portfolio, and runs
an independent backtest against XLK buy-and-hold using VectorBT.

Factor Weights (from settings.yaml)
------------------------------------
  gross_profitability    0.20
  fcf_yield              0.20
  revenue_acceleration   0.15
  earnings_surprise      0.20  (sue_score column)
  eps_revision_momentum  0.10  (eps_revision_1m column)
  deferred_revenue_yoy   0.05
  rd_intensity           0.05  (higher R&D is positive for tech — NOT inverted)
  finbert_sentiment      0.05  (finbert_score column)

Portfolio Construction
----------------------
  rebalance_freq: quarterly
  long_only: true
  top_n: 20
  min_score_threshold: 0.0

Outputs
-------
  backtests/results/fundamental_sprint2.pkl   (VectorBT portfolio objects)
  backtests/reports/fundamental_sprint2.html  (stats to HTML)
  data/processed/fundamental_scores.parquet   (quarterly composite scores)

CLI
---
  python -m src.strategies.fundamental.fundamental_scorer
  python -m src.strategies.fundamental.fundamental_scorer --train-only
  python -m src.strategies.fundamental.fundamental_scorer --tickers AAPL MSFT NVDA
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

# ── Config ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[3]
with open(ROOT / "config" / "settings.yaml") as f:
    CFG = yaml.safe_load(f)

FF_CFG = CFG["fundamental_factors"]
BT_CFG = CFG["backtest"]
TIMELINE = CFG["timeline"]

RESULTS_DIR = ROOT / "backtests" / "results"
REPORTS_DIR = ROOT / "backtests" / "reports"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Weight Mapping ────────────────────────────────────────────────────────────
# Maps settings.yaml factor names → actual column names from xbrl_features.py

_WEIGHT_MAP: dict[str, tuple[str, float]] = {
    # (column_name_in_feature_matrix, weight)
    "gross_profitability":    ("gross_profitability",    FF_CFG["weights"]["gross_profitability"]),
    "fcf_yield":              ("fcf_yield",              FF_CFG["weights"]["fcf_yield"]),
    "revenue_acceleration":   ("revenue_acceleration",   FF_CFG["weights"]["revenue_acceleration"]),
    "earnings_surprise":      ("sue_score",              FF_CFG["weights"]["earnings_surprise"]),
    "eps_revision_momentum":  ("eps_revision_1m",        FF_CFG["weights"]["eps_revision_momentum"]),
    "deferred_revenue_growth":("deferred_revenue_yoy",   FF_CFG["weights"]["deferred_revenue_growth"]),
    "rd_intensity_trend":     ("rd_intensity",           FF_CFG["weights"]["rd_intensity_trend"]),
    "finbert_sentiment":      ("finbert_score",          FF_CFG["weights"]["finbert_sentiment"]),
    "piotroski_f_score":      ("piotroski_f",            FF_CFG["weights"]["piotroski_f_score"]),
    "qmj_safety":             ("qmj_safety",             FF_CFG["weights"]["qmj_safety"]),
    "qmj_payout":             ("qmj_payout",             FF_CFG["weights"]["qmj_payout"]),
    "lm_sentiment_score":     ("lm_sentiment_score",     FF_CFG["weights"]["lm_sentiment_score"]),
}

# Portfolio construction params
REBALANCE_FREQ = FF_CFG.get("rebalance_freq", "quarterly")
LONG_ONLY = FF_CFG.get("long_only", True)
TOP_N = FF_CFG.get("top_n", 20)
MIN_SCORE_THRESHOLD = FF_CFG.get("min_score_threshold", 0.0)

# Benchmark
BENCHMARK_TICKER = CFG["benchmarks"].get("phase2", "XLK")


# ── 1. COMPOSITE SCORE ───────────────────────────────────────────────────────

def compute_composite_scores(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a weighted composite score for each (ticker, quarter_end) row.

    If a feature column is missing (NaN) for a given ticker/quarter,
    redistribute its weight proportionally among available features.
    This avoids penalising tickers with partial data coverage.

    Parameters
    ----------
    feature_matrix : DataFrame with columns [ticker, quarter_end, ...features]

    Returns
    -------
    DataFrame with columns [ticker, quarter_end, composite_score,
                            features_used, features_missing]
    """
    rows = []

    all_factor_names = list(_WEIGHT_MAP.keys())
    all_col_names = [col for _, (col, _) in _WEIGHT_MAP.items()]
    all_weights = [w for _, (_, w) in _WEIGHT_MAP.items()]

    for _, row in feature_matrix.iterrows():
        ticker = row["ticker"]
        qe = row["quarter_end"]

        available = []
        missing = []
        for factor_name, (col_name, weight) in _WEIGHT_MAP.items():
            val = row.get(col_name, np.nan)
            if pd.notna(val) and col_name in feature_matrix.columns:
                available.append((factor_name, col_name, weight, val))
            else:
                missing.append(factor_name)

        if not available:
            rows.append({
                "ticker": ticker,
                "quarter_end": qe,
                "composite_score": np.nan,
                "features_used": "",
                "features_missing": ",".join(missing),
            })
            continue

        # Redistribute missing weight proportionally
        total_available_weight = sum(w for _, _, w, _ in available)
        if total_available_weight <= 0:
            rows.append({
                "ticker": ticker,
                "quarter_end": qe,
                "composite_score": np.nan,
                "features_used": "",
                "features_missing": ",".join(missing),
            })
            continue

        redistribution_factor = 1.0 / total_available_weight
        composite = sum(w * redistribution_factor * val for _, _, w, val in available)

        rows.append({
            "ticker": ticker,
            "quarter_end": qe,
            "composite_score": composite,
            "features_used": ",".join(name for name, _, _, _ in available),
            "features_missing": ",".join(missing),
        })

    scores_df = pd.DataFrame(rows)

    # Log diagnostics per quarter
    if not scores_df.empty:
        for qe, grp in scores_df.groupby("quarter_end"):
            n = len(grp)
            n_valid = grp["composite_score"].notna().sum()
            miss_counts: dict[str, int] = {}
            for _, r in grp.iterrows():
                if r["features_missing"]:
                    for feat in r["features_missing"].split(","):
                        miss_counts[feat] = miss_counts.get(feat, 0) + 1
            miss_summary = {k: f"{v}/{n}" for k, v in miss_counts.items() if v > 0}
            if miss_summary:
                logger.info(
                    f"  {pd.Timestamp(qe).date()}  "
                    f"{n_valid}/{n} valid scores | "
                    f"missing: {miss_summary}"
                )
            else:
                logger.info(f"  {pd.Timestamp(qe).date()}  {n_valid}/{n} valid scores | all features present")

    return scores_df


# ── 2. SIGNAL CONSTRUCTION ───────────────────────────────────────────────────

def build_quarterly_signals(
    scores_df: pd.DataFrame,
    prices: pd.DataFrame,
    top_n: int = TOP_N,
    min_threshold: float = MIN_SCORE_THRESHOLD,
) -> pd.DataFrame:
    """
    Build daily position weights from quarterly composite scores.

    On each quarterly rebalance date:
      a) Filter to composite > min_score_threshold
      b) Select top_n by composite score
      c) Equal-weight the selected tickers

    Forward-fill signals to daily frequency (hold until next rebalance).

    Parameters
    ----------
    scores_df : DataFrame with [ticker, quarter_end, composite_score]
    prices    : Wide DataFrame (date × ticker) of adj_close prices
    top_n     : Number of top tickers to select
    min_threshold : Minimum composite score for inclusion

    Returns
    -------
    DataFrame (date × ticker) of target portfolio weights (0.0–1.0)
    """
    # Get unique quarter-end rebalance dates
    rebalance_dates = sorted(scores_df["quarter_end"].unique())
    all_tickers = sorted(prices.columns.tolist())

    # Build quarterly weight snapshots
    quarterly_weights = {}
    for qe in rebalance_dates:
        qe_ts = pd.Timestamp(qe)
        snapshot = scores_df[scores_df["quarter_end"] == qe].copy()

        # Filter: composite > threshold
        snapshot = snapshot[snapshot["composite_score"] > min_threshold]

        if snapshot.empty:
            quarterly_weights[qe_ts] = pd.Series(0.0, index=all_tickers)
            logger.warning(f"  {qe_ts.date()} — no tickers above threshold")
            continue

        # Sort by composite descending, pick top_n
        snapshot = snapshot.nlargest(min(top_n, len(snapshot)), "composite_score")
        selected = snapshot["ticker"].tolist()

        # Only include tickers that exist in price data
        selected = [t for t in selected if t in prices.columns]

        if not selected:
            quarterly_weights[qe_ts] = pd.Series(0.0, index=all_tickers)
            continue

        # Equal weight
        w = 1.0 / len(selected)
        weight_row = pd.Series(0.0, index=all_tickers)
        for t in selected:
            weight_row[t] = w

        quarterly_weights[qe_ts] = weight_row
        logger.info(
            f"  {qe_ts.date()} — selected {len(selected)} tickers  "
            f"(top: {selected[:5]}{'...' if len(selected) > 5 else ''})"
        )

    # Build weight DataFrame at quarterly frequency
    weight_df = pd.DataFrame(quarterly_weights).T
    weight_df.index.name = "date"

    # Forward-fill to daily frequency, aligned to price dates
    # Only use trading days from the price index
    daily_weights = weight_df.reindex(prices.index, method="ffill").fillna(0.0)

    return daily_weights


# ── 3. VECTORBT BACKTEST ─────────────────────────────────────────────────────

def _fetch_benchmark(ticker: str, start, end) -> pd.Series:
    """Download benchmark prices aligned to the strategy's date range."""
    import yfinance as yf

    raw = yf.download(
        ticker,
        start=pd.Timestamp(start).strftime("%Y-%m-%d"),
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw["Close"].rename(ticker)


def run_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    benchmark_ticker: str = BENCHMARK_TICKER,
    period_label: str = "full",
) -> dict:
    """
    Run two independent VectorBT portfolios:
      a) Fundamental strategy (quarterly rebalance, top-20)
      b) Benchmark (XLK) buy-and-hold

    Parameters
    ----------
    prices           : Wide DataFrame (date × ticker) of adj_close
    weights          : Daily weight matrix (date × ticker)
    benchmark_ticker : Benchmark ETF ticker
    period_label     : Label for logging

    Returns
    -------
    dict with keys: 'strategy', 'benchmark', 'benchmark_prices'
    """
    # Align dates
    common_dates = prices.index.intersection(weights.index)
    px = prices.loc[common_dates]
    wt = weights.loc[common_dates].reindex(columns=px.columns).fillna(0.0)

    logger.info(
        f"Running VectorBT backtest [{period_label}]: "
        f"{len(common_dates)} trading days, "
        f"{(wt > 0).any(axis=0).sum()} tickers ever held"
    )

    # Strategy portfolio
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

    # Benchmark: XLK buy-and-hold
    bench_prices = _fetch_benchmark(
        benchmark_ticker, common_dates[0], common_dates[-1]
    )
    # Align benchmark to strategy dates
    bench_prices = bench_prices.reindex(common_dates, method="ffill").dropna()
    common_bench_dates = common_dates.intersection(bench_prices.index)

    benchmark_pf = vbt.Portfolio.from_holding(
        close=bench_prices.loc[common_bench_dates],
        init_cash=BT_CFG["initial_capital"],
        freq=BT_CFG["freq"],
    )

    return {
        "strategy": strategy_pf,
        "benchmark": benchmark_pf,
        "benchmark_prices": bench_prices,
    }


# ── 4. METRICS TABLE ─────────────────────────────────────────────────────────

def _safe_metric(pf, method_name: str, fallback="N/A"):
    """Safely extract a scalar metric from a VectorBT portfolio."""
    try:
        val = getattr(pf, method_name)()
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        return val
    except Exception:
        return fallback


def _format_pct(val, fallback="N/A"):
    if val == "N/A" or val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val * 100:.2f}%"


def _format_ratio(val, fallback="N/A"):
    if val == "N/A" or val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.3f}"


def print_metrics(results: dict, period_label: str = "") -> None:
    """Print a side-by-side metrics comparison table."""
    strat = results["strategy"]
    bench = results["benchmark"]

    metrics = {
        "Total Return": (
            _format_pct(_safe_metric(strat, "total_return")),
            _format_pct(_safe_metric(bench, "total_return")),
        ),
        "CAGR": (
            _format_pct(_safe_metric(strat, "annualized_return")),
            _format_pct(_safe_metric(bench, "annualized_return")),
        ),
        "Sharpe Ratio": (
            _format_ratio(_safe_metric(strat, "sharpe_ratio")),
            _format_ratio(_safe_metric(bench, "sharpe_ratio")),
        ),
        "Sortino Ratio": (
            _format_ratio(_safe_metric(strat, "sortino_ratio")),
            _format_ratio(_safe_metric(bench, "sortino_ratio")),
        ),
        "Calmar Ratio": (
            _format_ratio(_safe_metric(strat, "calmar_ratio")),
            _format_ratio(_safe_metric(bench, "calmar_ratio")),
        ),
        "Max Drawdown": (
            _format_pct(_safe_metric(strat, "max_drawdown")),
            _format_pct(_safe_metric(bench, "max_drawdown")),
        ),
        "Win Rate": (
            _format_pct(_safe_metric(strat, "win_rate")),
            _format_pct(_safe_metric(bench, "win_rate")),
        ),
    }

    # Avg hold period — trades-level metric
    for label, pf in [("Strategy", strat), ("Benchmark", bench)]:
        try:
            trades = pf.trades.records_readable
            if not trades.empty and "Duration" in trades.columns:
                avg_dur = trades["Duration"].mean()
                metrics.setdefault("Avg Hold Period", [None, None])
                idx = 0 if label == "Strategy" else 1
                if isinstance(metrics["Avg Hold Period"], tuple):
                    metrics["Avg Hold Period"] = list(metrics["Avg Hold Period"])
                metrics["Avg Hold Period"][idx] = str(avg_dur)
        except Exception:
            pass

    if "Avg Hold Period" not in metrics:
        metrics["Avg Hold Period"] = ("N/A", "N/A")
    elif isinstance(metrics["Avg Hold Period"], list):
        metrics["Avg Hold Period"] = tuple(
            v if v is not None else "N/A" for v in metrics["Avg Hold Period"]
        )

    bench_name = f"{BENCHMARK_TICKER} B&H"
    title = f"  {period_label}  " if period_label else ""
    header = f"{'Metric':<25} {'Strategy':>18} {bench_name:>18}"
    sep = "═" * len(header)

    print(f"\n{sep}")
    if title:
        print(f"  {title.strip()}")
        print(f"{sep}")
    print(header)
    print(sep)
    for metric, vals in metrics.items():
        if isinstance(vals, (list, tuple)) and len(vals) >= 2:
            print(f"{metric:<25} {str(vals[0]):>18} {str(vals[1]):>18}")
    print(sep)


# ── 5. SAVE OUTPUTS ──────────────────────────────────────────────────────────

def save_results(
    results_train: Optional[dict],
    results_test: Optional[dict],
    scores_df: pd.DataFrame,
    tag: str = "fundamental_sprint2",
) -> None:
    """Save stats CSVs, equity curves, HTML report, and composite scores."""

    # Save stats and equity curves for each period
    for period_label, res in [("train", results_train), ("test", results_test)]:
        if res is None:
            continue
        try:
            strat_stats = res["strategy"].stats().to_frame("strategy")
            bench_stats = res["benchmark"].stats().to_frame("benchmark")
            stats_path = RESULTS_DIR / f"{tag}_{period_label}_stats.csv"
            pd.concat([strat_stats, bench_stats], axis=1).to_csv(stats_path)
            logger.info(f"Saved {period_label} stats → {stats_path}")
        except Exception as e:
            logger.warning(f"Could not save {period_label} stats: {e}")

        try:
            equity_path = RESULTS_DIR / f"{tag}_{period_label}_equity.csv"
            pd.DataFrame({
                "strategy": res["strategy"].value(),
                "benchmark": res["benchmark"].value(),
            }).to_csv(equity_path)
            logger.info(f"Saved {period_label} equity curves → {equity_path}")
        except Exception as e:
            logger.warning(f"Could not save {period_label} equity: {e}")

    # HTML report
    html_path = REPORTS_DIR / f"{tag}.html"
    html_parts = ["<html><head><title>Fundamental Sprint 2 Results</title>"]
    html_parts.append("""
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
               background: #1a1a2e; color: #e0e0e0; padding: 40px; }
        h1 { color: #00d4ff; }
        h2 { color: #7c83ff; margin-top: 30px; }
        table { border-collapse: collapse; margin: 15px 0; width: 100%; max-width: 700px; }
        th, td { border: 1px solid #333; padding: 10px 16px; text-align: right; }
        th { background: #16213e; color: #00d4ff; }
        td:first-child, th:first-child { text-align: left; }
        tr:nth-child(even) { background: #0f3460; }
        tr:hover { background: #1a4080; }
    </style>
    """)
    html_parts.append("</head><body>")
    html_parts.append(f"<h1>Fundamental Strategy — Sprint 2</h1>")

    for label, res in [("Train Period", results_train), ("Test Period", results_test)]:
        if res is None:
            continue
        html_parts.append(f"<h2>{label}</h2>")
        try:
            stats = res["strategy"].stats()
            if isinstance(stats, pd.Series):
                stats = stats.to_frame("Strategy")
            html_parts.append(stats.to_html())
        except Exception as e:
            html_parts.append(f"<p>Could not generate stats: {e}</p>")

    html_parts.append("</body></html>")
    html_path.write_text("\n".join(html_parts))
    logger.info(f"Saved HTML report → {html_path}")

    # Save composite scores as parquet
    parquet_path = PROCESSED_DIR / "fundamental_scores.parquet"
    scores_out = scores_df[["ticker", "quarter_end", "composite_score"]].copy()
    scores_out.to_parquet(parquet_path, index=False, engine="pyarrow")
    logger.info(f"Saved composite scores → {parquet_path}")


# ── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run_fundamental_strategy(
    tickers: Optional[list[str]] = None,
    train_only: bool = False,
    save: bool = True,
) -> dict:
    """
    Full fundamental scoring + backtest pipeline.

    Steps:
      1. Build z-scored feature matrix via xbrl_features.build_feature_matrix()
      2. Compute weighted composite scores with missing-weight redistribution
      3. Build quarterly rebalance signals (top-20, equal-weight)
      4. Run VectorBT backtest vs XLK (train + test periods)
      5. Print metrics table & save outputs

    Parameters
    ----------
    tickers    : Optional list of tickers to restrict universe
    train_only : If True, only backtest on train period (2015–2022)
    save       : If True, persist results to disk

    Returns
    -------
    dict with keys: 'scores', 'results_train', 'results_test' (if applicable)
    """
    from src.strategies.fundamental.xbrl_features import build_feature_matrix
    from src.data.quant_pipeline import load_prices, get_engine

    engine = get_engine()

    # ── Resolve tickers ──────────────────────────────────────────────
    if tickers is None:
        uni_path = ROOT / "data" / "universe" / "universe.csv"
        if uni_path.exists():
            uni = pd.read_csv(uni_path)
            tickers = uni["ticker"].tolist()
            logger.info(f"Loaded {len(tickers)} tickers from universe.csv")
        else:
            logger.error(f"Universe file not found: {uni_path}")
            return {}

    # ── Date ranges ──────────────────────────────────────────────────
    train_start = TIMELINE["train_start"]
    train_end = TIMELINE["train_end"]
    test_start = TIMELINE["test_start"]
    test_end = TIMELINE["test_end"]

    full_start = train_start
    full_end = train_end if train_only else test_end

    # ── Step 1: Build feature matrix ─────────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 1: Building z-scored feature matrix")
    logger.info("=" * 70)

    feature_matrix = build_feature_matrix(
        tickers=tickers,
        start=full_start,
        end=full_end,
        engine=engine,
    )

    if feature_matrix.empty:
        logger.error("Feature matrix is empty — cannot proceed")
        return {}

    logger.info(
        f"Feature matrix: {len(feature_matrix)} rows, "
        f"{feature_matrix['ticker'].nunique()} tickers, "
        f"{feature_matrix['quarter_end'].nunique()} quarters"
    )

    # ── Step 2: Compute composite scores ─────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 2: Computing composite scores")
    logger.info("=" * 70)

    scores_df = compute_composite_scores(feature_matrix)
    n_valid = scores_df["composite_score"].notna().sum()
    logger.info(
        f"Composite scores: {n_valid}/{len(scores_df)} valid "
        f"(mean={scores_df['composite_score'].mean():.4f}, "
        f"std={scores_df['composite_score'].std():.4f})"
    )

    # ── Step 3: Load prices for backtest ─────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 3: Loading prices & building signals")
    logger.info("=" * 70)

    prices = load_prices(
        tickers=tickers,
        start=full_start,
        end=full_end,
        engine=engine,
    )

    # Drop tickers with insufficient price history (< 126 days ≈ 6 months)
    min_obs = 126
    prices = prices.loc[:, prices.count() >= min_obs]
    logger.info(f"Tickers with sufficient price history: {prices.shape[1]}")

    if prices.empty:
        logger.error("No price data available — cannot proceed")
        return {}

    # ── Step 3b: Build daily weight signals ──────────────────────────
    weights = build_quarterly_signals(scores_df, prices)
    logger.info(f"Daily weight matrix: {weights.shape}")

    # ── Step 4: Run backtests ────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 4: Running VectorBT backtests")
    logger.info("=" * 70)

    output = {"scores": scores_df}

    # --- TRAIN PERIOD ---
    train_mask = (prices.index >= train_start) & (prices.index <= train_end)
    if train_mask.sum() > 0:
        train_prices = prices.loc[train_mask]
        train_weights = weights.loc[train_mask]
        results_train = run_backtest(
            train_prices, train_weights,
            period_label=f"Train ({train_start} → {train_end})",
        )
        output["results_train"] = results_train
        print_metrics(results_train, period_label=f"TRAIN PERIOD ({train_start} → {train_end})")
    else:
        logger.warning("No data in train period")
        output["results_train"] = None

    # --- TEST PERIOD ---
    if not train_only:
        test_mask = (prices.index >= test_start) & (prices.index <= test_end)
        if test_mask.sum() > 0:
            test_prices = prices.loc[test_mask]
            test_weights = weights.loc[test_mask]
            results_test = run_backtest(
                test_prices, test_weights,
                period_label=f"Test ({test_start} → {test_end})",
            )
            output["results_test"] = results_test
            print_metrics(results_test, period_label=f"TEST PERIOD ({test_start} → {test_end})")
        else:
            logger.warning("No data in test period")
            output["results_test"] = None
    else:
        output["results_test"] = None

    # ── Step 5: Save outputs ─────────────────────────────────────────
    if save:
        logger.info("=" * 70)
        logger.info("STEP 5: Saving outputs")
        logger.info("=" * 70)
        save_results(
            results_train=output.get("results_train"),
            results_test=output.get("results_test"),
            scores_df=scores_df,
            tag="fundamental_sprint2",
        )

    logger.info("=" * 70)
    logger.info("Fundamental scorer pipeline complete ✓")
    logger.info("=" * 70)

    return output


# ── 6. CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Fundamental Scorer — composite scoring + backtest vs XLK",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Override universe with specific tickers",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        default=False,
        help="Backtest only on train period (2015–2022)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Don't save results to disk",
    )
    args = parser.parse_args()

    run_fundamental_strategy(
        tickers=args.tickers,
        train_only=args.train_only,
        save=not args.no_save,
    )
