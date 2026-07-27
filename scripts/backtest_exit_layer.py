#!/usr/bin/env python
"""
Diagnostic backtest of the Markov exit layer over the 2025-26 window.

This is a PROTOTYPE QUALITY CHECK, not a production verdict. It compares three
monthly-return paths on the frozen-ensemble TOP_N book:

    baseline      buy-and-hold the TOP_N book until the next monthly rebalance
                  (the current production exit rule)
    experimental  baseline, but zero out (to cash) any held ticker whose
                  monthly exit review says SELL — do NOT re-buy on the
                  ensemble signal alone
    hard-stop     baseline, plus a daily Zhang Case II mandatory-sell: a held
                  ticker that breaches x*_0 on any trading day is sold at that
                  day's close and sits in cash for the remainder of the month

Assumptions (explicit)
----------------------
  - Zhang (f1,f2,λ1,λ2) recalibrated MONTHLY per ticker via
    exit_manager.calibrate_for_month (dt=1/252). Cached per (ticker, month);
    NOT recomputed for the daily hard-stop.
  - Andrade DHMMs retrained monthly per ticker with fixed random_state=42.
    Reproducibility of the Sharpe numbers is preferred over avoiding one
    seed's local optimum — no seed sweep in the monthly loop.
  - K = 0.01 flat transaction-cost approximation (loose; refine later).
  - rho = 0.03 risk-free assumption (≈ 2025-26 avg 3M T-bill).
  - Ignoring slippage, dividends, position-sizing. This diagnoses the exit
    SIGNAL alone.
  - Equal-weight book (1/n). When a ticker is exited its weight goes to cash
    (return 0); it is NOT redistributed to survivors.

Scale caveat (read the attribution before trusting the hard-stop path)
----------------------------------------------------------------------
  Zhang's paper uses K≈0.01 with prices near 1. Real prices here are ~$100+.
  Under K=0.01/rho=0.03, Case II only arises when the calibrated uptick rate
  f1 < rho = 0.03 (a weak/declining name), and there x*_0 = rho·K/(rho−f1) is
  ~0.03 — far below any real share price — so a Case II name hard-stops on the
  first trading day. Most names calibrate to Case I (f1 > 0.03) where the daily
  hard-stop never fires. Expect the hard-stop path to differ from baseline only
  on the handful of names that calibrate to Case II. This is a K/rho scale
  artifact, documented so the diagnostic is read correctly.

Usage
-----
    python scripts/backtest_exit_layer.py --start 2025-01-01 --end 2026-06-30
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # allow `python scripts/…` without install

from src.exit.exit_manager import (  # noqa: E402
    calibrate_for_month,
    daily_hard_stop,
    monthly_exit_review,
)
from src.live.scorer import score_months  # noqa: E402
from src.strategies.ensemble.portfolio_builder import MIN_SCORE, TOP_N  # noqa: E402
with open(ROOT / "config" / "settings.yaml") as f:
    _CFG = yaml.safe_load(f)
DB_PATH = ROOT / _CFG["data"]["paths"]["db"]
HOLDOUT_SCORES = ROOT / "data" / "processed" / "holdout_scores.parquet"
LOG_DIR = ROOT / "logs"

RHO = 0.03
K = 0.01
CALIB_LOOKBACK = 250

logger = logging.getLogger("backtest_exit_layer")


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_prices_wide() -> pd.DataFrame:
    """adj_close pivoted to (date × ticker), date-sorted DatetimeIndex."""
    with sqlite3.connect(str(DB_PATH)) as conn:
        df = pd.read_sql_query(
            "SELECT date, ticker, adj_close FROM prices ORDER BY date", conn
        )
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    return wide


def _ticker_closes(wide: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Per-ticker close frame (index=date, column 'close'), NaNs dropped."""
    s = wide[ticker].dropna()
    return pd.DataFrame({"close": s})


def _price_on_or_before(wide: pd.DataFrame, ticker: str, when: pd.Timestamp) -> float:
    col = wide[ticker].dropna()
    sub = col.loc[col.index <= when]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[-1])


# ── Return helpers ───────────────────────────────────────────────────────────

def _annualized_sharpe(monthly: pd.Series) -> float:
    m = monthly.dropna()
    if len(m) < 2:
        return float("nan")
    sd = m.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(m.mean() / sd * np.sqrt(12))


# ── Core backtest ────────────────────────────────────────────────────────────

def run_backtest(start: str, end: str) -> dict:
    logger.info("Regenerating holdout_scores.parquet for [%s, %s] …", start, end)
    score_months(start=start, end=end)  # overwrites HOLDOUT_SCORES

    scores = pd.read_parquet(HOLDOUT_SCORES)
    scores["date"] = pd.to_datetime(scores["date"])
    scores = scores.sort_values(["date", "ticker"])
    month_ends = sorted(scores["date"].unique())
    logger.info("Loaded %d score rows across %d month-ends",
                len(scores), len(month_ends))

    wide = _load_prices_wide()

    rows: list[dict] = []          # per-month path returns
    trigger_rows: list[dict] = []  # per-month trigger_source attribution
    big_moves: list[dict] = []     # |monthly return| > 5% highlights

    for i, me in enumerate(month_ends[:-1]):
        me = pd.Timestamp(me)
        me_next = pd.Timestamp(month_ends[i + 1])

        month_scores = scores[scores["date"] == me]
        above = month_scores[month_scores["ensemble_score"] > MIN_SCORE]
        book = above.nlargest(min(TOP_N, len(above)), "ensemble_score")
        tickers = [t for t in book["ticker"].tolist() if t in wide.columns]
        n = len(tickers)
        if n == 0:
            logger.warning("%s: empty book, skipping", me.date())
            continue

        base_rets, exp_rets, hs_rets = [], [], []
        cnt = {"zhang": 0, "andrade": 0, "both": 0, "none": 0}

        for t in tickers:
            p0 = _price_on_or_before(wide, t, me)
            p1 = _price_on_or_before(wide, t, me_next)
            if np.isnan(p0) or np.isnan(p1) or p0 <= 0:
                continue
            ret_full = p1 / p0 - 1.0

            # ── calibrate ONCE, reuse for both monthly + daily decisions ──
            closes_df = _ticker_closes(wide, t)
            try:
                calib = calibrate_for_month(t, me, closes_df, CALIB_LOOKBACK)
            except ValueError as exc:
                logger.warning("%s %s: calibration skipped (%s)", me.date(), t, exc)
                base_rets.append(ret_full)
                exp_rets.append(ret_full)
                hs_rets.append(ret_full)
                continue

            # ── monthly experimental review ──
            review = monthly_exit_review(calib, p0, rho=RHO, K=K)
            cnt[review.trigger_source] += 1
            exp_ret = 0.0 if review.action == "SELL" else ret_full

            # ── daily hard-stop over (me, me_next] using the SAME calib ──
            day_prices = wide[t].dropna()
            day_prices = day_prices.loc[
                (day_prices.index > me) & (day_prices.index <= me_next)
            ]
            hs_ret = ret_full
            for day, day_close in day_prices.items():
                hs = daily_hard_stop(calib, float(day_close), rho=RHO, K=K)
                if hs.action == "SELL":
                    hs_ret = float(day_close) / p0 - 1.0
                    break

            base_rets.append(ret_full)
            exp_rets.append(exp_ret)
            hs_rets.append(hs_ret)

            if abs(ret_full) > 0.05 and review.action == "SELL":
                big_moves.append({
                    "month": me.date().isoformat(),
                    "ticker": t,
                    "ret_full": ret_full,
                    "trigger_source": review.trigger_source,
                })

        if not base_rets:
            continue
        base_m = float(np.mean(base_rets))
        exp_m = float(np.sum(exp_rets) / n)   # exits → cash (weight kept at 1/n)
        hs_m = float(np.mean(hs_rets))
        rows.append({
            "month": me.date().isoformat(),
            "baseline": base_m,
            "experimental": exp_m,
            "hard_stop": hs_m,
            "n_held": n,
        })
        trigger_rows.append({"month": me.date().isoformat(), **cnt})
        logger.info(
            "%s  n=%2d  base=%+.4f  exp=%+.4f  hs=%+.4f  "
            "[zhang=%d andrade=%d both=%d]",
            me.date(), n, base_m, exp_m, hs_m,
            cnt["zhang"], cnt["andrade"], cnt["both"],
        )

    ret_df = pd.DataFrame(rows).set_index("month") if rows else pd.DataFrame()
    trig_df = pd.DataFrame(trigger_rows).set_index("month") if trigger_rows else pd.DataFrame()

    sharpe = {
        "baseline": _annualized_sharpe(ret_df["baseline"]) if not ret_df.empty else float("nan"),
        "experimental": _annualized_sharpe(ret_df["experimental"]) if not ret_df.empty else float("nan"),
        "hard_stop": _annualized_sharpe(ret_df["hard_stop"]) if not ret_df.empty else float("nan"),
    }
    return {
        "returns": ret_df,
        "triggers": trig_df,
        "sharpe": sharpe,
        "big_moves": big_moves,
    }


# ── Reporting ────────────────────────────────────────────────────────────────

def write_report(result: dict, start: str, end: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"exit_backtest_{ts}.log"

    ret_df: pd.DataFrame = result["returns"]
    trig_df: pd.DataFrame = result["triggers"]
    sharpe = result["sharpe"]
    big = result["big_moves"]

    d_exp = sharpe["experimental"] - sharpe["baseline"]
    d_hs = sharpe["hard_stop"] - sharpe["baseline"]

    lines: list[str] = []
    lines.append("═" * 72)
    lines.append(f"  Markov Exit Layer — DIAGNOSTIC backtest  [{start} → {end}]")
    lines.append(f"  generated {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"  rho={RHO}  K={K}  calibration_lookback={CALIB_LOOKBACK}d  "
                 f"MIN_SCORE={MIN_SCORE}  TOP_N={TOP_N}")
    lines.append("═" * 72)

    lines.append("\n── Annualized Sharpe (×√12) ──")
    lines.append(f"  baseline      : {sharpe['baseline']:+.4f}")
    lines.append(f"  experimental  : {sharpe['experimental']:+.4f}   "
                 f"(Δ vs baseline: {d_exp:+.4f})")
    lines.append(f"  hard_stop     : {sharpe['hard_stop']:+.4f}   "
                 f"(Δ vs baseline: {d_hs:+.4f})")

    lines.append("\n── Monthly returns (baseline / experimental / hard_stop) ──")
    if not ret_df.empty:
        lines.append(f"  {'month':<12}{'base':>10}{'exp':>10}{'hs':>10}"
                     f"{'exp-base':>11}{'hs-base':>10}{'n':>4}")
        for month, r in ret_df.iterrows():
            lines.append(
                f"  {month:<12}{r['baseline']:>+10.4f}{r['experimental']:>+10.4f}"
                f"{r['hard_stop']:>+10.4f}"
                f"{r['experimental']-r['baseline']:>+11.4f}"
                f"{r['hard_stop']-r['baseline']:>+10.4f}{int(r['n_held']):>4d}"
            )
        lines.append("  " + "-" * 60)
        lines.append(
            f"  {'MEAN':<12}{ret_df['baseline'].mean():>+10.4f}"
            f"{ret_df['experimental'].mean():>+10.4f}"
            f"{ret_df['hard_stop'].mean():>+10.4f}"
        )
    else:
        lines.append("  (no monthly returns computed)")

    lines.append("\n── Override attribution per month (monthly baseline layer) ──")
    lines.append("  zhang-only  = Zhang SELL, Andrade not STRONG_SELL")
    lines.append("  andrade-only= Andrade STRONG_SELL, Zhang HOLD")
    lines.append("  both        = both fired the same month")
    if not trig_df.empty:
        lines.append(f"  {'month':<12}{'zhang':>8}{'andrade':>9}{'both':>7}{'none':>7}")
        for month, r in trig_df.iterrows():
            lines.append(f"  {month:<12}{int(r['zhang']):>8d}{int(r['andrade']):>9d}"
                         f"{int(r['both']):>7d}{int(r['none']):>7d}")
        lines.append("  " + "-" * 42)
        lines.append(f"  {'TOTAL':<12}{int(trig_df['zhang'].sum()):>8d}"
                     f"{int(trig_df['andrade'].sum()):>9d}"
                     f"{int(trig_df['both'].sum()):>7d}{int(trig_df['none'].sum()):>7d}")
    else:
        lines.append("  (no overrides)")

    lines.append("\n── Large moves (|monthly return| > 5%) the exit layer would have exited ──")
    if big:
        for m in big:
            lines.append(f"  {m['month']}  {m['ticker']:<6}  "
                         f"ret={m['ret_full']:+.4f}  caught_by={m['trigger_source']}")
    else:
        lines.append("  (none — no exited ticker had a >5% monthly move)")

    lines.append("\n" + "═" * 72)
    lines.append(f"  VERDICT: experimental Sharpe {'≥' if d_exp >= 0 else '<'} baseline "
                 f"(Δ={d_exp:+.4f})")
    lines.append("═" * 72)

    report = "\n".join(lines)
    path.write_text(report + "\n")
    print(report)
    return path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description="Diagnostic exit-layer backtest")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-06-30")
    args = ap.parse_args()

    result = run_backtest(args.start, args.end)
    path = write_report(result, args.start, args.end)
    logger.info("Report written → %s", path)


if __name__ == "__main__":
    main()
