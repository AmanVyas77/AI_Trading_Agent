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
  - K = K_fraction × entry_price (REV 4 fix A): a FIXED dollar transaction cost
    per position, not the flat K=0.01 of REV 3. K_fraction = 0.001 (0.1%).
  - entry_price (backtest simplification): the first close of the ~250d
    calibration lookback window ending at month_end — a stable proxy for the
    position's cost basis at the start of the calibration period. A future LIVE
    wiring MUST replace this with the actual broker cost basis of the held
    position; the first-close proxy is only defensible for a diagnostic.
  - rho = 0.03 risk-free assumption (≈ 2025-26 avg 3M T-bill).
  - Ignoring slippage, dividends, position-sizing. This diagnoses the exit
    SIGNAL alone.
  - Equal-weight book (1/n). When a ticker is exited its weight goes to cash
    (return 0); it is NOT redistributed to survivors.

Scale note (REV 4 — why the hard-stop can now fire on real names)
-----------------------------------------------------------------
  REV 3 passed a flat K=0.01 (paper-scale) against ~$100 prices, so Case II
  x*_0 = rho·K/(rho−f1) collapsed to ~cents and a Case II name hard-stopped on
  the first trading day (or the condition degenerated to a price-independent
  constant — Opus's Prompt-2 flag). REV 4 references K to entry_price, so
  x*_0 = rho·(K_fraction·entry_price)/(rho−f1) is a genuine dollar LEVEL the
  live price crosses. Case I names (f1 > rho) still never hard-stop; Case II
  names now trigger only when the live price actually breaches x*_0. Report the
  hard-stop trigger count to see how often that happens on this slice.

Regime gate (REV 4 fix B)
-------------------------
  The Andrade STRONG_SELL override is suppressed when the cached regime
  multiplier > NEUTRAL_MULT (a RISK_ON tape).

  LOOKAHEAD FIX (2026-08-03). REV 4 called get_live_regime_signal() inside
  calibrate_for_month, i.e. inside this backtest loop — against that function's
  own docstring ("Only for live/forward use — never called inside the backtest
  loop"). It has no as-of parameter, so every (ticker, month) calibration cached
  the SAME run-time regime: today's macro gating 2025 decisions, all-or-nothing
  across the window, and silently different on every re-run as the macro tables
  refreshed. It now uses get_regime_signal_asof(month_end), which reads only
  macro observations dated on or before that month-end. Suppression is now a
  genuine per-month property; the summary reports the per-month reading.

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
    DEFAULT_K_FRACTION,
    calibrate_for_month,
    daily_hard_stop,
    monthly_exit_review,
)
from src.live.scorer import score_months  # noqa: E402
from src.strategies.ensemble.portfolio_builder import MIN_SCORE, TOP_N  # noqa: E402
from src.strategies.ensemble.regime_gate import (  # noqa: E402
    NEUTRAL_MULT,
    get_regime_signal_asof,
)
from src.utils.data_vintage import format_vintage, price_vintage  # noqa: E402
with open(ROOT / "config" / "settings.yaml") as f:
    _CFG = yaml.safe_load(f)
DB_PATH = ROOT / _CFG["data"]["paths"]["db"]
HOLDOUT_SCORES = ROOT / "data" / "processed" / "holdout_scores.parquet"
LOG_DIR = ROOT / "logs"

RHO = 0.03
# REV 4: transaction cost is now K_fraction × entry_price (a fixed dollar amount
# per position), computed inside calibrate_for_month — not a flat K=0.01. We pass
# K_fraction explicitly for auditability; it mirrors exit_manager.DEFAULT_K_FRACTION.
K_FRACTION = DEFAULT_K_FRACTION  # 0.001 = 0.1% of entry price
CALIB_LOOKBACK = 250

# REV 3 Sharpe numbers (commit 4a2a699) — the diagnostic FAIL we are re-testing.
REV3_SHARPE = {"baseline": 1.5871, "experimental": 1.0783, "hard_stop": 1.5871}

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
    # Pin the price vintage FIRST: the prices table is retroactively rewritten by
    # the AV backfill, so a result is only comparable against another run on the
    # same fingerprint. See src/utils/data_vintage.py.
    vintage = price_vintage(DB_PATH, start, end)
    logger.info("AUDIT: %s", format_vintage(vintage))

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
    suppression_rows: list[dict] = []  # per-month Andrade-suppressed-by-regime count
    k_abs_values: list[float] = []     # every K_absolute used (for distribution)
    hard_stop_rows: list[dict] = []    # per-month hard-stop trigger count
    regime_asof_rows: list[dict] = []  # per-month point-in-time regime reading

    for i, me in enumerate(month_ends[:-1]):
        me = pd.Timestamp(me)
        me_next = pd.Timestamp(month_ends[i + 1])

        # POINT-IN-TIME regime for this month (2026-08-03 lookahead fix).
        # Previously calibrate_for_month called get_live_regime_signal() per
        # (ticker, month), which reads the LATEST macro snapshot — today's macro
        # gating 2025 decisions, and a result that changed whenever the macro
        # tables were refreshed. Fetched once per month and reused across the
        # book: it does not vary by ticker.
        regime_asof = get_regime_signal_asof(me)
        regime_asof_rows.append({
            "month": me.date().isoformat(),
            "multiplier": regime_asof["multiplier"],
            "rule_signal": regime_asof["rule_signal"],
            "stale": regime_asof["stale"],
        })

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
        suppressed = 0   # Andrade STRONG_SELLs suppressed by RISK_ON regime
        hs_fired = 0     # hard-stop SELL triggers this month

        for t in tickers:
            p0 = _price_on_or_before(wide, t, me)
            p1 = _price_on_or_before(wide, t, me_next)
            if np.isnan(p0) or np.isnan(p1) or p0 <= 0:
                continue
            ret_full = p1 / p0 - 1.0

            # ── calibrate ONCE, reuse for both monthly + daily decisions ──
            closes_df = _ticker_closes(wide, t)
            # entry_price = first close of the ~250d calibration window ending at
            # me (cost-basis proxy for the diagnostic — see header). Mirrors the
            # window calibrate_for_month slices internally.
            window = closes_df.loc[closes_df.index <= me].iloc[-CALIB_LOOKBACK:]
            entry_price = float(window["close"].iloc[0]) if not window.empty else float("nan")
            try:
                calib = calibrate_for_month(
                    t, me, closes_df,
                    entry_price=entry_price,
                    K_fraction=K_FRACTION,
                    calibration_lookback=CALIB_LOOKBACK,
                    regime_signal=regime_asof,
                )
            except ValueError as exc:
                logger.warning("%s %s: calibration skipped (%s)", me.date(), t, exc)
                base_rets.append(ret_full)
                exp_rets.append(ret_full)
                hs_rets.append(ret_full)
                continue

            k_abs_values.append(calib.K_absolute)
            # A STRONG_SELL that the regime gate suppressed (RISK_ON tape).
            if (calib.andrade_signal.action == "STRONG_SELL"
                    and calib.regime_signal["multiplier"] > NEUTRAL_MULT):
                suppressed += 1

            # ── monthly experimental review ──
            review = monthly_exit_review(calib, p0, rho=RHO)
            cnt[review.trigger_source] += 1
            exp_ret = 0.0 if review.action == "SELL" else ret_full

            # ── daily hard-stop over (me, me_next] using the SAME calib ──
            day_prices = wide[t].dropna()
            day_prices = day_prices.loc[
                (day_prices.index > me) & (day_prices.index <= me_next)
            ]
            hs_ret = ret_full
            for day, day_close in day_prices.items():
                hs = daily_hard_stop(calib, float(day_close), rho=RHO)
                if hs.action == "SELL":
                    hs_ret = float(day_close) / p0 - 1.0
                    hs_fired += 1
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
        suppression_rows.append({"month": me.date().isoformat(), "suppressed": suppressed})
        hard_stop_rows.append({"month": me.date().isoformat(), "hs_fired": hs_fired})
        logger.info(
            "%s  n=%2d  base=%+.4f  exp=%+.4f  hs=%+.4f  "
            "[zhang=%d andrade=%d both=%d]  suppressed=%d  hs_fired=%d",
            me.date(), n, base_m, exp_m, hs_m,
            cnt["zhang"], cnt["andrade"], cnt["both"], suppressed, hs_fired,
        )

    ret_df = pd.DataFrame(rows).set_index("month") if rows else pd.DataFrame()
    trig_df = pd.DataFrame(trigger_rows).set_index("month") if trigger_rows else pd.DataFrame()
    supp_df = pd.DataFrame(suppression_rows).set_index("month") if suppression_rows else pd.DataFrame()
    hs_df = pd.DataFrame(hard_stop_rows).set_index("month") if hard_stop_rows else pd.DataFrame()

    sharpe = {
        "baseline": _annualized_sharpe(ret_df["baseline"]) if not ret_df.empty else float("nan"),
        "experimental": _annualized_sharpe(ret_df["experimental"]) if not ret_df.empty else float("nan"),
        "hard_stop": _annualized_sharpe(ret_df["hard_stop"]) if not ret_df.empty else float("nan"),
    }
    k_series = pd.Series(k_abs_values, dtype=float)
    k_abs_summary = {
        "n": int(k_series.size),
        "min": float(k_series.min()) if k_series.size else float("nan"),
        "median": float(k_series.median()) if k_series.size else float("nan"),
        "max": float(k_series.max()) if k_series.size else float("nan"),
    }
    return {
        "returns": ret_df,
        "triggers": trig_df,
        "suppression": supp_df,
        "hard_stop_triggers": hs_df,
        "k_abs_summary": k_abs_summary,
        "sharpe": sharpe,
        "big_moves": big_moves,
        "price_vintage": vintage,
        "regime_asof": regime_asof_rows,
    }


# ── Reporting ────────────────────────────────────────────────────────────────

def write_report(result: dict, start: str, end: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"exit_backtest_{ts}.log"

    ret_df: pd.DataFrame = result["returns"]
    trig_df: pd.DataFrame = result["triggers"]
    supp_df: pd.DataFrame = result["suppression"]
    hs_df: pd.DataFrame = result["hard_stop_triggers"]
    k_abs = result["k_abs_summary"]
    sharpe = result["sharpe"]
    big = result["big_moves"]

    d_exp = sharpe["experimental"] - sharpe["baseline"]
    d_hs = sharpe["hard_stop"] - sharpe["baseline"]

    lines: list[str] = []
    lines.append("═" * 72)
    lines.append(f"  Markov Exit Layer — DIAGNOSTIC backtest  [{start} → {end}]")
    lines.append(f"  generated {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"  rho={RHO}  K_fraction={K_FRACTION}  "
                 f"calibration_lookback={CALIB_LOOKBACK}d  "
                 f"MIN_SCORE={MIN_SCORE}  TOP_N={TOP_N}")
    v = result["price_vintage"]
    lines.append(f"  {format_vintage(v)}")
    lines.append(f"  price_vintage sha256 (full): {v['sha256']}")
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

    # ── Regime suppression (REV 4 fix B) ──────────────────────────────────
    lines.append("\n── Regime suppression of Andrade STRONG_SELL (REV 4 fix B) ──")
    lines.append("  A STRONG_SELL suppressed because the live-regime multiplier > "
                 f"NEUTRAL_MULT ({NEUTRAL_MULT}).")
    if not supp_df.empty:
        total_supp = int(supp_df["suppressed"].sum())
        months_with_supp = int((supp_df["suppressed"] > 0).sum())
        lines.append(f"  {'month':<12}{'suppressed':>12}")
        for month, r in supp_df.iterrows():
            lines.append(f"  {month:<12}{int(r['suppressed']):>12d}")
        lines.append("  " + "-" * 24)
        lines.append(f"  {'TOTAL':<12}{total_supp:>12d}")
        lines.append(f"  months with ≥1 suppression: {months_with_supp} / {len(supp_df)}")
    else:
        lines.append("  (no calibrations)")

    # ── Hard-stop trigger count ───────────────────────────────────────────
    lines.append("\n── Daily hard-stop triggers (REV 3 = 0) ──")
    if not hs_df.empty:
        total_hs = int(hs_df["hs_fired"].sum())
        months_with_hs = int((hs_df["hs_fired"] > 0).sum())
        lines.append(f"  {'month':<12}{'hs_fired':>10}")
        for month, r in hs_df.iterrows():
            lines.append(f"  {month:<12}{int(r['hs_fired']):>10d}")
        lines.append("  " + "-" * 22)
        lines.append(f"  {'TOTAL':<12}{total_hs:>10d}")
        lines.append(f"  months with ≥1 hard-stop: {months_with_hs} / {len(hs_df)}")
    else:
        lines.append("  (no calibrations)")

    # ── Point-in-time regime per month (2026-08-03 lookahead fix) ─────────
    lines.append("\n── Point-in-time regime per month (as-of, NOT run-time) ──")
    lines.append("  Was: get_live_regime_signal() — latest snapshot applied to every month.")
    lines.append("  Now: get_regime_signal_asof(month_end) — macro dated ≤ month_end only.")
    ra = result["regime_asof"]
    if ra:
        lines.append(f"  {'month':<12}{'mult':>7}{'rule':>10}{'stale':>8}")
        for r in ra:
            lines.append(f"  {r['month']:<12}{r['multiplier']:>7.2f}"
                         f"{r['rule_signal']:>10}{str(r['stale']):>8}")
        n_supp_possible = sum(1 for r in ra if r["multiplier"] > NEUTRAL_MULT)
        lines.append(f"  months where regime could suppress Andrade "
                     f"(mult > {NEUTRAL_MULT}): {n_supp_possible} / {len(ra)}")
    else:
        lines.append("  (none)")

    # ── K_absolute distribution ───────────────────────────────────────────
    lines.append("\n── K_absolute distribution across all (ticker, month) calibrations ──")
    lines.append(f"  K_fraction = {K_FRACTION}  (K_absolute = K_fraction × entry_price)")
    lines.append(f"  n={k_abs['n']}  min={k_abs['min']:.4f}  "
                 f"median={k_abs['median']:.4f}  max={k_abs['max']:.4f}")
    lines.append(f"  (median should be ≈ {K_FRACTION} × median entry price in the universe)")

    # ── REV 3 vs REV 4 comparison ─────────────────────────────────────────
    lines.append("\n── REV 3 vs REV 4 comparison ──")
    lines.append(f"  {'Path':<14}{'REV 3 Sharpe':>14}{'REV 4 Sharpe':>14}"
                 f"{'Δ (REV4−REV3)':>16}")
    for path_label, key in (("baseline", "baseline"),
                            ("experimental", "experimental"),
                            ("hard-stop", "hard_stop")):
        r3 = REV3_SHARPE[key]
        r4 = sharpe[key]
        lines.append(f"  {path_label:<14}{r3:>+14.4f}{r4:>+14.4f}{r4 - r3:>+16.4f}")

    # ── Verdict against the REV 4 rules ───────────────────────────────────
    exp_r4 = sharpe["experimental"]
    base_drift = abs(sharpe["baseline"] - REV3_SHARPE["baseline"])
    rule2 = exp_r4 > REV3_SHARPE["experimental"]        # improve on REV 3
    rule3 = exp_r4 >= REV3_SHARPE["baseline"]           # meet baseline
    if rule3:
        verdict = "FULL PASS (rules 2 & 3)"
    elif rule2:
        verdict = "MINIMUM PASS (rule 2 only)"
    else:
        verdict = "FAIL (below rule 2)"

    lines.append("\n" + "═" * 72)
    lines.append(f"  baseline drift vs REV 3: {base_drift:.6f} "
                 f"({'OK' if base_drift <= 1e-4 else 'DRIFT > 1e-4 — INVESTIGATE'})")
    lines.append(f"  rule 2 (exp > {REV3_SHARPE['experimental']:+.4f}): "
                 f"{'PASS' if rule2 else 'FAIL'}")
    lines.append(f"  rule 3 (exp ≥ {REV3_SHARPE['baseline']:+.4f}): "
                 f"{'PASS' if rule3 else 'FAIL'}")
    lines.append(f"  VERDICT: {verdict}   experimental Sharpe {exp_r4:+.4f} "
                 f"(Δ vs baseline {d_exp:+.4f})")
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
