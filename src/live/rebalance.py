"""
Live target-book construction (broker-agnostic).

Two public entry points:

    build_targets(as_of_month_end=None) -> dict
        1. Score the given month-end with the frozen model.
        2. Select TOP_N by ensemble_score with MIN_SCORE floor
           (mirrors portfolio_builder._select_monthly_holdings — see
           inline citation).
        3. Apply the live regime multiplier from get_live_regime_signal()
           (rules-only default per Sprint 8 Prompt 2).
        4. Persist the resulting target book to
           data/live/targets_<YYYY-MM>.json AND SQLite live_targets
           (INSERT OR REPLACE by month).

    diff_orders(targets, current_positions, equity) -> list[dict]
        Pure function. Given a target book, the current broker book,
        and account equity, return the order list that moves current →
        target. Do-not-trade band: skip any order whose notional is
        below 0.25% of equity (kills dust churn and preserves
        idempotency across a monthly cycle).

The constants MIN_SCORE, TOP_N, NEUTRAL_MULT, RISK_ON_MULT are imported
from their canonical modules — this file is *not* a source of truth.

CLI
---
    python -m src.live.rebalance [--as-of YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from src.live.scorer import score_months
from src.strategies.ensemble.portfolio_builder import MIN_SCORE, TOP_N
from src.strategies.ensemble.regime_gate import (
    NEUTRAL_MULT, RISK_ON_MULT, get_live_regime_signal,
)

ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config" / "settings.yaml") as f:
    _CFG = yaml.safe_load(f)

DB_PATH = ROOT / _CFG["data"]["paths"]["db"]
LIVE_DIR = ROOT / "data" / "live"

# 0.25% of equity is the smallest position change we will actually
# submit — orders below this threshold get skipped so a monthly cycle
# re-run on the same target book generates zero orders.
DO_NOT_TRADE_BAND = 0.0025

logger = logging.getLogger(__name__)


# ── Persistence ──────────────────────────────────────────────────────────────

def _ensure_live_targets_table() -> None:
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS live_targets ("
            "  month TEXT PRIMARY KEY, "
            "  generated_at TEXT NOT NULL, "
            "  payload TEXT NOT NULL"
            ")"
        )


def _persist_targets(targets: dict) -> None:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    month_key = pd.Timestamp(targets["as_of"]).strftime("%Y-%m")
    json_path = LIVE_DIR / f"targets_{month_key}.json"
    with open(json_path, "w") as f:
        json.dump(targets, f, indent=2, default=str)

    _ensure_live_targets_table()
    payload = json.dumps(targets, default=str)
    with sqlite3.connect(str(DB_PATH)) as conn:
        existing = conn.execute(
            "SELECT 1 FROM live_targets WHERE month = ?", (month_key,)
        ).fetchone()
        if existing:
            logger.info(
                "live_targets row for %s exists — INSERT OR REPLACE",
                month_key,
            )
        conn.execute(
            "INSERT OR REPLACE INTO live_targets "
            "(month, generated_at, payload) VALUES (?, ?, ?)",
            (month_key, datetime.now().isoformat(timespec="seconds"), payload),
        )
    logger.info("Persisted targets → %s (+ live_targets['%s'])", json_path, month_key)


# ── build_targets ────────────────────────────────────────────────────────────

def _resolve_month_end(as_of_month_end: Optional[str]) -> pd.Timestamp:
    if as_of_month_end:
        return pd.Timestamp(as_of_month_end)
    today = pd.Timestamp(date.today())
    return today.to_period("M").to_timestamp(how="start") - pd.Timedelta(days=1)


def build_targets(as_of_month_end: Optional[str] = None) -> dict:
    """Score the given month-end, select TOP_N, apply regime gate, persist."""
    month_end = _resolve_month_end(as_of_month_end)
    me_str = month_end.strftime("%Y-%m-%d")
    logger.info("build_targets: month_end=%s", me_str)

    scores = score_months(me_str, me_str)
    if scores.empty:
        raise RuntimeError(
            f"No scores for {me_str} — rebuild feature_matrix / run refresh"
        )
    scores = scores[scores["date"] == month_end].copy()

    # Mirrors portfolio_builder._select_monthly_holdings steps (a)-(d)
    # at src/strategies/ensemble/portfolio_builder.py:128-156:
    #   (a) filter by MIN_SCORE
    #   (b) nlargest TOP_N by ensemble_score
    #   (d) equal weight = 1/n_selected among chosen names
    # then scale by the regime multiplier. This is the VALIDATED
    # weighting — it concentrates when breadth is thin (2 names ×
    # RISK_ON = 60% each) and matches the holdout backtest. Σweights
    # equals the multiplier: 1.0 at neutral, 1.2 gross under RISK_ON
    # (uses Alpaca 2× paper margin — verified in paper_runner).
    above = scores[scores["ensemble_score"] > MIN_SCORE]
    selected = above.nlargest(min(TOP_N, len(above)), "ensemble_score")
    n_selected = len(selected)

    regime = get_live_regime_signal()
    logger.info("Regime signal (verbatim): %s", regime)

    if n_selected == 0:
        scaled_weight = 0.0
    else:
        scaled_weight = regime["multiplier"] / n_selected
    total_equity_weight = scaled_weight * n_selected
    # cash_weight = 1 - Σweights by definition. Negative under RISK_ON
    # (gross exposure > 1.0) — paper_runner enforces buying_power ≥ gross
    # before submission so we can't overshoot margin.
    cash_weight = 1.0 - total_equity_weight

    targets_list = [
        {
            "ticker": row["ticker"],
            "score": round(float(row["ensemble_score"]), 6),
            "weight": round(scaled_weight, 6),
        }
        for _, row in selected.iterrows()
    ]

    payload = {
        "as_of": me_str,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "min_score": MIN_SCORE,
        "top_n": TOP_N,
        "n_selected": n_selected,
        "per_name_weight": round(scaled_weight, 6),
        "gross_exposure": round(total_equity_weight, 6),
        "regime": {
            "multiplier": regime["multiplier"],
            "rule_signal": regime["rule_signal"],
            "llm_signal": regime["llm_signal"],
            "stale": regime["stale"],
            "stale_series": regime["stale_series"],
            "reasoning": regime["reasoning"],
            "macro_snapshot": regime["macro_snapshot"],
        },
        "targets": targets_list,
        "cash_weight": round(cash_weight, 6),
    }

    _persist_targets(payload)
    return payload


# ── diff_orders ──────────────────────────────────────────────────────────────

def diff_orders(
    targets: dict,
    current_positions: list[dict],
    equity: float,
) -> list[dict]:
    """Pure function: current book → target book → order list.

    Parameters
    ----------
    targets           : dict from build_targets(); uses ``targets`` and
                        ``cash_weight``.
    current_positions : [{"ticker", "qty", "market_value"}]. Missing
                        tickers count as 0 exposure.
    equity            : account equity (cash + market value of positions).

    Returns
    -------
    list[{"ticker", "side", "notional"}] — one order per ticker whose
    target notional differs from current by more than the do-not-trade
    band (default 0.25% of equity). Ordered sells first, then buys, so a
    single-pass submission frees cash before consuming it.
    """
    if equity <= 0:
        raise ValueError(f"equity must be positive, got {equity}")

    target_notional = {
        t["ticker"]: t["weight"] * equity for t in targets.get("targets", [])
    }
    current_notional = {
        p["ticker"]: float(p.get("market_value", 0.0)) for p in current_positions
    }

    all_tickers = set(target_notional) | set(current_notional)
    band = DO_NOT_TRADE_BAND * equity

    orders: list[dict] = []
    for tkr in all_tickers:
        tgt = target_notional.get(tkr, 0.0)
        cur = current_notional.get(tkr, 0.0)
        delta = tgt - cur
        if abs(delta) < band:
            continue
        side = "buy" if delta > 0 else "sell"
        orders.append({
            "ticker": tkr,
            "side": side,
            "notional": round(abs(delta), 2),
        })

    orders.sort(key=lambda o: (0 if o["side"] == "sell" else 1, o["ticker"]))
    return orders


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ap = argparse.ArgumentParser(description="Build live target book (no broker)")
    ap.add_argument("--as-of", default=None,
                    help="Month-end date (YYYY-MM-DD); defaults to last completed")
    args = ap.parse_args()

    payload = build_targets(as_of_month_end=args.as_of)

    print("\n" + "═" * 72)
    print(f"  Target book — as of {payload['as_of']}")
    print("═" * 72)
    print(f"  Regime:      mult={payload['regime']['multiplier']}  "
          f"rule={payload['regime']['rule_signal']}  "
          f"llm={payload['regime']['llm_signal']}  "
          f"stale={payload['regime']['stale']}")
    print(f"  Selected:    {payload['n_selected']}/{payload['top_n']} "
          f"(min_score={payload['min_score']})")
    print(f"  Cash weight: {payload['cash_weight']:.2%}")
    print("  Top 5 by weight (equal-weighted within regime):")
    for t in payload["targets"][:5]:
        print(f"    {t['ticker']:<6s}  score={t['score']:.4f}  weight={t['weight']:.4f}")
    print("═" * 72)


if __name__ == "__main__":
    main()
