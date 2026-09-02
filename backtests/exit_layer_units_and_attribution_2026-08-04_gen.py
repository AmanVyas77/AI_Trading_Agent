"""Prompt 1D — Zhang units resolution (Task A) + pivotality attribution (Task B).

Replays the exact decision path of scripts/backtest_exit_layer.run_backtest over
the frozen vintage, but records, per (ticker, month):

  * the full fitted parameter set (f1, f2, lam1, lam2, phi, case)
  * the Zhang thresholds actually used (x*, x*_0) and the dollar quantities they
    are compared against (p0, entry_price, K_absolute)
  * the REAL decision from src.exit.exit_manager.monthly_exit_review
  * counterfactual re-evaluations for pivotality (Task B items 5-6)

The counterfactual evaluator `_compose()` is a re-implementation of the SELL
composition in monthly_exit_review. Every row asserts _compose() under the
observed conditions reproduces the real decision, so a drift between the two
fails loudly rather than silently producing a wrong attribution table.

Output: units_pivotality_rows.csv (one row per calibration) + a printed summary.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/aman/dev/Ai Trading Agent")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import backtest_exit_layer as bt  # noqa: E402
from src.exit.exit_manager import (  # noqa: E402
    NEUTRAL_MULT,
    _ANDRADE_TO_STATE,
    calibrate_for_month,
    monthly_exit_review,
)
from src.exit.zhang_optimal import (  # noqa: E402
    beta2,
    case1_threshold,
    case2_thresholds,
    phi,
)
from src.live.scorer import score_months  # noqa: E402
from src.strategies.ensemble.portfolio_builder import MIN_SCORE, TOP_N  # noqa: E402
from src.strategies.ensemble.regime_gate import get_regime_signal_asof  # noqa: E402

OUT = Path(__file__).resolve().parent
RHO = bt.RHO
K_FRACTION = bt.K_FRACTION
CALIB_LOOKBACK = bt.CALIB_LOOKBACK

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("units_pivotality")


# ── Counterfactual composition ───────────────────────────────────────────────

def _compose(case1: bool, phi_pos: bool, state2: bool, price_ok: bool,
             andrade_strong: bool, regime_permits: bool) -> bool:
    """Final monthly SELL as a function of the primitive conditions.

    Mirrors monthly_exit_review + zhang_optimal.decide for the regimes that
    occur on this window (Case I and never-sell; Case II is empty — see report).
    `price_ok` is the state-appropriate price predicate already evaluated.
    """
    zhang_fired = phi_pos and case1 and state2 and price_ok
    andrade_fired = andrade_strong and regime_permits
    return zhang_fired or andrade_fired


def main() -> None:
    bt._assert_interpreter()
    manifest = bt.use_vintage("backtests/vintage_2026-08-04")
    start, end = "2025-01-01", "2026-06-30"

    vintage = bt.price_vintage(bt.DB_PATH, start, end)
    log.info("AUDIT: %s", bt.format_vintage(vintage))

    scores_out = ROOT / "logs" / "holdout_scores_run_prompt1d.parquet"
    score_months(start=start, end=end,
                 feature_matrix_path=bt.FEATURE_MATRIX, output_path=scores_out)
    frozen = pd.read_parquet(bt.VINTAGE_DIR / "holdout_scores.parquet")
    regen = pd.read_parquet(scores_out)
    a = frozen.sort_values(["date", "ticker"]).reset_index(drop=True)
    b = regen.sort_values(["date", "ticker"]).reset_index(drop=True)
    same = (len(a) == len(b)
            and bool((a["ensemble_score"].values == b["ensemble_score"].values).all()))
    if not same:
        raise SystemExit("ABORT: regenerated scores differ from frozen snapshot")
    log.info("VINTAGE: regenerated scores match frozen copy bit-for-bit: %s", same)

    scores = pd.read_parquet(scores_out)
    scores["date"] = pd.to_datetime(scores["date"])
    scores = scores.sort_values(["date", "ticker"])
    month_ends = sorted(scores["date"].unique())
    wide = bt._load_prices_wide()

    rows: list[dict] = []
    for i, me in enumerate(month_ends[:-1]):
        me = pd.Timestamp(me)
        regime_asof = get_regime_signal_asof(me)
        month_scores = scores[scores["date"] == me]
        above = month_scores[month_scores["ensemble_score"] > MIN_SCORE]
        book = above.nlargest(min(TOP_N, len(above)), "ensemble_score")
        tickers = [t for t in book["ticker"].tolist() if t in wide.columns]

        for t in tickers:
            p0 = bt._price_on_or_before(wide, t, me)
            p1 = bt._price_on_or_before(wide, t, pd.Timestamp(month_ends[i + 1]))
            if np.isnan(p0) or np.isnan(p1) or p0 <= 0:
                continue
            closes_df = bt._ticker_closes(wide, t)
            window = closes_df.loc[closes_df.index <= me].iloc[-CALIB_LOOKBACK:]
            entry_price = float(window["close"].iloc[0]) if not window.empty else float("nan")
            try:
                calib = calibrate_for_month(
                    t, me, closes_df, entry_price=entry_price,
                    K_fraction=K_FRACTION, calibration_lookback=CALIB_LOOKBACK,
                    regime_signal=regime_asof)
            except ValueError as exc:
                log.warning("%s %s: calibration skipped (%s)", me.date(), t, exc)
                continue

            zp = calib.zhang_params
            args = (RHO, zp.f1, zp.f2, zp.lam1, zp.lam2)
            ph = phi(*args)
            phi_pos = ph > 0.0
            case1 = phi_pos and RHO <= zp.f1
            case2 = phi_pos and RHO > zp.f1

            x_star = float("nan")
            x0_star = float("nan")
            b2 = float("nan")
            if case1:
                x_star = case1_threshold(*args, calib.K_absolute)
                b2 = beta2(*args)
            elif case2:
                x_star, x0_star = case2_thresholds(*args, calib.K_absolute)
                b2 = beta2(*args)

            state = _ANDRADE_TO_STATE[calib.andrade_signal.action]
            state2 = state == 2
            andrade_strong = calib.andrade_signal.action == "STRONG_SELL"
            regime_permits = regime_asof["multiplier"] <= NEUTRAL_MULT

            # Observed price predicate, state-appropriate.
            if case1:
                price_ok = bool(p0 >= x_star)
            elif case2:
                price_ok = bool(p0 >= (x_star if state2 else x0_star))
            else:
                price_ok = False  # never-sell: no threshold exists

            review = monthly_exit_review(calib, p0, rho=RHO, allow_andrade=True)
            real_sell = review.action == "SELL"

            # Self-check: the counterfactual composer must reproduce reality.
            recon = _compose(case1, phi_pos, state2, price_ok,
                             andrade_strong, regime_permits)
            if recon != real_sell:
                raise SystemExit(
                    f"ABORT: _compose mismatch {me.date()} {t}: "
                    f"recon={recon} real={real_sell} case1={case1} phi={ph} "
                    f"state2={state2} price_ok={price_ok}")

            # ── Pivotality (Task B item 5) ───────────────────────────────────
            # A condition is pivotal iff forcing it True vs forcing it False
            # changes the final decision, holding every other condition fixed.
            price_piv = (_compose(case1, phi_pos, state2, True,
                                  andrade_strong, regime_permits)
                         != _compose(case1, phi_pos, state2, False,
                                     andrade_strong, regime_permits))
            # State flip: re-evaluate the price predicate under the flipped
            # state, since Case II uses a different threshold per state. In
            # Case I state 1 never sells, so price_ok under state 1 is moot.
            price_ok_s2 = price_ok if state2 else (
                bool(p0 >= x_star) if (case1 or case2) else False)
            price_ok_s1 = (False if case1 else
                           (bool(p0 >= x0_star) if case2 else False))
            state_piv = (_compose(case1, phi_pos, True, price_ok_s2,
                                  andrade_strong, regime_permits)
                         != _compose(case1, phi_pos, False, price_ok_s1,
                                     andrade_strong, regime_permits))
            case_piv = (_compose(True, phi_pos, state2, price_ok,
                                 andrade_strong, regime_permits)
                        != _compose(False, phi_pos, state2, price_ok,
                                    andrade_strong, regime_permits))

            rows.append({
                "month": me.date().isoformat(), "ticker": t,
                "f1": zp.f1, "f2": zp.f2, "lam1": zp.lam1, "lam2": zp.lam2,
                "mu": zp.mu, "sigma0": zp.sigma0, "phi": ph, "beta2": b2,
                "case": "case1" if case1 else ("case2" if case2 else "never_sell"),
                "state": state, "andrade_action": calib.andrade_signal.action,
                "regime_mult": regime_asof["multiplier"],
                "regime_permits": regime_permits,
                "p0": p0, "entry_price": calib.entry_price,
                "K_absolute": calib.K_absolute,
                "x_star": x_star, "x0_star": x0_star,
                "x_star_over_p0": x_star / p0 if np.isfinite(x_star) else float("nan"),
                "x_star_over_K": x_star / calib.K_absolute if np.isfinite(x_star) else float("nan"),
                "price_ok": price_ok,
                "real_sell": real_sell,
                "trigger_source": review.trigger_source,
                "price_test_pivotal": price_piv,
                "state_test_pivotal": state_piv,
                "case_gate_pivotal": case_piv,
            })

    df = pd.DataFrame(rows)
    out_csv = OUT / "units_pivotality_rows.csv"
    df.to_csv(out_csv, index=False)
    log.info("wrote %s  (%d rows)", out_csv, len(df))
    print("\nMANIFEST git_head:", manifest["git_head"])
    print("price_vintage sha256:", vintage.get("sha256", vintage))


if __name__ == "__main__":
    main()
