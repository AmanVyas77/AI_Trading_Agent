"""
Exit manager: fuse Zhang optimal-stopping thresholds with Andrade DHMM regime
signals into a single per-ticker exit decision, on a hybrid cadence.

Cadence (Aman, 2026-07-25): monthly baseline + daily hard-stop.

Separation of calibration from decision
---------------------------------------
Both cadences depend on expensive per-ticker parameters that DO NOT change
intra-month:
    - Andrade's three DHMMs (Baum-Welch fits), and
    - Zhang's (f1, f2, λ1, λ2) from the sign-flip calibration.
These are fitted ONCE per (ticker, month_end) by :func:`calibrate_for_month`,
returned in a :class:`MonthlyCalibration` bundle, and reused by every cheap
decision call — :func:`monthly_exit_review` and :func:`daily_hard_stop` — for
the rest of that month. Neither decision function re-fits.

Andrade → Zhang state mapping
-----------------------------
Andrade's next-observation forecast IS the observable Markov state that Zhang's
model conditions on. Rise-prediction ⇔ uptick regime (state 1); drop-prediction
⇔ downtick regime (state 2):

    Andrade STRONG_BUY  → state 1  (uptick, positive drift)
    Andrade HOLD        → state 1  (default: no clear sell → normal hold)
    Andrade SELL        → state 2  (downtick, negative drift)
    Andrade STRONG_SELL → state 2  (downtick)

Sell rule (monthly baseline)
----------------------------
Force target weight to 0 for a held ticker iff EITHER
    - Zhang :func:`decide` returns SELL, OR
    - Andrade signal is STRONG_SELL.
``trigger_source`` records which layer(s) fired so the diagnostic backtest can
attribute the Sharpe delta per driver.

Daily hard-stop
---------------
Once per trading day, only the Zhang Case II mandatory-sell threshold x*_0 is
checked (Case I is a soft signal the monthly baseline already handles). SELL iff
the position is in a Case II regime and today's close ≥ x*_0, regardless of
state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src.exit.andrade_dhmm import Signal, next_signal
from src.exit.zhang_optimal import (
    CalibratedParams,
    case2_thresholds,
    decide,
    estimate_parameters,
    phi,
)

TriggerSource = Literal["zhang", "andrade", "both", "none"]
Trigger = Literal["monthly_baseline", "daily_hard_stop", "no_action"]

# Andrade forecast action → Zhang observable Markov state (see module docstring).
_ANDRADE_TO_STATE: dict[str, int] = {
    "STRONG_BUY": 1,
    "HOLD": 1,
    "SELL": 2,
    "STRONG_SELL": 2,
}


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MonthlyCalibration:
    """Everything expensive to compute per (ticker, month). Cache and reuse."""
    ticker: str
    month_end: pd.Timestamp
    zhang_params: CalibratedParams
    andrade_signal: Signal
    calibration_lookback: int


@dataclass(frozen=True)
class ExitDecision:
    ticker: str
    action: Literal["HOLD", "SELL"]
    trigger: Trigger
    trigger_source: TriggerSource
    zhang_reason: str
    andrade_reason: str
    current_price: float
    calibration: MonthlyCalibration


# ── Calibration (the ONLY fitting entry point) ───────────────────────────────

def calibrate_for_month(
    ticker: str,
    month_end: pd.Timestamp,
    closes_df: pd.DataFrame,
    calibration_lookback: int = 250,
) -> MonthlyCalibration:
    """Fit Zhang params + Andrade signal once for (ticker, month_end).

    ``closes_df`` must be indexed by date with a ``close`` column and hold at
    least ``calibration_lookback`` rows ending on or before ``month_end``. Only
    the trailing ``calibration_lookback`` rows are used.

    This is the ONLY function here that performs DHMM / parameter fitting —
    the decision functions consume the returned bundle without re-fitting.
    """
    if not isinstance(closes_df.index, pd.DatetimeIndex):
        raise ValueError("closes_df must have a DatetimeIndex")
    if "close" not in closes_df.columns:
        raise ValueError("closes_df must have a 'close' column")

    month_end = pd.Timestamp(month_end)
    window = closes_df.loc[closes_df.index <= month_end]
    if len(window) < calibration_lookback:
        raise ValueError(
            f"{ticker}: need ≥ {calibration_lookback} closes on/before "
            f"{month_end.date()}, got {len(window)}"
        )
    window = window.iloc[-calibration_lookback:]

    zhang_params = estimate_parameters(window["close"].to_numpy(dtype=float))
    andrade_signal = next_signal(window, prev_mode="daily", random_state=42)

    return MonthlyCalibration(
        ticker=ticker,
        month_end=month_end,
        zhang_params=zhang_params,
        andrade_signal=andrade_signal,
        calibration_lookback=calibration_lookback,
    )


# ── Decision functions (cheap; no fitting) ───────────────────────────────────

def monthly_exit_review(
    calibration: MonthlyCalibration,
    current_price: float,
    rho: float = 0.03,
    K: float = 0.01,
) -> ExitDecision:
    """Monthly baseline exit decision from a pre-computed calibration.

    Runs Zhang :func:`decide` under the Andrade-derived state, then combines:
    SELL iff Zhang says SELL OR Andrade says STRONG_SELL.
    """
    cp = calibration.zhang_params
    andrade = calibration.andrade_signal
    state = _ANDRADE_TO_STATE[andrade.action]

    zhang_decision = decide(
        price=current_price,
        state=state,
        rho=rho,
        f1=cp.f1,
        f2=cp.f2,
        lam1=cp.lam1,
        lam2=cp.lam2,
        K=K,
    )

    zhang_fired = zhang_decision.action == "SELL"
    andrade_fired = andrade.action == "STRONG_SELL"

    if zhang_fired and andrade_fired:
        source: TriggerSource = "both"
    elif zhang_fired:
        source = "zhang"
    elif andrade_fired:
        source = "andrade"
    else:
        source = "none"

    action: Literal["HOLD", "SELL"] = "SELL" if (zhang_fired or andrade_fired) else "HOLD"
    trigger: Trigger = "monthly_baseline" if action == "SELL" else "no_action"

    return ExitDecision(
        ticker=calibration.ticker,
        action=action,
        trigger=trigger,
        trigger_source=source,
        zhang_reason=zhang_decision.reason,
        andrade_reason=andrade.reason,
        current_price=current_price,
        calibration=calibration,
    )


def daily_hard_stop(
    calibration: MonthlyCalibration,
    current_price: float,
    rho: float = 0.03,
    K: float = 0.01,
) -> ExitDecision:
    """Daily mandatory-sell check on the Zhang Case II threshold x*_0.

    Uses only the cached ``calibration.zhang_params`` — never re-fits. Case I
    regimes (ρ ≤ f1) and never-sell regimes (Φ ≤ 0) return no_action; the
    monthly baseline owns those softer signals.
    """
    cp = calibration.zhang_params
    args = (rho, cp.f1, cp.f2, cp.lam1, cp.lam2)

    no_action = ExitDecision(
        ticker=calibration.ticker,
        action="HOLD",
        trigger="no_action",
        trigger_source="none",
        zhang_reason="daily hard-stop: not a Case II mandatory-sell",
        andrade_reason=calibration.andrade_signal.reason,
        current_price=current_price,
        calibration=calibration,
    )

    # Case I (ρ ≤ f1) or never-sell (Φ ≤ 0) → no hard threshold to enforce.
    if rho <= cp.f1 or phi(*args) <= 0.0:
        return no_action

    _x_star, x0_star = case2_thresholds(*args, K)
    if current_price >= x0_star:
        return ExitDecision(
            ticker=calibration.ticker,
            action="SELL",
            trigger="daily_hard_stop",
            trigger_source="zhang",
            zhang_reason=(
                f"daily hard-stop: price {current_price:.6g} ≥ x*_0={x0_star:.6g}"
            ),
            andrade_reason=calibration.andrade_signal.reason,
            current_price=current_price,
            calibration=calibration,
        )
    return ExitDecision(
        ticker=calibration.ticker,
        action="HOLD",
        trigger="no_action",
        trigger_source="none",
        zhang_reason=(
            f"daily hard-stop: price {current_price:.6g} < x*_0={x0_star:.6g}"
        ),
        andrade_reason=calibration.andrade_signal.reason,
        current_price=current_price,
        calibration=calibration,
    )
