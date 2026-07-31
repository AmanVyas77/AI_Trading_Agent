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
    - Andrade signal is STRONG_SELL *and the regime permits the override*
      (see the regime-gate section below).
``trigger_source`` records which layer(s) fired so the diagnostic backtest can
attribute the Sharpe delta per driver.

Regime gate on the Andrade override (REV 4 fix B)
-------------------------------------------------
REV 3 let a STRONG_SELL fire the monthly exit unconditionally. In a bull tape
that repeatedly cut winners on transient overbought DHMM forecasts and dragged
the experimental Sharpe *below* the frozen baseline — the "Andrade-in-bull-tape"
finding recorded against REV 3. REV 4 gates the override behind the live macro
regime: the Andrade STRONG_SELL override only fires when the cached regime
multiplier is ≤ :data:`~src.strategies.ensemble.regime_gate.NEUTRAL_MULT`
(i.e. NEUTRAL or RISK_OFF). In a RISK_ON tape (multiplier > NEUTRAL_MULT) the
override is suppressed and only Zhang's optimal-stopping decision can force a
sell. The regime reading is fetched once per calibration
(:func:`~src.strategies.ensemble.regime_gate.get_live_regime_signal`) and cached
on :class:`MonthlyCalibration`, since it does not change intra-month for our
purposes.

Zhang's decision path is UNCHANGED by the gate — its threshold math (now correct
under entry-price K, see below) fires regardless of regime.

Daily hard-stop
---------------
Once per trading day, only the Zhang Case II mandatory-sell threshold x*_0 is
checked (Case I is a soft signal the monthly baseline already handles). SELL iff
the position is in a Case II regime and today's close ≥ x*_0, regardless of
state.

Transaction-cost scaling (K) — REV 4 fix A (entry_price reference)
------------------------------------------------------------------
Zhang's fixed transaction cost K is a *price-dimensioned* quantity: every
threshold (Case I x*, Case II x*_0 = ρK/(ρ−f1)) is linear in K and therefore
carries the units of K. REV 3 passed a flat ``K = 0.01`` into
:mod:`src.exit.zhang_optimal` regardless of the share price, which is
dimensionally wrong for real equities trading at $50–$500: a 1-cent absolute
cost made x*/x*_0 collapse to a tiny sub-cent level, so the daily hard-stop
fired at prices barely above zero. That is the diagnostic FAIL recorded in
commit ``4a2a699``.

Prompt 2 (REV 1 of this remediation) replaced the flat K with
``K = K_fraction × current_price``. That fixed the dimensional bug but Opus
flagged a subtler defect: with K tied to the *current* price, Zhang's Case II
hard-stop condition ``current_price ≥ x*_0`` algebraically simplifies to

    current_price ≥ ρ·(K_fraction·current_price)/(ρ − f1)
    ⇔ 1 ≥ ρ·K_fraction/(ρ − f1)                    (price cancels)

— a *constant* with no price dependency. The price-crossing threshold collapses
into a regime yes/no gate, defeating Zhang's paper intent (the whole point is a
price level the asset crosses).

REV 4 fix A references K to the **entry price** instead: ``entry_price`` is fixed
at calibration time and does NOT evolve with the market, so
``K_absolute = K_fraction × entry_price`` is a fixed dollar amount per position
("cost basis + transaction cost" semantics). Zhang's thresholds are then fixed
dollar levels the *live* price genuinely crosses, restoring real price-crossing
behaviour: x*_0 = ρ·K_absolute/(ρ − f1) is a constant *level*, and
``current_price ≥ x*_0`` is a genuine crossing test again.

``K_absolute`` and ``entry_price`` are computed once in
:func:`calibrate_for_month` and cached on :class:`MonthlyCalibration`; the
decision functions consume ``calibration.K_absolute`` directly and NEVER
recompute K from ``current_price``. Both are echoed onto :class:`ExitDecision`
(alongside ``K_fraction``, retained for audit/logging) so the diagnostic backtest
can log the exact thresholds used per (ticker, month). ``zhang_optimal`` itself is
unchanged — the paper math was always correct; only this caller layer chose the
wrong reference price. In backtests ``entry_price`` is the first close of the
calibration window; live trading passes the actual position cost basis when
available.
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
from src.strategies.ensemble.regime_gate import (
    get_live_regime_signal,
    NEUTRAL_MULT,
)

TriggerSource = Literal["zhang", "andrade", "both", "none"]
Trigger = Literal["monthly_baseline", "daily_hard_stop", "no_action"]

# Transaction cost as a fraction of the ENTRY price (fixed at calibration),
# multiplied by ``entry_price`` in calibrate_for_month to get the absolute K
# handed to Zhang. Referencing entry_price (not current_price) keeps K a fixed
# dollar amount per position so Zhang's thresholds stay genuine price-crossing
# levels — see the "REV 4 fix A" section of the module docstring for why the
# earlier current_price reference collapsed the hard-stop into a constant.
DEFAULT_K_FRACTION = 0.001  # 0.1% transaction cost

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
    # REV 4 fix A — K referenced to a FIXED entry price, not current_price.
    entry_price: float          # reference price for K, fixed at calibration
                                # (backtest: first close of the window; live:
                                # position cost basis when available).
    K_fraction: float           # retained for audit / logging
    K_absolute: float           # K_fraction × entry_price, FIXED for the life
                                # of this calibration; what Zhang's decide()
                                # consumes (never recomputed from current_price).
    # REV 4 fix B — live macro regime, cached (doesn't change intra-month for
    # our purposes). Gates the Andrade STRONG_SELL override.
    regime_signal: dict


@dataclass(frozen=True)
class ExitDecision:
    ticker: str
    action: Literal["HOLD", "SELL"]
    trigger: Trigger
    trigger_source: TriggerSource
    zhang_reason: str
    andrade_reason: str
    current_price: float
    # Transaction-cost inputs actually handed to Zhang for this decision, echoed
    # from the calibration. K_absolute = K_fraction × entry_price is FIXED at
    # calibration time (REV 4 fix A) — it does NOT depend on current_price.
    entry_price: float
    K_fraction: float
    K_absolute: float
    calibration: MonthlyCalibration


# ── Calibration (the ONLY fitting entry point) ───────────────────────────────

def calibrate_for_month(
    ticker: str,
    month_end: pd.Timestamp,
    closes_df: pd.DataFrame,
    entry_price: float,
    K_fraction: float = DEFAULT_K_FRACTION,
    calibration_lookback: int = 250,
) -> MonthlyCalibration:
    """Fit Zhang params + Andrade signal once for (ticker, month_end).

    ``closes_df`` must be indexed by date with a ``close`` column and hold at
    least ``calibration_lookback`` rows ending on or before ``month_end``. Only
    the trailing ``calibration_lookback`` rows are used.

    ``entry_price`` is the reference price for the transaction cost K and is
    FIXED for the lifetime of this calibration (REV 4 fix A). The backtest passes
    the first close of the calibration window; live trading passes the actual
    position cost basis when available. ``K_absolute = K_fraction × entry_price``
    is what Zhang's :func:`decide` / :func:`case2_thresholds` consume — never a
    quantity derived from the evolving ``current_price``.

    The live macro regime is fetched once here via
    :func:`~src.strategies.ensemble.regime_gate.get_live_regime_signal` and cached
    on the bundle (fix B); it gates the Andrade STRONG_SELL override in
    :func:`monthly_exit_review`. ``get_live_regime_signal`` does not accept a
    timestamp, so historical backtests use the *current regime as of run time* as
    an approximation — defensible for a short diagnostic window where the macro
    regime is effectively constant.

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

    K_absolute = K_fraction * entry_price
    regime_signal = get_live_regime_signal()

    return MonthlyCalibration(
        ticker=ticker,
        month_end=month_end,
        zhang_params=zhang_params,
        andrade_signal=andrade_signal,
        calibration_lookback=calibration_lookback,
        entry_price=entry_price,
        K_fraction=K_fraction,
        K_absolute=K_absolute,
        regime_signal=regime_signal,
    )


# ── Decision functions (cheap; no fitting) ───────────────────────────────────

def monthly_exit_review(
    calibration: MonthlyCalibration,
    current_price: float,
    rho: float = 0.03,
) -> ExitDecision:
    """Monthly baseline exit decision from a pre-computed calibration.

    Runs Zhang :func:`decide` under the Andrade-derived state, then combines:
    SELL iff Zhang says SELL OR (Andrade says STRONG_SELL **and the regime
    permits the override**).

    The transaction cost is the FIXED ``calibration.K_absolute`` (=
    ``K_fraction × entry_price``), never recomputed from ``current_price`` — see
    the "REV 4 fix A" section of the module docstring.

    The Andrade STRONG_SELL override is gated on the cached macro regime
    (fix B): it only fires when ``calibration.regime_signal["multiplier"] ≤
    NEUTRAL_MULT``. In a RISK_ON tape (multiplier > NEUTRAL_MULT) it is
    suppressed and only Zhang can force a sell. Zhang's decision path is
    unaffected by the gate.
    """
    cp = calibration.zhang_params
    andrade = calibration.andrade_signal
    state = _ANDRADE_TO_STATE[andrade.action]

    K_absolute = calibration.K_absolute

    zhang_decision = decide(
        price=current_price,
        state=state,
        rho=rho,
        f1=cp.f1,
        f2=cp.f2,
        lam1=cp.lam1,
        lam2=cp.lam2,
        K=K_absolute,
    )

    # ── Regime gate on the Andrade override (REV 4 fix B) ────────────────────
    mult = calibration.regime_signal["multiplier"]
    if mult > NEUTRAL_MULT:
        # RISK_ON — suppress Andrade override; Zhang decision only.
        andrade_override_active = False
        andrade_reason = (
            f"Andrade suppressed by regime (multiplier={mult}): {andrade.reason}"
        )
    else:
        andrade_override_active = True
        andrade_reason = andrade.reason

    zhang_fired = zhang_decision.action == "SELL"
    andrade_fired = andrade.action == "STRONG_SELL" and andrade_override_active

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
        andrade_reason=andrade_reason,
        current_price=current_price,
        entry_price=calibration.entry_price,
        K_fraction=calibration.K_fraction,
        K_absolute=K_absolute,
        calibration=calibration,
    )


def daily_hard_stop(
    calibration: MonthlyCalibration,
    current_price: float,
    rho: float = 0.03,
) -> ExitDecision:
    """Daily mandatory-sell check on the Zhang Case II threshold x*_0.

    Uses only the cached ``calibration.zhang_params`` — never re-fits. Case I
    regimes (ρ ≤ f1) and never-sell regimes (Φ ≤ 0) return no_action; the
    monthly baseline owns those softer signals.

    The Case II threshold is computed from the FIXED ``calibration.K_absolute``
    (= ``K_fraction × entry_price``), so x*_0 = ρ·K_absolute/(ρ − f1) is a fixed
    dollar *level* and ``current_price ≥ x*_0`` is a genuine price crossing —
    the behaviour REV 4 fix A restores (see module docstring for why the earlier
    current_price reference collapsed this into a constant).
    """
    cp = calibration.zhang_params
    args = (rho, cp.f1, cp.f2, cp.lam1, cp.lam2)

    K_absolute = calibration.K_absolute

    no_action = ExitDecision(
        ticker=calibration.ticker,
        action="HOLD",
        trigger="no_action",
        trigger_source="none",
        zhang_reason="daily hard-stop: not a Case II mandatory-sell",
        andrade_reason=calibration.andrade_signal.reason,
        current_price=current_price,
        entry_price=calibration.entry_price,
        K_fraction=calibration.K_fraction,
        K_absolute=K_absolute,
        calibration=calibration,
    )

    # Case I (ρ ≤ f1) or never-sell (Φ ≤ 0) → no hard threshold to enforce.
    if rho <= cp.f1 or phi(*args) <= 0.0:
        return no_action

    _x_star, x0_star = case2_thresholds(*args, K_absolute)
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
            entry_price=calibration.entry_price,
            K_fraction=calibration.K_fraction,
            K_absolute=K_absolute,
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
        entry_price=calibration.entry_price,
        K_fraction=calibration.K_fraction,
        K_absolute=K_absolute,
        calibration=calibration,
    )
