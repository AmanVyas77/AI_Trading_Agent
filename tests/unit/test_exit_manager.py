"""
Unit tests for :mod:`src.exit.exit_manager` — the Zhang+Andrade exit fusion.

Decision tests construct MonthlyCalibration bundles directly (with hand-checked
Zhang params) so they run fast and isolated — no DHMM fitting. The calibration
tests do exercise the real fitting path once on a synthetic uptrend (with the
regime lookup stubbed so they stay DB-independent).

K reference (REV 4 fix A, entry_price)
--------------------------------------
The decision functions no longer compute K from ``current_price``. K is fixed at
calibration time: ``K_absolute = K_fraction × entry_price`` is stored on the
:class:`MonthlyCalibration` bundle and consumed verbatim by Zhang. Because
``entry_price`` does not evolve with the market, every Zhang threshold is a fixed
dollar *level* the live price genuinely crosses. In particular the Case II
hard-stop test ``current_price ≥ x*_0`` is a real price crossing again, where

    x*_0 = ρ·K_absolute/(ρ − f1)  (a constant level, NOT proportional to price).

This is the property that would FAIL under the REV 1 (current_price) K, where the
crossing test cancelled the price and reduced to the constant ``1 ≥
ρ·K_fraction/(ρ − f1)`` — a regime yes/no gate rather than a price threshold.

Regime gate (REV 4 fix B)
-------------------------
``monthly_exit_review`` reads ``calibration.regime_signal["multiplier"]`` (cached
at calibration time). The Andrade STRONG_SELL override only fires when the
multiplier ≤ NEUTRAL_MULT; in a RISK_ON tape (multiplier > NEUTRAL_MULT) it is
suppressed and only Zhang can force a sell. The decision tests therefore set the
regime multiplier directly on the synthetic calibration (default NEUTRAL) — the
decision path never re-calls get_live_regime_signal.

Hand-checked param sets (rho=0.03, entry_price=100, K_fraction=0.001 → K_abs=0.1)
---------------------------------------------------------------------------------
CASE_I      : f1=0.05, f2=-2.0, lam1=lam2=0.1
              Φ=+0.1604 > 0, rho<f1 → Case I. x* ≈ 0.101 (fixed dollar level,
              linear in K_abs). SELL in state 2 iff price ≥ x*.
CASE_II     : f1=0.01, f2=-0.5, lam1=lam2=1.0
              Φ=+0.5606 > 0, rho>f1 → Case II. x*_0 = ρ·K_abs/(ρ−f1)
              = 0.03·0.1/0.02 = 0.15 (fixed level). Daily hard-stop fires iff
              current_price ≥ 0.15.
CASE_II_HOLD: f1=0.029985, f2=-0.5, lam1=lam2=1.0
              Φ=+0.53 > 0, rho>f1 → Case II. x*_0 = 0.03·0.1/0.000015 = 200,
              so the daily hard-stop HOLDs for any current_price < 200.
NEVER       : f1=0.5,  f2=-0.3, lam1=lam2=1.0
              Φ=-0.295 ≤ 0 → never-sell regime (decide always HOLD)
"""
from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.exit import exit_manager
from src.exit.andrade_dhmm import Signal
from src.exit.exit_manager import (
    ExitDecision,
    MonthlyCalibration,
    calibrate_for_month,
    daily_hard_stop,
    monthly_exit_review,
)
from src.exit.zhang_optimal import CalibratedParams
from src.strategies.ensemble.regime_gate import NEUTRAL_MULT, RISK_ON_MULT

MONTH_END = pd.Timestamp("2025-06-30")


def _regime(multiplier: float = NEUTRAL_MULT, label: str = "NEUTRAL") -> dict:
    return {
        "multiplier": multiplier,
        "rule_signal": label,
        "llm_signal": "SKIPPED",
        "reasoning": "test stub",
    }


def _calib(
    zhang: CalibratedParams,
    andrade_action: str,
    *,
    entry_price: float = 100.0,
    K_fraction: float = 0.001,
    regime_mult: float = NEUTRAL_MULT,
) -> MonthlyCalibration:
    sig = Signal(
        action=andrade_action, mode="daily", rsi=50.0,
        reason=f"dummy andrade={andrade_action}",
    )
    return MonthlyCalibration(
        ticker="TST",
        month_end=MONTH_END,
        zhang_params=zhang,
        andrade_signal=sig,
        calibration_lookback=250,
        entry_price=entry_price,
        K_fraction=K_fraction,
        K_absolute=K_fraction * entry_price,
        regime_signal=_regime(regime_mult),
    )


CASE_I = CalibratedParams(f1=0.05, f2=-2.0, lam1=0.1, lam2=0.1, mu=0.0, sigma0=0.01)
CASE_II = CalibratedParams(f1=0.01, f2=-0.5, lam1=1.0, lam2=1.0, mu=0.0, sigma0=0.01)
# f1 just below rho=0.03 → x*_0 = ρ·K_abs/(ρ−f1) = 0.03·0.1/0.000015 = 200 at the
# default entry_price=100, so the daily hard-stop HOLDs for any current_price
# below 200. Keeps a genuine Case-II HOLD reachable under the fixed-level K.
CASE_II_HOLD = CalibratedParams(
    f1=0.029985, f2=-0.5, lam1=1.0, lam2=1.0, mu=0.0, sigma0=0.01
)
NEVER = CalibratedParams(f1=0.5, f2=-0.3, lam1=1.0, lam2=1.0, mu=0.0, sigma0=0.01)


# ── TestCalibrateForMonth ────────────────────────────────────────────────────

def _uptrend_closes(n: int = 260, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.003, 0.008, size=n - 1)  # clear positive drift, both signs
    prices = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], returns])))
    idx = pd.date_range("2024-06-01", periods=n, freq="B")
    return pd.DataFrame({"close": prices}, index=idx)


class TestCalibrateForMonth:
    @pytest.fixture(autouse=True)
    def _stub_regime(self, monkeypatch):
        # Keep calibration DB-independent and deterministic: the regime lookup is
        # an integration concern exercised elsewhere.
        monkeypatch.setattr(
            exit_manager, "get_live_regime_signal", lambda *a, **k: _regime()
        )

    def test_uptrend_params_and_signal(self):
        closes = _uptrend_closes()
        me = closes.index[-1]
        entry = float(closes["close"].iloc[0])
        calib = calibrate_for_month("TST", me, closes, entry, calibration_lookback=250)
        assert calib.zhang_params.f1 > 0
        assert calib.zhang_params.f2 < 0
        assert calib.andrade_signal.action in {"STRONG_BUY", "HOLD"}
        assert calib.calibration_lookback == 250
        assert calib.ticker == "TST"
        assert calib.entry_price == entry
        assert calib.K_absolute == pytest.approx(exit_manager.DEFAULT_K_FRACTION * entry)
        assert isinstance(calib.regime_signal, dict)

    def test_insufficient_history_raises(self):
        closes = _uptrend_closes(n=100)  # < 250
        me = closes.index[-1]
        with pytest.raises(ValueError):
            calibrate_for_month("TST", me, closes, 100.0, calibration_lookback=250)

    def test_determinism(self):
        closes = _uptrend_closes()
        me = closes.index[-1]
        entry = float(closes["close"].iloc[0])
        c1 = calibrate_for_month("TST", me, closes, entry, calibration_lookback=250)
        c2 = calibrate_for_month("TST", me, closes, entry, calibration_lookback=250)
        assert c1 == c2


# ── TestMonthlyExitReview (regime defaults to NEUTRAL via _calib) ─────────────

class TestMonthlyExitReview:
    def test_case_I_uptrend_high_price_holds(self):
        calib = _calib(CASE_I, "STRONG_BUY")  # → state 1, Case I never sells st.1
        d = monthly_exit_review(calib, current_price=500.0)
        assert d.action == "HOLD"
        assert d.trigger == "no_action"
        assert d.trigger_source == "none"

    def test_andrade_strong_sell_triggers(self):
        calib = _calib(NEVER, "STRONG_SELL")  # Zhang never-sells → only Andrade fires
        d = monthly_exit_review(calib, current_price=100.0)
        assert d.action == "SELL"
        assert d.trigger == "monthly_baseline"
        assert d.trigger_source in {"andrade", "both"}
        assert d.trigger_source == "andrade"

    def test_zhang_only_trigger(self):
        # Case II uptick branch: state=1 (Andrade HOLD) + price ≥ x*_0 → SELL.
        # x*_0 = ρ·K_abs/(ρ−f1) = 0.15 (K_abs=0.1), so 100 ≥ 0.15 → SELL.
        calib = _calib(CASE_II, "HOLD")
        d = monthly_exit_review(calib, current_price=100.0)
        assert d.action == "SELL"
        assert d.trigger_source == "zhang"
        assert d.K_absolute == pytest.approx(0.1)  # 0.001 × entry_price(100)

    def test_both_trigger(self):
        # Case II downtick: state=2 (Andrade STRONG_SELL) + price ≥ x* → Zhang SELL,
        # and Andrade STRONG_SELL fires too (NEUTRAL regime) → "both".
        calib = _calib(CASE_II, "STRONG_SELL")
        d = monthly_exit_review(calib, current_price=100.0)
        assert d.action == "SELL"
        assert d.trigger_source == "both"

    def test_neither_holds(self):
        calib = _calib(NEVER, "HOLD")
        d = monthly_exit_review(calib, current_price=100.0)
        assert d.action == "HOLD"
        assert d.trigger_source == "none"


# ── TestDailyHardStop ────────────────────────────────────────────────────────

class TestDailyHardStop:
    def test_case_I_never_fires(self):
        calib = _calib(CASE_I, "STRONG_BUY")
        d = daily_hard_stop(calib, current_price=1e6)
        assert d.action == "HOLD"
        assert d.trigger == "no_action"

    def test_case_II_above_threshold_sells(self):
        # CASE_II: x*_0 = 0.15 (fixed level, K_abs=0.1). price 100 ≥ 0.15 → SELL.
        calib = _calib(CASE_II, "HOLD")
        d = daily_hard_stop(calib, current_price=100.0)
        assert d.action == "SELL"
        assert d.trigger == "daily_hard_stop"
        assert d.trigger_source == "zhang"

    def test_case_II_below_threshold_holds(self):
        # CASE_II_HOLD: x*_0 = 200 (fixed level), so current_price 100 < 200 → HOLD.
        # Under the fixed entry-price K the threshold is a genuine dollar level,
        # not proportional to the live price.
        calib = _calib(CASE_II_HOLD, "HOLD")
        d = daily_hard_stop(calib, current_price=100.0)
        assert d.action == "HOLD"
        assert d.trigger == "no_action"

    def test_no_refitting(self):
        # SELL from CASE_II (x*_0=0.15), HOLD from CASE_II_HOLD (x*_0=200);
        # neither re-fits.
        sell_calib = _calib(CASE_II, "HOLD")
        hold_calib = _calib(CASE_II_HOLD, "HOLD")
        with mock.patch.object(exit_manager, "estimate_parameters") as m_fit:
            d1 = daily_hard_stop(sell_calib, current_price=100.0)
            d2 = daily_hard_stop(hold_calib, current_price=100.0)
        assert m_fit.call_count == 0
        assert isinstance(d1, ExitDecision) and isinstance(d2, ExitDecision)
        assert d1.action == "SELL" and d2.action == "HOLD"


# ── TestKScalingEntryPrice (REV 4 fix A: K referenced to entry_price) ─────────

class TestKScalingEntryPrice:
    """K = K_fraction × entry_price (fixed at calibration) replaces the REV 1
    ``K_fraction × current_price``. Thresholds are fixed dollar levels again."""

    def test_K_absolute_fixed_at_calibration(self):
        # One calibration, two decide() calls at different current_prices.
        # K_absolute reported is IDENTICAL and equals calibration.K_absolute;
        # live price moves do NOT change K.
        calib = _calib(CASE_II, "HOLD", entry_price=100.0, K_fraction=0.001)
        d_low = monthly_exit_review(calib, current_price=60.0)
        d_high = monthly_exit_review(calib, current_price=600.0)
        assert d_low.K_absolute == d_high.K_absolute == calib.K_absolute
        assert calib.K_absolute == pytest.approx(0.1)
        assert d_low.entry_price == d_high.entry_price == 100.0

    def test_K_absolute_scales_with_entry_price(self):
        # Same ticker/K_fraction, two entry prices → K_absolute tracks entry.
        c50 = _calib(CASE_II, "HOLD", entry_price=50.0, K_fraction=0.001)
        c500 = _calib(CASE_II, "HOLD", entry_price=500.0, K_fraction=0.001)
        assert c50.K_absolute == pytest.approx(0.05)
        assert c500.K_absolute == pytest.approx(0.50)

    def test_case_II_x0_matches_entry_based_formula(self):
        # x*_0 = ρ·K_fraction·entry/(ρ − f1) to 1e-9 — the algebra the fix restores.
        from src.exit.zhang_optimal import case2_thresholds

        rho, entry, Kf = 0.03, 250.0, 0.001
        cp = CASE_II
        K_abs = Kf * entry
        _x_star, x0 = case2_thresholds(rho, cp.f1, cp.f2, cp.lam1, cp.lam2, K_abs)
        expected = rho * Kf * entry / (rho - cp.f1)
        assert x0 == pytest.approx(expected, abs=1e-9)

    def test_daily_hard_stop_is_price_crossing(self):
        # Sweep current_price across x*_0. SELL iff current_price ≥ x*_0. This is
        # the test that FAILS under the REV 1 (current_price) K, where the
        # condition is constant across price levels.
        from src.exit.zhang_optimal import case2_thresholds

        calib = _calib(CASE_II, "HOLD", entry_price=100.0, K_fraction=0.001)
        cp = CASE_II
        _x_star, x0 = case2_thresholds(
            0.03, cp.f1, cp.f2, cp.lam1, cp.lam2, calib.K_absolute
        )
        assert x0 == pytest.approx(0.15)

        for price in (0.05, 0.10, x0 * 0.99):
            d = daily_hard_stop(calib, current_price=price)
            assert d.action == "HOLD", f"expected HOLD below x*_0 at price={price}"

        for price in (x0, x0 * 1.01, 0.5, 500.0):
            d = daily_hard_stop(calib, current_price=price)
            assert d.action == "SELL", f"expected SELL at/above x*_0 at price={price}"


# ── TestRegimeGate (REV 4 fix B: Andrade override gated on macro regime) ──────

class TestRegimeGate:
    def test_andrade_override_suppressed_in_risk_on(self):
        # RISK_ON tape (multiplier > NEUTRAL_MULT) + synthetic STRONG_SELL, Zhang
        # never-sells (NEVER). Override suppressed → HOLD / "none".
        calib = _calib(NEVER, "STRONG_SELL", regime_mult=RISK_ON_MULT)
        d = monthly_exit_review(calib, current_price=100.0)
        assert d.action == "HOLD"
        assert d.trigger_source == "none"
        assert "suppressed by regime" in d.andrade_reason

    def test_andrade_override_active_in_neutral(self):
        # NEUTRAL tape + STRONG_SELL, Zhang never-sells → Andrade override fires.
        calib = _calib(NEVER, "STRONG_SELL", regime_mult=NEUTRAL_MULT)
        d = monthly_exit_review(calib, current_price=100.0)
        assert d.action == "SELL"
        assert d.trigger_source in {"andrade", "both"}

    def test_zhang_fires_regardless_of_regime(self):
        # RISK_ON suppresses the Andrade override, but Zhang's decision is
        # unaffected: Case I, state 2 (Andrade STRONG_SELL), price ≫ x* → SELL.
        calib = _calib(CASE_I, "STRONG_SELL", regime_mult=RISK_ON_MULT)
        d = monthly_exit_review(calib, current_price=500.0)
        assert d.action == "SELL"
        assert d.trigger_source == "zhang"

    def test_regime_signal_cached_in_calibration(self, monkeypatch):
        # calibrate_for_month caches a regime_signal dict on the bundle.
        monkeypatch.setattr(
            exit_manager, "get_live_regime_signal", lambda *a, **k: _regime()
        )
        closes = _uptrend_closes()
        me = closes.index[-1]
        entry = float(closes["close"].iloc[0])
        c1 = calibrate_for_month("TST", me, closes, entry, calibration_lookback=250)
        c2 = calibrate_for_month("TST", me, closes, entry, calibration_lookback=250)
        assert isinstance(c1.regime_signal, dict)
        assert isinstance(c2.regime_signal, dict)
        assert "multiplier" in c1.regime_signal
