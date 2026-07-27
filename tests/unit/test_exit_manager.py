"""
Unit tests for :mod:`src.exit.exit_manager` — the Zhang+Andrade exit fusion.

Decision tests construct MonthlyCalibration bundles directly (with hand-checked
Zhang params) so they run fast and isolated — no DHMM fitting. The calibration
test does exercise the real fitting path once on a synthetic uptrend.

Hand-checked param sets (rho=0.03, K=0.01, lam1=lam2=1 unless noted)
--------------------------------------------------------------------
CASE_I  : f1=0.05, f2=-2.0, lam1=lam2=0.1
          Φ=+0.1604 > 0, rho<f1 → Case I, β2≈2.562, κ2≈0.019, x*≈0.0101
CASE_II : f1=0.01, f2=-0.5, lam1=lam2=1.0
          Φ=+0.5606 > 0, rho>f1 → Case II, x*_0 = rho·K/(rho−f1) = 0.015
NEVER   : f1=0.5,  f2=-0.3, lam1=lam2=1.0
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

MONTH_END = pd.Timestamp("2025-06-30")


def _calib(zhang: CalibratedParams, andrade_action: str) -> MonthlyCalibration:
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
    )


CASE_I = CalibratedParams(f1=0.05, f2=-2.0, lam1=0.1, lam2=0.1, mu=0.0, sigma0=0.01)
CASE_II = CalibratedParams(f1=0.01, f2=-0.5, lam1=1.0, lam2=1.0, mu=0.0, sigma0=0.01)
NEVER = CalibratedParams(f1=0.5, f2=-0.3, lam1=1.0, lam2=1.0, mu=0.0, sigma0=0.01)


# ── TestCalibrateForMonth ────────────────────────────────────────────────────

def _uptrend_closes(n: int = 260, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.003, 0.008, size=n - 1)  # clear positive drift, both signs
    prices = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], returns])))
    idx = pd.date_range("2024-06-01", periods=n, freq="B")
    return pd.DataFrame({"close": prices}, index=idx)


class TestCalibrateForMonth:
    def test_uptrend_params_and_signal(self):
        closes = _uptrend_closes()
        me = closes.index[-1]
        calib = calibrate_for_month("TST", me, closes, calibration_lookback=250)
        assert calib.zhang_params.f1 > 0
        assert calib.zhang_params.f2 < 0
        assert calib.andrade_signal.action in {"STRONG_BUY", "HOLD"}
        assert calib.calibration_lookback == 250
        assert calib.ticker == "TST"

    def test_insufficient_history_raises(self):
        closes = _uptrend_closes(n=100)  # < 250
        me = closes.index[-1]
        with pytest.raises(ValueError):
            calibrate_for_month("TST", me, closes, calibration_lookback=250)

    def test_determinism(self):
        closes = _uptrend_closes()
        me = closes.index[-1]
        c1 = calibrate_for_month("TST", me, closes, calibration_lookback=250)
        c2 = calibrate_for_month("TST", me, closes, calibration_lookback=250)
        assert c1 == c2


# ── TestMonthlyExitReview ────────────────────────────────────────────────────

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
        # Case II uptick branch: state=1 (Andrade HOLD) + price ≥ x*_0=0.015 → SELL.
        calib = _calib(CASE_II, "HOLD")
        d = monthly_exit_review(calib, current_price=100.0)
        assert d.action == "SELL"
        assert d.trigger_source == "zhang"

    def test_both_trigger(self):
        # Case II downtick: state=2 (Andrade STRONG_SELL) + price ≥ x* → Zhang SELL,
        # and Andrade STRONG_SELL fires too → "both".
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
        calib = _calib(CASE_II, "HOLD")  # x*_0 = 0.015
        d = daily_hard_stop(calib, current_price=100.0)
        assert d.action == "SELL"
        assert d.trigger == "daily_hard_stop"
        assert d.trigger_source == "zhang"

    def test_case_II_below_threshold_holds(self):
        calib = _calib(CASE_II, "HOLD")  # x*_0 = 0.015
        d = daily_hard_stop(calib, current_price=0.001)
        assert d.action == "HOLD"
        assert d.trigger == "no_action"

    def test_no_refitting(self):
        calib = _calib(CASE_II, "HOLD")
        with mock.patch.object(exit_manager, "estimate_parameters") as m_fit:
            d1 = daily_hard_stop(calib, current_price=100.0)
            d2 = daily_hard_stop(calib, current_price=0.001)
        assert m_fit.call_count == 0
        assert isinstance(d1, ExitDecision) and isinstance(d2, ExitDecision)
        assert d1.action == "SELL" and d2.action == "HOLD"
