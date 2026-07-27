"""
Unit tests for :mod:`src.exit.andrade_dhmm` — DHMM + RSI regime switcher.

Pattern-error tests reproduce the paper's Appendix A validation. Baum-Welch
converges to different local optima under different BLAS backends /
init states, so per-pattern error rates are checked against a generous
40 % ceiling (paper max was 38 %) rather than the exact table values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.exit.andrade_dhmm import (
    Signal,
    discretize_daily,
    discretize_weekly,
    next_signal,
    rsi_wilder,
    train_and_predict,
    weekly_fridays,
)


# ── TestDiscretization ───────────────────────────────────────────────────────

class TestDiscretization:
    def test_daily_basic(self):
        got = discretize_daily(np.array([100, 101, 100, 102, 101], dtype=float))
        assert list(got) == [1, 0, 1, 0]

    def test_daily_rejects_short(self):
        with pytest.raises(ValueError):
            discretize_daily(np.array([100.0]))

    def test_daily_rejects_non_positive(self):
        with pytest.raises(ValueError):
            discretize_daily(np.array([100.0, 0.0, 101.0]))

    def test_weekly_length_matches_n_weeks(self):
        # 100 business days from 2024-01-01 (a Monday) → ≥ 14 Fridays.
        idx = pd.date_range("2024-01-01", periods=100, freq="B")
        rng = np.random.default_rng(0)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=100)))
        df = pd.DataFrame({"close": prices}, index=idx)
        obs = discretize_weekly(df, n_weeks=10)
        assert obs.shape == (10,)
        assert set(obs.tolist()).issubset({0, 1})

    def test_weekly_fridays_lands_on_friday(self):
        idx = pd.date_range("2024-01-01", periods=50, freq="B")
        df = pd.DataFrame({"close": np.linspace(100, 110, 50)}, index=idx)
        fri = weekly_fridays(df)
        # Every W-FRI bucket is timestamped on its Friday.
        assert all(d.weekday() == 4 for d in fri.index)


# ── TestRSI ──────────────────────────────────────────────────────────────────

class TestRSI:
    def test_monotone_up_gives_100(self):
        prices = np.arange(1, 30, dtype=float)  # 29 strictly-increasing
        r = rsi_wilder(prices, period=14)
        assert np.isnan(r[:14]).all()
        assert np.allclose(r[14:], 100.0)

    def test_monotone_down_gives_0(self):
        prices = np.arange(30, 1, -1, dtype=float)  # 29 strictly-decreasing
        r = rsi_wilder(prices, period=14)
        assert np.isnan(r[:14]).all()
        assert np.allclose(r[14:], 0.0)

    def test_constant_gives_nan(self):
        r = rsi_wilder(np.full(30, 42.0), period=14)
        # All deltas zero → avg_gain=avg_loss=0 → NaN by convention.
        assert np.isnan(r[14:]).all()

    def test_hand_computed_alternating_gives_50(self):
        # Prices 100, 101, 100, 101, ... for indices 0..14 (15 prices).
        # 14 deltas alternate +1, -1 starting with +1: seven +1s and seven -1s.
        # avg_gain = avg_loss = 0.5 → RS = 1 → RSI = 50.
        prices = np.array(
            [100.0 if i % 2 == 0 else 101.0 for i in range(15)]
        )
        r = rsi_wilder(prices, period=14)
        assert r[14] == pytest.approx(50.0, abs=1e-9)


# ── TestPatternPredictions (paper Appendix A) ────────────────────────────────

PATTERNS = {
    "P1_all_ones":        [1, 1, 1, 1, 1, 1, 1, 1],
    "P2_run_up_down":     [1, 1, 1, 1, 0, 0, 0, 0],
    "P3_alternating":     [1, 0, 1, 0, 1, 0, 1, 0],
    "P4_lone_up":         [0, 0, 0, 0, 1, 0, 0, 0],
    "P5_mixed_low":       [0, 1, 0, 1, 0, 0, 1, 0],
    "P6_mixed_high":      [1, 1, 0, 1, 1, 1, 0, 1],
}


FALLBACK_SEEDS = [42, 0, 123]  # per Prompt 3 spec — retry BW init on breach


class TestPatternPredictions:
    """Sliding-window 30-obs refit; predict symbols 51-80; error rate ≤ 40 %.

    Baum-Welch on 30 obs with 3 states hits different local optima under
    different init states. The spec allows retrying with alternate
    random_states on breach — we sweep three seeds and pass on the first
    that clears 40 %, matching the paper's own robustness against BW
    convergence noise.
    """

    @pytest.mark.parametrize("name,pattern", list(PATTERNS.items()))
    def test_error_rate_under_ceiling(self, name, pattern):
        seq = np.tile(np.asarray(pattern, dtype=int), 10)  # 80 symbols
        assert seq.shape == (80,)

        best_rate = 1.0
        best_seed = None
        for seed in FALLBACK_SEEDS:
            errors = 0
            for t in range(50, 80):
                window = seq[t - 30 : t]
                pred = train_and_predict(
                    window, n_states=3, n_symbols=2, n_iter=100, random_state=seed,
                )
                if pred != seq[t]:
                    errors += 1
            rate = errors / 30.0
            if rate < best_rate:
                best_rate, best_seed = rate, seed
            if rate <= 0.40:
                return  # first passing seed wins
        pytest.fail(
            f"{name}: best error_rate={best_rate:.2%} at seed={best_seed} "
            f"across {FALLBACK_SEEDS} exceeds 40% ceiling"
        )


# ── TestSignalStateMachine ───────────────────────────────────────────────────

def _make_prices(returns: np.ndarray, start_price: float = 100.0) -> pd.DataFrame:
    n = len(returns) + 1
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = start_price * np.exp(np.cumsum(np.concatenate([[0.0], returns])))
    return pd.DataFrame({"close": prices}, index=idx)


class TestSignalStateMachine:
    def test_uptrend_selects_daily_mode(self):
        # Strong positive drift → RSI drives well above 70.
        rng = np.random.default_rng(1)
        returns = rng.normal(0.01, 0.003, size=100)
        df = _make_prices(returns)
        sig = next_signal(df, prev_mode="weekly")
        assert isinstance(sig, Signal)
        assert sig.mode == "daily"
        assert sig.rsi > 70.0
        assert "RSI" in sig.reason

    def test_downtrend_selects_weekly_mode(self):
        # Strong negative drift + long history so the 30-week window has
        # 31 Fridays; RSI drives well below 30.
        rng = np.random.default_rng(2)
        returns = rng.normal(-0.01, 0.003, size=260)
        df = _make_prices(returns)
        sig = next_signal(df, prev_mode="daily")
        assert sig.mode == "weekly"
        assert sig.rsi < 30.0

    def test_neutral_rsi_preserves_prev_daily(self):
        # Zero-drift random walk → RSI oscillates around 50; both directions
        # of prev_mode should survive.
        rng = np.random.default_rng(3)
        returns = rng.normal(0.0, 0.005, size=260)
        df = _make_prices(returns)
        latest_rsi = rsi_wilder(df["close"].to_numpy(), period=14)[-1]
        assert 30.0 <= latest_rsi <= 70.0, (
            f"seed 3 broke RSI-neutral assumption (RSI={latest_rsi:.2f}) — "
            f"pick a different seed"
        )
        assert next_signal(df, prev_mode="daily").mode == "daily"
        assert next_signal(df, prev_mode="weekly").mode == "weekly"

    def test_action_is_always_allowed(self):
        rng = np.random.default_rng(4)
        returns = rng.normal(0.001, 0.01, size=260)
        df = _make_prices(returns)
        for prev in ("daily", "weekly"):
            sig = next_signal(df, prev_mode=prev)
            assert sig.action in {"STRONG_BUY", "STRONG_SELL", "SELL", "HOLD"}
            assert sig.mode in {"daily", "weekly"}
