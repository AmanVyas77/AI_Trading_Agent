"""
Unit tests for :mod:`src.exit.zhang_optimal` — Zhang (arXiv:1309.7507v1)
closed-form optimal stopping.

Fixtures come straight from the paper: Example 2 (page 19-20) covers the
Case II thresholds; Table 1 (page 22) provides 8 half-year AAPL windows
whose Φ(0.03) values are known and whose 2H-2012 row falls in Case I with
a published x* = 0.017213 at K = 0.01.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.exit.zhang_optimal import (
    CalibratedParams,
    X0_upper_bound,
    _d1,
    _d2,
    beta2,
    case1_threshold,
    case2_thresholds,
    decide,
    estimate_parameters,
    kappa2,
    phi,
)


# ── Paper Table 1 fixture (AAPL, ρ = 0.03, K = 0.01) ─────────────────────────

AAPL_TABLE_1 = [
    # (period, f1,     f2,      λ1,     λ2,     expected_Φ)
    ("1H-2009", 10.45, -10.61, 100.48, 124.23, -336.06),
    ("2H-2009",  3.21,  -2.32, 102.15, 141.44, -217.41),
    ("1H-2010",  3.06,  -3.15,  97.98, 127.02,  -83.18),
    ("2H-2010",  2.27,  -1.92, 103.57, 134.25, -103.09),
    ("1H-2011",  1.80,  -1.85, 117.19, 125.00,   -5.02),
    ("2H-2011",  3.01,  -2.72,  97.98, 107.95,  -60.56),
    ("1H-2012",  5.39,  -4.80, 108.21, 127.19, -185.32),
    ("2H-2012",  4.89,  -5.13, 135.25, 130.95,   35.79),
]
AAPL_RHO = 0.03
AAPL_K = 0.01
AAPL_NEG_ROWS = [row for row in AAPL_TABLE_1 if row[5] < 0]  # first 7


# ── TestExample2 ─────────────────────────────────────────────────────────────

class TestExample2:
    """Paper Example 2 (page 19-20) — Case II."""

    EX2 = dict(f1=0.07, f2=-0.03, lam1=1.0, lam2=1.0, rho=0.10, K=0.01)

    def _args(self):
        return (
            self.EX2["rho"], self.EX2["f1"], self.EX2["f2"],
            self.EX2["lam1"], self.EX2["lam2"],
        )

    def test_phi_positive(self):
        assert phi(*self._args()) > 0

    def test_is_case_II(self):
        assert self.EX2["rho"] > self.EX2["f1"]

    def test_beta2_gt_one(self):
        assert beta2(*self._args()) > 1.0

    def test_kappa2_in_unit_interval(self):
        k2 = kappa2(*self._args())
        assert 0.0 < k2 < 1.0

    def test_thresholds_match_paper(self):
        x_star, x0_star = case2_thresholds(*self._args(), self.EX2["K"])
        assert x_star == pytest.approx(0.012478, abs=1e-4)
        assert x0_star == pytest.approx(0.033333, abs=1e-4)

    def test_x0_star_closed_form(self):
        _, x0_star = case2_thresholds(*self._args(), self.EX2["K"])
        expected = self.EX2["rho"] * self.EX2["K"] / (self.EX2["rho"] - self.EX2["f1"])
        assert x0_star == pytest.approx(expected, rel=1e-12)

    def test_X0_upper_bound(self):
        X0 = X0_upper_bound(*self._args(), self.EX2["K"])
        assert X0 == pytest.approx(0.013326, abs=1e-4)

    def test_x_star_within_X0(self):
        x_star, _ = case2_thresholds(*self._args(), self.EX2["K"])
        X0 = X0_upper_bound(*self._args(), self.EX2["K"])
        assert x_star <= X0


# ── TestAAPLTable1 ───────────────────────────────────────────────────────────

class TestAAPLTable1:
    """Paper Table 1 (page 22) — 8 half-year AAPL windows, ρ = 0.03."""

    @pytest.mark.parametrize("period,f1,f2,lam1,lam2,expected_phi", AAPL_TABLE_1)
    def test_phi_matches_paper(self, period, f1, f2, lam1, lam2, expected_phi):
        got = phi(AAPL_RHO, f1, f2, lam1, lam2)
        # Paper inputs are 2dp, so ±2 units on the last digit is generous.
        assert got == pytest.approx(expected_phi, abs=2.0), (
            f"{period}: Φ={got}, expected≈{expected_phi}"
        )

    @pytest.mark.parametrize("period,f1,f2,lam1,lam2,_expected", AAPL_NEG_ROWS)
    def test_negative_phi_always_holds(self, period, f1, f2, lam1, lam2, _expected):
        for state in (1, 2):
            d = decide(
                price=1.0, state=state, rho=AAPL_RHO,
                f1=f1, f2=f2, lam1=lam1, lam2=lam2, K=AAPL_K,
            )
            assert d.action == "HOLD", f"{period} state={state}: {d}"

    def test_2H_2012_case_I_threshold(self):
        _, f1, f2, lam1, lam2, _ = AAPL_TABLE_1[-1]
        assert AAPL_RHO <= f1, "2H-2012 should be Case I"
        x_star = case1_threshold(AAPL_RHO, f1, f2, lam1, lam2, AAPL_K)
        assert x_star == pytest.approx(0.017213, abs=1e-4)


# ── TestDecideCase1 ──────────────────────────────────────────────────────────

class TestDecideCase1:
    """Case I decision rule using 2H-2012 AAPL params."""

    _, F1, F2, LAM1, LAM2, _ = AAPL_TABLE_1[-1]
    RHO, K = AAPL_RHO, AAPL_K

    def _xstar(self):
        return case1_threshold(self.RHO, self.F1, self.F2, self.LAM1, self.LAM2, self.K)

    def test_downtick_at_or_above_threshold_sells(self):
        x = self._xstar()
        d = decide(
            x * 1.5, 2, self.RHO, self.F1, self.F2, self.LAM1, self.LAM2, self.K,
        )
        assert d.action == "SELL"

    def test_downtick_below_threshold_holds(self):
        x = self._xstar()
        d = decide(
            x * 0.5, 2, self.RHO, self.F1, self.F2, self.LAM1, self.LAM2, self.K,
        )
        assert d.action == "HOLD"

    def test_uptick_never_sells(self):
        x = self._xstar()
        d = decide(
            x * 1000.0, 1, self.RHO, self.F1, self.F2, self.LAM1, self.LAM2, self.K,
        )
        assert d.action == "HOLD"

    def test_bad_state_raises(self):
        with pytest.raises(ValueError):
            decide(1.0, 0, self.RHO, self.F1, self.F2, self.LAM1, self.LAM2, self.K)


# ── TestDecideCase2 ──────────────────────────────────────────────────────────

class TestDecideCase2:
    """Case II decision rule (Example 2 params)."""

    F1, F2 = 0.07, -0.03
    LAM1, LAM2 = 1.0, 1.0
    RHO, K = 0.10, 0.01

    def _thresholds(self):
        return case2_thresholds(self.RHO, self.F1, self.F2, self.LAM1, self.LAM2, self.K)

    def test_downtick_in_band_sells(self):
        x_star, x0_star = self._thresholds()
        mid = 0.5 * (x_star + x0_star)
        d = decide(mid, 2, self.RHO, self.F1, self.F2, self.LAM1, self.LAM2, self.K)
        assert d.action == "SELL"

    def test_uptick_above_x0_sells(self):
        _, x0_star = self._thresholds()
        d = decide(
            x0_star * 1.01, 1, self.RHO, self.F1, self.F2, self.LAM1, self.LAM2, self.K,
        )
        assert d.action == "SELL"

    def test_uptick_between_thresholds_holds(self):
        x_star, x0_star = self._thresholds()
        mid = 0.5 * (x_star + x0_star)
        d = decide(mid, 1, self.RHO, self.F1, self.F2, self.LAM1, self.LAM2, self.K)
        assert d.action == "HOLD"

    def test_downtick_below_xstar_holds(self):
        x_star, _ = self._thresholds()
        d = decide(
            x_star * 0.5, 2, self.RHO, self.F1, self.F2, self.LAM1, self.LAM2, self.K,
        )
        assert d.action == "HOLD"


# ── TestDecidePhiNegative ────────────────────────────────────────────────────

class TestDecidePhiNegative:
    """Any Φ<0 row must return HOLD across states {1,2} × any positive price."""

    # 1H-2011 is the closest-to-zero negative Φ in the table — hardest case.
    _, F1, F2, LAM1, LAM2, _ = AAPL_TABLE_1[4]
    RHO, K = AAPL_RHO, AAPL_K

    @pytest.mark.parametrize("state", [1, 2])
    @pytest.mark.parametrize("price", [0.001, 1.0, 1e6])
    def test_holds_regardless(self, state, price):
        d = decide(
            price, state, self.RHO, self.F1, self.F2, self.LAM1, self.LAM2, self.K,
        )
        assert d.action == "HOLD"
        assert "never-sell" in d.reason or "Φ" in d.reason


# ── TestEstimateParameters ───────────────────────────────────────────────────

class TestEstimateParameters:
    """Calibration from a synthetic geometric Brownian walk."""

    def test_synthetic_gaussian_walk(self):
        rng = np.random.default_rng(0)
        n = 500
        dt = 1.0 / 252.0
        drift, vol = 1e-4, 1e-2
        returns = rng.normal(drift, vol, size=n - 1)
        prices = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], returns])))
        cp = estimate_parameters(prices, dt=dt)
        assert isinstance(cp, CalibratedParams)
        assert cp.f1 > 0
        assert cp.f2 < 0
        assert cp.lam1 > 0
        assert cp.lam2 > 0
        assert cp.sigma0 > 0

    def test_rejects_non_positive_prices(self):
        with pytest.raises(ValueError):
            estimate_parameters(np.array([100.0, -1.0, 105.0]))

    def test_rejects_short_series(self):
        with pytest.raises(ValueError):
            estimate_parameters(np.array([100.0, 101.0]))

    def test_rejects_monotone_returns(self):
        # Strictly increasing prices → all-positive log-returns → no neg sign.
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        with pytest.raises(ValueError):
            estimate_parameters(prices)


# ── TestPhiSignConventions ───────────────────────────────────────────────────

class TestPhiSignConventions:
    """Cross-check Φ against Φ = D2 - D1 + f1·f2 (algebraic expansion).

    (ρ+λ1-f1)(ρ+λ2-f2) - λ1λ2
        = (ρ+λ1)(ρ+λ2) - (ρ+λ1)f2 - (ρ+λ2)f1 + f1f2 - λ1λ2
        = D2 - D1 + f1f2.
    """

    @pytest.mark.parametrize("period,f1,f2,lam1,lam2,_expected", AAPL_TABLE_1)
    def test_phi_equals_D_expansion(self, period, f1, f2, lam1, lam2, _expected):
        got = phi(AAPL_RHO, f1, f2, lam1, lam2)
        alt = (
            _d2(AAPL_RHO, f1, f2, lam1, lam2)
            - _d1(AAPL_RHO, f1, f2, lam1, lam2)
            + f1 * f2
        )
        assert got == pytest.approx(alt, rel=1e-12), f"{period}: got={got}, alt={alt}"
