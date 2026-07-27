"""
Zhang (2013) closed-form optimal stopping for a two-state Markov-modulated
geometric Brownian motion, applied to selling one asset with a fixed
transaction cost K.

Paper
-----
Q. Zhang, "Stock Trading: An Optimal Selling Rule",
arXiv:1309.7507v1 (August 2013). Equation numbers cited below refer to
that document.

Model (§2)
----------
    dS_t / S_t = f(α_t) dt,  α_t ∈ {1, 2} a continuous-time Markov chain
    f(1) = f1 > 0                 (uptick log-return rate)
    f(2) = f2 < 0                 (downtick log-return rate)
    Generator Q = [[-λ1, λ1],
                   [ λ2, -λ2]]
    Discount rate ρ > 0, fixed transaction cost K > 0.
    Objective: choose stopping time τ that maximizes
               E[ e^(-ρτ) (S_τ - K) ].

Two exercise regimes (paper §3):

  Case I  (ρ ≤ f1):   sell iff  state = 2  AND  price ≥ x*
                      x* closed-form via eq. 9.

  Case II (ρ > f1):   sell iff  (state=2 AND price ≥ x*)
                              OR (state=1 AND price ≥ x*_0)
                      x*_0 closed-form; x* via bisection on eq. 27.

If Φ(ρ) ≤ 0 the "never-sell" regime holds and :func:`decide` returns HOLD
regardless of state and price.

The unit tests in ``tests/unit/test_zhang_optimal.py`` reproduce the
paper's Example 2 (page 19-20) and AAPL Table 1 (page 22) as executable
regression fixtures — treat them as the reference for expected numerics.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal, Optional

import numpy as np


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Decision:
    action: Literal["HOLD", "SELL"]
    reason: str
    x_star: Optional[float] = None
    x0_star: Optional[float] = None


@dataclass(frozen=True)
class CalibratedParams:
    f1: float
    f2: float
    lam1: float
    lam2: float
    mu: float
    sigma0: float


# ── Core equations (paper §3) ────────────────────────────────────────────────

def phi(rho: float, f1: float, f2: float, lam1: float, lam2: float) -> float:
    """Φ(ρ) = (ρ+λ1-f1)(ρ+λ2-f2) - λ1·λ2.  Assumption A2 requires Φ > 0."""
    return (rho + lam1 - f1) * (rho + lam2 - f2) - lam1 * lam2


def _d1(rho: float, f1: float, f2: float, lam1: float, lam2: float) -> float:
    # D1 (eq. 3): (ρ+λ1)·f2 + (ρ+λ2)·f1
    return (rho + lam1) * f2 + (rho + lam2) * f1


def _d2(rho: float, f1: float, f2: float, lam1: float, lam2: float) -> float:
    # D2 (eq. 3): (ρ+λ1)(ρ+λ2) - λ1·λ2
    return (rho + lam1) * (rho + lam2) - lam1 * lam2


def beta2(rho: float, f1: float, f2: float, lam1: float, lam2: float) -> float:
    """β2 (eq. 5).

    Under A1 (f1>0, f2<0 so f1·f2<0) and A2 (Φ>0), Lemma 1 gives β2 > 1.
    The minus-root branch of the quadratic is the positive one because
    the denominator 2·f1·f2 is negative.
    """
    if f1 * f2 >= 0:
        raise ValueError(
            f"Assumption A1 violated: need f1 > 0 and f2 < 0, got f1={f1}, f2={f2}"
        )
    D1 = _d1(rho, f1, f2, lam1, lam2)
    D2 = _d2(rho, f1, f2, lam1, lam2)
    disc = D1 * D1 - 4.0 * f1 * f2 * D2
    if disc < 0.0:
        raise ValueError(f"β2 discriminant negative ({disc}); parameters infeasible")
    return (D1 - sqrt(disc)) / (2.0 * f1 * f2)


def kappa2(rho: float, f1: float, f2: float, lam1: float, lam2: float) -> float:
    """κ2 = (ρ + λ1 - f1·β2) / λ1  (Lemma 2 → 0 < κ2 < 1 under A1–A2)."""
    b2 = beta2(rho, f1, f2, lam1, lam2)
    return (rho + lam1 - f1 * b2) / lam1


# ── Case I ────────────────────────────────────────────────────────────────────

def case1_threshold(
    rho: float, f1: float, f2: float, lam1: float, lam2: float, K: float
) -> float:
    """x* for Case I (ρ ≤ f1), eq. 9.

        x* = ((ρ+λ1-f1)/(ρ+λ1)) · (K·β2 / (β2-1)).

    Sell iff state=2 AND price ≥ x*.  In Case I state=1 is always HOLD.
    """
    if rho > f1:
        raise ValueError(f"Case I requires ρ ≤ f1, got ρ={rho}, f1={f1}")
    if phi(rho, f1, f2, lam1, lam2) <= 0.0:
        raise ValueError("Φ(ρ) ≤ 0 — never-sell regime, no Case I threshold")
    b2 = beta2(rho, f1, f2, lam1, lam2)
    if b2 <= 1.0:
        raise ValueError(f"β2 ≤ 1 (got {b2}); Lemma 1 violated")
    return ((rho + lam1 - f1) / (rho + lam1)) * (K * b2 / (b2 - 1.0))


# ── Case II ───────────────────────────────────────────────────────────────────

def _phi_star_at(
    x: float,
    x0_star: float,
    K: float,
    A0: float,
    b0: float,
    gamma1: float,
    kap2: float,
) -> float:
    """φ*(x) from eq. 27, evaluated stably.

    The literal form φ*(x) = C1·x^γ1 + φ0(x) - (x-K)/κ2 needs C1 that can
    reach ~1e19 when x*_0 << 1. Using the identity
        C1 · x*_0^γ1 = x*_0 - K - φ0(x*_0)      (eq. 25)
    we substitute
        C1·x^γ1 = (x*_0 - K - φ0(x*_0)) · (x/x*_0)^γ1
    which keeps every intermediate near unit magnitude.
    """
    phi0_x = A0 * x + b0
    phi0_x0 = A0 * x0_star + b0
    residual_at_x0 = x0_star - K - phi0_x0  # = C1 · x*_0^γ1  (eq. 25)
    scaled_power = (x / x0_star) ** gamma1
    return residual_at_x0 * scaled_power + phi0_x - (x - K) / kap2


def case2_thresholds(
    rho: float,
    f1: float,
    f2: float,
    lam1: float,
    lam2: float,
    K: float,
    tol: float = 1e-10,
) -> tuple[float, float]:
    """Return (x*, x*_0) for Case II (ρ > f1).

    x*_0 = ρK / (ρ - f1) is the closed-form uptick threshold.
    x* is the unique root of φ* (eq. 27) on [K, x*_0]; φ* is monotone
    decreasing with φ*(K) > 0 and φ*(x*_0) < 0 under A1–A2, so a
    bisection with ≤ 200 iterations converges to ``tol``.
    """
    if rho <= f1:
        raise ValueError(f"Case II requires ρ > f1, got ρ={rho}, f1={f1}")
    if phi(rho, f1, f2, lam1, lam2) <= 0.0:
        raise ValueError("Φ(ρ) ≤ 0 — never-sell regime, no Case II threshold")

    x0_star = rho * K / (rho - f1)
    A0 = lam1 / (rho + lam1 - f1)             # eq. 6, particular soln slope
    b0 = -lam1 * K / (rho + lam1)             # eq. 6, particular soln intercept
    gamma1 = (rho + lam1) / f1                # eq. 25 exponent
    kap2 = kappa2(rho, f1, f2, lam1, lam2)

    lo, hi = K, x0_star
    f_lo = _phi_star_at(lo, x0_star, K, A0, b0, gamma1, kap2)
    f_hi = _phi_star_at(hi, x0_star, K, A0, b0, gamma1, kap2)
    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"Bisection endpoints have same sign: φ*({lo})={f_lo}, "
            f"φ*({hi})={f_hi} — cannot bracket root"
        )

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = _phi_star_at(mid, x0_star, K, A0, b0, gamma1, kap2)
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid, x0_star
        if f_lo * f_mid < 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi), x0_star


# ── Sufficient-condition upper bound (paper page 15) ─────────────────────────

def X0_upper_bound(
    rho: float, f1: float, f2: float, lam1: float, lam2: float, K: float
) -> float:
    """X0 upper bound.  Any valid Case II threshold x* must satisfy x* ≤ X0."""
    b2 = beta2(rho, f1, f2, lam1, lam2)
    kap2 = kappa2(rho, f1, f2, lam1, lam2)
    term1 = K * b2 / (b2 - 1.0)
    num = (lam2 - kap2 * (rho + lam2)) * K
    den = lam2 - kap2 * (rho + lam2 - f2)
    term2 = num / den
    return min(term1, term2)


# ── Unified decision rule ────────────────────────────────────────────────────

def decide(
    price: float,
    state: int,
    rho: float,
    f1: float,
    f2: float,
    lam1: float,
    lam2: float,
    K: float,
) -> Decision:
    """Return the optimal action at (price, state).

    Dispatches on ρ vs f1 to Case I / Case II thresholds. Short-circuits
    to HOLD when Φ(ρ) ≤ 0 (never-sell regime), before touching β2 or the
    Case-II bisection.
    """
    if state not in (1, 2):
        raise ValueError(f"state must be 1 or 2, got {state}")

    if phi(rho, f1, f2, lam1, lam2) <= 0.0:
        return Decision(action="HOLD", reason="Φ(ρ) ≤ 0 — never-sell regime")

    if rho <= f1:
        x_star = case1_threshold(rho, f1, f2, lam1, lam2, K)
        if state == 2 and price >= x_star:
            return Decision(
                action="SELL",
                reason=f"Case I downtick: price {price:.6g} ≥ x*={x_star:.6g}",
                x_star=x_star,
            )
        reason = (
            "Case I uptick — never sell in state 1"
            if state == 1
            else f"Case I downtick: price {price:.6g} < x*={x_star:.6g}"
        )
        return Decision(action="HOLD", reason=reason, x_star=x_star)

    # Case II
    x_star, x0_star = case2_thresholds(rho, f1, f2, lam1, lam2, K)
    if state == 2 and price >= x_star:
        return Decision(
            action="SELL",
            reason=f"Case II downtick: price {price:.6g} ≥ x*={x_star:.6g}",
            x_star=x_star,
            x0_star=x0_star,
        )
    if state == 1 and price >= x0_star:
        return Decision(
            action="SELL",
            reason=f"Case II uptick: price {price:.6g} ≥ x*_0={x0_star:.6g}",
            x_star=x_star,
            x0_star=x0_star,
        )
    reason = (
        f"Case II downtick: price {price:.6g} < x*={x_star:.6g}"
        if state == 2
        else f"Case II uptick: price {price:.6g} < x*_0={x0_star:.6g}"
    )
    return Decision(action="HOLD", reason=reason, x_star=x_star, x0_star=x0_star)


# ── Calibration (paper §Example 3) ───────────────────────────────────────────

def estimate_parameters(
    prices: np.ndarray, dt: float = 1.0 / 252.0
) -> CalibratedParams:
    """Fit (f1, f2, λ1, λ2) from a daily price series via sign-flip counting.

    Steps (paper §Example 3):
      1. ΔZ_k = log(S_{k+1}) - log(S_k)
      2. μ̂ = mean(ΔZ) / dt,  σ0² = var(ΔZ, ddof=1)
      3. R1 = # (ΔZ_k < 0 AND ΔZ_{k+1} ≥ 0)   (down → up flip)
         R2 = # (ΔZ_k > 0 AND ΔZ_{k+1} ≤ 0)   (up   → down flip)
      4. R  = #{ΔZ > 0} / #{ΔZ < 0}
      5. T  = n·dt   (n = len(ΔZ))
         λ1 = (R1 + R2/R) / T,  λ2 = (R·R1 + R2) / T
      6. σ1 = σ0/√dt · √(λ1(λ1+λ2) / (2·λ2))
         σ2 = σ0/√dt · √(λ2(λ1+λ2) / (2·λ1))
      7. f1 = μ + σ1,  f2 = μ - σ2
    """
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1:
        raise ValueError(f"prices must be 1-D, got shape {prices.shape}")
    if len(prices) < 3:
        raise ValueError(
            f"need ≥ 3 prices for one sign-flip pair, got {len(prices)}"
        )
    if np.any(prices <= 0):
        raise ValueError("prices must be strictly positive")

    dz = np.diff(np.log(prices))
    n = len(dz)
    mu = float(dz.mean() / dt)
    sigma0 = float(dz.std(ddof=1))

    pos = int((dz > 0).sum())
    neg = int((dz < 0).sum())
    if pos == 0 or neg == 0:
        raise ValueError("returns are all one sign; cannot estimate two-state chain")

    prev, nxt = dz[:-1], dz[1:]
    R1 = int(((prev < 0) & (nxt >= 0)).sum())
    R2 = int(((prev > 0) & (nxt <= 0)).sum())
    if R1 == 0 and R2 == 0:
        raise ValueError("no sign flips detected; cannot estimate λ")

    R = pos / neg
    T = n * dt
    lam1 = (R1 + R2 / R) / T
    lam2 = (R * R1 + R2) / T
    if lam1 <= 0 or lam2 <= 0:
        raise ValueError(f"non-positive λ estimates: λ1={lam1}, λ2={lam2}")

    sigma_annual = sigma0 / sqrt(dt)
    sigma1 = sigma_annual * sqrt(lam1 * (lam1 + lam2) / (2.0 * lam2))
    sigma2 = sigma_annual * sqrt(lam2 * (lam1 + lam2) / (2.0 * lam1))
    f1 = mu + sigma1
    f2 = mu - sigma2

    return CalibratedParams(
        f1=float(f1),
        f2=float(f2),
        lam1=float(lam1),
        lam2=float(lam2),
        mu=float(mu),
        sigma0=float(sigma0),
    )
