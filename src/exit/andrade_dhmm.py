"""
Andrade (2017 IST Lisboa MSc) discrete-HMM + RSI regime state machine.

Paper: A. Andrade, "Stock Market Index Trading Algorithm Using Discrete
       Hidden Markov Models and Technical Analysis", MSc dissertation,
       Instituto Superior Técnico, Lisboa (2017). Section 3.4-3.5 describe
       the RSI switcher and forecasting rule.

Model summary
-------------
Three discrete HMMs, each with N=3 hidden states and 2 discrete symbols
(strict maintenance: dZ ≤ 0 → 0, dZ > 0 → 1 — paper §4.4.5, approach 1,
ROR 80.3% — best of the four discretization schemes tested):

    DHMM_daily_30   30-day  window of daily obs
    DHMM_daily_60   60-day  window of daily obs
    DHMM_weekly_30  30-week window of weekly-Friday-close obs

RSI(14) — Wilder smoothing — chooses which DHMM(s) to consult:

    RSI > 70 → daily mode   (short-term sensitivity, overbought)
    RSI < 30 → weekly mode  (long-term sensitivity,  oversold)
    30 ≤ RSI ≤ 70 → keep prev_mode  (never flip on noise)

Signal assembly (paper §3.4, Figure 17):

    daily mode:  f30 = predict_next(DHMM_daily_30), f60 = predict_next(DHMM_daily_60)
                 f30 == f60 == 1 → STRONG_BUY
                 f30 == f60 == 0 → STRONG_SELL
                 else            → HOLD
    weekly mode: fw = predict_next(DHMM_weekly_30)
                 fw == 1 → STRONG_BUY
                 else    → SELL   (softer than STRONG_SELL — shorts in an
                                   oversold weekly regime are risky, paper §3.4)

Next-observation prediction (paper §3.5)
----------------------------------------
Given a fitted DHMM and an observation window:
    1. Viterbi-decode the window → last state s_T
    2. Most probable next state s_{T+1} = argmax_j A[s_T, j]
    3. Most probable next observation o = argmax_k B[s_{T+1}, k]

hmmlearn class
--------------
Uses ``CategoricalHMM`` from hmmlearn 0.3+. Older releases exposed the
same functionality as ``MultinomialHMM`` — this module falls back to
that name if the newer class isn't available. Baum-Welch is not fully
deterministic across BLAS backends; ``random_state`` fixes the init
but the local optimum can still drift by a few percentage-point
prediction-error rates. Treat the paper's Appendix A error rates as a
soft target, not an exact match — pattern-error tests use a generous
40% ceiling per pattern (paper max was 38%).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import CategoricalHMM as _DiscreteHMM
    _HMM_CLASS_NAME = "CategoricalHMM"
except ImportError:  # older hmmlearn
    from hmmlearn.hmm import MultinomialHMM as _DiscreteHMM  # type: ignore
    _HMM_CLASS_NAME = "MultinomialHMM"


# ── Discretization ───────────────────────────────────────────────────────────

def discretize_daily(prices: np.ndarray) -> np.ndarray:
    """Encode daily log-returns as 0 (fall/flat) / 1 (rise).

    Length of the output is ``len(prices) - 1`` (one obs per return).
    Strict-maintenance convention: dZ ≤ 0 → 0.
    """
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1:
        raise ValueError(f"prices must be 1-D, got shape {prices.shape}")
    if len(prices) < 2:
        raise ValueError(f"need ≥ 2 prices to compute one return, got {len(prices)}")
    if np.any(prices <= 0):
        raise ValueError("prices must be strictly positive")
    dz = np.diff(np.log(prices))
    return (dz > 0).astype(int)


def weekly_fridays(prices_df: pd.DataFrame) -> pd.Series:
    """Return the last close on or before each Friday.

    ``prices_df`` must have a ``DatetimeIndex`` and a ``close`` column.
    The pandas ``W-FRI`` resample rule anchors each week on Friday and
    ``.last()`` keeps the final observation in that week, silently
    handling Friday holidays by falling back to Thursday's close.
    """
    if not isinstance(prices_df.index, pd.DatetimeIndex):
        raise ValueError("prices_df must have a DatetimeIndex")
    if "close" not in prices_df.columns:
        raise ValueError("prices_df must have a 'close' column")
    return prices_df["close"].resample("W-FRI").last().dropna()


def discretize_weekly(prices_df: pd.DataFrame, n_weeks: int = 30) -> np.ndarray:
    """30 (or n_weeks) most-recent weekly returns, encoded 0/1.

    Uses the last ``n_weeks + 1`` Friday closes so the output length is
    exactly ``n_weeks``.
    """
    fridays = weekly_fridays(prices_df)
    if len(fridays) < n_weeks + 1:
        raise ValueError(
            f"need ≥ {n_weeks + 1} Friday closes for a {n_weeks}-week window, "
            f"got {len(fridays)}"
        )
    recent = fridays.iloc[-(n_weeks + 1):].to_numpy(dtype=float)
    dz = np.diff(np.log(recent))
    return (dz > 0).astype(int)


# ── RSI ──────────────────────────────────────────────────────────────────────

def rsi_wilder(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder-smoothed RSI. Output length matches input; first ``period``
    entries are NaN (need ``period`` deltas to seed the average).

    Convention:
      - All losses zero and gains > 0 → RSI = 100
      - All gains zero and losses > 0 → RSI = 0
      - Both zero (constant window)   → RSI = NaN
    """
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1:
        raise ValueError(f"prices must be 1-D, got shape {prices.shape}")
    n = len(prices)
    if n <= period:
        return np.full(n, np.nan)

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    rsi = np.full(n, np.nan)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    def _rsi_from(avg_g: float, avg_l: float) -> float:
        if avg_l == 0.0 and avg_g == 0.0:
            return np.nan
        if avg_l == 0.0:
            return 100.0
        rs = avg_g / avg_l
        return 100.0 - 100.0 / (1.0 + rs)

    rsi[period] = _rsi_from(avg_gain, avg_loss)
    for i in range(period + 1, n):
        g, l = gains[i - 1], losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        rsi[i] = _rsi_from(avg_gain, avg_loss)
    return rsi


# ── HMM training + next-obs prediction ───────────────────────────────────────

def _fit_hmm(
    obs: np.ndarray,
    n_states: int,
    n_symbols: int,
    n_iter: int,
    random_state: int,
) -> _DiscreteHMM:
    """Fit CategoricalHMM/MultinomialHMM on a 1-D int obs sequence."""
    if obs.ndim != 1:
        raise ValueError(f"obs must be 1-D, got shape {obs.shape}")
    if obs.dtype.kind not in ("i", "u"):
        obs = obs.astype(int)
    x = obs.reshape(-1, 1)
    model = _DiscreteHMM(
        n_components=n_states,
        n_iter=n_iter,
        random_state=random_state,
        init_params="ste",  # start-prob, transmat, emissionprob randomly init
        params="ste",
    )
    # hmmlearn 0.3+ CategoricalHMM needs n_features (number of symbols).
    # MultinomialHMM (older) infers from data.
    if _HMM_CLASS_NAME == "CategoricalHMM":
        model.n_features = n_symbols
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ConvergenceWarning is expected on tiny data
        model.fit(x)
    return model


def _predict_next(model: _DiscreteHMM, obs: np.ndarray) -> int:
    """One-step forecast of the next observation symbol.

    Uses the forward-filtered posterior over states at time T (obtained via
    ``predict_proba``, which runs forward-backward internally), rolls it one
    step through the transition matrix, then through the emission matrix:

        p(o_{T+1}=k) = Σ_i Σ_j  p(s_T=i | o_1..o_T) · A[i,j] · B[j,k]

    This is the standard mathematically-correct HMM next-obs distribution —
    the paper's ψ_T pointer formulation reduces to it. A naïve
    "argmax next state from Viterbi last state" (see earlier draft) loses
    the cycle-phase information carried by the posterior and predicts poorly
    on repetitive short patterns.
    """
    x = obs.reshape(-1, 1)
    posterior = model.predict_proba(x)          # (T, n_states)
    alpha_T = posterior[-1]                     # (n_states,)
    next_state_probs = alpha_T @ model.transmat_        # (n_states,)
    next_obs_probs = next_state_probs @ model.emissionprob_  # (n_symbols,)
    return int(np.argmax(next_obs_probs))


def train_and_predict(
    obs: np.ndarray,
    n_states: int = 3,
    n_symbols: int = 2,
    n_iter: int = 100,
    random_state: int = 42,
) -> int:
    """Fit a DHMM on ``obs`` and return the most-likely next observation."""
    if len(obs) < n_states:
        raise ValueError(
            f"need ≥ n_states={n_states} observations to fit, got {len(obs)}"
        )
    model = _fit_hmm(obs, n_states, n_symbols, n_iter, random_state)
    return _predict_next(model, obs)


# ── State machine ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Signal:
    action: Literal["STRONG_BUY", "STRONG_SELL", "SELL", "HOLD"]
    mode: Literal["daily", "weekly"]
    rsi: float
    reason: str


_ALLOWED_ACTIONS = {"STRONG_BUY", "STRONG_SELL", "SELL", "HOLD"}
_ALLOWED_MODES = {"daily", "weekly"}


def _select_mode(
    rsi_latest: float,
    prev_mode: Literal["daily", "weekly"],
) -> Literal["daily", "weekly"]:
    if np.isnan(rsi_latest):
        return prev_mode
    if rsi_latest > 70.0:
        return "daily"
    if rsi_latest < 30.0:
        return "weekly"
    return prev_mode


def next_signal(
    closes_df: pd.DataFrame,
    prev_mode: Literal["daily", "weekly"] = "daily",
    random_state: int = 42,
    n_iter: int = 100,
) -> Signal:
    """Full Andrade pipeline on a daily close series.

    ``closes_df`` must have a ``DatetimeIndex`` and a ``close`` column.
    Requires enough history to build the required window in the chosen
    mode (60 daily prices for daily mode; ≥ 31 Fridays for weekly mode).
    Also needs ≥ 15 daily prices to seed RSI(14).
    """
    if prev_mode not in _ALLOWED_MODES:
        raise ValueError(f"prev_mode must be 'daily' or 'weekly', got {prev_mode!r}")
    if not isinstance(closes_df.index, pd.DatetimeIndex):
        raise ValueError("closes_df must have a DatetimeIndex")
    if "close" not in closes_df.columns:
        raise ValueError("closes_df must have a 'close' column")

    closes = closes_df["close"].to_numpy(dtype=float)
    if len(closes) < 15:
        raise ValueError(f"need ≥ 15 daily prices to seed RSI(14), got {len(closes)}")

    rsi = rsi_wilder(closes, period=14)
    rsi_latest = float(rsi[-1])
    mode = _select_mode(rsi_latest, prev_mode)

    if mode == "daily":
        if len(closes) < 61:
            raise ValueError(
                f"daily mode needs ≥ 61 prices (60 returns for DHMM_60), got {len(closes)}"
            )
        obs30 = discretize_daily(closes[-31:])
        obs60 = discretize_daily(closes[-61:])
        f30 = train_and_predict(
            obs30, n_states=3, n_symbols=2, n_iter=n_iter, random_state=random_state,
        )
        f60 = train_and_predict(
            obs60, n_states=3, n_symbols=2, n_iter=n_iter, random_state=random_state,
        )
        if f30 == 1 and f60 == 1:
            action: Literal["STRONG_BUY", "STRONG_SELL", "SELL", "HOLD"] = "STRONG_BUY"
        elif f30 == 0 and f60 == 0:
            action = "STRONG_SELL"
        else:
            action = "HOLD"
        reason = (
            f"daily mode (RSI={rsi_latest:.2f}); "
            f"DHMM_30 forecast={f30}, DHMM_60 forecast={f60}"
        )
    else:  # weekly
        obs_w = discretize_weekly(closes_df, n_weeks=30)
        fw = train_and_predict(
            obs_w, n_states=3, n_symbols=2, n_iter=n_iter, random_state=random_state,
        )
        action = "STRONG_BUY" if fw == 1 else "SELL"
        reason = f"weekly mode (RSI={rsi_latest:.2f}); DHMM_weekly forecast={fw}"

    return Signal(action=action, mode=mode, rsi=rsi_latest, reason=reason)
