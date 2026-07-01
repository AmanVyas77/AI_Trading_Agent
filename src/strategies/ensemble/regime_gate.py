"""
Regime Gate — scales portfolio weights by macro regime.

Two modes:
  1. Historical (backtest): get_historical_regime_multipliers() returns a
     monthly Series of multipliers derived purely from rule-based VIX /
     yield-spread thresholds.  No LLM calls.
  2. Live (forward): get_live_regime_signal() blends the same rule-based
     classification with an LLM assessment for a single present-day
     reading.  Only called outside the backtest loop.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
with open(ROOT / "config" / "settings.yaml") as f:
    _CFG = yaml.safe_load(f)

DB_PATH = ROOT / _CFG["data"]["paths"]["db"]

# ── Regime thresholds (match regime_analysis._classify_regimes) ──────────────
VIX_RISK_OFF = 25.0
VIX_RISK_ON = 20.0
SPREAD_FLOOR = 0.0

# ── Multipliers ──────────────────────────────────────────────────────────────
RISK_OFF_MULT = 0.5
NEUTRAL_MULT = 1.0
RISK_ON_MULT = 1.2

_LABEL_TO_MULT = {
    "RISK_OFF": RISK_OFF_MULT,
    "NEUTRAL": NEUTRAL_MULT,
    "RISK_ON": RISK_ON_MULT,
}

_VALID_LABELS = set(_LABEL_TO_MULT)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _classify_row(vix: Optional[float], spread: Optional[float]) -> str:
    """Classify a single month.  risk_off beats risk_on when both fire."""
    if vix is None and spread is None:
        return "NEUTRAL"
    if (vix is not None and vix > VIX_RISK_OFF) or (
        spread is not None and spread < SPREAD_FLOOR
    ):
        return "RISK_OFF"
    if (vix is not None and vix < VIX_RISK_ON) and (
        spread is not None and spread > SPREAD_FLOOR
    ):
        return "RISK_ON"
    return "NEUTRAL"


def _load_macro_snapshot() -> dict[str, float]:
    """Load the most recent value per macro series from SQLite.

    Mirrors context_builder.ContextBuilder._load_macro().
    """
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql(
            "SELECT * FROM macro_series ORDER BY date DESC LIMIT 7",
            conn,
            parse_dates=["date"],
        )
        conn.close()
    except Exception:
        logger.warning("Could not connect to macro_series table")
        return {}

    if df.empty:
        return {}

    if "series_name" in df.columns and "value" in df.columns:
        latest = df.sort_values("date").groupby("series_name")["value"].last()
        return {k.lower().replace(" ", "_"): round(v, 4) for k, v in latest.items()}

    return df.iloc[0].dropna().to_dict()


# ── Public API ───────────────────────────────────────────────────────────────

def get_historical_regime_multipliers(
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Return a monthly Series of regime multipliers for [start_date, end_date].

    Purely rule-based — no LLM calls.  Suitable for backtest use.

    Parameters
    ----------
    start_date, end_date : ISO date strings ("YYYY-MM-DD").

    Returns
    -------
    pd.Series with DatetimeIndex (month-end) and float values in
    {RISK_OFF_MULT, NEUTRAL_MULT, RISK_ON_MULT}.
    """
    from src.strategies.ensemble.regime_analysis import _load_raw_macro

    macro = _load_raw_macro()  # month-end DatetimeIndex, wide columns

    if macro.empty:
        logger.warning("No macro data — returning all-neutral multipliers")
        idx = pd.date_range(start_date, end_date, freq="ME")
        return pd.Series(NEUTRAL_MULT, index=idx, name="regime_mult")

    mask = (macro.index >= pd.Timestamp(start_date)) & (
        macro.index <= pd.Timestamp(end_date)
    )
    macro = macro.loc[mask]

    vix = macro.get("vix")
    spread = macro.get("yield_spread_10y2y")

    labels = []
    for dt in macro.index:
        v = vix.get(dt) if vix is not None else None
        s = spread.get(dt) if spread is not None else None
        if pd.isna(v):
            v = None
        if pd.isna(s):
            s = None
        labels.append(_classify_row(v, s))

    multipliers = pd.Series(
        [_LABEL_TO_MULT[lbl] for lbl in labels],
        index=macro.index,
        name="regime_mult",
    )

    counts = pd.Series(labels).value_counts()
    logger.info(
        f"Regime multipliers {start_date}→{end_date}: "
        + ", ".join(f"{k}={v}" for k, v in counts.items())
    )

    return multipliers


def get_live_regime_signal() -> dict:
    """Return the current regime multiplier, blending rules + LLM.

    Only for live/forward use — never called inside the backtest loop.

    Returns
    -------
    dict with keys: multiplier, rule_signal, llm_signal, reasoning,
    macro_snapshot.
    """
    snapshot = _load_macro_snapshot()

    vix = snapshot.get("vix")
    spread = snapshot.get("yield_spread_10y2y")
    rule_signal = _classify_row(vix, spread)

    # ── LLM assessment ───────────────────────────────────────────────────
    prompt = (
        "You are a macro regime classifier for a systematic equity strategy.\n"
        "Given the following macro indicators, classify the current regime "
        "as exactly one of: RISK_ON, RISK_OFF, or NEUTRAL.\n"
        "Start your response with exactly that label on its own line, "
        "then explain briefly.\n\n"
        "Current macro snapshot:\n"
    )
    for key in ("vix", "yield_spread_10y2y", "fed_funds_rate", "cpi"):
        val = snapshot.get(key)
        prompt += f"  {key}: {val if val is not None else 'N/A'}\n"

    llm_signal = "NEUTRAL"
    reasoning = ""

    try:
        from src.rag.query.llm_client import LLMClient

        raw = LLMClient().generate(prompt)
        reasoning = raw.strip()

        for line in reasoning.splitlines():
            token = line.strip().upper()
            if token in _VALID_LABELS:
                llm_signal = token
                break
    except Exception as exc:
        logger.warning(f"LLM call failed, defaulting to NEUTRAL: {exc}")
        reasoning = f"LLM error: {exc}"

    # ── Blend: agree → use that signal; disagree → NEUTRAL ───────────────
    if rule_signal == llm_signal:
        final = rule_signal
    else:
        logger.info(
            f"Rule ({rule_signal}) and LLM ({llm_signal}) disagree → NEUTRAL"
        )
        final = "NEUTRAL"

    return {
        "multiplier": _LABEL_TO_MULT[final],
        "rule_signal": rule_signal,
        "llm_signal": llm_signal,
        "reasoning": reasoning,
        "macro_snapshot": snapshot,
    }
