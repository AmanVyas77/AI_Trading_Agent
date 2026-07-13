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

import datetime as _dt
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

# ── Live staleness limits (calendar days per series) ─────────────────────────
# Sprint 8 Prompt 2: never silently trade off a stale macro reading.
MAX_STALENESS_DAYS = {
    "vix": 5,
    "yield_spread_10y2y": 5,
    "fed_funds_rate": 10,
    # CPI is reference-month dated; BLS releases ~2-3 weeks after
    # month-end so age normally oscillates 40-75d. Amended 60 → 75
    # in Prompt 4 FIX 2 (calibration only).
    "cpi": 75,
}

# vix / yield spread are load-bearing for the rule signal — stale on
# either forces NEUTRAL regardless of what the rules say.
_CRITICAL_SERIES = ("vix", "yield_spread_10y2y")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _classify_row(vix: Optional[float], spread: Optional[float]) -> str:
    """Classify a single month.  risk_off beats risk_on when both fire."""
    if vix is None and spread is None:
        return "NEUTRAL"
    if (vix is not None and vix > VIX_RISK_OFF) or (
        vix is not None and vix > VIX_RISK_ON
        and spread is not None and spread < SPREAD_FLOOR
    ):
        return "RISK_OFF"
    if (vix is not None and vix < VIX_RISK_ON) and (
        spread is not None and spread > SPREAD_FLOOR
    ):
        return "RISK_ON"
    return "NEUTRAL"


def _load_macro_snapshot() -> dict[str, dict]:
    """Load the most recent value per macro series from SQLite.

    Uses a per-series MAX(date) join so every series contributes its own
    latest observation (monthly cpi/industrial_production are otherwise
    starved out by daily prints from a naive `ORDER BY date DESC LIMIT n`).

    Returns
    -------
    dict[series_name, {"value": float, "date": "YYYY-MM-DD"}]
        Empty dict on connect/read failure. Series with a NULL value are
        skipped. Use ``_snapshot_values(snap)`` for the legacy values-only
        view expected by the prompt builder.
    """
    sql = (
        "SELECT m.series_name, m.value, m.date "
        "FROM macro_series m "
        "JOIN ( "
        "  SELECT series_name, MAX(date) AS max_date "
        "  FROM macro_series "
        "  WHERE value IS NOT NULL "
        "  GROUP BY series_name "
        ") latest "
        "ON m.series_name = latest.series_name AND m.date = latest.max_date"
    )
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql(sql, conn, parse_dates=["date"])
        conn.close()
    except Exception as exc:
        logger.warning("Could not read macro_series table: %s", exc)
        return {}

    if df.empty:
        return {}

    snap: dict[str, dict] = {}
    for _, row in df.iterrows():
        name = str(row["series_name"]).lower().replace(" ", "_")
        val = row["value"]
        if pd.isna(val):
            continue
        snap[name] = {
            "value": round(float(val), 4),
            "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
        }
    return snap


def _snapshot_values(snap: dict[str, dict]) -> dict[str, float]:
    """Values-only view over a snapshot — backward-compat for the prompt
    builder and any consumer that just wants numbers."""
    return {k: v["value"] for k, v in snap.items()}


def _snapshot_staleness(
    snap: dict[str, dict],
    today: Optional[_dt.date] = None,
) -> dict[str, int]:
    """Return {series: age_in_days} for every series in the snapshot."""
    today = today or _dt.date.today()
    ages: dict[str, int] = {}
    for name, cell in snap.items():
        try:
            d = _dt.date.fromisoformat(cell["date"])
        except (KeyError, TypeError, ValueError):
            continue
        ages[name] = (today - d).days
    return ages


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


def _run_llm_regime(prompt: str) -> tuple[str, str]:
    """Ask the configured LLM for a regime label.

    Hard-blocks the ollama backend — running local models on this Mac
    triggered kernel panics during Sprint 4; that path stays disabled
    until the analyst work brings it back with a safer setup.

    Returns (label, reasoning). Label is one of _VALID_LABELS or
    ``"NEUTRAL"`` on parse miss. Raises RuntimeError if ollama is the
    resolved backend so the caller can force NEUTRAL cleanly.
    """
    from src.rag.query.llm_client import LLMClient  # local import — no cost when unused

    client = LLMClient()
    if client.backend == "ollama":
        raise RuntimeError(
            "ollama backend is disabled in live regime signal — "
            "local Ollama runs caused kernel panics on this Mac. "
            "Set TRADING_AGENT_LLM_BACKEND to anthropic/gemini/deepseek "
            "or call get_live_regime_signal(use_llm=False)."
        )

    raw = client.generate(prompt).strip()
    for line in raw.splitlines():
        token = line.strip().upper()
        if token in _VALID_LABELS:
            return token, raw
    return "NEUTRAL", raw


def get_live_regime_signal(use_llm: bool = False) -> dict:
    """Return the current regime multiplier, primarily rule-based.

    Only for live/forward use — never called inside the backtest loop.

    Parameters
    ----------
    use_llm : bool, default False
        When True (and the resolved LLM backend is not ollama), consult
        the LLM and blend with the rule signal (agree → that signal;
        disagree → NEUTRAL). When False the LLM path is skipped entirely
        and the final signal equals the rule signal.

    Returns
    -------
    dict with keys:
        multiplier      : float — final regime multiplier
        rule_signal     : "RISK_ON" | "NEUTRAL" | "RISK_OFF"
        llm_signal      : same labels, or "SKIPPED" when use_llm=False
                          / backend disabled
        reasoning       : str — LLM output or short status message
        macro_snapshot  : dict[series, {"value", "date"}] (as-of dates)
        stale           : bool — True if any tracked series exceeded its
                          MAX_STALENESS_DAYS budget
        stale_series    : list[str] — names of stale/missing series
    """
    snapshot = _load_macro_snapshot()
    values = _snapshot_values(snapshot)
    ages = _snapshot_staleness(snapshot)

    # ── Staleness check ──────────────────────────────────────────────────
    stale_series: list[str] = []
    for name, limit in MAX_STALENESS_DAYS.items():
        if name not in snapshot:
            logger.warning("Macro series '%s' missing from snapshot", name)
            stale_series.append(name)
            continue
        age = ages.get(name, 10**6)
        if age > limit:
            logger.warning(
                "Macro series '%s' is %d days old (limit %d)",
                name, age, limit,
            )
            stale_series.append(name)

    stale = bool(stale_series)

    vix = values.get("vix")
    spread = values.get("yield_spread_10y2y")
    rule_signal = _classify_row(vix, spread)

    # A stale critical series (vix / yield spread) forces NEUTRAL — never
    # silently RISK_ON/OFF on old data.
    critical_stale = [s for s in stale_series if s in _CRITICAL_SERIES]
    if critical_stale:
        return {
            "multiplier": NEUTRAL_MULT,
            "rule_signal": rule_signal,
            "llm_signal": "SKIPPED",
            "reasoning": "stale macro — defaulting to neutral",
            "macro_snapshot": snapshot,
            "stale": True,
            "stale_series": stale_series,
        }

    # ── LLM assessment (opt-in, ollama-blocked) ──────────────────────────
    llm_signal = "SKIPPED"
    reasoning = "LLM skipped — rules-only live signal"

    if use_llm:
        prompt = (
            "You are a macro regime classifier for a systematic equity strategy.\n"
            "Given the following macro indicators, classify the current regime "
            "as exactly one of: RISK_ON, RISK_OFF, or NEUTRAL.\n"
            "Start your response with exactly that label on its own line, "
            "then explain briefly.\n\n"
            "Current macro snapshot:\n"
        )
        for key in ("vix", "yield_spread_10y2y", "fed_funds_rate", "cpi"):
            val = values.get(key)
            prompt += f"  {key}: {val if val is not None else 'N/A'}\n"

        try:
            llm_signal, reasoning = _run_llm_regime(prompt)
        except Exception as exc:
            logger.warning("LLM call failed, blend falls through to rule: %s", exc)
            llm_signal = "SKIPPED"
            reasoning = f"LLM error: {exc}"

    # ── Blend ────────────────────────────────────────────────────────────
    if llm_signal == "SKIPPED" or llm_signal == rule_signal:
        final = rule_signal
    else:
        logger.info(
            "Rule (%s) and LLM (%s) disagree → NEUTRAL",
            rule_signal, llm_signal,
        )
        final = "NEUTRAL"

    return {
        "multiplier": _LABEL_TO_MULT[final],
        "rule_signal": rule_signal,
        "llm_signal": llm_signal,
        "reasoning": reasoning,
        "macro_snapshot": snapshot,
        "stale": stale,
        "stale_series": stale_series,
    }
