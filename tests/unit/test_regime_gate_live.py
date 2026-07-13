"""Sprint 8 Prompt 2 — regime_gate live path + sentiment default tests.

Offline-safe: only touches the local SQLite DB and module-level defaults.
No network, no LLM calls, no Ollama import.
"""
from __future__ import annotations

import datetime as dt
import sys

import pytest

from src.strategies.ensemble import regime_gate as rg


# ── 1. Snapshot: per-series latest read ──────────────────────────────────────


def test_snapshot_covers_every_series():
    snap = rg._load_macro_snapshot()

    # Every macro series that the DB carries should show up — the old
    # LIMIT-7 query starved monthly cpi / industrial_production.
    for name in ("vix", "yield_spread_10y2y", "cpi", "fed_funds_rate"):
        assert name in snap, f"missing series: {name}"
        assert "value" in snap[name] and "date" in snap[name]
        assert isinstance(snap[name]["value"], float)

    cpi_date = dt.date.fromisoformat(snap["cpi"]["date"])
    # CPI is monthly and typically stamped to the first of the month.
    # Sprint 7 baseline was 2026-07-04 with CPI up to 2026-05-01.
    assert cpi_date.year == 2026
    assert cpi_date.month in (4, 5, 6)

    today = dt.date.today()
    for critical in ("vix", "yield_spread_10y2y"):
        age = (today - dt.date.fromisoformat(snap[critical]["date"])).days
        assert age <= 14, f"{critical} unexpectedly stale in DB ({age}d)"


# ── 2. Staleness guard: old vix forces NEUTRAL ───────────────────────────────


def test_stale_vix_forces_neutral(monkeypatch):
    today = dt.date.today()
    stale_date = (today - dt.timedelta(days=90)).isoformat()
    fresh_date = today.isoformat()

    fake_snap = {
        "vix": {"value": 32.0, "date": stale_date},  # would say RISK_OFF if fresh
        "yield_spread_10y2y": {"value": 0.5, "date": fresh_date},
        "fed_funds_rate": {"value": 4.5, "date": fresh_date},
        "cpi": {"value": 3.1, "date": fresh_date},
    }
    monkeypatch.setattr(rg, "_load_macro_snapshot", lambda: fake_snap)

    out = rg.get_live_regime_signal()

    assert out["multiplier"] == rg.NEUTRAL_MULT
    assert out["stale"] is True
    assert "vix" in out["stale_series"]
    assert out["reasoning"] == "stale macro — defaulting to neutral"
    assert out["llm_signal"] == "SKIPPED"
    # rule_signal is still reported for diagnostics
    assert out["rule_signal"] == "RISK_OFF"


# ── 3. Rules-only default: no LLM import, no ollama import ───────────────────


def test_rules_only_default_never_imports_ollama(monkeypatch):
    today = dt.date.today().isoformat()
    fresh_snap = {
        "vix": {"value": 18.0, "date": today},
        "yield_spread_10y2y": {"value": 0.5, "date": today},
        "fed_funds_rate": {"value": 4.5, "date": today},
        "cpi": {"value": 3.1, "date": today},
    }
    monkeypatch.setattr(rg, "_load_macro_snapshot", lambda: fresh_snap)

    # Force a clean slate — llm_client / ollama must not be pulled in
    # by the default (use_llm=False) path.
    for mod in list(sys.modules):
        if mod.startswith("ollama") or mod == "src.rag.query.llm_client":
            monkeypatch.delitem(sys.modules, mod, raising=False)

    out = rg.get_live_regime_signal()  # default use_llm=False

    assert out["llm_signal"] == "SKIPPED"
    assert out["multiplier"] in (
        rg.RISK_OFF_MULT, rg.NEUTRAL_MULT, rg.RISK_ON_MULT,
    )
    # Rule signal for (VIX=18, spread=+0.5) is RISK_ON per _classify_row.
    assert out["rule_signal"] == "RISK_ON"
    assert out["multiplier"] == rg.RISK_ON_MULT
    assert out["stale"] is False
    assert out["stale_series"] == []

    assert "ollama" not in sys.modules, "ollama must not be imported on the default path"
    assert "src.rag.query.llm_client" not in sys.modules, (
        "LLMClient must not be imported when use_llm=False"
    )


# ── 4. Sentiment DATE_END default = today ────────────────────────────────────


def test_sentiment_date_end_default_is_today():
    from src.strategies.fundamental import sentiment_pipeline as sp

    assert sp.DATE_START == "2015-01-01"
    assert sp.DATE_END == dt.datetime.now().strftime("%Y-%m-%d")
