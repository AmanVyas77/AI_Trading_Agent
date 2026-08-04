"""Tests for the 2026-08-03 lookahead fix and price-vintage pin.

Context: `backtest_exit_layer.py` called `get_live_regime_signal()` inside its
per-(ticker, month) loop, applying the *current* macro snapshot to decisions made
months in the past. These tests lock in the point-in-time replacement and the
vintage fingerprint that makes a rewritten price table detectable.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from src.strategies.ensemble import regime_gate as rg
from src.utils.data_vintage import price_vintage


# ── as-of regime signal ──────────────────────────────────────────────────────

class TestRegimeSignalAsOf:
    def test_matches_historical_multipliers_over_holdout(self):
        """The as-of path and the independently-written historical path must
        agree month for month — they encode the same rules."""
        hist = rg.get_historical_regime_multipliers("2025-01-31", "2026-06-30")
        if hist.empty:
            pytest.skip("no macro data in DB")
        for dt in hist.index:
            asof = rg.get_regime_signal_asof(dt)
            assert asof["multiplier"] == pytest.approx(float(hist.loc[dt])), (
                f"{dt.date()}: as-of {asof['multiplier']} != historical "
                f"{float(hist.loc[dt])}"
            )

    def test_returns_live_signal_dict_shape(self):
        """Must be drop-in compatible with the live signal it replaces."""
        asof = rg.get_regime_signal_asof("2025-06-30")
        for key in ("multiplier", "rule_signal", "llm_signal", "reasoning",
                    "macro_snapshot", "stale", "stale_series"):
            assert key in asof, f"missing key {key}"
        assert asof["llm_signal"] == "SKIPPED"  # never reconstructible historically

    def test_uses_no_data_after_as_of(self):
        """The whole point: no macro observation may postdate as_of."""
        as_of = pd.Timestamp("2025-06-30")
        asof = rg.get_regime_signal_asof(as_of)
        for name, entry in asof["macro_snapshot"].items():
            assert pd.Timestamp(entry["date"]) <= as_of, (
                f"{name} dated {entry['date']} is AFTER as_of {as_of.date()}"
            )

    def test_is_not_constant_across_the_window(self):
        """Regression guard for the bug itself: the old code returned the same
        run-time reading for every month, so a flat series is the signature."""
        months = pd.date_range("2025-01-31", "2026-06-30", freq="ME")
        mults = {rg.get_regime_signal_asof(m)["multiplier"] for m in months}
        if len(months) > 1:
            assert len(mults) > 1, (
                "as-of regime is constant across the window — the run-time "
                "lookahead bug may have regressed"
            )

    def test_missing_macro_forces_neutral(self):
        """Before macro coverage begins there is nothing to classify on, so the
        signal must degrade to NEUTRAL and say so — never guess a regime."""
        asof = rg.get_regime_signal_asof("1990-01-01")
        assert asof["multiplier"] == rg.NEUTRAL_MULT
        assert asof["stale"] is True
        assert set(asof["stale_series"]) == set(rg._CRITICAL_SERIES)

    def test_stale_macro_forces_neutral(self):
        """A real observation that is too old must also degrade to NEUTRAL
        rather than silently trading off it — mirrors the live path's
        critical-staleness rule. 2015-01-04 is a Sunday: the nearest observation
        is Friday 2015-01-02, so a 1-day budget is genuinely breached."""
        asof = rg.get_regime_signal_asof("2015-01-04", max_staleness_days=1)
        assert asof["stale"] is True
        assert asof["multiplier"] == rg.NEUTRAL_MULT
        # ... and the same date inside the budget classifies normally.
        fresh = rg.get_regime_signal_asof("2015-01-04", max_staleness_days=5)
        assert fresh["stale"] is False
        assert fresh["multiplier"] != rg.NEUTRAL_MULT


# ── price vintage ────────────────────────────────────────────────────────────

class TestPriceVintage:
    def test_deterministic(self, tmp_path):
        db = self._make_db(tmp_path, [("2025-01-02", "AAA", 10.0)])
        a = price_vintage(db, "2025-01-01", "2025-12-31")
        b = price_vintage(db, "2025-01-01", "2025-12-31")
        assert a["sha256"] == b["sha256"]

    def test_detects_a_revised_price(self, tmp_path):
        """The failure mode that broke Sprint 7 reproducibility: same row count,
        same window, one rewritten adj_close."""
        db1 = self._make_db(tmp_path / "a", [("2025-01-02", "AAA", 10.0)])
        db2 = self._make_db(tmp_path / "b", [("2025-01-02", "AAA", 10.01)])
        v1, v2 = (price_vintage(d, "2025-01-01", "2025-12-31") for d in (db1, db2))
        assert v1["n_rows"] == v2["n_rows"] == 1
        assert v1["sha256"] != v2["sha256"], "revised price did not change the digest"

    def test_detects_an_added_row(self, tmp_path):
        db1 = self._make_db(tmp_path / "a", [("2025-01-02", "AAA", 10.0)])
        db2 = self._make_db(tmp_path / "b", [("2025-01-02", "AAA", 10.0),
                                             ("2025-01-03", "AAA", 11.0)])
        v1, v2 = (price_vintage(d, "2025-01-01", "2025-12-31") for d in (db1, db2))
        assert v1["sha256"] != v2["sha256"]
        assert (v1["n_rows"], v2["n_rows"]) == (1, 2)

    def test_window_is_respected(self, tmp_path):
        db = self._make_db(tmp_path, [("2025-01-02", "AAA", 10.0),
                                      ("2026-01-02", "AAA", 20.0)])
        assert price_vintage(db, "2025-01-01", "2025-12-31")["n_rows"] == 1

    @staticmethod
    def _make_db(path, rows) -> str:
        path.mkdir(parents=True, exist_ok=True)
        db = path / "t.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE prices "
                         "(date TEXT, ticker TEXT, adj_close REAL)")
            conn.executemany("INSERT INTO prices VALUES (?,?,?)", rows)
        return str(db)
