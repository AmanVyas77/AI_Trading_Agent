"""
Unit Tests — AlphaVantage news backfill integrity
==================================================
Regression cover for the two data-quality defects found on 2026-08-11 after
the backfill loop reported BACKFILL COMPLETE:

  1. Windows returning AV's 1000-item cap were logged as done, silently
     dropping every article after the cut (NVDA's 2026 window held 9 days of
     news out of 223).
  2. A window whose fetch raised was still written to news_ingest_log as
     completed, so it was never retried — and `items_fetched` recorded the
     run-cumulative total, which made the dead window look productive.

All tests stub the network; no API key or live request is involved.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from src.data import news_pipeline as np_mod


# ── helpers ──────────────────────────────────────────────────────────────────

class _KeepAlive(sqlite3.Connection):
    """
    av_backfill() closes its connection on exit, which would discard an
    in-memory DB before the test could assert on it. Ignore close(); the
    connection is garbage-collected with the test.
    """

    def close(self):    # noqa: D102
        pass


def _mem_db(keepalive: bool = False) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:",
                          factory=_KeepAlive if keepalive else sqlite3.Connection)
    con.executescript(np_mod.DDL)
    np_mod._migrate(con)
    return con


def _item(ts: str, title: str) -> dict:
    """Minimal AV feed item."""
    return {"time_published": ts, "title": title, "summary": "s",
            "source_domain": "example.com",
            "ticker_sentiment": [{"ticker": "TEST", "relevance_score": "0.5"}]}


def _page(n: int, start: datetime, step_min: int = 10) -> list[dict]:
    """n synthetic items, `step_min` apart, ascending (AV sort=EARLIEST)."""
    out = []
    for i in range(n):
        ts = (start.timestamp() + i * step_min * 60)
        d = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
        out.append(_item(d.strftime("%Y%m%dT%H%M%S"), f"a{i}"))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 1. Truncation → pagination
# ══════════════════════════════════════════════════════════════════════════════

class TestTruncationPagination:
    """A full page means 'there is more', not 'that is everything'."""

    def test_full_page_triggers_another_fetch(self, monkeypatch):
        con = _mem_db()
        calls = []

        def fake_fetch(ticker, tf, tt, key, limit=np_mod.AV_ITEM_CAP,
                       attempts=np_mod.AV_MAX_ATTEMPTS):
            calls.append(tf)
            if len(calls) == 1:
                # exactly the cap → truncated
                return {"feed": _page(np_mod.AV_ITEM_CAP,
                                      datetime(2026, 1, 1))}, False
            return {"feed": _page(12, datetime(2026, 3, 1))}, False

        monkeypatch.setattr(np_mod, "_av_fetch", fake_fetch)
        monkeypatch.setattr(np_mod, "AV_SLEEP_S", 0)

        res = np_mod._av_collect(con, "TEST", "20260101T0000",
                                 "20261231T2359", "k", budget_left=10)

        assert len(calls) == 2, "truncated page must be followed by another fetch"
        assert res["status"] == "ok"
        assert res["pages"] == 2
        assert res["inserted"] == np_mod.AV_ITEM_CAP + 12
        # second call resumed at the last item of page 1, not at the window start
        assert calls[1] > calls[0]

    def test_short_page_stops_immediately(self, monkeypatch):
        con = _mem_db()
        calls = []

        def fake_fetch(ticker, tf, tt, key, limit=np_mod.AV_ITEM_CAP,
                       attempts=np_mod.AV_MAX_ATTEMPTS):
            calls.append(tf)
            return {"feed": _page(5, datetime(2026, 1, 1))}, False

        monkeypatch.setattr(np_mod, "_av_fetch", fake_fetch)
        monkeypatch.setattr(np_mod, "AV_SLEEP_S", 0)

        res = np_mod._av_collect(con, "TEST", "20260101T0000",
                                 "20261231T2359", "k", budget_left=10)
        assert len(calls) == 1
        assert res["status"] == "ok"

    def test_budget_exhaustion_marks_partial_not_ok(self, monkeypatch):
        """Running out of quota mid-window must not look like success."""
        con = _mem_db()

        def fake_fetch(ticker, tf, tt, key, limit=np_mod.AV_ITEM_CAP,
                       attempts=np_mod.AV_MAX_ATTEMPTS):
            return {"feed": _page(np_mod.AV_ITEM_CAP, datetime(2026, 1, 1))}, False

        monkeypatch.setattr(np_mod, "_av_fetch", fake_fetch)
        monkeypatch.setattr(np_mod, "AV_SLEEP_S", 0)

        res = np_mod._av_collect(con, "TEST", "20260101T0000",
                                 "20261231T2359", "k", budget_left=2)
        assert res["status"] == "partial"
        assert res["status"] not in np_mod.AV_DONE_STATES

    def test_cursor_stall_is_forced_forward(self, monkeypatch):
        """≥threshold items sharing one minute must not spin forever."""
        con = _mem_db()
        calls = []

        def fake_fetch(ticker, tf, tt, key, limit=np_mod.AV_ITEM_CAP,
                       attempts=np_mod.AV_MAX_ATTEMPTS):
            calls.append(tf)
            if len(calls) >= 3:
                return {"feed": []}, False
            # every item in the same minute → cursor cannot advance naturally
            return {"feed": [_item("20260101T000000", f"x{i}")
                             for i in range(np_mod.AV_ITEM_CAP)]}, False

        monkeypatch.setattr(np_mod, "_av_fetch", fake_fetch)
        monkeypatch.setattr(np_mod, "AV_SLEEP_S", 0)

        res = np_mod._av_collect(con, "TEST", "20260101T0000",
                                 "20260102T0000", "k", budget_left=6)
        assert len(calls) == 3
        assert calls[1] != calls[0], "stalled cursor must be stepped forward"
        assert res["pages"] == 3


# ══════════════════════════════════════════════════════════════════════════════
# 2. Failed fetch must not be recorded as a completed window
# ══════════════════════════════════════════════════════════════════════════════

class TestFailureNotMarkedComplete:
    """The NVDA 2026 blackout: one dropped connection, window lost forever."""

    def test_fetch_exception_yields_error_status(self, monkeypatch):
        con = _mem_db()

        def boom(*a, **k):
            raise ConnectionResetError("Remote end closed connection")

        monkeypatch.setattr(np_mod, "_av_fetch", boom)
        monkeypatch.setattr(np_mod, "AV_SLEEP_S", 0)

        res = np_mod._av_collect(con, "NVDA", "20260101T0000",
                                 "20260811T2359", "k", budget_left=10)
        assert res["status"] == "error"
        assert res["status"] not in np_mod.AV_DONE_STATES
        assert res["inserted"] == 0
        assert "ConnectionResetError" in res["error"]

    def test_errored_window_stays_pending(self, monkeypatch):
        """_av_pending must return a window whose log row says 'error'."""
        con = _mem_db()
        monkeypatch.setattr(np_mod, "_av_ticker_set", lambda: ["NVDA"])
        np_mod._av_log_window(con, "NVDA", "2026-01..2026-12",
                              "error", 3, 0, "ConnectionResetError")

        pending = np_mod._av_pending(con)
        assert any(t == "NVDA" and tf.startswith("2026")
                   for t, tf, _ in pending), \
            "an errored window must be retried, not treated as done"

    def test_partial_window_stays_pending(self, monkeypatch):
        con = _mem_db()
        monkeypatch.setattr(np_mod, "_av_ticker_set", lambda: ["MSFT"])
        np_mod._av_log_window(con, "MSFT", "2025-01..2025-12",
                              "partial", 5, 900, None)
        pending = np_mod._av_pending(con)
        assert any(t == "MSFT" and tf.startswith("2025")
                   for t, tf, _ in pending)

    def test_ok_window_is_not_refetched(self, monkeypatch):
        con = _mem_db()
        monkeypatch.setattr(np_mod, "_av_ticker_set", lambda: ["MSFT"])
        np_mod._av_log_window(con, "MSFT", "2023-01..2023-12", "ok", 2, 500, None)
        pending = np_mod._av_pending(con)
        assert not any(t == "MSFT" and tf.startswith("2023")
                       for t, tf, _ in pending)

    def test_empty_window_is_not_refetched(self, monkeypatch):
        """Genuine no-coverage must settle, or it burns quota every night."""
        con = _mem_db()
        monkeypatch.setattr(np_mod, "_av_ticker_set", lambda: ["APPF"])
        np_mod._av_log_window(con, "APPF", "2022-01..2022-12", "empty", 1, 0, None)
        pending = np_mod._av_pending(con)
        assert not any(t == "APPF" and tf.startswith("2022")
                       for t, tf, _ in pending)


class TestDailyCap:
    """Hitting the free tier's 25/day must never fabricate a completed window."""

    CAP_MSG = ("We have detected your API key as XXX and our standard API "
               "rate limit is 25 requests per day.")

    def test_capped_window_is_not_logged(self, monkeypatch):
        con = _mem_db(keepalive=True)
        monkeypatch.setattr(np_mod, "_av_ticker_set", lambda: ["AAA"])
        monkeypatch.setattr(np_mod, "_av_key", lambda: "k")
        monkeypatch.setattr(np_mod, "_connect", lambda: con)
        monkeypatch.setattr(np_mod, "_append_state", lambda payload: None)
        monkeypatch.setattr(np_mod, "AV_SLEEP_S", 0)
        monkeypatch.setattr(
            np_mod, "_av_year_windows",
            lambda today=None: [("20230101T0000", "20231231T2359")])
        monkeypatch.setattr(
            np_mod, "_av_fetch",
            lambda *a, **k: ({"Information": self.CAP_MSG}, True))

        np_mod.av_backfill(budget=10)
        assert con.execute("SELECT COUNT(*) FROM news_ingest_log").fetchone()[0] == 0

    def test_cap_stops_the_run_immediately(self, monkeypatch):
        """Once quota is gone, further windows must not each burn a request."""
        con = _mem_db(keepalive=True)
        monkeypatch.setattr(np_mod, "_av_ticker_set",
                            lambda: ["AAA", "BBB", "CCC"])
        monkeypatch.setattr(np_mod, "_av_key", lambda: "k")
        monkeypatch.setattr(np_mod, "_connect", lambda: con)
        monkeypatch.setattr(np_mod, "_append_state", lambda payload: None)
        monkeypatch.setattr(np_mod, "AV_SLEEP_S", 0)
        monkeypatch.setattr(
            np_mod, "_av_year_windows",
            lambda today=None: [("20230101T0000", "20231231T2359")])

        seen = []

        def fake_fetch(ticker, tf, tt, key, limit=np_mod.AV_ITEM_CAP,
                       attempts=np_mod.AV_MAX_ATTEMPTS):
            seen.append(ticker)
            return {"Information": self.CAP_MSG}, True

        monkeypatch.setattr(np_mod, "_av_fetch", fake_fetch)
        res = np_mod.av_backfill(budget=10)

        assert seen == ["AAA"], f"cap must halt the run, but tried {seen}"
        assert res["cap_hit_early"] is True

    def test_cap_after_partial_progress_keeps_rows_and_stays_pending(
            self, monkeypatch):
        """Rows fetched before the cap are kept; the window remains resumable."""
        con = _mem_db(keepalive=True)
        monkeypatch.setattr(np_mod, "_av_ticker_set", lambda: ["NVDA"])
        monkeypatch.setattr(np_mod, "_av_key", lambda: "k")
        monkeypatch.setattr(np_mod, "_connect", lambda: con)
        monkeypatch.setattr(np_mod, "_append_state", lambda payload: None)
        monkeypatch.setattr(np_mod, "AV_SLEEP_S", 0)
        monkeypatch.setattr(
            np_mod, "_av_year_windows",
            lambda today=None: [("20250101T0000", "20251231T2359")])

        calls = []

        def fake_fetch(ticker, tf, tt, key, limit=np_mod.AV_ITEM_CAP,
                       attempts=np_mod.AV_MAX_ATTEMPTS):
            calls.append(tf)
            if len(calls) == 1:
                return {"feed": _page(np_mod.AV_ITEM_CAP,
                                      datetime(2025, 1, 1))}, False
            return {"Information": self.CAP_MSG}, True

        monkeypatch.setattr(np_mod, "_av_fetch", fake_fetch)
        np_mod.av_backfill(budget=10)

        row = con.execute("SELECT status, items_fetched FROM news_ingest_log "
                          "WHERE ticker='NVDA'").fetchone()
        assert row[0] == "partial"
        assert row[1] == np_mod.AV_ITEM_CAP
        assert row[0] not in np_mod.AV_DONE_STATES
        kept = con.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]
        assert kept == np_mod.AV_ITEM_CAP, "partial progress must be durable"


# ══════════════════════════════════════════════════════════════════════════════
# 3. items_fetched is per-window, not run-cumulative
# ══════════════════════════════════════════════════════════════════════════════

class TestPerWindowItemAccounting:
    """NVDA's dead window logged 6835 — the previous window's running total."""

    def test_each_window_logs_its_own_delta(self, monkeypatch):
        con = _mem_db(keepalive=True)
        monkeypatch.setattr(np_mod, "_av_ticker_set", lambda: ["AAA", "BBB"])
        monkeypatch.setattr(np_mod, "_av_key", lambda: "k")
        monkeypatch.setattr(np_mod, "_connect", lambda: con)
        monkeypatch.setattr(np_mod, "_append_state", lambda payload: None)
        monkeypatch.setattr(np_mod, "AV_SLEEP_S", 0)
        monkeypatch.setattr(
            np_mod, "_av_year_windows",
            lambda today=None: [("20230101T0000", "20231231T2359")])

        counts = {"AAA": 7, "BBB": 3}

        def fake_fetch(ticker, tf, tt, key, limit=np_mod.AV_ITEM_CAP,
                       attempts=np_mod.AV_MAX_ATTEMPTS):
            base = datetime(2023, 1, 1) if ticker == "AAA" else datetime(2023, 6, 1)
            return {"feed": _page(counts[ticker], base)}, False

        monkeypatch.setattr(np_mod, "_av_fetch", fake_fetch)
        np_mod.av_backfill(budget=10)

        logged = dict(con.execute(
            "SELECT ticker, items_fetched FROM news_ingest_log "
            "WHERE source='alphavantage'"))
        assert logged["AAA"] == 7
        assert logged["BBB"] == 3, \
            "second window must log its own 3, not the cumulative 10"

    def test_failed_window_logs_zero_items(self, monkeypatch):
        con = _mem_db(keepalive=True)
        monkeypatch.setattr(np_mod, "_av_ticker_set", lambda: ["AAA", "BBB"])
        monkeypatch.setattr(np_mod, "_av_key", lambda: "k")
        monkeypatch.setattr(np_mod, "_connect", lambda: con)
        monkeypatch.setattr(np_mod, "_append_state", lambda payload: None)
        monkeypatch.setattr(np_mod, "AV_SLEEP_S", 0)
        monkeypatch.setattr(
            np_mod, "_av_year_windows",
            lambda today=None: [("20230101T0000", "20231231T2359")])

        def fake_fetch(ticker, tf, tt, key, limit=np_mod.AV_ITEM_CAP,
                       attempts=np_mod.AV_MAX_ATTEMPTS):
            if ticker == "BBB":
                raise TimeoutError("read timed out")
            return {"feed": _page(7, datetime(2023, 1, 1))}, False

        monkeypatch.setattr(np_mod, "_av_fetch", fake_fetch)
        np_mod.av_backfill(budget=20)

        row = con.execute(
            "SELECT items_fetched, status FROM news_ingest_log "
            "WHERE ticker='BBB'").fetchone()
        assert row[0] == 0, "a failed window must not inherit AAA's 7"
        assert row[1] == "error"


# ══════════════════════════════════════════════════════════════════════════════
# 4. Resume cursor
# ══════════════════════════════════════════════════════════════════════════════

class TestResumeCursor:
    """Repairing a truncated window costs only the missing tail."""

    def test_resumes_from_newest_stored_row(self):
        con = _mem_db()
        con.execute(
            "INSERT INTO news_articles (ticker, published_at, title, snippet, "
            "site, source, relevance_score, article_hash) VALUES "
            "('NVDA','2026-01-09 12:03:00','t',NULL,NULL,'alphavantage',NULL,'h1')")
        con.commit()
        cur = np_mod._av_resume_cursor(con, "NVDA", "20260101T0000",
                                       "20260811T2359")
        assert cur == "20260109T1203"

    def test_empty_window_starts_at_window_open(self):
        con = _mem_db()
        cur = np_mod._av_resume_cursor(con, "NVDA", "20260101T0000",
                                       "20260811T2359")
        assert cur == "20260101T0000"

    def test_other_sources_do_not_move_the_cursor(self):
        """FNSPID rows must not be mistaken for AV coverage."""
        con = _mem_db()
        con.execute(
            "INSERT INTO news_articles (ticker, published_at, title, snippet, "
            "site, source, relevance_score, article_hash) VALUES "
            "('NVDA','2026-05-01 00:00:00','t',NULL,NULL,'fnspid',NULL,'h2')")
        con.commit()
        cur = np_mod._av_resume_cursor(con, "NVDA", "20260101T0000",
                                       "20260811T2359")
        assert cur == "20260101T0000"


# ══════════════════════════════════════════════════════════════════════════════
# 5. Persist-layer hygiene
# ══════════════════════════════════════════════════════════════════════════════

class TestPersist:

    def test_unparseable_timestamps_are_dropped_and_counted(self, caplog):
        con = _mem_db()
        feed = [_item("20260101T000000", "good"),
                _item("2026-01-01 00:00:00", "bad-format"),
                _item("", "empty")]
        with caplog.at_level("WARNING"):
            inserted, max_tp = np_mod._av_persist(con, "TEST", feed)
        assert inserted == 1
        assert max_tp == "20260101T000000"
        assert "dropped 2/3" in caplog.text, \
            "silent drops are what hid the failure class in the first place"

    def test_reinsert_is_idempotent(self):
        con = _mem_db()
        feed = _page(5, datetime(2026, 1, 1))
        first, _ = np_mod._av_persist(con, "TEST", feed)
        second, _ = np_mod._av_persist(con, "TEST", feed)
        assert first == 5
        assert second == 0, "article_hash PK must make re-fetch a no-op"


# ══════════════════════════════════════════════════════════════════════════════
# 6. Period key stability
# ══════════════════════════════════════════════════════════════════════════════

class TestPeriodKey:

    def test_current_year_key_does_not_drift_with_today(self):
        """'2026-01..2026-07' → '2026-01..2026-08' minted a new window monthly."""
        july = np_mod._av_period_key("20260101T0000", "20260731T2359")
        august = np_mod._av_period_key("20260101T0000", "20260811T0405")
        assert july == august == "2026-01..2026-12"

    def test_full_year_key(self):
        assert np_mod._av_period_key("20220101T0000",
                                     "20221231T2359") == "2022-01..2022-12"
