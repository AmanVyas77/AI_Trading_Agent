"""
Shared pytest fixtures for unit and integration tests.
All fixtures use in-memory SQLite (no disk I/O) or synthetic DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text


# ── In-memory SQLite engine with all project tables ──────────────────────────

@pytest.fixture()
def engine():
    """
    In-memory SQLAlchemy engine with every table the project uses
    (prices, macro_series, xbrl_facts, xbrl_derived, analyst_estimates,
    eps_revisions, short_interest, sentiment_scores, pipeline_runs).
    """
    eng = create_engine("sqlite:///:memory:", echo=False)
    with eng.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS prices (
                ticker    TEXT NOT NULL,
                date      TEXT NOT NULL,
                open      REAL,
                high      REAL,
                low       REAL,
                close     REAL,
                volume    REAL,
                adj_close REAL,
                PRIMARY KEY (ticker, date)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS macro_series (
                series_id   TEXT NOT NULL,
                series_name TEXT,
                date        TEXT NOT NULL,
                value       REAL,
                PRIMARY KEY (series_id, date)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS xbrl_facts (
                ticker              TEXT NOT NULL,
                end_date            TEXT NOT NULL,
                revenue             REAL,
                gross_profit        REAL,
                rd_expense          REAL,
                operating_income    REAL,
                net_income          REAL,
                eps_diluted         REAL,
                total_assets        REAL,
                total_liabilities   REAL,
                stockholders_equity REAL,
                deferred_revenue    REAL,
                cash                REAL,
                long_term_debt      REAL,
                operating_cf        REAL,
                capex               REAL,
                PRIMARY KEY (ticker, end_date)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS xbrl_derived (
                ticker               TEXT NOT NULL,
                end_date             TEXT NOT NULL,
                free_cash_flow       REAL,
                enterprise_value     REAL,
                gross_profitability  REAL,
                fcf_yield            REAL,
                revenue_yoy          REAL,
                revenue_acceleration REAL,
                deferred_revenue_yoy REAL,
                rd_intensity         REAL,
                PRIMARY KEY (ticker, end_date)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS analyst_estimates (
                ticker        TEXT NOT NULL,
                fiscal_period TEXT NOT NULL,
                estimate_date TEXT NOT NULL,
                eps_estimate  REAL,
                eps_actual    REAL,
                eps_surprise  REAL,
                sue_score     REAL,
                source        TEXT,
                PRIMARY KEY (ticker, fiscal_period, estimate_date)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS eps_revisions (
                ticker      TEXT NOT NULL,
                date        TEXT NOT NULL,
                revision_1m REAL,
                revision_3m REAL,
                direction   TEXT,
                magnitude   REAL,
                PRIMARY KEY (ticker, date)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS short_interest (
                ticker        TEXT NOT NULL,
                date          TEXT NOT NULL,
                short_volume  REAL,
                total_volume  REAL,
                short_ratio   REAL,
                days_to_cover REAL,
                PRIMARY KEY (ticker, date)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sentiment_scores (
                ticker           TEXT NOT NULL,
                filing_date      TEXT NOT NULL,
                period_of_report TEXT,
                finbert_score    REAL,
                num_chunks       INTEGER,
                source           TEXT,
                PRIMARY KEY (ticker, filing_date)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at       TEXT,
                run_type     TEXT,
                tickers      TEXT,
                rows_written INTEGER
            )
        """))
    return eng


# ── Sample prices: 30 trading days, 5 tickers ────────────────────────────────

@pytest.fixture()
def sample_prices() -> pd.DataFrame:
    """
    Wide DataFrame (date × ticker) of synthetic adj_close prices.
    30 business days, 5 tickers: AAAA, BBBB, CCCC, DDDD, EEEE.
    Prices start at 100 and follow a simple random walk (seeded).
    """
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-02", periods=30, freq="B")
    tickers = ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"]
    data = {}
    for t in tickers:
        returns = np.random.normal(0.001, 0.02, size=30)
        prices = 100.0 * np.cumprod(1 + returns)
        data[t] = prices
    return pd.DataFrame(data, index=dates)


# ── Sample XBRL: 8 quarters, 5 tickers ──────────────────────────────────────

@pytest.fixture()
def sample_xbrl() -> pd.DataFrame:
    """
    Synthetic xbrl_facts DataFrame: 8 quarters × 5 tickers = 40 rows.
    Quarter ends: 2021-Q1 through 2022-Q4.
    """
    np.random.seed(99)
    tickers = ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"]
    quarter_ends = pd.date_range("2021-03-31", periods=8, freq="QE")

    rows = []
    for t in tickers:
        base_rev = np.random.uniform(500, 2000)
        for i, qe in enumerate(quarter_ends):
            rev = base_rev * (1 + 0.02 * i + np.random.normal(0, 0.01))
            gp = rev * np.random.uniform(0.30, 0.60)
            ta = rev * np.random.uniform(3.0, 6.0)
            rows.append({
                "ticker": t,
                "end_date": qe.strftime("%Y-%m-%d"),
                "revenue": round(rev, 2),
                "gross_profit": round(gp, 2),
                "rd_expense": round(rev * np.random.uniform(0.05, 0.20), 2),
                "operating_income": round(gp * np.random.uniform(0.20, 0.50), 2),
                "net_income": round(gp * np.random.uniform(0.10, 0.40), 2),
                "eps_diluted": round(np.random.uniform(0.50, 3.00), 2),
                "total_assets": round(ta, 2),
                "total_liabilities": round(ta * np.random.uniform(0.30, 0.60), 2),
                "stockholders_equity": round(ta * np.random.uniform(0.30, 0.60), 2),
                "deferred_revenue": round(rev * np.random.uniform(0.05, 0.15), 2),
                "cash": round(ta * np.random.uniform(0.05, 0.20), 2),
                "long_term_debt": round(ta * np.random.uniform(0.05, 0.25), 2),
                "operating_cf": round(gp * np.random.uniform(0.30, 0.70), 2),
                "capex": round(gp * np.random.uniform(0.05, 0.20), 2),
            })
    return pd.DataFrame(rows)
