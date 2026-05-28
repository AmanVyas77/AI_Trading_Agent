"""
XBRL Feature Engineering Module
================================
Loads all fundamental, sentiment, and short-interest signals from SQLite,
aligns them to a quarterly calendar, z-scores cross-sectionally, and
produces a clean feature matrix for the fundamental strategy scorer.

Tables consumed
---------------
  xbrl_facts, xbrl_derived, analyst_estimates, eps_revisions,
  short_interest, sentiment_scores, prices

Output features (per ticker per quarter)
----------------------------------------
  gross_profitability, fcf_yield, revenue_yoy, revenue_acceleration,
  deferred_revenue_yoy, rd_intensity, sue_score, eps_revision_1m,
  eps_revision_3m, short_interest_dtc, finbert_score

Public API
----------
  build_feature_matrix(tickers, start, end, engine) -> pd.DataFrame
  get_feature_snapshot(date, engine)                -> pd.DataFrame
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
with open(ROOT / "config" / "settings.yaml") as f:
    CFG = yaml.safe_load(f)

DB_PATH = ROOT / CFG["data"]["paths"]["db"]
DB_URL = f"sqlite:///{DB_PATH}"
TIMELINE = CFG["timeline"]

# Features that appear in the final matrix
FEATURE_COLS = [
    "gross_profitability",
    "fcf_yield",
    "revenue_yoy",
    "revenue_acceleration",
    "deferred_revenue_yoy",
    "rd_intensity",
    "sue_score",
    "eps_revision_1m",
    "eps_revision_3m",
    "short_interest_dtc",
    "finbert_score",
    "piotroski_f",
    "qmj_safety",
    "qmj_payout",
]

# Maximum forward-fill horizon for fundamental data (2 quarters ≈ 180 days)
MAX_FFILL_QUARTERS = 2
# Minimum non-null observations to compute a valid z-score on a given date
MIN_OBS_FOR_ZSCORE = 10
# Z-score winsorisation cap
ZSCORE_CAP = 3.0


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_engine(engine=None):
    if engine is not None:
        return engine
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(DB_URL, echo=False)


def _quarter_end_index(start: str, end: str) -> pd.DatetimeIndex:
    """Return a DatetimeIndex of quarter-end dates between *start* and *end*."""
    return pd.date_range(start=start, end=end, freq="QE")


def _ticker_where(tickers: Optional[list[str]], alias: str = "") -> tuple[str, dict]:
    """Build a WHERE clause + params dict for a ticker list."""
    if not tickers:
        return "", {}
    col = f"{alias}.ticker" if alias else "ticker"
    ph = ",".join(f":t{i}" for i in range(len(tickers)))
    return f"{col} IN ({ph})", {f"t{i}": t for i, t in enumerate(tickers)}


# ── loaders ───────────────────────────────────────────────────────────────────

def _load_derived(tickers, engine) -> pd.DataFrame:
    """Load xbrl_derived table → long DataFrame with (ticker, end_date)."""
    clause, params = _ticker_where(tickers)
    where = f"WHERE {clause}" if clause else ""
    sql = f"SELECT * FROM xbrl_derived {where} ORDER BY ticker, end_date"
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, params=params)
    if not df.empty:
        df["end_date"] = pd.to_datetime(df["end_date"])
    return df


def _load_facts(tickers, engine) -> pd.DataFrame:
    """Load xbrl_facts (needed for shares_outstanding / EV approximation)."""
    clause, params = _ticker_where(tickers)
    where = f"WHERE {clause}" if clause else ""
    sql = f"SELECT ticker, end_date, total_assets, stockholders_equity, cash, long_term_debt, operating_cf, capex, gross_profit FROM xbrl_facts {where} ORDER BY ticker, end_date"
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, params=params)
    if not df.empty:
        df["end_date"] = pd.to_datetime(df["end_date"])
    return df


def _load_sue(tickers, engine) -> pd.DataFrame:
    clause, params = _ticker_where(tickers)
    where = f"WHERE {clause}" if clause else ""
    sql = f"SELECT ticker, fiscal_period, sue_score FROM analyst_estimates {where} ORDER BY ticker, fiscal_period"
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, params=params)
    if not df.empty:
        df["fiscal_period"] = pd.to_datetime(df["fiscal_period"])
    return df


def _load_eps_revisions(tickers, engine) -> pd.DataFrame:
    clause, params = _ticker_where(tickers)
    where = f"WHERE {clause}" if clause else ""
    sql = f"SELECT ticker, date, revision_1m, revision_3m FROM eps_revisions {where} ORDER BY ticker, date"
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, params=params)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _load_short_interest(tickers, engine) -> pd.DataFrame:
    # Check table exists — FINRA data may be unavailable
    try:
        with engine.connect() as conn:
            tables = conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='short_interest'"
            )).fetchall()
        if not tables:
            logger.warning("short_interest table not found — skipping DTC signal")
            return pd.DataFrame(columns=["ticker", "date", "days_to_cover"])
    except Exception:
        return pd.DataFrame(columns=["ticker", "date", "days_to_cover"])

    clause, params = _ticker_where(tickers)
    where = f"WHERE {clause}" if clause else ""
    sql = f"SELECT ticker, date, days_to_cover FROM short_interest {where} ORDER BY ticker, date"
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, params=params)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _load_sentiment(tickers, engine) -> pd.DataFrame:
    # Check table exists first — FinBERT pipeline may not have run yet
    try:
        with engine.connect() as conn:
            tables = conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sentiment_scores'"
            )).fetchall()
        if not tables:
            logger.warning("sentiment_scores table not found — skipping FinBERT signal")
            return pd.DataFrame(columns=["ticker", "filing_date", "finbert_score"])
    except Exception:
        return pd.DataFrame(columns=["ticker", "filing_date", "finbert_score"])

    clause, params = _ticker_where(tickers)
    where = f"WHERE {clause}" if clause else ""
    sql = f"SELECT ticker, filing_date, finbert_score FROM sentiment_scores {where} ORDER BY ticker, filing_date"
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, params=params)
    if not df.empty:
        df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df


def _load_prices_wide(tickers, start, end, engine) -> pd.DataFrame:
    """Load adj_close prices as a wide (date × ticker) DataFrame."""
    conditions, params = [], {}
    clause, tp = _ticker_where(tickers)
    if clause:
        conditions.append(clause)
        params.update(tp)
    if start:
        conditions.append("date >= :start"); params["start"] = start
    if end:
        conditions.append("date <= :end"); params["end"] = end
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"SELECT ticker, date, adj_close FROM prices {where} ORDER BY date"
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, params=params)
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="ticker", values="adj_close")


# ── alignment helpers ─────────────────────────────────────────────────────────

def _to_quarterly(df: pd.DataFrame, date_col: str, value_cols: list[str],
                  ticker_col: str = "ticker") -> pd.DataFrame:
    """
    Resample a (ticker, date, value) frame to quarter-end frequency.
    Takes the *last* observation within each quarter for each ticker.
    Returns a DataFrame indexed by quarter_end with ticker + value columns.
    """
    if df.empty:
        return pd.DataFrame(columns=[ticker_col, "quarter_end"] + value_cols)
    df = df.copy()
    df["quarter_end"] = df[date_col].dt.to_period("Q").dt.to_timestamp("Q")
    # Last observation per ticker per quarter
    agg = df.groupby([ticker_col, "quarter_end"])[value_cols].last().reset_index()
    return agg


# ── Enterprise Value approximation ───────────────────────────────────────────

def _compute_fcf_yield(facts: pd.DataFrame, prices_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate FCF Yield = FreeCashFlow / EnterpriseValue.

    Enterprise Value ≈ MarketCap + LongTermDebt − Cash
    MarketCap ≈ adj_close × shares_outstanding

    Shares outstanding is *not* directly in XBRL facts for most filers,
    so we approximate:
        shares_out ≈ stockholders_equity / (stockholders_equity / total_assets * adj_close)
    This is unreliable, so as a **serviceable proxy** when shares data is
    unavailable we fall back to:
        fcf_yield_proxy = gross_profit / total_assets   (Novy-Marx style)

    The function returns a (ticker, quarter_end, fcf_yield) DataFrame.
    """
    if facts.empty:
        return pd.DataFrame(columns=["ticker", "quarter_end", "fcf_yield"])

    facts = facts.copy()
    facts["free_cash_flow"] = facts["operating_cf"] - facts["capex"].abs()
    facts["quarter_end"] = facts["end_date"].dt.to_period("Q").dt.to_timestamp("Q")

    # Get quarter-end prices per ticker
    if not prices_wide.empty:
        qe_prices = prices_wide.resample("QE").last().stack()
        qe_prices.index.names = ["quarter_end", "ticker"]
        qe_prices = qe_prices.rename("adj_close").reset_index()
    else:
        qe_prices = pd.DataFrame(columns=["quarter_end", "ticker", "adj_close"])

    merged = facts.merge(qe_prices, on=["ticker", "quarter_end"], how="left")

    # Fallback proxy: gross_profit / total_assets (Novy-Marx profitability)
    merged["fcf_yield"] = merged["gross_profit"] / merged["total_assets"].replace(0, np.nan)

    return merged[["ticker", "quarter_end", "fcf_yield"]]


# ── Piotroski F-Score (Piotroski 2000) ────────────────────────────────────────

def compute_piotroski_f(
    tickers: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    engine=None,
) -> pd.DataFrame:
    """
    Piotroski F-Score for each (ticker, quarter_end) — Piotroski (2000),
    "Value Investing: The Use of Historical Financial Statement Information."

    The F-Score is the sum of 9 binary signals (1 point if positive, else 0):

      Profitability
        F1: ROA > 0              (net_income / total_assets)
        F2: CFO > 0              (operating_cf)
        F3: ΔROA > 0             (ROA improved YoY)
        F4: Accruals < 0         (CFO/assets − ROA, i.e. cash > accrual earnings)

      Leverage / Liquidity
        F5: ΔLeverage < 0        (long_term_debt/total_assets decreased YoY)
        F6: ΔLiquidity > 0       (current_ratio improved YoY)
        F7: No dilution          (shares_outstanding did not increase YoY)

      Operating Efficiency
        F8: ΔGross margin > 0    (gross_profit/revenue improved YoY)
        F9: ΔAsset turnover > 0  (revenue/total_assets improved YoY)

    Missing-data policy
    -------------------
    A NULL input deterministically fails its signal (0).  If every raw input
    for a row is NULL, the row's F-Score is returned as NaN so downstream
    code can distinguish "no data" from "score of 0".

    Schema notes
    ------------
    Uses the actual xbrl_facts column name ``operating_cf`` (not
    ``operating_cash_flow``).  ``current_assets``, ``current_liabilities``,
    and ``shares_outstanding`` are recent schema additions; if a column is
    not yet present in xbrl_facts it is treated as NULL via ``NULL AS <col>``
    so this function does not crash on legacy ingestion data.

    Returns
    -------
    pd.DataFrame with columns [ticker, quarter_end, piotroski_f]
    piotroski_f is an integer 0–9 (or NaN when all inputs are missing).
    Z-scoring is intentionally NOT applied here — build_feature_matrix()
    handles cross-sectional standardisation for every feature in FEATURE_COLS.
    """
    engine = _get_engine(engine)

    # Introspect xbrl_facts to know which optional columns we can SELECT.
    try:
        with engine.connect() as conn:
            existing = {
                r[1] for r in conn.execute(text("PRAGMA table_info(xbrl_facts)")).fetchall()
            }
    except Exception as e:
        logger.warning(f"Could not introspect xbrl_facts schema for Piotroski: {e}")
        return pd.DataFrame(columns=["ticker", "quarter_end", "piotroski_f"])

    required = {"net_income", "total_assets", "operating_cf",
                "long_term_debt", "gross_profit", "revenue"}
    optional = ["current_assets", "current_liabilities", "shares_outstanding"]

    missing_required = required - existing
    if missing_required:
        logger.warning(
            f"xbrl_facts missing required columns for Piotroski: {sorted(missing_required)} "
            "— returning empty F-Score frame"
        )
        return pd.DataFrame(columns=["ticker", "quarter_end", "piotroski_f"])

    missing_optional = [c for c in optional if c not in existing]
    if missing_optional:
        logger.warning(
            f"xbrl_facts missing optional Piotroski columns: {missing_optional} "
            "— F6/F7 will be 0 until schema is extended and EDGAR re-ingested"
        )

    select_cols = [
        "ticker", "end_date",
        "net_income", "total_assets", "operating_cf",
        "long_term_debt", "gross_profit", "revenue",
    ]
    for col in optional:
        select_cols.append(col if col in existing else f"NULL AS {col}")

    clause, params = _ticker_where(tickers)
    where = f"WHERE {clause}" if clause else ""
    sql = (
        f"SELECT {', '.join(select_cols)} FROM xbrl_facts "
        f"{where} ORDER BY ticker, end_date"
    )

    try:
        with engine.connect() as conn:
            facts = pd.read_sql_query(text(sql), conn, params=params)
    except Exception as e:
        logger.warning(f"Piotroski SQL load failed: {e}")
        return pd.DataFrame(columns=["ticker", "quarter_end", "piotroski_f"])

    if facts.empty:
        logger.warning("xbrl_facts returned no rows for Piotroski computation")
        return pd.DataFrame(columns=["ticker", "quarter_end", "piotroski_f"])

    facts["end_date"] = pd.to_datetime(facts["end_date"])

    # Collapse to one row per (ticker, quarter_end) — last filing within quarter wins,
    # matching the convention used by _to_quarterly().
    raw_inputs = [
        "net_income", "total_assets", "operating_cf", "long_term_debt",
        "gross_profit", "revenue", "current_assets", "current_liabilities",
        "shares_outstanding",
    ]
    qe_facts = _to_quarterly(facts, "end_date", raw_inputs)
    qe_facts = qe_facts.sort_values(["ticker", "quarter_end"]).reset_index(drop=True)

    # ── Derived ratios at quarter T ───────────────────────────────────────
    def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        return num / den.replace(0, np.nan)

    qe_facts["roa"]             = _safe_div(qe_facts["net_income"],     qe_facts["total_assets"])
    qe_facts["leverage"]        = _safe_div(qe_facts["long_term_debt"], qe_facts["total_assets"])
    qe_facts["gross_margin"]    = _safe_div(qe_facts["gross_profit"],   qe_facts["revenue"])
    qe_facts["asset_turnover"]  = _safe_div(qe_facts["revenue"],        qe_facts["total_assets"])
    qe_facts["current_ratio"]   = _safe_div(qe_facts["current_assets"], qe_facts["current_liabilities"])
    qe_facts["cfo_over_assets"] = _safe_div(qe_facts["operating_cf"],   qe_facts["total_assets"])

    # ── YoY (4-quarter) lags per ticker ───────────────────────────────────
    lag_cols = ["roa", "leverage", "gross_margin", "asset_turnover",
                "current_ratio", "shares_outstanding"]
    grouped = qe_facts.groupby("ticker", sort=False)
    for c in lag_cols:
        qe_facts[f"{c}_lag4"] = grouped[c].shift(4)

    # ── 9 binary signals ──────────────────────────────────────────────────
    # NaN comparisons evaluate to False (numpy semantics) → failed signal = 0,
    # matching requirement #4: missing input deterministically fails its signal.
    def _signal(cond: pd.Series) -> pd.Series:
        return cond.fillna(False).astype(int)

    f1 = _signal(qe_facts["roa"] > 0)
    f2 = _signal(qe_facts["operating_cf"] > 0)
    f3 = _signal(qe_facts["roa"] > qe_facts["roa_lag4"])
    f4 = _signal((qe_facts["cfo_over_assets"] - qe_facts["roa"]) < 0)
    f5 = _signal(qe_facts["leverage"] < qe_facts["leverage_lag4"])
    f6 = _signal(qe_facts["current_ratio"] > qe_facts["current_ratio_lag4"])
    f7 = _signal(qe_facts["shares_outstanding"] <= qe_facts["shares_outstanding_lag4"])
    f8 = _signal(qe_facts["gross_margin"] > qe_facts["gross_margin_lag4"])
    f9 = _signal(qe_facts["asset_turnover"] > qe_facts["asset_turnover_lag4"])

    qe_facts["piotroski_f"] = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9

    # ── If every raw input is NULL, surface that as NaN (per requirement #4) ─
    all_null = qe_facts[raw_inputs].isna().all(axis=1)
    qe_facts.loc[all_null, "piotroski_f"] = np.nan

    out = qe_facts[["ticker", "quarter_end", "piotroski_f"]].copy()

    # ── Window filter (applied AFTER YoY so the lag-4 lookback has history) ─
    if start is not None:
        out = out[out["quarter_end"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["quarter_end"] <= pd.Timestamp(end)]

    n_total = len(out)
    n_valid = int(out["piotroski_f"].notna().sum())
    if n_total:
        logger.info(
            f"Piotroski F-Score: {n_valid}/{n_total} rows, "
            f"mean={out['piotroski_f'].mean():.2f}, "
            f"tickers={out['ticker'].nunique()}"
        )
    else:
        logger.warning("Piotroski F-Score: no rows in requested window")

    return out.reset_index(drop=True)


# ── QMJ Quality Factor (Asness, Frazzini & Pedersen 2019) ────────────────────

def _xs_zscore_series(values: pd.Series, group_keys: pd.Series) -> pd.Series:
    """
    Cross-sectional z-score of ``values`` grouped by ``group_keys`` (typically
    quarter_end).  Mirrors `_zscore_cross_section` semantics: winsorises at
    ±ZSCORE_CAP and returns NaN when a group has fewer than
    MIN_OBS_FOR_ZSCORE non-null observations.  Used inside compute_qmj_safety
    and compute_qmj_payout to standardise sub-signals onto a comparable scale
    before averaging — QMJ sub-signals (e.g. leverage in units vs beta unitless)
    cannot be averaged raw without one dominating the composite.
    """
    def _g(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if len(valid) < MIN_OBS_FOR_ZSCORE:
            return pd.Series(np.nan, index=s.index)
        mu, sigma = valid.mean(), valid.std()
        if sigma == 0 or np.isnan(sigma):
            return pd.Series(0.0, index=s.index)
        z = (s - mu) / sigma
        return z.clip(-ZSCORE_CAP, ZSCORE_CAP)

    return values.groupby(group_keys, sort=False).transform(_g)


def _compute_qe_beta(
    stock_tickers: list[str],
    engine,
    window_days: int = 756,   # ~36 months of trading days (252 × 3)
    min_periods: int = 252,   # require at least 12 months before producing a beta
) -> pd.DataFrame:
    """
    Rolling 36-month CAPM beta of each ticker's daily returns vs SPY,
    sampled at quarter-end.

    Returns DataFrame [ticker, quarter_end, beta_36m].  If SPY is not in the
    prices table (universe never loaded the benchmark series), logs a warning
    and returns an empty frame — compute_qmj_safety will then leave the beta
    sub-signal NaN and the pillar will be averaged from the two remaining
    sub-signals.
    """
    universe = sorted(set(stock_tickers) | {"SPY"})
    px = _load_prices_wide(universe, start=None, end=None, engine=engine)
    if px.empty or "SPY" not in px.columns:
        logger.warning("QMJ Safety: SPY prices not in DB — beta sub-signal will be NaN")
        return pd.DataFrame(columns=["ticker", "quarter_end", "beta_36m"])

    rets = px.pct_change()
    spy_rets = rets["SPY"]
    spy_var = spy_rets.rolling(window_days, min_periods=min_periods).var()

    frames = []
    for tk in stock_tickers:
        if tk == "SPY" or tk not in rets.columns:
            continue
        stk_rets = rets[tk]
        rolling_cov = stk_rets.rolling(window_days, min_periods=min_periods).cov(spy_rets)
        beta = rolling_cov / spy_var
        qe_beta = (
            beta.resample("QE").last()
                .rename_axis("quarter_end")
                .to_frame("beta_36m")
                .reset_index()
        )
        qe_beta["ticker"] = tk
        frames.append(qe_beta[["ticker", "quarter_end", "beta_36m"]])

    if not frames:
        return pd.DataFrame(columns=["ticker", "quarter_end", "beta_36m"])
    return pd.concat(frames, ignore_index=True)


def compute_qmj_safety(
    tickers: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    engine=None,
) -> pd.DataFrame:
    """
    QMJ Safety pillar — Asness, Frazzini & Pedersen (2019), "Quality Minus Junk".

    Three sub-signals, each cross-sectionally z-scored per quarter_end and
    sign-inverted so "lower-is-safer" maps to higher score:

      1. Leverage     = long_term_debt / stockholders_equity
      2. Earnings vol = rolling 5-quarter std of ROE (net_income / equity)
      3. Beta         = rolling 36-month CAPM beta vs SPY (daily returns)

    Equally weighted (1/3 each) within the pillar.  Row-wise mean of the
    available z-scored sub-signals (``skipna=True``) — a ticker missing one
    sub-signal is scored on the other two, matching the missing-weight
    redistribution convention already used by
    ``fundamental_scorer.compute_composite_scores``.

    Negative-equity rows are excluded from leverage and ROE (debt-to-equity
    is undefined when E ≤ 0).  If SPY is not in the prices table the beta
    sub-signal is dropped from the average.

    Returns
    -------
    pd.DataFrame [ticker, quarter_end, qmj_safety]
    Already roughly standardised; ``build_feature_matrix``'s outer z-score
    pass is approximately idempotent on it.
    """
    engine = _get_engine(engine)

    clause, params = _ticker_where(tickers)
    where = f"WHERE {clause}" if clause else ""
    sql = (
        "SELECT ticker, end_date, net_income, stockholders_equity, "
        f"long_term_debt FROM xbrl_facts {where} ORDER BY ticker, end_date"
    )
    try:
        with engine.connect() as conn:
            facts = pd.read_sql_query(text(sql), conn, params=params)
    except Exception as e:
        logger.warning(f"QMJ Safety SQL load failed: {e}")
        return pd.DataFrame(columns=["ticker", "quarter_end", "qmj_safety"])

    if facts.empty:
        logger.warning("xbrl_facts returned no rows for QMJ Safety computation")
        return pd.DataFrame(columns=["ticker", "quarter_end", "qmj_safety"])

    facts["end_date"] = pd.to_datetime(facts["end_date"])
    raw_inputs = ["net_income", "stockholders_equity", "long_term_debt"]
    qe_facts = _to_quarterly(facts, "end_date", raw_inputs)
    qe_facts = qe_facts.sort_values(["ticker", "quarter_end"]).reset_index(drop=True)

    # ── Sub-signal 1: Leverage = LT debt / equity (neg equity invalidates) ─
    eq = qe_facts["stockholders_equity"]
    eq_safe = eq.where(eq > 0, np.nan)
    qe_facts["leverage_de"] = qe_facts["long_term_debt"] / eq_safe

    # ── Sub-signal 2: Earnings variability — 5-quarter rolling std of ROE ──
    qe_facts["roe"] = qe_facts["net_income"] / eq_safe
    qe_facts["earnings_vol"] = (
        qe_facts.groupby("ticker", sort=False)["roe"]
                .transform(lambda s: s.rolling(window=5, min_periods=5).std())
    )

    # ── Sub-signal 3: 36-month rolling beta vs SPY ─────────────────────────
    qe_beta = _compute_qe_beta(qe_facts["ticker"].unique().tolist(), engine)
    qe_facts = qe_facts.merge(qe_beta, on=["ticker", "quarter_end"], how="left")

    # ── Z-score each sub-signal per quarter, invert (lower = safer = higher z)
    qe_facts["leverage_z"]     = -_xs_zscore_series(qe_facts["leverage_de"],   qe_facts["quarter_end"])
    qe_facts["earnings_vol_z"] = -_xs_zscore_series(qe_facts["earnings_vol"],  qe_facts["quarter_end"])
    qe_facts["beta_z"]         = -_xs_zscore_series(qe_facts["beta_36m"],      qe_facts["quarter_end"])

    qe_facts["qmj_safety"] = qe_facts[["leverage_z", "earnings_vol_z", "beta_z"]].mean(axis=1, skipna=True)

    out = qe_facts[["ticker", "quarter_end", "qmj_safety"]].copy()
    if start is not None:
        out = out[out["quarter_end"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["quarter_end"] <= pd.Timestamp(end)]

    n_total = len(out)
    n_valid = int(out["qmj_safety"].notna().sum())
    if n_total:
        logger.info(
            f"QMJ Safety: {n_valid}/{n_total} rows, "
            f"mean={out['qmj_safety'].mean():.3f}, "
            f"tickers={out['ticker'].nunique()}"
        )
    else:
        logger.warning("QMJ Safety: no rows in requested window")

    return out.reset_index(drop=True)


def compute_qmj_payout(
    tickers: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    engine=None,
) -> pd.DataFrame:
    """
    QMJ Payout pillar — Asness, Frazzini & Pedersen (2019).

    Two sub-signals, equally weighted (1/2 each), z-scored cross-sectionally
    per quarter:

      1. Anti-dilution    — −((shares_T − shares_{T-4}) / shares_{T-4})
                             so buybacks score positive, dilution negative
                             (requires `shares_outstanding`)
      2. Total payout yld — (dividends_paid + share_repurchases) / total_assets
                             (denominator change vs. classical QMJ: total_assets
                             is a simpler scaling that doesn't require a price
                             merge or shares_outstanding — Session 3 only needs
                             to add dividends_paid + share_repurchases to
                             activate this signal, not shares_outstanding too.)

    Schema gaps (Session-3 deferral)
    -------------------------------
    `shares_outstanding`, `dividends_paid`, and `share_repurchases` are
    recent additions to `xbrl_facts` and are NOT yet present at the time of
    Session 2.  Optional columns are substituted with `NULL AS <col>` so
    this function does not crash on legacy data:

      • Anti-dilution sub-signal activates when shares_outstanding lands.
      • Total payout yield activates when dividends_paid + share_repurchases
        land (independent of shares_outstanding).

    Within-row missing-value policy
    -------------------------------
    When the columns ARE present but a row has NULL dividends or NULL
    repurchases, those values are treated as 0 (assumed no payout that
    quarter).  A NULL `total_assets` invalidates payout_yield for that
    row; a NULL `shares_outstanding` invalidates anti-dilution.

    Returns
    -------
    pd.DataFrame [ticker, quarter_end, qmj_payout]
    """
    engine = _get_engine(engine)

    try:
        with engine.connect() as conn:
            existing = {
                r[1] for r in conn.execute(text("PRAGMA table_info(xbrl_facts)")).fetchall()
            }
    except Exception as e:
        logger.warning(f"Could not introspect xbrl_facts schema for QMJ Payout: {e}")
        return pd.DataFrame(columns=["ticker", "quarter_end", "qmj_payout"])

    required = {"total_assets"}
    optional = ["shares_outstanding", "dividends_paid", "share_repurchases"]

    missing_required = required - existing
    if missing_required:
        logger.warning(
            f"xbrl_facts missing required columns for QMJ Payout: "
            f"{sorted(missing_required)} — returning empty frame"
        )
        return pd.DataFrame(columns=["ticker", "quarter_end", "qmj_payout"])

    missing_optional = [c for c in optional if c not in existing]
    if missing_optional:
        logger.warning(
            f"xbrl_facts missing optional QMJ Payout columns: {missing_optional} "
            "— affected sub-signals will be NaN until Session 3 schema extension"
        )

    select_cols = ["ticker", "end_date", "total_assets"]
    for col in optional:
        select_cols.append(col if col in existing else f"NULL AS {col}")

    clause, params = _ticker_where(tickers)
    where = f"WHERE {clause}" if clause else ""
    sql = (
        f"SELECT {', '.join(select_cols)} FROM xbrl_facts "
        f"{where} ORDER BY ticker, end_date"
    )
    try:
        with engine.connect() as conn:
            facts = pd.read_sql_query(text(sql), conn, params=params)
    except Exception as e:
        logger.warning(f"QMJ Payout SQL load failed: {e}")
        return pd.DataFrame(columns=["ticker", "quarter_end", "qmj_payout"])

    if facts.empty:
        logger.warning("xbrl_facts returned no rows for QMJ Payout computation")
        return pd.DataFrame(columns=["ticker", "quarter_end", "qmj_payout"])

    facts["end_date"] = pd.to_datetime(facts["end_date"])
    raw_inputs = ["total_assets", "shares_outstanding",
                  "dividends_paid", "share_repurchases"]
    qe_facts = _to_quarterly(facts, "end_date", raw_inputs)
    qe_facts = qe_facts.sort_values(["ticker", "quarter_end"]).reset_index(drop=True)

    # ── Sub-signal 1: anti-dilution = -(ΔShares_YoY / Shares_lag4) ────────
    qe_facts["shares_lag4"] = qe_facts.groupby("ticker", sort=False)["shares_outstanding"].shift(4)
    qe_facts["dilution_yoy"] = (
        (qe_facts["shares_outstanding"] - qe_facts["shares_lag4"])
        / qe_facts["shares_lag4"].replace(0, np.nan)
    )
    qe_facts["anti_dilution"] = -qe_facts["dilution_yoy"]

    # ── Sub-signal 2: total payout yield = (div + repo) / total_assets ────
    # Treat NULL dividends/repurchases as 0 *only* when the column is in the
    # schema (then NULL means "no payout").  When the column itself is missing
    # (NULL AS), the values stay NaN and propagate so payout_yield is NaN
    # rather than a spurious 0 / total_assets.
    if "dividends_paid" in existing:
        qe_facts["dividends_paid"] = qe_facts["dividends_paid"].fillna(0)
    if "share_repurchases" in existing:
        qe_facts["share_repurchases"] = qe_facts["share_repurchases"].fillna(0)
    qe_facts["gross_payout"] = qe_facts["dividends_paid"] + qe_facts["share_repurchases"]

    qe_facts["payout_yield"] = qe_facts["gross_payout"] / qe_facts["total_assets"].replace(0, np.nan)

    # ── Z-score sub-signals cross-sectionally per quarter ─────────────────
    z_anti  = _xs_zscore_series(qe_facts["anti_dilution"], qe_facts["quarter_end"])
    z_yield = _xs_zscore_series(qe_facts["payout_yield"],  qe_facts["quarter_end"])

    qe_facts["qmj_payout"] = pd.concat([z_anti, z_yield], axis=1).mean(axis=1, skipna=True)

    out = qe_facts[["ticker", "quarter_end", "qmj_payout"]].copy()
    if start is not None:
        out = out[out["quarter_end"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["quarter_end"] <= pd.Timestamp(end)]

    n_total = len(out)
    n_valid = int(out["qmj_payout"].notna().sum())
    if n_total:
        logger.info(
            f"QMJ Payout: {n_valid}/{n_total} rows, "
            f"mean={out['qmj_payout'].mean():.3f}, "
            f"tickers={out['ticker'].nunique()}"
        )
    else:
        logger.warning("QMJ Payout: no rows in requested window")

    return out.reset_index(drop=True)


# ── cross-sectional z-scoring ─────────────────────────────────────────────────

def _zscore_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each feature column cross-sectionally (across tickers) for each
    quarter_end date.  Caps at ±ZSCORE_CAP.

    Input:  DataFrame indexed by (ticker, quarter_end) with feature columns.
    Output: same shape, z-scored values (or NaN where < MIN_OBS_FOR_ZSCORE).
    """
    result = df.copy()
    for col in FEATURE_COLS:
        if col not in result.columns:
            result[col] = np.nan
            continue

        def _zscore_group(s: pd.Series) -> pd.Series:
            valid = s.dropna()
            if len(valid) < MIN_OBS_FOR_ZSCORE:
                return pd.Series(np.nan, index=s.index)
            mu, sigma = valid.mean(), valid.std()
            if sigma == 0 or np.isnan(sigma):
                return pd.Series(0.0, index=s.index)
            z = (s - mu) / sigma
            return z.clip(-ZSCORE_CAP, ZSCORE_CAP)

        result[col] = result.groupby("quarter_end")[col].transform(_zscore_group)

    return result


# ── missing data diagnostics ─────────────────────────────────────────────────

def _log_missing_rates(df: pd.DataFrame) -> None:
    """Log the fraction of NaN per feature per quarter_end."""
    if df.empty:
        return
    for qe, grp in df.groupby("quarter_end"):
        n = len(grp)
        missing = {col: grp[col].isna().sum() / n for col in FEATURE_COLS if col in grp.columns}
        high = {k: f"{v:.0%}" for k, v in missing.items() if v > 0.3}
        if high:
            logger.warning(f"  {qe.date()}  high missing rates: {high}")


# ── public API ────────────────────────────────────────────────────────────────

def build_feature_matrix(
    tickers: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    engine=None,
) -> pd.DataFrame:
    """
    Build the z-scored fundamental feature matrix.

    Returns
    -------
    pd.DataFrame with columns: ticker, quarter_end, + FEATURE_COLS
    All feature columns are cross-sectionally z-scored per quarter_end.
    MultiIndex columns version: pivot with .set_index(["quarter_end","ticker"])
    """
    engine = _get_engine(engine)
    start = start or TIMELINE["train_start"]
    end = end or TIMELINE["test_end"]

    logger.info(f"Building feature matrix  [{start} → {end}]")

    # 1. Load all source tables
    derived = _load_derived(tickers, engine)
    facts   = _load_facts(tickers, engine)
    sue     = _load_sue(tickers, engine)
    revs    = _load_eps_revisions(tickers, engine)
    si      = _load_short_interest(tickers, engine)
    sent    = _load_sentiment(tickers, engine)
    prices  = _load_prices_wide(tickers, start, end, engine)

    # 2. Align derived features to quarterly calendar
    qe_derived = _to_quarterly(
        derived, "end_date",
        ["gross_profitability", "revenue_yoy", "revenue_acceleration",
         "deferred_revenue_yoy", "rd_intensity"],
    ) if not derived.empty else pd.DataFrame()

    # 3. FCF yield (with EV approximation)
    qe_fcf = _compute_fcf_yield(facts, prices)

    # 4. SUE
    qe_sue = _to_quarterly(sue, "fiscal_period", ["sue_score"]) if not sue.empty else pd.DataFrame()

    # 5. EPS revisions (quarterly last obs)
    qe_rev = _to_quarterly(revs, "date", ["revision_1m", "revision_3m"]) if not revs.empty else pd.DataFrame()
    if not qe_rev.empty:
        qe_rev = qe_rev.rename(columns={"revision_1m": "eps_revision_1m", "revision_3m": "eps_revision_3m"})

    # 6. Short interest → forward-fill daily then take quarter-end snapshot
    qe_si = pd.DataFrame()
    if not si.empty:
        si_pivot = si.pivot_table(index="date", columns="ticker", values="days_to_cover")
        si_pivot = si_pivot.resample("D").last().ffill()
        si_qe = si_pivot.resample("QE").last().stack().rename("short_interest_dtc").reset_index()
        si_qe.columns = ["quarter_end", "ticker", "short_interest_dtc"]
        qe_si = si_qe

    # 7. Sentiment → forward-fill then quarter-end snapshot
    qe_sent = pd.DataFrame()
    if not sent.empty:
        se_pivot = sent.pivot_table(index="filing_date", columns="ticker", values="finbert_score")
        se_pivot = se_pivot.resample("D").last().ffill()
        se_qe = se_pivot.resample("QE").last().stack().rename("finbert_score").reset_index()
        se_qe.columns = ["quarter_end", "ticker", "finbert_score"]
        qe_sent = se_qe

    # 7b. Piotroski F-Score (already aligned to quarter_end internally)
    qe_pio = compute_piotroski_f(
        tickers=tickers, start=start, end=end, engine=engine,
    )

    # 7c. QMJ Safety pillar (Asness, Frazzini & Pedersen 2019)
    qe_qmj_safety = compute_qmj_safety(
        tickers=tickers, start=start, end=end, engine=engine,
    )

    # 7d. QMJ Payout pillar
    qe_qmj_payout = compute_qmj_payout(
        tickers=tickers, start=start, end=end, engine=engine,
    )

    # 8. Merge everything on (ticker, quarter_end)
    base_qe = _quarter_end_index(start, end)
    all_tickers = set()
    for frame in [qe_derived, qe_fcf, qe_sue, qe_rev, qe_si, qe_sent,
                  qe_pio, qe_qmj_safety, qe_qmj_payout]:
        if isinstance(frame, pd.DataFrame) and not frame.empty and "ticker" in frame.columns:
            all_tickers.update(frame["ticker"].unique())
    if tickers:
        all_tickers = all_tickers.intersection(set(t.upper() for t in tickers)) or set(tickers)

    if not all_tickers:
        logger.warning("No tickers found across any source table")
        return pd.DataFrame(columns=["ticker", "quarter_end"] + FEATURE_COLS)

    # Scaffold: all (ticker, quarter_end) combinations
    scaffold = pd.MultiIndex.from_product(
        [sorted(all_tickers), base_qe], names=["ticker", "quarter_end"]
    ).to_frame(index=False)

    def _merge(base, right, on_cols=("ticker", "quarter_end")):
        if right is None or (isinstance(right, pd.DataFrame) and right.empty):
            return base
        return base.merge(right, on=list(on_cols), how="left")

    matrix = scaffold.copy()
    matrix = _merge(matrix, qe_derived)
    matrix = _merge(matrix, qe_fcf)
    matrix = _merge(matrix, qe_sue)
    matrix = _merge(matrix, qe_rev)
    matrix = _merge(matrix, qe_si)
    matrix = _merge(matrix, qe_sent)
    matrix = _merge(matrix, qe_pio)
    matrix = _merge(matrix, qe_qmj_safety)
    matrix = _merge(matrix, qe_qmj_payout)

    # 9. Missing-data handling
    #  a. Forward-fill up to 2 quarters per ticker for fundamental cols
    fund_cols = [c for c in FEATURE_COLS if c in matrix.columns]
    matrix = matrix.sort_values(["ticker", "quarter_end"])
    matrix[fund_cols] = (
        matrix.groupby("ticker")[fund_cols]
        .apply(lambda g: g.ffill(limit=MAX_FFILL_QUARTERS))
        .reset_index(drop=True)
    )

    #  b. Log missing rates before filling with 0
    _log_missing_rates(matrix)

    #  c. Fill remaining NaN with 0.0 (cross-sectional neutral)
    for col in FEATURE_COLS:
        if col not in matrix.columns:
            matrix[col] = 0.0
    matrix[FEATURE_COLS] = matrix[FEATURE_COLS].fillna(0.0)

    # 10. Cross-sectional z-score
    matrix = _zscore_cross_section(matrix)

    logger.info(
        f"Feature matrix built: {len(matrix)} rows, "
        f"{matrix['ticker'].nunique()} tickers, "
        f"{matrix['quarter_end'].nunique()} quarters"
    )

    return matrix[["ticker", "quarter_end"] + FEATURE_COLS]


def get_feature_snapshot(
    date: str,
    tickers: Optional[list[str]] = None,
    engine=None,
) -> pd.DataFrame:
    """
    Return a single-quarter feature snapshot (tickers × features).

    Finds the latest quarter_end on or before *date*, then returns the
    z-scored feature row for every ticker.

    Parameters
    ----------
    date    : reference date string (YYYY-MM-DD)
    tickers : optional ticker filter
    engine  : SQLAlchemy engine

    Returns
    -------
    pd.DataFrame indexed by ticker with FEATURE_COLS columns
    """
    engine = _get_engine(engine)
    ref = pd.Timestamp(date)
    # Build a small window so we don't load the entire history
    lookback_start = (ref - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    matrix = build_feature_matrix(
        tickers=tickers, start=lookback_start, end=date, engine=engine
    )
    if matrix.empty:
        return pd.DataFrame(columns=FEATURE_COLS)

    # Pick the latest quarter_end <= date
    latest_qe = matrix.loc[matrix["quarter_end"] <= ref, "quarter_end"].max()
    if pd.isna(latest_qe):
        return pd.DataFrame(columns=FEATURE_COLS)

    snap = matrix[matrix["quarter_end"] == latest_qe].set_index("ticker")[FEATURE_COLS]
    logger.info(f"Snapshot for {latest_qe.date()}: {len(snap)} tickers")
    return snap


# ── CLI (convenience) ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Build XBRL feature matrix")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--snapshot", default=None,
                        help="Return a single-date snapshot instead of full matrix")
    args = parser.parse_args()

    if args.snapshot:
        snap = get_feature_snapshot(args.snapshot, tickers=args.tickers)
        print(snap.to_string())
    else:
        mat = build_feature_matrix(tickers=args.tickers, start=args.start, end=args.end)
        print(mat.describe().to_string())
        print(f"\nShape: {mat.shape}")
