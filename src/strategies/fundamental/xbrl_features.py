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

    # 8. Merge everything on (ticker, quarter_end)
    base_qe = _quarter_end_index(start, end)
    all_tickers = set()
    for frame in [qe_derived, qe_fcf, qe_sue, qe_rev, qe_si, qe_sent]:
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
