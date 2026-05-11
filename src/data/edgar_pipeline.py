"""
SEC EDGAR XBRL Pipeline
=======================
Downloads 10-K and 10-Q XBRL financial facts for universe tickers via the
SEC EDGAR Company Facts API (free, no key required).

XBRL tags extracted
-------------------
Income Statement:
  Revenues / RevenueFromContractWithCustomerExcludingAssessedTax
  GrossProfit
  ResearchAndDevelopmentExpense
  OperatingIncomeLoss
  NetIncomeLoss
  EarningsPerShareDiluted

Balance Sheet:
  Assets (TotalAssets)
  Liabilities (TotalLiabilities)
  StockholdersEquity
  DeferredRevenue / DeferredRevenueNoncurrent
  CashAndCashEquivalentsAtCarryingValue

Cash Flow:
  NetCashProvidedByUsedInOperatingActivities
  CapitalExpenditureDiscontinuedOperation / PaymentsToAcquirePropertyPlantAndEquipment

Derived / computed:
  FreeCashFlow = OperatingCF - CapEx
  EnterpriseValue = MarketCap + TotalDebt - Cash  (joined from price data)
  GrossProfitability = GrossProfit / TotalAssets   (Novy-Marx)
  FCFYield = FreeCashFlow / EnterpriseValue
  RevenueAcceleration = Δ(YoY revenue growth), QoQ
  DeferredRevenueGrowth = YoY growth in DeferredRevenue
  RDIntensity = R&D / Revenue

Storage: SQLite table `xbrl_facts` and `xbrl_derived`

Usage:
  python -m src.data.edgar_pipeline                      # full universe
  python -m src.data.edgar_pipeline --tickers AAPL MSFT  # specific tickers
  python -m src.data.edgar_pipeline --derive-only        # recompute derived features
"""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config" / "settings.yaml") as f:
    CFG = yaml.safe_load(f)

DB_PATH = ROOT / CFG["data"]["paths"]["db"]
DB_URL = f"sqlite:///{DB_PATH}"
SEC_HEADERS = {
    "User-Agent": os.getenv("SEC_USER_AGENT", "QuantResearch research@example.com"),
    "Accept-Encoding": "gzip, deflate",
}

# ── XBRL tag groups ───────────────────────────────────────────────────────────
# Multiple possible tags per concept (SEC filers use different tags over time)
XBRL_TAGS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "gross_profit": ["GrossProfit"],
    "rd_expense": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "eps_diluted": [
        "EarningsPerShareDiluted",
        "IncomeLossFromContinuingOperationsPerDilutedShare",
    ],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "deferred_revenue": [
        "DeferredRevenue",
        "DeferredRevenueNoncurrent",
        "ContractWithCustomerLiability",
        "ContractWithCustomerLiabilityCurrent",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
    ],
    "long_term_debt": [
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "DebtAndCapitalLeaseObligations",
    ],
    "operating_cf": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpendituresIncurredButNotYetPaid",
        "PaymentsForProceedsFromProductiveAssets",
    ],
}


# ── SEC EDGAR API helpers ─────────────────────────────────────────────────────

def get_cik_map() -> dict[str, str]:
    """Download the full SEC ticker→CIK mapping (cached in data/universe/)."""
    cache_path = ROOT / "data" / "universe" / "cik_map.json"
    if cache_path.exists():
        import json
        with open(cache_path) as f:
            return json.load(f)

    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    mapping = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in data.values()}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(cache_path, "w") as f:
        json.dump(mapping, f)

    logger.info(f"CIK map downloaded: {len(mapping)} tickers")
    return mapping


def fetch_company_facts(cik: str) -> dict:
    """Fetch full XBRL company facts JSON from SEC EDGAR."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def extract_tag_series(facts: dict, tags: list[str]) -> pd.DataFrame:
    """
    Extract a time series for a list of possible XBRL tags.
    Returns the first tag found with data, preferring quarterly (10-Q) filings.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    for tag in tags:
        if tag not in us_gaap:
            continue

        units = us_gaap[tag].get("units", {})
        # Prefer USD units; some tags use shares or pure numbers
        unit_data = units.get("USD") or units.get("shares") or next(iter(units.values()), [])

        rows = []
        for entry in unit_data:
            form = entry.get("form", "")
            if form not in ("10-K", "10-Q"):
                continue
            rows.append({
                "end":   entry.get("end"),
                "val":   entry.get("val"),
                "form":  form,
                "frame": entry.get("frame", ""),
            })

        if rows:
            df = pd.DataFrame(rows)
            df["end"] = pd.to_datetime(df["end"])
            df = df.sort_values("end").drop_duplicates(subset=["end", "form"], keep="last")
            return df

    return pd.DataFrame()


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_ticker_facts(ticker: str, cik: str) -> pd.DataFrame:
    """
    Pull all XBRL tags for a single ticker and return a wide quarterly DataFrame.
    Columns: end_date, form, revenue, gross_profit, rd_expense, ..., ticker
    """
    try:
        facts = fetch_company_facts(cik)
    except Exception as e:
        logger.warning(f"[{ticker}] Failed to fetch facts: {e}")
        return pd.DataFrame()

    series_dict = {}
    for concept, tags in XBRL_TAGS.items():
        s = extract_tag_series(facts, tags)
        if not s.empty:
            series_dict[concept] = s.set_index("end")["val"]

    if not series_dict:
        logger.warning(f"[{ticker}] No XBRL data extracted")
        return pd.DataFrame()

    combined = pd.DataFrame(series_dict)
    combined.index.name = "end_date"
    combined["ticker"] = ticker
    combined = combined.reset_index()

    logger.info(f"  [{ticker}] {len(combined)} quarterly rows, "
                f"{combined.dropna(how='all', subset=list(series_dict.keys())).shape[0]} non-null")
    return combined


# ── Database ──────────────────────────────────────────────────────────────────

def get_engine():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(DB_URL, echo=False)
    _create_tables(engine)
    return engine


def _create_tables(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS xbrl_facts (
                ticker          TEXT NOT NULL,
                end_date        TEXT NOT NULL,
                revenue         REAL,
                gross_profit    REAL,
                rd_expense      REAL,
                operating_income REAL,
                net_income      REAL,
                eps_diluted     REAL,
                total_assets    REAL,
                total_liabilities REAL,
                stockholders_equity REAL,
                deferred_revenue REAL,
                cash            REAL,
                long_term_debt  REAL,
                operating_cf    REAL,
                capex           REAL,
                PRIMARY KEY (ticker, end_date)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS xbrl_derived (
                ticker                  TEXT NOT NULL,
                end_date                TEXT NOT NULL,
                free_cash_flow          REAL,
                enterprise_value        REAL,
                gross_profitability     REAL,
                fcf_yield               REAL,
                revenue_yoy             REAL,
                revenue_acceleration    REAL,
                deferred_revenue_yoy    REAL,
                rd_intensity            REAL,
                PRIMARY KEY (ticker, end_date)
            )
        """))


def upsert_facts(df: pd.DataFrame, engine) -> None:
    cols = [
        "ticker", "end_date", "revenue", "gross_profit", "rd_expense",
        "operating_income", "net_income", "eps_diluted", "total_assets",
        "total_liabilities", "stockholders_equity", "deferred_revenue",
        "cash", "long_term_debt", "operating_cf", "capex",
    ]
    df = df.rename(columns={"end_date": "end_date"})
    for col in cols:
        if col not in df.columns:
            df[col] = None

    with engine.begin() as conn:
        for _, row in df[cols].iterrows():
            conn.execute(text("""
                INSERT OR REPLACE INTO xbrl_facts
                    (ticker, end_date, revenue, gross_profit, rd_expense,
                     operating_income, net_income, eps_diluted, total_assets,
                     total_liabilities, stockholders_equity, deferred_revenue,
                     cash, long_term_debt, operating_cf, capex)
                VALUES
                    (:ticker, :end_date, :revenue, :gross_profit, :rd_expense,
                     :operating_income, :net_income, :eps_diluted, :total_assets,
                     :total_liabilities, :stockholders_equity, :deferred_revenue,
                     :cash, :long_term_debt, :operating_cf, :capex)
            """), row.to_dict())


def load_xbrl_facts(
    tickers: Optional[list[str]] = None,
    engine=None,
) -> pd.DataFrame:
    """Load raw XBRL facts from DB."""
    if engine is None:
        engine = get_engine()

    where = ""
    params: dict = {}
    if tickers:
        ph = ",".join(f":t{i}" for i in range(len(tickers)))
        where = f"WHERE ticker IN ({ph})"
        params = {f"t{i}": t for i, t in enumerate(tickers)}

    sql = f"SELECT * FROM xbrl_facts {where} ORDER BY ticker, end_date"
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, params=params)
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df


# ── Derived feature computation ───────────────────────────────────────────────

def compute_derived_features(
    facts: pd.DataFrame,
    prices: Optional[pd.DataFrame] = None,
    engine=None,
) -> pd.DataFrame:
    """
    Compute all derived fundamental features from raw XBRL facts.
    prices: wide DataFrame (date × ticker) of market cap or adj_close — used for EV.
    """
    if engine is None:
        engine = get_engine()

    derived_all = []

    for ticker, grp in facts.groupby("ticker"):
        grp = grp.sort_values("end_date").copy()

        # Free Cash Flow
        grp["free_cash_flow"] = grp["operating_cf"] - grp["capex"].abs()

        # Gross Profitability (Novy-Marx): GP / Total Assets
        grp["gross_profitability"] = grp["gross_profit"] / grp["total_assets"]

        # Revenue YoY growth
        grp["revenue_yoy"] = grp["revenue"].pct_change(4)  # 4 quarters back

        # Revenue Acceleration: change in YoY growth rate, QoQ
        grp["revenue_acceleration"] = grp["revenue_yoy"].diff(1)

        # Deferred Revenue YoY growth
        grp["deferred_revenue_yoy"] = grp["deferred_revenue"].pct_change(4)

        # R&D Intensity: R&D / Revenue
        grp["rd_intensity"] = grp["rd_expense"] / grp["revenue"]

        # Enterprise Value & FCF Yield (requires market data)
        # EV = Market Cap + Long-Term Debt - Cash
        # For now, leave as NaN until price data is joined in scorer
        grp["enterprise_value"] = None
        grp["fcf_yield"] = None

        derived_cols = [
            "ticker", "end_date", "free_cash_flow", "enterprise_value",
            "gross_profitability", "fcf_yield", "revenue_yoy",
            "revenue_acceleration", "deferred_revenue_yoy", "rd_intensity",
        ]
        derived_all.append(grp[derived_cols])

    if not derived_all:
        return pd.DataFrame()

    derived = pd.concat(derived_all, ignore_index=True)

    # Upsert to DB
    with engine.begin() as conn:
        for _, row in derived.iterrows():
            conn.execute(text("""
                INSERT OR REPLACE INTO xbrl_derived
                    (ticker, end_date, free_cash_flow, enterprise_value,
                     gross_profitability, fcf_yield, revenue_yoy,
                     revenue_acceleration, deferred_revenue_yoy, rd_intensity)
                VALUES
                    (:ticker, :end_date, :free_cash_flow, :enterprise_value,
                     :gross_profitability, :fcf_yield, :revenue_yoy,
                     :revenue_acceleration, :deferred_revenue_yoy, :rd_intensity)
            """), {k: (None if pd.isna(v) else v) for k, v in row.items()})

    logger.info(f"Derived features computed and saved: {len(derived)} rows")
    return derived


# ── Pipeline orchestrator ─────────────────────────────────────────────────────

def run_edgar_pipeline(
    tickers: Optional[list[str]] = None,
    derive_only: bool = False,
) -> None:
    """
    Full EDGAR pipeline:
      1. Resolve tickers → CIKs
      2. Download XBRL company facts from SEC
      3. Extract and upsert raw facts to DB
      4. Compute and upsert derived features
    """
    from src.universe.screener import load_universe

    engine = get_engine()

    if tickers is None:
        uni_path = ROOT / "data" / "universe" / "universe.csv"
        uni = load_universe(uni_path)
        tickers = uni["ticker"].tolist()

    if not derive_only:
        cik_map = get_cik_map()
        logger.info(f"Downloading XBRL facts for {len(tickers)} tickers…")

        all_facts = []
        for i, ticker in enumerate(tickers):
            cik = cik_map.get(ticker.upper())
            if not cik:
                logger.warning(f"  [{ticker}] CIK not found, skipping")
                continue

            df = extract_ticker_facts(ticker, cik)
            if not df.empty:
                upsert_facts(df, engine)
                all_facts.append(df)

            if (i + 1) % 10 == 0:
                logger.info(f"  {i+1}/{len(tickers)} tickers done")
            time.sleep(0.15)   # SEC rate limit: max ~10 req/s

        logger.info(f"XBRL facts downloaded for {len(all_facts)} tickers")

    # Compute derived features
    logger.info("Computing derived features…")
    facts = load_xbrl_facts(tickers=tickers, engine=engine)
    if facts.empty:
        logger.warning("No XBRL facts in DB — run without --derive-only first")
        return

    compute_derived_features(facts, engine=engine)
    logger.info("EDGAR pipeline complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="SEC EDGAR XBRL pipeline")
    parser.add_argument("--tickers", nargs="+", help="Override universe")
    parser.add_argument("--derive-only", action="store_true",
                        help="Skip download; only recompute derived features from existing DB data")
    args = parser.parse_args()

    run_edgar_pipeline(tickers=args.tickers, derive_only=args.derive_only)
