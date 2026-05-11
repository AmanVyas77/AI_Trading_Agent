"""
Universe Screener
=================
Filters the investable tech universe to ~80 US-listed stocks matching the
project spec:

  - S&P 500 Tech + Semiconductors + Enterprise Software + Mid-cap Growth
  - Market cap >= $2B
  - Avg daily volume >= $5M (USD)
  - >= 8 quarters of XBRL filing history on SEC EDGAR
  - NYSE / NASDAQ only, USD reporting
  - No IPOs before 2017 in training data

Output: data/universe/universe.csv  (ticker, name, exchange, sector, market_cap, ipo_date)
"""

from __future__ import annotations

import os
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
import requests
import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = ROOT / "config" / "settings.yaml"

with open(SETTINGS_PATH) as f:
    CFG = yaml.safe_load(f)

UNI_CFG = CFG["universe"]
TIMELINE = CFG["timeline"]
OUTPUT_DIR = ROOT / "data" / "universe"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Seed ticker list ──────────────────────────────────────────────────────────
# Hand-curated starting pool; screener will prune to spec.
# Covers: S&P 500 Tech, SOX components, enterprise software, mid-cap growth.
SEED_TICKERS = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AVGO", "ORCL", "CRM", "AMD",
    # Semiconductors / SOX
    "QCOM", "TXN", "INTC", "MU", "LRCX", "AMAT", "KLAC", "MRVL", "ON", "SWKS",
    "MPWR", "WOLF", "ENTG", "ACLS", "COHU", "MKSI", "ONTO", "FORM", "ICHR", "CAMT",
    # Enterprise Software
    "ADBE", "NOW", "INTU", "WDAY", "PANW", "CRWD", "ZS", "FTNT", "SNOW", "DDOG",
    "MDB", "HUBS", "VEEV", "CDNS", "SNPS", "ANSS", "PTC", "MANH", "PAYC", "PCTY",
    # Internet / Cloud
    "AMZN", "NFLX", "UBER", "LYFT", "ABNB", "SHOP", "TWLO", "ZM", "OKTA", "CFLT",
    # IT Services / Infrastructure
    "ACN", "IBM", "ANET", "CSCO", "HPE", "DELL", "STX", "WDC", "NTAP",
    # Mid-cap growth tech
    "APP", "GTLB", "PATH", "SMAR", "DOCN", "FSLY", "NET", "ESTC", "APPF", "ASAN",
    "TOST", "DXCM", "SAMSF", "IOT", "TASK", "AI",
]

# ── SEC EDGAR XBRL Check ──────────────────────────────────────────────────────

SEC_HEADERS = {
    "User-Agent": os.getenv("SEC_USER_AGENT", "QuantResearch research@example.com"),
    "Accept-Encoding": "gzip, deflate",
}
SEC_CIK_URL = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2010-01-01&enddt=2024-12-31&forms=10-K"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"


def get_cik(ticker: str) -> Optional[int]:
    """Resolve ticker → SEC CIK via EDGAR company search."""
    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=10-K"
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=10)
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])
        if hits:
            return int(hits[0]["_source"]["entity_id"])
    except Exception as e:
        logger.warning(f"CIK lookup failed for {ticker}: {e}")
    return None


def count_xbrl_quarters(cik: int) -> int:
    """
    Count quarterly XBRL filings (10-Q + 10-K) for a given CIK.
    Each 10-K ≈ 4 quarters; each 10-Q = 1 quarter.
    Returns an estimate of total XBRL quarters available.
    """
    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        n_10k = sum(1 for f in forms if f == "10-K")
        n_10q = sum(1 for f in forms if f == "10-Q")
        return n_10q + (n_10k * 4)
    except Exception as e:
        logger.warning(f"XBRL count failed for CIK {cik}: {e}")
    return 0


# ── Market Data Screen ────────────────────────────────────────────────────────

def fetch_market_data(tickers: list[str]) -> pd.DataFrame:
    """
    Pull market cap, avg daily volume, exchange, IPO date, currency via yfinance.
    Returns a DataFrame indexed by ticker.
    """
    records = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            records.append({
                "ticker": ticker,
                "name": info.get("longName", ""),
                "exchange": info.get("exchange", ""),
                "currency": info.get("currency", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0) or 0,
                "avg_volume": info.get("averageVolume", 0) or 0,
                "avg_price_approx": info.get("regularMarketPrice", 0) or 0,
                "ipo_date": info.get("ipoExpectedDate") or _infer_ipo_date(ticker),
            })
            time.sleep(0.3)   # gentle rate limit
        except Exception as e:
            logger.warning(f"yfinance fetch failed for {ticker}: {e}")
    return pd.DataFrame(records).set_index("ticker")


def _infer_ipo_date(ticker: str) -> Optional[str]:
    """Fall back: look at oldest available price date as IPO proxy."""
    try:
        hist = yf.Ticker(ticker).history(period="max", auto_adjust=True)
        if not hist.empty:
            return hist.index[0].strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


# ── Main Screener ─────────────────────────────────────────────────────────────

def run_screener(
    seed_tickers: list[str] = SEED_TICKERS,
    check_xbrl: bool = True,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run the full universe screen.

    Parameters
    ----------
    seed_tickers : list of ticker strings to start from
    check_xbrl   : if True, hit SEC EDGAR to verify XBRL filing history
    output_path  : save CSV here; defaults to data/universe/universe.csv

    Returns
    -------
    pd.DataFrame with passing tickers and metadata
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "universe.csv"

    logger.info(f"Starting universe screen on {len(seed_tickers)} seed tickers…")

    # Step 1 — market data
    df = fetch_market_data(seed_tickers)
    logger.info(f"Fetched market data for {len(df)} tickers")

    # Step 2 — filter: exchange
    valid_exchanges = {"NYQ", "NMS", "NGM", "NCM", "NYSE", "NASDAQ"}
    df = df[df["exchange"].isin(valid_exchanges)]
    logger.info(f"After exchange filter: {len(df)}")

    # Step 3 — filter: currency
    df = df[df["currency"] == "USD"]
    logger.info(f"After currency filter: {len(df)}")

    # Step 4 — filter: market cap >= $2B
    min_mcap = UNI_CFG["min_market_cap_b"] * 1e9
    df = df[df["market_cap"] >= min_mcap]
    logger.info(f"After market cap filter (>=${UNI_CFG['min_market_cap_b']}B): {len(df)}")

    # Step 5 — filter: avg daily volume >= $5M
    # Approximate dollar volume = avg_volume * price
    df["approx_dollar_volume"] = df["avg_volume"] * df["avg_price_approx"]
    min_vol = UNI_CFG["min_avg_daily_volume_m"] * 1e6
    df = df[df["approx_dollar_volume"] >= min_vol]
    logger.info(f"After volume filter (>=${UNI_CFG['min_avg_daily_volume_m']}M/day): {len(df)}")

    # Step 6 — filter: IPO cutoff (no IPOs before 2017 in training data)
    ipo_cutoff = pd.Timestamp(UNI_CFG["ipo_cutoff"])
    df["ipo_date"] = pd.to_datetime(df["ipo_date"], errors="coerce")
    df = df[df["ipo_date"].isna() | (df["ipo_date"] <= ipo_cutoff)]
    logger.info(f"After IPO cutoff filter: {len(df)}")

    # Step 7 — XBRL check (optional — slow, hits SEC API)
    if check_xbrl:
        logger.info("Checking XBRL filing history on SEC EDGAR…")
        xbrl_counts = {}
        for ticker in df.index:
            cik = get_cik(ticker)
            if cik:
                count = count_xbrl_quarters(cik)
                xbrl_counts[ticker] = count
            else:
                xbrl_counts[ticker] = 0
            time.sleep(0.5)  # SEC rate limit: 10 req/s max
        df["xbrl_quarters"] = pd.Series(xbrl_counts)
        df = df[df["xbrl_quarters"] >= UNI_CFG["min_xbrl_quarters"]]
        logger.info(f"After XBRL filter (>={UNI_CFG['min_xbrl_quarters']} qtrs): {len(df)}")
    else:
        df["xbrl_quarters"] = None
        logger.info("Skipping XBRL check (check_xbrl=False)")

    # Step 8 — save
    df_out = df.reset_index()[
        ["ticker", "name", "exchange", "sector", "industry",
         "market_cap", "approx_dollar_volume", "ipo_date", "xbrl_quarters"]
    ].sort_values("market_cap", ascending=False)

    df_out.to_csv(output_path, index=False)
    logger.info(f"Universe saved → {output_path}  ({len(df_out)} tickers)")
    return df_out


def load_universe(path: Optional[Path] = None) -> pd.DataFrame:
    """Load saved universe CSV."""
    if path is None:
        path = OUTPUT_DIR / "universe.csv"
    return pd.read_csv(path, parse_dates=["ipo_date"])


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Run universe screener")
    parser.add_argument("--no-xbrl", action="store_true",
                        help="Skip SEC EDGAR XBRL check (faster, for testing)")
    args = parser.parse_args()

    universe = run_screener(check_xbrl=not args.no_xbrl)
    print(universe.to_string(index=False))
    print(f"\n✓ {len(universe)} tickers passed all filters")
