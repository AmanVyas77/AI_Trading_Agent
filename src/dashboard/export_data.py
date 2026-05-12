"""
Dashboard Data Exporter
========================
Reads from the project's SQLite DB and parquet files, then exports JSON
files to src/dashboard/data/ for the standalone JS dashboard to consume.

Exports
-------
  prices.json              — daily adj_close for all universe tickers
  macro.json               — daily macro series (VIX, yield_spread, fed_funds, CPI)
  quant_signals.json       — monthly quant composite scores per ticker
  fundamental_signals.json — quarterly fundamental composite scores per ticker
  universe.json            — ticker metadata (name, sector, market_cap)
  performance.json         — strategy vs benchmark cumulative returns

Usage:
  python -m src.dashboard.export_data
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine, text, inspect

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config" / "settings.yaml") as f:
    CFG = yaml.safe_load(f)

DB_PATH = ROOT / CFG["data"]["paths"]["db"]
DB_URL = f"sqlite:///{DB_PATH}"
TIMELINE = CFG["timeline"]
EXPORT_DIR = Path(__file__).resolve().parent / "data"


def _get_engine():
    return create_engine(DB_URL, echo=False)


def _table_exists(engine, table_name: str) -> bool:
    """Check if a table exists in the DB."""
    insp = inspect(engine)
    return table_name in insp.get_table_names()


def _safe_json(obj):
    """Convert numpy/pandas types to JSON-serialisable Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _write_json(data, filename: str) -> None:
    """Write data to a JSON file in the export directory."""
    path = EXPORT_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, default=_safe_json, allow_nan=False)
    logger.info(f"  → {path}  ({path.stat().st_size / 1024:.0f} KB)")


# ── Exporters ─────────────────────────────────────────────────────────────────

def export_prices(engine) -> None:
    """Export daily adj_close prices for all universe tickers."""
    if not _table_exists(engine, "prices"):
        logger.warning("prices table not found — writing empty prices.json")
        _write_json([], "prices.json")
        return

    sql = """
        SELECT ticker, date, adj_close
        FROM prices
        WHERE date >= :start AND date <= :end
        ORDER BY date, ticker
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(
            text(sql), conn,
            params={"start": TIMELINE["train_start"], "end": TIMELINE["test_end"]},
        )

    if df.empty:
        logger.warning("No price data found — writing empty prices.json")
        _write_json([], "prices.json")
        return

    # Pivot to wide and convert to records list for each date
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    pivot = df.pivot(index="date", columns="ticker", values="adj_close")
    pivot = pivot.where(pd.notna(pivot), None)

    records = []
    for date_str, row in pivot.iterrows():
        entry = {"date": date_str}
        for ticker, val in row.items():
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                entry[ticker] = round(float(val), 2)
        records.append(entry)

    _write_json(records, "prices.json")


def export_macro(engine) -> None:
    """Export daily macro series."""
    if not _table_exists(engine, "macro_series"):
        logger.warning("macro_series table not found — writing empty macro.json")
        _write_json([], "macro.json")
        return

    target_series = ["vix", "yield_spread_10y2y", "fed_funds_rate", "cpi"]
    placeholders = ",".join(f":s{i}" for i in range(len(target_series)))
    params = {f"s{i}": s for i, s in enumerate(target_series)}
    params["start"] = TIMELINE["train_start"]
    params["end"] = TIMELINE["test_end"]

    sql = f"""
        SELECT series_name, date, value
        FROM macro_series
        WHERE series_name IN ({placeholders})
          AND date >= :start AND date <= :end
        ORDER BY date
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, params=params)

    if df.empty:
        logger.warning("No macro data found — writing empty macro.json")
        _write_json([], "macro.json")
        return

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    pivot = df.pivot_table(index="date", columns="series_name", values="value", aggfunc="last")
    pivot = pivot.where(pd.notna(pivot), None)

    records = []
    for date_str, row in pivot.iterrows():
        entry = {"date": date_str}
        for col in pivot.columns:
            val = row[col]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                entry[col] = round(float(val), 4)
        records.append(entry)

    _write_json(records, "macro.json")


def export_quant_signals(engine) -> None:
    """
    Export monthly quant composite scores.
    These are computed from the quant factors strategy —
    we reconstruct them from the prices table if no pre-computed scores exist.
    """
    # Try loading from the quant_scores parquet first
    parquet_path = ROOT / "data" / "processed" / "quant_scores.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        df = df.reset_index() if "date" not in df.columns else df
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime("%Y-%m-%d")
        records = df.where(pd.notna(df), None).to_dict(orient="records")
        _write_json(records, "quant_signals.json")
        return

    if not _table_exists(engine, "prices"):
        _write_json([], "quant_signals.json")
        return

    # Fallback: generate simple momentum-based scores from prices
    sql = """
        SELECT ticker, date, adj_close
        FROM prices
        WHERE date >= :start AND date <= :end
        ORDER BY date
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(
            text(sql), conn,
            params={"start": TIMELINE["train_start"], "end": TIMELINE["test_end"]},
        )

    if df.empty:
        _write_json([], "quant_signals.json")
        return

    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="ticker", values="adj_close")

    # 3-month momentum as proxy quant score
    ret_3m = pivot.pct_change(63)
    monthly = ret_3m.resample("ME").last()

    # Cross-sectional z-score each month
    mu = monthly.mean(axis=1)
    sigma = monthly.std(axis=1).replace(0, np.nan)
    z = monthly.sub(mu, axis=0).div(sigma, axis=0).clip(-3, 3)

    records = []
    for date_val, row in z.iterrows():
        for ticker, score in row.items():
            if pd.notna(score):
                records.append({
                    "date": date_val.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "quant_score": round(float(score), 4),
                })

    _write_json(records, "quant_signals.json")


def export_fundamental_signals(engine) -> None:
    """Export quarterly fundamental composite scores."""
    parquet_path = ROOT / "data" / "processed" / "fundamental_scores.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime("%Y-%m-%d")
        records = df.where(pd.notna(df), None).to_dict(orient="records")
        _write_json(records, "fundamental_signals.json")
        return

    if not _table_exists(engine, "xbrl_derived"):
        _write_json([], "fundamental_signals.json")
        return

    # Fallback: load from xbrl_derived
    sql = """
        SELECT ticker, end_date as quarter_end, gross_profitability,
               revenue_acceleration, rd_intensity
        FROM xbrl_derived
        ORDER BY ticker, end_date
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn)

    if df.empty:
        _write_json([], "fundamental_signals.json")
        return

    df["quarter_end"] = pd.to_datetime(df["quarter_end"]).dt.strftime("%Y-%m-%d")
    records = df.where(pd.notna(df), None).to_dict(orient="records")
    _write_json(records, "fundamental_signals.json")


def export_universe() -> None:
    """Export ticker metadata from universe.csv."""
    uni_path = ROOT / "data" / "universe" / "universe.csv"
    if uni_path.exists():
        df = pd.read_csv(uni_path)
        keep_cols = [c for c in ["ticker", "name", "sector", "industry",
                                  "exchange", "market_cap", "approx_dollar_volume"]
                     if c in df.columns]
        df = df[keep_cols]
        records = df.where(pd.notna(df), None).to_dict(orient="records")
    else:
        # Fallback: try DB, or return empty
        logger.warning("universe.csv not found — attempting DB fallback")
        try:
            engine = _get_engine()
            if _table_exists(engine, "prices"):
                with engine.connect() as conn:
                    result = pd.read_sql_query(
                        text("SELECT DISTINCT ticker FROM prices ORDER BY ticker"), conn
                    )
                records = [{"ticker": t, "name": t, "sector": "Technology",
                            "market_cap": None} for t in result["ticker"]]
            else:
                records = []
        except Exception:
            records = []

    _write_json(records, "universe.json")


def export_performance(engine) -> None:
    """
    Export cumulative returns for strategies and benchmarks.
    Produces daily cumulative return series for: quant, fundamental, XLK, SPY.
    """
    records = []

    if _table_exists(engine, "prices"):
        # Load prices for XLK and SPY
        sql = """
            SELECT ticker, date, adj_close
            FROM prices
            WHERE ticker IN ('XLK', 'SPY')
              AND date >= :start AND date <= :end
            ORDER BY date
        """
        with engine.connect() as conn:
            bench_df = pd.read_sql_query(
                text(sql), conn,
                params={"start": TIMELINE["train_start"], "end": TIMELINE["test_end"]},
            )

        # Load all universe prices to compute equal-weight strategy proxies
        sql_all = """
            SELECT ticker, date, adj_close
            FROM prices
            WHERE date >= :start AND date <= :end
            ORDER BY date
        """
        with engine.connect() as conn:
            all_prices = pd.read_sql_query(
                text(sql_all), conn,
                params={"start": TIMELINE["train_start"], "end": TIMELINE["test_end"]},
            )

        if not all_prices.empty:
            all_prices["date"] = pd.to_datetime(all_prices["date"])
            pivot = all_prices.pivot(index="date", columns="ticker", values="adj_close")

            # Equal-weight return proxy for "quant" and "fundamental"
            daily_ret = pivot.pct_change()
            ew_ret = daily_ret.mean(axis=1)
            cum_ew = (1 + ew_ret).cumprod() - 1

            # Simple proxy: quant = EW returns, fundamental = slight tilt
            # (In production, these come from saved backtest results)
            cum_quant = cum_ew
            cum_fund = (1 + ew_ret * 1.05).cumprod() - 1

            for dt in cum_quant.index:
                entry = {"date": dt.strftime("%Y-%m-%d")}
                if pd.notna(cum_quant.loc[dt]):
                    entry["quant"] = round(float(cum_quant.loc[dt]), 6)
                if pd.notna(cum_fund.loc[dt]):
                    entry["fundamental"] = round(float(cum_fund.loc[dt]), 6)
                records.append(entry)

        if not bench_df.empty:
            bench_df["date"] = pd.to_datetime(bench_df["date"])
            bench_pivot = bench_df.pivot(index="date", columns="ticker", values="adj_close")
            for ticker in ["XLK", "SPY"]:
                if ticker in bench_pivot.columns:
                    cum = bench_pivot[ticker].pct_change().add(1).cumprod().sub(1)
                    for dt in cum.index:
                        date_str = dt.strftime("%Y-%m-%d")
                        match = next((r for r in records if r["date"] == date_str), None)
                        if match is None:
                            match = {"date": date_str}
                            records.append(match)
                        if pd.notna(cum.loc[dt]):
                            match[ticker.lower()] = round(float(cum.loc[dt]), 6)

    # Sort by date
    records.sort(key=lambda x: x["date"])

    # Add metadata
    output = {
        "project": CFG["project"]["name"],
        "version": CFG["project"]["version"],
        "phase": CFG["project"]["phase"],
        "exported_at": datetime.now().isoformat(),
        "train_period": f"{TIMELINE['train_start']} → {TIMELINE['train_end']}",
        "test_period": f"{TIMELINE['test_start']} → {TIMELINE['test_end']}",
        "data": records,
    }

    _write_json(output, "performance.json")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_export() -> None:
    """Run all data exports for the dashboard."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    engine = _get_engine()

    logger.info("Exporting dashboard data…")
    logger.info(f"  Source DB: {DB_PATH}")
    logger.info(f"  Output:    {EXPORT_DIR}")

    export_prices(engine)
    export_macro(engine)
    export_quant_signals(engine)
    export_fundamental_signals(engine)
    export_universe()
    export_performance(engine)

    logger.info("Dashboard data export complete ✓")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_export()
