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

    # Drop benchmark columns before the cross-sectional z-score below.
    from src.utils.benchmarks import strip_benchmarks
    pivot = pivot[strip_benchmarks(pivot.columns)]

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
                # `prices` also holds benchmark series (SPY) — not universe members.
                from src.utils.benchmarks import strip_benchmarks
                records = [{"ticker": t, "name": t, "sector": "Technology",
                            "market_cap": None}
                           for t in strip_benchmarks(result["ticker"])]
            else:
                records = []
        except Exception:
            records = []

    _write_json(records, "universe.json")


def export_performance(engine) -> None:
    """
    Export cumulative returns for strategies and benchmarks.
    Reads from the real VectorBT equity curve CSVs saved by each strategy.
    Falls back to equal-weight proxy if CSVs are not found.
    """
    RESULTS_DIR = ROOT / "backtests" / "results"
    records_map: dict[str, dict] = {}  # date_str -> {key: value}

    def _add_series(csv_path: Path, col: str, out_key: str) -> None:
        """Read a CSV equity curve and add it to records_map as cumulative return."""
        if not csv_path.exists():
            logger.warning(f"  Equity CSV not found: {csv_path.name}")
            return
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if col not in df.columns:
            logger.warning(f"  Column '{col}' not in {csv_path.name}")
            return
        s = df[col].dropna()
        if s.empty:
            return
        # Normalise to cumulative return (start = 0%)
        cum = s / s.iloc[0] - 1
        for dt, val in cum.items():
            date_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
            if date_str not in records_map:
                records_map[date_str] = {"date": date_str}
            records_map[date_str][out_key] = round(float(val), 6)
        logger.info(f"  Loaded {out_key} from {csv_path.name} ({len(cum)} rows)")

    # ── Real backtest equity curves ───────────────────────────────────
    # Quant: combine train + test into one series
    quant_train = RESULTS_DIR / "quant_sprint1_train_equity.csv"
    quant_full  = RESULTS_DIR / "quant_sprint1_full_equity.csv"
    quant_src   = quant_full if quant_full.exists() else quant_train

    fund_train  = RESULTS_DIR / "fundamental_sprint2_train_equity.csv"
    fund_test   = RESULTS_DIR / "fundamental_sprint2_test_equity.csv"

    ens_train   = RESULTS_DIR / "ensemble_sprint3_train_equity.csv"
    ens_test    = RESULTS_DIR / "ensemble_sprint3_test_equity.csv"

    _add_series(quant_src,  "strategy",  "quant")
    _add_series(quant_src,  "benchmark", "spy")      # quant benchmarks vs SPY

    # For fundamental, stitch train + test into one continuous series
    def _stitch_train_test(train_path, test_path, col, out_key):
        if train_path.exists() and test_path.exists():
            train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
            test_df  = pd.read_csv(test_path,  index_col=0, parse_dates=True)
            if col in train_df.columns and col in test_df.columns:
                s_train = train_df[col].dropna()
                s_test  = test_df[col].dropna()
                scale = s_train.iloc[-1] / s_test.iloc[0] if s_test.iloc[0] != 0 else 1.0
                s_combined = pd.concat([s_train, s_test * scale])
                cum = s_combined / s_combined.iloc[0] - 1
                for dt, val in cum.items():
                    date_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
                    if date_str not in records_map:
                        records_map[date_str] = {"date": date_str}
                    records_map[date_str][out_key] = round(float(val), 6)
                logger.info(f"  Loaded {out_key} from {train_path.name}+{test_path.name}")
                return True
        return False

    if not _stitch_train_test(fund_train, fund_test, "strategy", "fundamental"):
        _add_series(fund_train, "strategy",  "fundamental")
    if not _stitch_train_test(fund_train, fund_test, "benchmark", "xlk"):
        _add_series(fund_train, "benchmark", "xlk")

    # Ensemble: stitch train + test
    if not _stitch_train_test(ens_train, ens_test, "strategy", "ensemble"):
        _add_series(ens_train, "strategy", "ensemble")

    # ── Fallback: equal-weight proxy if no CSVs found ─────────────────
    if not records_map and _table_exists(engine, "prices"):
        logger.warning("No equity CSVs found — falling back to equal-weight proxy")
        sql_all = """
            SELECT ticker, date, adj_close FROM prices
            WHERE date >= :start AND date <= :end ORDER BY date
        """
        with engine.connect() as conn:
            all_prices = pd.read_sql_query(
                text(sql_all), conn,
                params={"start": TIMELINE["train_start"], "end": TIMELINE["test_end"]},
            )
        if not all_prices.empty:
            all_prices["date"] = pd.to_datetime(all_prices["date"])
            pivot = all_prices.pivot(index="date", columns="ticker", values="adj_close")
            # Benchmark series must not become a member of the equal-weight proxy.
            from src.utils.benchmarks import strip_benchmarks
            pivot = pivot[strip_benchmarks(pivot.columns)]
            ew_ret = pivot.pct_change().mean(axis=1)
            cum_ew = (1 + ew_ret).cumprod() - 1
            for dt, val in cum_ew.items():
                date_str = dt.strftime("%Y-%m-%d")
                records_map[date_str] = {
                    "date": date_str,
                    "quant": round(float(val), 6),
                    "fundamental": round(float(val), 6),
                }

    records = sorted(records_map.values(), key=lambda x: x["date"])

    output = {
        "project": CFG["project"]["name"],
        "version": CFG["project"]["version"],
        "phase": CFG["project"]["phase"],
        "exported_at": datetime.now().isoformat(),
        "train_period": f"{TIMELINE['train_start']} → {TIMELINE['train_end']}",
        "test_period":  f"{TIMELINE['test_start']} → {TIMELINE['test_end']}",
        "data": records,
    }

    _write_json(output, "performance.json")


# ── Ensemble export ───────────────────────────────────────────────────────────

def export_ensemble() -> None:
    """
    Export ensemble-specific data for the dashboard:
      - ensemble_scores.json    (monthly scores per ticker)
      - regime_attribution.json (copy to dashboard data dir)
    """
    # Ensemble scores
    scores_path = ROOT / "data" / "processed" / "ensemble_scores.parquet"
    if scores_path.exists():
        df = pd.read_parquet(scores_path)
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime("%Y-%m-%d")
        records = df.where(pd.notna(df), None).to_dict(orient="records")
        _write_json(records, "ensemble_scores.json")
    else:
        logger.warning("ensemble_scores.parquet not found — skipping")

    # Regime attribution — copy from processed to dashboard dir
    attr_src = ROOT / "data" / "processed" / "regime_attribution.json"
    if attr_src.exists():
        import shutil
        attr_dst = EXPORT_DIR / "regime_attribution.json"
        shutil.copy2(attr_src, attr_dst)
        logger.info(f"  → {attr_dst}  ({attr_dst.stat().st_size / 1024:.0f} KB)")
    else:
        logger.warning("regime_attribution.json not found — skipping")


# ── Dashboard v2 exports (ensemble equity, sprints, holdings, breadth, regime) ─

RESULTS_DIR = ROOT / "backtests" / "results"

# Portfolio-construction constants mirrored from portfolio_builder (module-level
# there too; keep in sync manually — do NOT move to settings.yaml).
V2_TOP_N = 20
V2_MIN_SCORE = 0.52


def export_ensemble_performance() -> None:
    """ensemble_performance.json — Sprint 5 production equity as cumulative %.

    Source: backtests/results/ensemble_timesfm_{train,test}_equity.csv
    (tracked Sprint 5 artifacts). The two CSVs are non-overlapping slices
    of ONE continuous backtest run (train ends 2022-12-30 at 164,513.79;
    test opens 2023-01-03 at 163,921.10 — the gap is the real overnight
    return), so a plain concat reproduces sprint5 total_return_full
    129.0257% exactly. Do NOT rescale the test segment.
    """
    train_p = RESULTS_DIR / "ensemble_timesfm_train_equity.csv"
    test_p = RESULTS_DIR / "ensemble_timesfm_test_equity.csv"
    if not (train_p.exists() and test_p.exists()):
        logger.warning("ensemble_timesfm_*.csv not found — skipping ensemble_performance")
        return

    train = pd.read_csv(train_p, index_col=0, parse_dates=True)
    test = pd.read_csv(test_p, index_col=0, parse_dates=True)

    out = {}
    for col in ("strategy", "benchmark"):
        if col not in train.columns or col not in test.columns:
            continue
        full = pd.concat([train[col], test[col]])
        out[col] = (full / full.iloc[0] - 1.0) * 100.0

    df = pd.DataFrame(out).round(3)
    records = [
        {"date": d.strftime("%Y-%m-%d"),
         **{k: (None if pd.isna(v) else v) for k, v in row.items()}}
        for d, row in df.iterrows()
    ]
    payload = {
        "test_start": TIMELINE["test_start"],
        "source": "ensemble_timesfm_train/test_equity.csv (Sprint 5 production)",
        "data": records,
    }
    _write_json(payload, "ensemble_performance.json")


def export_sprints() -> None:
    """sprints.json — sprint comparison table + verbatim verdicts.

    Sources: sprint5_results.json (holds sprint0/sprint4/sprint5 blocks),
    sprint4_recal_results.json, sprint6_results.json. Verdict notes are
    copied VERBATIM — the dashboard must not paraphrase experiment records.
    """
    s5_p = RESULTS_DIR / "sprint5_results.json"
    s6_p = RESULTS_DIR / "sprint6_results.json"
    if not s5_p.exists():
        logger.warning("sprint5_results.json not found — skipping sprints export")
        return
    s5 = json.loads(s5_p.read_text())
    s6 = json.loads(s6_p.read_text()) if s6_p.exists() else {}

    sprints = [
        {
            "id": "sprint0", "name": "Sprint 0", "change": "Frozen baseline",
            "verdict": "BASELINE", "metrics": s5.get("sprint0_baseline", {}),
            "verdict_notes": "Frozen Sprint 0 baseline (commit c0f9c65). Full-period "
                             "Sharpe 0.600 from ensemble_baseline_stats.csv; the 0.721 "
                             "in baseline_metrics.json is train-period only.",
        },
        {
            "id": "sprint4", "name": "Sprint 4", "change": "Graded regime gate",
            "verdict": "FAIL", "metrics": s5.get("sprint4_graded_gate", {}),
            "verdict_notes": "",
        },
        {
            "id": "sprint5", "name": "Sprint 5", "change": "+TimesFM 23rd feature",
            "verdict": "PASS", "metrics": s5.get("sprint5_timesfm", {}),
            "verdict_notes": s5.get("verdict_notes", ""),
        },
    ]
    s4_p = RESULTS_DIR / "sprint4_recal_results.json"
    if s4_p.exists():
        try:
            s4 = json.loads(s4_p.read_text())
            # In sprint4_recal_results.json the long verdict PARAGRAPH lives
            # under "verdict" (unlike sprint5/6 where "verdict" is PASS/FAIL
            # and the paragraph is "verdict_notes"); its "notes" key is a dict.
            v = s4.get("verdict", "")
            if isinstance(v, str) and len(v) > 20:
                sprints[1]["verdict_notes"] = v
        except Exception:
            pass
    if s6:
        verdict = s6.get("verdict", "")
        sprints.append({
            "id": "sprint6", "name": "Sprint 6", "change": "Rolling 36-month window",
            "verdict": "REFUTED" if verdict == "FAIL" else verdict,
            "metrics": s6.get("sprint6_rolling", {}),
            "verdict_notes": s6.get("verdict_notes", ""),
            "fold_diagnostics": s6.get("fold_diagnostics", {}),
            "checks": s6.get("checks", {}),
        })

    _write_json({"generated_from": ["sprint4_recal_results.json", "sprint5_results.json",
                                    "sprint6_results.json"],
                 "sprints": sprints}, "sprints.json")


def export_regime() -> None:
    """regime.json — monthly regime multipliers for band shading + chips."""
    try:
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from src.strategies.ensemble.regime_gate import get_historical_regime_multipliers
    except Exception as exc:
        logger.warning(f"regime_gate import failed ({exc}) — skipping regime export")
        return

    mults = get_historical_regime_multipliers(TIMELINE["train_start"], TIMELINE["test_end"])
    label = {0.5: "RISK_OFF", 1.0: "NEUTRAL", 1.2: "RISK_ON"}
    records = [
        {"month": d.strftime("%Y-%m"), "multiplier": float(v),
         "label": label.get(round(float(v), 2), "NEUTRAL")}
        for d, v in mults.items()
    ]
    _write_json(records, "regime.json")


def export_holdings_and_breadth() -> None:
    """holdings.json + breadth.json — from production ensemble scores.

    Mirrors portfolio_builder selection: score > V2_MIN_SCORE, top V2_TOP_N,
    equal weight. Breadth = names above threshold per month (production /
    Sprint 5 state; the Sprint 6 rolling comparison numbers live in
    sprints.json fold_diagnostics).
    """
    scores_path = ROOT / "data" / "processed" / "ensemble_scores.parquet"
    if not scores_path.exists():
        logger.warning("ensemble_scores.parquet not found — skipping holdings/breadth")
        return
    df = pd.read_parquet(scores_path)
    df["date"] = pd.to_datetime(df["date"])

    # Universe metadata for names/industries
    meta = {}
    uni_p = EXPORT_DIR / "universe.json"
    if uni_p.exists():
        try:
            for row in json.loads(uni_p.read_text()):
                meta[row["ticker"]] = {"name": row.get("name", ""),
                                       "industry": row.get("industry", "")}
        except Exception:
            pass

    holdings = {}
    breadth = []
    for date, g in df.groupby("date"):
        month = date.strftime("%Y-%m")
        above = g[g["ensemble_score"] > V2_MIN_SCORE]
        breadth.append({"month": month, "names_above_threshold": int(len(above))})
        sel = above.nlargest(min(V2_TOP_N, len(above)), "ensemble_score")
        n = len(sel)
        holdings[month] = [
            {"ticker": r["ticker"],
             "name": meta.get(r["ticker"], {}).get("name", r["ticker"]),
             "industry": meta.get(r["ticker"], {}).get("industry", ""),
             "score": round(float(r["ensemble_score"]), 4),
             "weight": round(1.0 / n, 4) if n else 0.0}
            for _, r in sel.iterrows()
        ]

    _write_json({"top_n": V2_TOP_N, "min_score": V2_MIN_SCORE, "months": holdings},
                "holdings.json")
    _write_json({"min_score": V2_MIN_SCORE, "max_positions": V2_TOP_N, "data": breadth},
                "breadth.json")


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
    export_ensemble()

    # Dashboard v2 exports
    export_ensemble_performance()
    export_sprints()
    export_regime()
    export_holdings_and_breadth()

    logger.info("Dashboard data export complete ✓")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_export()
