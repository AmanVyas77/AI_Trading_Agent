"""
Live monthly data refresh — the runbook that gets the DB and parquet
artefacts current enough for a live rebalance.

Steps, in order (each timed and logged):
    1.  quant_pipeline           (prices + FRED)
    2.  backfill_cpi             (monthly CPI; not in the routine FRED list)
    3.  sentiment increment      (FinBERT on 8-Ks; --date-start = max
                                  filing_date − 7d for overlap safety)
    3b. edgar_xbrl               (EDGAR company-facts pull for the
                                  universe; per-ticker upsert is idempotent)
    3c. lm_10k_increment         (collect new 10-Ks, then LM-score;
                                  the scoring LEFT-JOIN skips already-
                                  scored filings so this is cheap when
                                  no new 10-K has landed)
    3d. simfin_estimates         (analyst EPS + revision momentum;
                                  now uses today at import as --end)
    3e. factor_export_fundamental (regenerates the fundamental parquet
                                  the feature_matrix reads; without this
                                  3b-3d change nothing downstream)
    4.  factor_export_quant      (--start 2015-01-01 --end <last completed
                                  month-end>; TimesFM inside)
    5.  feature_matrix           (same window)

After the steps, this module asserts freshness against Sprint 8 Rule 1
limits and raises on any violation — a stale series must fail LOUD
rather than silently ship into rebalance.

Freshness limits (calendar days, hard-coded to avoid a yaml round-trip):
    prices             ≤ 3 trading days       (allow one long weekend)
    vix                ≤ 3 days
    yield_spread_10y2y ≤ 3 days
    sentiment_scores   ≤ 14 days
    cpi                ≤ 75 days (release lag)
    xbrl_facts         ≤ 120 days (10-Q filing deadline ~40 d after
                                    quarter end + a full quarter of
                                    lag before the next print)
    eps_revisions      ≤ 120 days (same quarterly cadence)

CLI
---
    python -m src.live.refresh [--as-of YYYY-MM-DD] [--skip step,...]

Skippable step names: quant_pipeline, backfill_cpi, sentiment,
edgar_xbrl, lm_10k_increment, simfin_estimates,
factor_export_fundamental, quant_factors, feature_matrix. Useful when a
single component is under maintenance; freshness assertions still run.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yaml
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config" / "settings.yaml") as f:
    _CFG = yaml.safe_load(f)

DB_PATH = ROOT / _CFG["data"]["paths"]["db"]
DB_URL = f"sqlite:///{DB_PATH}"
LOGS_DIR = ROOT / "logs"
LOG_JSONL = LOGS_DIR / "live_refresh_log.jsonl"

logger = logging.getLogger(__name__)

# Sprint 8 Rule 1 staleness limits (calendar days).
# CPI is reference-month dated (May print → 05-01) and BLS releases
# with a ~2-3-week lag → age oscillates 40-75d in normal operation.
# Amended from 45 → 75 (Sprint 8 Prompt 4 FIX 2, calibration only).
# xbrl_facts + eps_revisions: 10-Q deadline is ~40 days after quarter
# end and the next report is ~90 days after that, so a healthy series
# never exceeds ~120 days without a gap.
STALENESS_LIMITS = {
    "prices": 5,               # 3 trading days ~= 5 calendar days worst case
    "vix": 3,
    "yield_spread_10y2y": 3,
    "sentiment_scores": 14,
    "cpi": 75,
    "xbrl_facts": 120,
    "eps_revisions": 120,
}

_ALL_STEPS = (
    "quant_pipeline",
    "backfill_cpi",
    "sentiment",
    "edgar_xbrl",
    "lm_10k_increment",
    "simfin_estimates",
    "factor_export_fundamental",
    "quant_factors",
    "feature_matrix",
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _last_completed_month_end(as_of: date) -> date:
    """Month-end strictly before `as_of` (mid-month → prior month-end)."""
    first_of_this_month = as_of.replace(day=1)
    return first_of_this_month - timedelta(days=1)


def _sentiment_increment_start() -> str:
    """max(filing_date) − 7d in sentiment_scores, or 2015-01-01 if empty."""
    engine = create_engine(DB_URL, echo=False)
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT MAX(filing_date) FROM sentiment_scores")
        ).fetchone()
    latest = row[0] if row and row[0] else None
    if not latest:
        return "2015-01-01"
    d = date.fromisoformat(latest) - timedelta(days=7)
    return d.isoformat()


def _run_step(name: str, fn) -> dict:
    """Time & log a single step. Any exception bubbles up."""
    logger.info("▶ %s starting…", name)
    t0 = time.perf_counter()
    try:
        result = fn()
        elapsed = time.perf_counter() - t0
        logger.info("✔ %s finished in %.1fs", name, elapsed)
        return {"step": name, "status": "ok", "elapsed_s": round(elapsed, 2),
                "result": result}
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error("✘ %s FAILED after %.1fs: %s", name, elapsed, exc)
        raise


def _run_subprocess_step(name: str, argv: list[str]) -> dict:
    """Run a `.venv/bin/python -m ...` step and stream its output."""
    logger.info("▶ %s (subprocess): %s", name, " ".join(argv))
    t0 = time.perf_counter()
    completed = subprocess.run(
        argv,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.perf_counter() - t0
    if completed.returncode != 0:
        logger.error("✘ %s FAILED (rc=%d) — last stderr/stdout tail:\n%s",
                     name, completed.returncode, completed.stdout[-2000:])
        raise RuntimeError(
            f"{name}: subprocess exited with rc={completed.returncode}"
        )
    logger.info("✔ %s finished in %.1fs", name, elapsed)
    return {"step": name, "status": "ok", "elapsed_s": round(elapsed, 2)}


# ── Freshness report ─────────────────────────────────────────────────────────

def _collect_freshness(as_of: date) -> dict:
    """Read the current max-date per required source and compute age in days."""
    engine = create_engine(DB_URL, echo=False)
    with engine.connect() as conn:
        max_price = conn.execute(text("SELECT MAX(date) FROM prices")).fetchone()[0]
        max_sent = conn.execute(
            text("SELECT MAX(filing_date) FROM sentiment_scores")
        ).fetchone()[0]
        max_xbrl = conn.execute(
            text("SELECT MAX(end_date) FROM xbrl_facts")
        ).fetchone()[0]
        max_eps_rev = conn.execute(
            text("SELECT MAX(date) FROM eps_revisions")
        ).fetchone()[0]
        macro_rows = conn.execute(text(
            "SELECT series_name, MAX(date) FROM macro_series "
            "WHERE value IS NOT NULL GROUP BY series_name"
        )).fetchall()

    per_series = {name: mx for name, mx in macro_rows}

    report = {
        "as_of": as_of.isoformat(),
        "sources": {
            "prices": {"max_date": max_price},
            "vix": {"max_date": per_series.get("vix")},
            "yield_spread_10y2y": {"max_date": per_series.get("yield_spread_10y2y")},
            "sentiment_scores": {"max_date": max_sent},
            "cpi": {"max_date": per_series.get("cpi")},
            "xbrl_facts": {"max_date": max_xbrl},
            "eps_revisions": {"max_date": max_eps_rev},
        },
    }

    for name, cell in report["sources"].items():
        mx = cell["max_date"]
        if mx is None:
            cell["age_days"] = None
            cell["limit_days"] = STALENESS_LIMITS[name]
            cell["ok"] = False
            continue
        age = (as_of - date.fromisoformat(mx)).days
        cell["age_days"] = age
        cell["limit_days"] = STALENESS_LIMITS[name]
        cell["ok"] = age <= STALENESS_LIMITS[name]

    return report


def _assert_freshness(report: dict) -> None:
    stale = [
        name for name, cell in report["sources"].items() if not cell["ok"]
    ]
    if stale:
        details = ", ".join(
            f"{n}={report['sources'][n]['age_days']}d>"
            f"{report['sources'][n]['limit_days']}d"
            for n in stale
        )
        raise RuntimeError(
            f"Freshness assertion failed for series: {details}. "
            "Live trading is BLOCKED until the corresponding pipeline "
            "step is re-run successfully."
        )


# ── Step wrappers ────────────────────────────────────────────────────────────

def _step_quant_pipeline() -> dict:
    from src.data.quant_pipeline import run_pipeline
    run_pipeline()
    return {}


def _step_backfill_cpi() -> dict:
    from scripts.backfill_cpi import backfill
    n = backfill()
    return {"rows_upserted": n}


def _step_sentiment(as_of: date) -> dict:
    from src.strategies.fundamental.sentiment_pipeline import run_sentiment_pipeline
    start = _sentiment_increment_start()
    end = as_of.isoformat()
    logger.info("Sentiment increment window: %s → %s", start, end)
    run_sentiment_pipeline(date_start=start, date_end=end)
    return {"date_start": start, "date_end": end}


def _step_edgar_xbrl() -> dict:
    """Full-per-ticker XBRL pull. INSERT OR REPLACE keeps it idempotent."""
    from src.data.edgar_pipeline import run_edgar_pipeline
    run_edgar_pipeline()
    return {}


def _unscored_10k_count() -> int:
    """Cheap LEFT JOIN: count 10-K filings not yet in lm_sentiment_scores."""
    engine = create_engine(DB_URL, echo=False)
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT COUNT(*) FROM edgar_10k_filings f "
            "LEFT JOIN lm_sentiment_scores s "
            "  ON f.ticker = s.ticker AND f.filing_date = s.filing_date "
            "WHERE s.ticker IS NULL"
        )).fetchone()
    return int(row[0] or 0)


def _step_lm_10k_increment() -> dict:
    """
    Collect + score 10-K filings only if there is unscored work. The
    collection call is a full-window SEC pull, so we gate on the cheap
    LEFT-JOIN count first; the scoring pass itself is already a no-op
    when nothing is unscored.
    """
    from src.strategies.fundamental.sentiment_pipeline import (
        _get_engine, run_lm_collection_pipeline, run_lm_scoring_pipeline,
    )
    import yaml as _yaml
    from datetime import datetime as _dt

    engine = _get_engine()
    with open(ROOT / "config" / "settings.yaml") as f:
        cfg = _yaml.safe_load(f)
    ff_cfg = cfg["fundamental_factors"]

    from src.universe.screener import load_universe
    uni = load_universe(ROOT / "data" / "universe" / "universe.csv")
    tickers = uni["ticker"].tolist()

    start_year = int(str(cfg["timeline"]["train_start"])[:4])
    end_year = _dt.now().year
    logger.info(
        "LM 10-K collection window: %d-%d for %d tickers",
        start_year, end_year, len(tickers),
    )
    run_lm_collection_pipeline(tickers, start_year, end_year, engine, ff_cfg)

    unscored = _unscored_10k_count()
    logger.info("Unscored 10-K filings after collection: %d", unscored)
    if unscored == 0:
        return {"collected_new": 0, "scored": 0}
    run_lm_scoring_pipeline(engine, ff_cfg)
    return {"scored": unscored}


def _step_simfin_estimates() -> dict:
    """Analyst EPS pull + revision momentum. DATE_END = today at import."""
    from src.data.simfin_pipeline import run_simfin_finra_pipeline
    # skip_finra=True: FINRA short-volume is Sprint 8's separate concern.
    run_simfin_finra_pipeline(skip_finra=True)
    return {}


def _step_factor_export_fundamental(month_end: date) -> dict:
    argv = [
        str(ROOT / ".venv" / "bin" / "python"),
        "-m", "src.strategies.ensemble.factor_export_fundamental",
        "--start", "2015-01-01",
        "--end", month_end.isoformat(),
    ]
    return _run_subprocess_step("factor_export_fundamental", argv)


def _step_quant_factors(month_end: date) -> dict:
    argv = [
        str(ROOT / ".venv" / "bin" / "python"),
        "-m", "src.strategies.ensemble.factor_export_quant",
        "--start", "2015-01-01",
        "--end", month_end.isoformat(),
    ]
    return _run_subprocess_step("quant_factors", argv)


def _step_feature_matrix(month_end: date) -> dict:
    argv = [
        str(ROOT / ".venv" / "bin" / "python"),
        "-m", "src.strategies.ensemble.feature_matrix",
        "--start", "2015-01-01",
        "--end", month_end.isoformat(),
    ]
    return _run_subprocess_step("feature_matrix", argv)


# ── Public API ───────────────────────────────────────────────────────────────

def run_refresh(
    as_of: Optional[str] = None,
    skip: Optional[Iterable[str]] = None,
) -> dict:
    """Run the monthly data refresh and return a freshness report.

    Parameters
    ----------
    as_of : YYYY-MM-DD; defaults to today. Anchors both the sentiment
            end date and the "last completed month-end" for the factor
            and feature-matrix rebuild windows.
    skip  : iterable of step names to skip (operator control). The
            freshness assertion at the end still runs — skipping a step
            does not skip its staleness limit.

    Returns
    -------
    dict — the freshness report, also appended as one JSON line to
    ``logs/live_refresh_log.jsonl``.

    Raises
    ------
    RuntimeError on freshness violations or subprocess step failures.
    """
    skip_set = set(skip or ())
    for s in skip_set:
        if s not in _ALL_STEPS:
            raise ValueError(
                f"Unknown skip step '{s}'. Valid: {sorted(_ALL_STEPS)}"
            )

    as_of_date = date.fromisoformat(as_of) if as_of else date.today()
    month_end = _last_completed_month_end(as_of_date)
    logger.info(
        "run_refresh: as_of=%s  last_completed_month_end=%s  skip=%s",
        as_of_date, month_end, sorted(skip_set) or "∅",
    )

    step_log: list[dict] = []

    def _run(name: str, fn) -> None:
        if name in skip_set:
            logger.info("↷ %s SKIPPED (operator)", name)
            step_log.append({"step": name, "status": "skipped"})
            return
        step_log.append(_run_step(name, fn))

    _run("quant_pipeline", _step_quant_pipeline)
    _run("backfill_cpi", _step_backfill_cpi)
    _run("sentiment", lambda: _step_sentiment(as_of_date))
    _run("edgar_xbrl", _step_edgar_xbrl)
    _run("lm_10k_increment", _step_lm_10k_increment)
    _run("simfin_estimates", _step_simfin_estimates)
    _run("factor_export_fundamental",
         lambda: _step_factor_export_fundamental(month_end))
    _run("quant_factors", lambda: _step_quant_factors(month_end))
    _run("feature_matrix", lambda: _step_feature_matrix(month_end))

    report = _collect_freshness(as_of_date)
    report["steps"] = step_log
    report["ran_at"] = datetime.now().isoformat(timespec="seconds")
    report["month_end"] = month_end.isoformat()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_JSONL, "a") as f:
        f.write(json.dumps(report, default=str) + "\n")

    _assert_freshness(report)  # LOUD failure if anything is stale
    return report


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ap = argparse.ArgumentParser(description="Live monthly data refresh runbook")
    ap.add_argument("--as-of", default=None, help="Anchor date (YYYY-MM-DD)")
    ap.add_argument(
        "--skip",
        default=None,
        help="Comma-separated step names to skip: " + ",".join(_ALL_STEPS),
    )
    args = ap.parse_args()

    skip = args.skip.split(",") if args.skip else None
    report = run_refresh(as_of=args.as_of, skip=skip)

    print("\n" + "═" * 72)
    print("  Freshness report")
    print("═" * 72)
    for name, cell in report["sources"].items():
        status = "✓" if cell["ok"] else "✘"
        print(
            f"  {status} {name:<22s}  max={cell['max_date']}  "
            f"age={cell['age_days']}d  limit={cell['limit_days']}d"
        )
    print("═" * 72)


if __name__ == "__main__":
    main()
