"""
CPI backfill (CPIAUCSL → macro_series['cpi']).

Sprint 7 Prompt 3 Part A0: CPIAUCSL is not in config/settings.yaml's FRED
list, so the routine quant_pipeline never advances macro_series['cpi']
past whatever was seeded historically. The frozen model uses `cpi` as
one of its 23 features and the feature-matrix z-scores it over a 36-mo
rolling window — a stale cpi value would ffill onto the holdout rows.

Idempotent: `INSERT OR REPLACE` on (series_id, date). Rerun-safe.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fredapi import Fred
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

with open(ROOT / "config" / "settings.yaml") as f:
    CFG = yaml.safe_load(f)

DB_PATH = ROOT / CFG["data"]["paths"]["db"]
DB_URL = f"sqlite:///{DB_PATH}"

SERIES_ID = "CPIAUCSL"
SERIES_NAME = "cpi"


def backfill(start: str = "2026-01-01", end: str | None = None) -> int:
    end = end or datetime.today().strftime("%Y-%m-%d")

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        print("[backfill_cpi] FRED_API_KEY missing — skipping", file=sys.stderr)
        return 0

    engine = create_engine(DB_URL, echo=False)
    with engine.connect() as conn:
        before = conn.execute(
            text("SELECT MAX(date) FROM macro_series WHERE series_id=:sid"),
            {"sid": SERIES_ID},
        ).fetchone()[0]
    print(f"[backfill_cpi] before: max({SERIES_ID}) = {before}")

    try:
        fred = Fred(api_key=api_key)
        s = fred.get_series(SERIES_ID, observation_start=start, observation_end=end)
    except Exception as e:
        print(f"[backfill_cpi] FRED fetch failed: {e} — skipping", file=sys.stderr)
        return 0

    rows = [
        {
            "series_id": SERIES_ID,
            "series_name": SERIES_NAME,
            "date": d.strftime("%Y-%m-%d"),
            "value": float(v),
        }
        for d, v in s.items()
        if v == v
    ]

    if not rows:
        print("[backfill_cpi] FRED returned 0 rows for window — nothing to upsert")
        return 0

    with engine.begin() as conn:
        for r in rows:
            conn.execute(
                text(
                    "INSERT OR REPLACE INTO macro_series "
                    "(series_id, series_name, date, value) "
                    "VALUES (:series_id, :series_name, :date, :value)"
                ),
                r,
            )

    with engine.connect() as conn:
        after = conn.execute(
            text("SELECT MAX(date) FROM macro_series WHERE series_id=:sid"),
            {"sid": SERIES_ID},
        ).fetchone()[0]
    print(f"[backfill_cpi] after:  max({SERIES_ID}) = {after}  (+{len(rows)} rows upserted)")
    return len(rows)


if __name__ == "__main__":
    backfill()
