"""News sentiment factor (R3 / R3-A) — monthly, shrunk, smoothed, standardised.

Corpus (REV 2, 2026-09-09)
--------------------------
Alpha Vantage ONLY, ``published_at >= 2022-01-01``. Prompt 2 reached the
decision table's STOP (rho = 0.208, 95% CI straddling the 0.20 line; the
text-shape control cell was structurally uncomputable) and the FNSPID/AV
stitch was abandoned. FNSPID rows remain in the DB and remain scored — they
are simply not read here. The source filter is explicit and asserted.

The pipeline, in order
----------------------
1. ``raw(i,m)``    arithmetic mean of ticker i's article scores in month m.
2. ``xsec(m)``     EQUAL-WEIGHTED mean of the per-ticker raw values for month
                   m — not the pooled mean over articles. NVDA carries ~223x
                   TWLO's article count, so a pooled mean would make xsec(m)
                   largely a reading of NVDA.
3. ``shrunk``      n/(n+k)*raw + k/(n+k)*xsec, k = 10 fixed, never tuned.
                   n = 0 is an EXPLICIT branch: raw is NaN there and numpy
                   evaluates 0 * NaN as NaN, not 0.
4. ``smoothed``    per-ticker EWM over months, span = 4 (the Arratia
                   convention already used by _ewm_smooth for the 8-K and LM
                   columns). SHRINK FIRST, SMOOTH SECOND — smoothing first
                   would apply one month's article count to a value already
                   mixing four months.
5. ``news_sentiment`` (R3-A) cross-sectional z-score of ``smoothed`` within
                   each month, clipped to factor_export_quant.ZSCORE_CAP, so
                   the column reaches the matrix on the same scale as its 23
                   neighbours rather than on FinBERT's raw [-1, +1].

Point-in-time discipline
------------------------
* Every month's aggregation filters ``published_at <= month-end 23:59:59
  UTC`` explicitly, rather than trusting the month bucket alone, so the
  intent is legible to a later reader.
* A PARTIAL trailing month is dropped entirely and logged. Without this a
  build run mid-month emits a row from a few days of articles that looks
  exactly like a complete one.
* ``xsec(m)`` reads only month m's own articles. This is CONTEMPORANEOUS,
  not forward-looking: at month-end m every article it averages has already
  been published. It resembles cross-sectional leakage at a glance, which is
  why this comment exists.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine, text

from src.strategies.ensemble.factor_export_quant import ZSCORE_CAP
from src.strategies.fundamental.sentiment_pipeline import EWM_SPAN

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
with open(ROOT / "config" / "settings.yaml") as f:
    CFG = yaml.safe_load(f)
DB_PATH = ROOT / CFG["data"]["paths"]["db"]
DB_URL = f"sqlite:///{DB_PATH}"
PROCESSED = ROOT / "data" / "processed"

SOURCE = "alphavantage"          # REV 2: AV only. FNSPID is not read.
AV_ERA_START = "2022-01-01"
SHRINK_K = 10                    # fixed in advance, never tuned
OUT_PARQUET = PROCESSED / "news_sentiment_scores.parquet"


def _get_engine(engine=None):
    return engine if engine is not None else create_engine(DB_URL, echo=False)


def _universe() -> list[str]:
    """Canonical ticker spine (universe.csv), per the Prompt 0 guard.

    Used rather than "tickers that have articles" so that every (date,
    ticker) the feature matrix asks for finds a row: a ticker with no
    articles is a legitimate n = 0, handled by the shrinkage branch, not a
    missing key that would merge to NaN.
    """
    import csv as _csv
    with open(ROOT / "data" / "universe" / "universe.csv", newline="") as f:
        return [r["ticker"] for r in _csv.DictReader(f)]


def _build_panel(start: str, end: str, engine=None) -> pd.DataFrame:
    """Full working panel (n, raw, xsec, w_own, shrunk, smoothed, news_sentiment).

    Separate from :func:`load_news_sentiment` so the public loader can return
    exactly the three contracted columns while the audit parquet and the
    shrinkage honesty report still see every intermediate.
    """
    engine = _get_engine(engine)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    # ── month spine, with the partial trailing month dropped ──────────
    months = pd.date_range(
        start_ts.to_period("M").to_timestamp("M"), end_ts, freq="ME"
    )
    last_full = end_ts.to_period("M").to_timestamp("M")
    if end_ts < last_full:
        logger.warning(
            f"DROPPED PARTIAL MONTH {last_full.strftime('%Y-%m')}: end={end_ts.date()} "
            f"is before that month's end ({last_full.date()}). A partial month must "
            "never be emitted as if complete."
        )
        months = months[months < last_full]
    if len(months) == 0:
        return pd.DataFrame(columns=["date", "ticker", "n", "raw", "xsec",
                                     "w_own", "shrunk", "smoothed",
                                     "news_sentiment"])

    # ── articles: AV only, AV era only, and never after the window ────
    cutoff = months.max() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    with engine.connect() as conn:
        arts = pd.read_sql_query(
            text(
                "SELECT ticker, published_at, source, finbert_score "
                "FROM news_sentiment_scores "
                "WHERE source = :src "
                "  AND published_at >= :era "
                "  AND published_at <= :cutoff"
            ),
            conn,
            params={"src": SOURCE, "era": AV_ERA_START,
                    "cutoff": cutoff.strftime("%Y-%m-%d %H:%M:%S")},
        )

    # Assert the filter caught exactly what REV 2 says it should.
    got = set(arts["source"].unique())
    assert got <= {SOURCE}, f"source filter leaked: {got - {SOURCE}}"
    n_fnspid = int((arts["source"] == "fnspid").sum())
    assert n_fnspid == 0, f"expected 0 fnspid rows, got {n_fnspid}"
    logger.info(
        f"News corpus after filter: {len(arts):,} rows, sources={sorted(got)}, "
        f"tickers={arts['ticker'].nunique()}"
    )

    arts["published_at"] = pd.to_datetime(arts["published_at"])
    # .dt.normalize() is load-bearing: MonthEnd(0) PRESERVES time-of-day, so
    # 2026-07-07 15:38:54 becomes 2026-07-31 15:38:54, which does not equal the
    # midnight month-end spine and would silently drop every article not
    # published at exactly 00:00:00 on a month end.
    arts["date"] = (arts["published_at"] + pd.offsets.MonthEnd(0)).dt.normalize()

    # Explicit point-in-time guard: no article may inform its own month
    # from beyond that month's end. (Implied by the bucketing, asserted
    # anyway so the invariant is checked rather than assumed.)
    assert (arts["published_at"] <= arts["date"]
            + pd.Timedelta(hours=23, minutes=59, seconds=59)).all()
    # Guard the bucketing itself: every article must land in its own month.
    assert (arts["date"].dt.to_period("M")
            == arts["published_at"].dt.to_period("M")).all(), "month bucketing drifted"

    tickers = _universe()
    spine = pd.MultiIndex.from_product([months, tickers], names=["date", "ticker"])

    # ── 1. raw + n ────────────────────────────────────────────────────
    g = (arts.groupby(["date", "ticker"])["finbert_score"]
              .agg(raw="mean", n="count"))
    panel = g.reindex(spine)
    panel["n"] = panel["n"].fillna(0).astype(int)

    # ── 1b. R2 GOOG -> GOOGL mirror (ruled by Aman, 2026-09-09) ───────
    # Alpha Vantage is fetched for GOOG only (news_pipeline._av_ticker_set
    # excludes GOOGL), so without this GOOGL would carry zero articles in every
    # month and ride the shrinkage prior forever.
    #
    # The mirror copies n and raw, and it happens HERE — after step 1, before
    # xsec(m) — so Alphabet contributes TWO rows to the equal-weighted
    # cross-sectional mean and to the z-score moments. That is intended, not an
    # oversight: the feature matrix already carries GOOG and GOOGL as two
    # distinct securities (separate prices, separate quant factors, labels that
    # disagree in 4 of 131 months), and all 12 fundamental columns — including
    # finbert_score and lm_sentiment_score, the two columns news_sentiment was
    # built to sit alongside on the same scale — already give Alphabet two
    # identical votes. Mirroring at the article level instead would collapse it
    # to one vote and make news the single column treating Alphabet differently.
    #
    # universe.csv and the spine are deliberately untouched: universe.csv is the
    # canonical guard spine for the Prompt 0 benchmark equality check.
    _MIRROR_SRC, _MIRROR_DST = "GOOG", "GOOGL"
    panel = panel.reset_index()
    _src = panel.loc[panel["ticker"] == _MIRROR_SRC, ["date", "raw", "n"]]
    if len(_src):
        _m = panel["ticker"] == _MIRROR_DST
        _aligned = panel.loc[_m, ["date"]].merge(_src, on="date", how="left")
        panel.loc[_m, "raw"] = _aligned["raw"].values
        panel.loc[_m, "n"] = _aligned["n"].values
        _a = panel.loc[panel["ticker"] == _MIRROR_SRC].set_index("date")[["n", "raw"]]
        _b = panel.loc[_m].set_index("date")[["n", "raw"]]
        assert _a["n"].equals(_b["n"]), "GOOG->GOOGL mirror: n diverged"
        assert _a["raw"].equals(_b["raw"]), "GOOG->GOOGL mirror: raw diverged"
        logger.info(
            f"  R2 mirror {_MIRROR_SRC} -> {_MIRROR_DST}: "
            f"{int(_a['n'].sum()):,} articles copied across "
            f"{int((_a['n'] > 0).sum())} months"
        )
    panel = panel.set_index(["date", "ticker"])

    # ── 2. xsec(m): EQUAL weight per ticker, tickers with n >= 1 only ─
    covered = panel[panel["n"] >= 1]
    xsec = covered.groupby("date")["raw"].mean()
    # Months with no articles at all (the whole pre-2022 span) have no
    # cross-section to read; 0.0 is the neutral point of the FinBERT scale.
    xsec = xsec.reindex(months).fillna(0.0)
    panel = panel.join(xsec.rename("xsec"), on="date")

    # ── 3. shrinkage, with the n == 0 branch made explicit ────────────
    n = panel["n"].to_numpy(dtype=float)
    raw = panel["raw"].to_numpy(dtype=float)
    xs = panel["xsec"].to_numpy(dtype=float)
    w = n / (n + SHRINK_K)
    shrunk = np.where(n == 0, xs, w * raw + (1.0 - w) * xs)
    panel["w_own"] = w
    panel["shrunk"] = shrunk
    assert not np.isnan(shrunk).any(), "shrinkage emitted NaN"

    # ── 4. EWM over months, per ticker (shrink first, smooth second) ──
    panel = panel.reset_index().sort_values(["ticker", "date"])
    panel["smoothed"] = (panel.groupby("ticker")["shrunk"]
                              .transform(lambda x: x.ewm(span=EWM_SPAN,
                                                         min_periods=1).mean()))

    # ── 5. R3-A cross-sectional z-score, clipped to ZSCORE_CAP ────────
    grp = panel.groupby("date")["smoothed"]
    mu, sd = grp.transform("mean"), grp.transform("std")
    # A month whose smoothed values are identical (every pre-2022 month, where
    # every ticker fell to xsec) has sd == 0. The deviation is then exactly 0,
    # so the z-score is exactly 0 — not NaN, which is what 0/0 would give.
    z = np.where(sd.to_numpy() > 0,
                 (panel["smoothed"] - mu).to_numpy() / sd.to_numpy().clip(min=1e-12),
                 0.0)
    panel["news_sentiment"] = np.clip(z, -ZSCORE_CAP, ZSCORE_CAP)
    assert not panel["news_sentiment"].isna().any(), "news_sentiment emitted NaN"

    logger.info(
        f"news_sentiment: {len(panel):,} rows, {panel['ticker'].nunique()} tickers, "
        f"{panel['date'].nunique()} months [{panel['date'].min().date()} → "
        f"{panel['date'].max().date()}]"
    )
    return panel


def load_news_sentiment(start: str, end: str, engine=None) -> pd.DataFrame:
    """Monthly news-sentiment factor.

    Returns [ticker, date (month-end), news_sentiment]; never emits NaN.
    """
    panel = _build_panel(start, end, engine)
    if panel.empty:
        return pd.DataFrame(columns=["ticker", "date", "news_sentiment"])
    return (panel[["ticker", "date", "news_sentiment"]]
            .sort_values(["date", "ticker"]).reset_index(drop=True))


def export(start: str, end: str, engine=None) -> pd.DataFrame:
    """Compute and write data/processed/news_sentiment_scores.parquet."""
    panel = _build_panel(start, end, engine)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    panel[["date", "ticker", "n", "raw", "xsec", "w_own", "shrunk",
           "smoothed", "news_sentiment"]].sort_values(
        ["date", "ticker"]).reset_index(drop=True).to_parquet(OUT_PARQUET, index=False)
    logger.info(f"wrote {OUT_PARQUET}")
    return load_news_sentiment(start, end, engine)
