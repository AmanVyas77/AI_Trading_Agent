#!/usr/bin/env python
"""Score every row of `news_articles` with the existing FinBERTScorer.

Why a separate table
--------------------
`sentiment_scores` is keyed (ticker, filing_date) and is 100% sec_8k today. It
physically cannot hold multiple articles per ticker per day, and mixing news
into it would silently corrupt the existing `finbert_score` feature. News lands
in `news_sentiment_scores`, keyed by `article_hash`.

Immutability
------------
Rows are written with INSERT OR IGNORE, never INSERT OR REPLACE. Once an
article_hash has a score, that score is frozen. A re-score must not rewrite
history: the moment news feeds the ensemble, a rewritten past score is a silent
lookahead. A model change means a new `model_name` and a new table, not an
overwrite.

Scored text
-----------
`title + ". " + snippet` when a snippet exists, else `title` alone.
`text_shape` records which, because Prompt 2 must test whether text shape —
rather than `source` — drives the seam between FNSPID and Alpha Vantage.

Resumability
------------
Work is selected by LEFT JOIN ... WHERE nss.article_hash IS NULL and committed
per batch, so the job can be killed and restarted at any point with no loss and
no double-scoring.

Usage
-----
    .venv/bin/python scripts/score_news_sentiment.py --max-rows 500   # probe
    .venv/bin/python scripts/score_news_sentiment.py                  # full
    .venv/bin/python scripts/score_news_sentiment.py --cpu-only
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strategies.fundamental.sentiment_pipeline import (  # noqa: E402
    FinBERTScorer, MODEL_NAME, DB_PATH,
)

logger = logging.getLogger("score_news")

DDL = """
CREATE TABLE IF NOT EXISTS news_sentiment_scores (
    article_hash   TEXT PRIMARY KEY,
    ticker         TEXT NOT NULL,
    published_at   TEXT NOT NULL,
    source         TEXT NOT NULL,
    finbert_score  REAL NOT NULL,
    text_shape     TEXT NOT NULL,
    model_name     TEXT NOT NULL,
    scored_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_nss_ticker_pub
    ON news_sentiment_scores(ticker, published_at);
"""

# Work is drained one (source, has-snippet) class at a time, ordered by text
# length. This is a throughput fix, not a semantic one: MPS compiles a fresh
# Metal kernel for every distinct padded batch width, so the original
# "fetch 2,000 arbitrary rows, sort within the chunk" produced ~63 distinct
# shapes per chunk and spent most of its time recompiling — the 1.5 rows/s
# collapse that tripped the ETA gate. Draining a length-homogeneous class
# reuses one compiled shape across many batches.
FETCH_SQL = """
SELECT na.article_hash, na.ticker, na.published_at, na.source,
       na.title, na.snippet
FROM news_articles na
LEFT JOIN news_sentiment_scores nss ON nss.article_hash = na.article_hash
WHERE nss.article_hash IS NULL
  AND na.source = :source
  AND (CASE WHEN na.snippet IS NULL OR TRIM(na.snippet) = '' THEN 0 ELSE 1 END) = :has_snip
ORDER BY LENGTH(COALESCE(na.title,'')) + LENGTH(COALESCE(na.snippet,''))
LIMIT :lim
"""

CLASSES_SQL = """
SELECT na.source,
       CASE WHEN na.snippet IS NULL OR TRIM(na.snippet) = '' THEN 0 ELSE 1 END AS has_snip,
       COUNT(*)
FROM news_articles na
LEFT JOIN news_sentiment_scores nss ON nss.article_hash = na.article_hash
WHERE nss.article_hash IS NULL
GROUP BY 1, 2
ORDER BY 3 DESC
"""

INSERT_SQL = """
INSERT OR IGNORE INTO news_sentiment_scores
    (article_hash, ticker, published_at, source,
     finbert_score, text_shape, model_name, scored_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""


def build_text(title: str | None, snippet: str | None) -> tuple[str, str]:
    """Return (text_to_score, text_shape)."""
    t = (title or "").strip()
    s = (snippet or "").strip()
    if s:
        return f"{t}. {s}", "title_snippet"
    return t, "title_only"


def _rss_mb() -> float:
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024 * 1024)  # macOS reports bytes
    except Exception:
        return float("nan")


def _fmt_eta(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-only", action="store_true",
                    help="force CPU (MPS is flaky for some transformer ops)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=2000,
                    help="rows fetched + committed per transaction")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="stop after N rows (probe mode)")
    ap.add_argument("--eta-abort-hours", type=float, default=8.0,
                    help="abort if the extrapolated full run exceeds this")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    con = sqlite3.connect(str(DB_PATH), timeout=60.0)
    con.executescript(DDL)
    con.commit()

    total = con.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]
    done = con.execute("SELECT COUNT(*) FROM news_sentiment_scores").fetchone()[0]
    remaining = total - done
    logger.info(f"news_articles={total:,}  already scored={done:,}  remaining={remaining:,}")
    if remaining <= 0:
        logger.info("nothing to do")
        return

    scorer = FinBERTScorer(cpu_only=args.cpu_only)
    scorer._load()

    t0 = time.time()
    n_done = 0
    reported_500 = False

    classes = con.execute(CLASSES_SQL).fetchall()
    logger.info("remaining work by class (drained in this order):")
    for src, has_snip, n in classes:
        logger.info(f"  {src:14s} {'title_snippet' if has_snip else 'title_only':14s} {n:>8,}")

    ci = 0
    while ci < len(classes):
        if args.max_rows is not None and n_done >= args.max_rows:
            break

        src, has_snip, _ = classes[ci]
        want = args.chunk
        if args.max_rows is not None:
            want = min(want, args.max_rows - n_done)

        rows = con.execute(
            FETCH_SQL, {"source": src, "has_snip": has_snip, "lim": want}
        ).fetchall()
        if not rows:
            ci += 1          # class drained; move to the next
            continue

        prepared = []
        for h, tic, pub, src, title, snip in rows:
            txt, shape = build_text(title, snip)
            prepared.append((h, tic, pub, src, txt, shape))

        scores = scorer.score_texts([p[4] for p in prepared],
                                    batch_size=args.batch_size)
        stamp = datetime.now(timezone.utc).isoformat()
        con.executemany(INSERT_SQL, [
            (p[0], p[1], p[2], p[3], float(sc), p[5], MODEL_NAME, stamp)
            for p, sc in zip(prepared, scores)
        ])
        con.commit()

        n_done += len(rows)
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0.0

        if not reported_500 and n_done >= 500:
            reported_500 = True
            eta = remaining / rate if rate else float("inf")
            logger.info("  NOTE: rate is class-dependent (long FNSPID "
                        "title_snippet rows are ~4x slower than AV); this ETA "
                        "reflects the class being drained right now.")
            logger.info("=" * 68)
            logger.info(f"THROUGHPUT GATE after {n_done:,} rows")
            logger.info(f"  elapsed        : {elapsed:.1f}s")
            logger.info(f"  rate           : {rate:.1f} rows/s")
            logger.info(f"  remaining      : {remaining:,} rows")
            logger.info(f"  full-run ETA   : {_fmt_eta(eta)}")
            logger.info(f"  peak RSS       : {_rss_mb():.0f} MB")
            logger.info("=" * 68)
            if eta > args.eta_abort_hours * 3600:
                logger.error(
                    f"ABORT: ETA {_fmt_eta(eta)} exceeds the "
                    f"{args.eta_abort_hours}h budget — stopping rather than "
                    "running blind. Re-run with --cpu-only or a larger "
                    "--batch-size, or reduce scope.")
                con.close()
                sys.exit(2)

        if n_done % 20000 < args.chunk:
            eta = (remaining - n_done) / rate if rate else 0
            logger.info(f"  {n_done:,}/{remaining:,}  {rate:.1f} rows/s  "
                        f"ETA {_fmt_eta(eta)}  RSS {_rss_mb():.0f} MB")

    elapsed = time.time() - t0
    final = con.execute("SELECT COUNT(*) FROM news_sentiment_scores").fetchone()[0]
    logger.info(f"scored {n_done:,} rows this run in {_fmt_eta(elapsed)} "
                f"({n_done/elapsed:.1f} rows/s); table now {final:,}/{total:,}")
    con.close()


if __name__ == "__main__":
    main()
