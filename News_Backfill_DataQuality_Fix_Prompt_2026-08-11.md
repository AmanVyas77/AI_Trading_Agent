# Prompt: Fix AV news-backfill data-quality gaps found after BACKFILL COMPLETE

## Context

Quant trading project ("Ai Trading Agent"), currently at
`/Users/aman/dev/Ai Trading Agent`. A nightly launchd job
(`scripts/av_backfill_wrapper.sh` → `src.data.news_pipeline av-backfill`)
has been backfilling AlphaVantage `NEWS_SENTIMENT` data into
`data/quant_research.db` (table `news_articles`, source='alphavantage').
The launchd reliability problem (stale hardcoded paths breaking the job on
every project move) was fixed on 2026-07-28 and has held through a 4th
project move on 2026-08-03 — that part is solid, not what this prompt is
about.

The backfill loop itself finished on 2026-08-11 04:05:05
(`av_backfill_state.jsonl` logs `BACKFILL COMPLETE`, all 288 ticker/period
windows attempted). But "loop finished" does not mean "data is clean" — two
real problems were found by querying the database directly after
completion.

## Problem 1: 1000-item truncation on high-volume 2026 windows

AlphaVantage's `NEWS_SENTIMENT` endpoint caps each call at 1000 items
(`AV_ITEM_CAP` in `src/data/news_pipeline.py`). The script already has
half-year splitting for known high-density tickers
(`_maybe_split()`/`AV_SPLIT_THRESHOLD = 950`), and a `WARNING` log line
fires when a window returns ≥950 items ("may be truncated; consider
quarterly split") — but nothing automatically re-splits and re-fetches when
that warning fires. Confirmed truncated (2026-window row counts sitting at
or near 1000): MSFT (991), MU (998), QCOM (993), ORCL (1000). There may be
others among the 53-ticker universe not yet checked.

**Fix:** for every ticker whose most recent (2026 YTD) window is at or near
the 1000-item cap, re-fetch that window split into quarters (or months if a
quarter also hits the cap) instead of a single year/half-year call, so no
items are silently dropped. Use `INSERT OR IGNORE` (already the pattern via
`article_hash`) so re-fetching is safe/idempotent against already-inserted
rows. After re-fetching, verify row counts are no longer sitting exactly at
a round-number ceiling.

## Problem 2: NVDA has a 10-month total blackout, not just truncation — and a logging bug is hiding this class of failure

NVDA is the single highest-volume ticker in the universe (11,043 FNSPID
rows historically). Its most recent AlphaVantage article is dated
**2025-10-13** — nothing since, despite `news_ingest_log` showing the
`2026-01..2026-08` window as `completed_at = 2026-08-06 04:49:18` with
`items_fetched = 6835`.

That `items_fetched` number is misleading — trace through `av_backfill()`
in `news_pipeline.py`: the `INSERT OR REPLACE INTO news_ingest_log` call
logs the `added` variable, which is accumulated across **every window
processed in that entire nightly run**, not reset per ticker/period. NVDA's
logged value (6835) is identical to the prior window's cumulative total,
meaning this specific call almost certainly added zero new rows to
`news_articles` — but there's no way to tell that from the log as currently
written.

**Investigate first, don't guess:**
1. Find NVDA's actual API response for the `2026-01..2026-08` window. Check
   `~/Library/Logs/aitrading/av_backfill.log` for log lines around
   2026-08-06 04:15:54–04:49:18 ET (the window before and after NVDA in
   ticker order should bound it). If nothing useful is there, re-run just
   that one window manually (a small script or REPL call to `_av_fetch`)
   and inspect the raw JSON — specifically the `time_published` field
   format on a few items.
2. `_av_persist()` in `news_pipeline.py` silently drops any item where
   `datetime.strptime(tp, "%Y%m%dT%H%M%S")` raises `ValueError` (bare
   `except: continue`, no logging of what was dropped or why). If NVDA's
   2026 response has timestamps in an unexpected format, or the `feed` key
   was empty/malformed for another reason (e.g. a throttle response that
   wasn't caught by the existing `throttled` check), that would explain a
   silent full-window loss.
3. Confirm the actual root cause before fixing it blind.

**Fix:**
1. Re-fetch NVDA's 2026 window (and re-check 2022-2025 windows too, in case
   the same failure mode hit other periods — the `_maybe_split` warnings
   suggest 2025 may also be near/at the cap based on FNSPID's volume
   pattern, worth checking even though it wasn't flagged here).
2. Fix the `items_fetched` logging bug: log the per-window delta (items
   actually inserted for *this* ticker/period), not the run-cumulative
   `added` total. This is what let the NVDA blackout go undetected —
   without this fix, the same silent-failure class will hit some other
   ticker next time and nobody will notice from the logs alone.
3. Also check SWKS and TWLO (zero AlphaVantage rows for either, still
   unexplained after the full backfill) — likely the same silent-drop
   mechanism, or these tickers may simply have no AV coverage; determine
   which and report.

## Constraints

- Do not delete any files without explicit confirmation.
- Do not run `git commit` — describe changes, leave them staged/unstaged
  for the project owner to commit from their own terminal.
- Don't fix blind — Problem 2 in particular needs the actual API response
  or log evidence before deciding what's wrong, not just a plausible-sounding
  guess.
- After fixes, report concrete before/after row counts per affected ticker
  (query `news_articles` directly), not just "should be fixed now."
