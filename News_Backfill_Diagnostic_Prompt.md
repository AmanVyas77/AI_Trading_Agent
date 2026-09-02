# Prompt: Verify and fix AlphaVantage news-backfill wrapper path bug

## Context

This is a quant trading research project ("Ai Trading Agent"). One of its data
pipelines backfills historical news/sentiment data from AlphaVantage's
`NEWS_SENTIMENT` API into a local SQLite database, run nightly via macOS
launchd. The project's canonical root directory as of 2026-07-23 is:

```
/Users/aman/Projects/Ai Trading Agent
```

It was previously located at `/Users/aman/Desktop/Stock_Project/Ai Trading Agent`
and was moved because iCloud Desktop-sync was evicting files (including the
live SQLite database), causing disk I/O errors. The old location may still
exist on disk under a renamed folder (something like
`Stock_Project_OLD_DO_NOT_USE`) as a rollback copy, kept temporarily.

## What was found (needs independent verification)

1. **`scripts/av_backfill_wrapper.sh`** (the shell script launchd invokes)
   still hardcodes the *old* stale path:

   ```bash
   PROJECT_DIR="/Users/aman/Desktop/Stock_Project/Ai Trading Agent"
   ```

   It should point at `/Users/aman/Projects/Ai Trading Agent` instead.

2. Despite that stale path in the tracked script, the backfill job has
   clearly still been writing new rows into the *new* location's database
   (`data/quant_research.db`, table `news_articles`, `source='alphavantage'`)
   as recently as 2026-07-24 00:02 and 04:05, per `logs/av_backfill_state.jsonl`.
   This is inconsistent — if `scripts/av_backfill_wrapper.sh` really has the
   old path and `set -eu`, `cd` into a nonexistent (or stale) directory should
   fail before python ever runs.

3. **`logs/av_backfill_launchd.log`** — the file the wrapper's own `echo`
   line writes to — has not been updated since 2026-07-17, even though the
   job has clearly fired multiple times since (per the state jsonl and the
   database row counts). This means whatever is actually executing right now
   is not sending stdout to this tracked log file, OR is a different
   copy/version of the wrapper/plist than what's checked into git.

## What to figure out

Please investigate and report back on:

- Where the *actual* live launchd plist lives (likely
  `~/Library/LaunchAgents/*.plist`) and what `ProgramArguments` / working
  directory / script path it currently points to. Confirm whether it matches
  the tracked `scripts/av_backfill_wrapper.sh` or a different, unversioned
  copy (e.g. one still sitting in the old Desktop location, or one that was
  hand-edited outside of git).
- Whether `/Users/aman/Desktop/Stock_Project/Ai Trading Agent` (the exact
  path, without any `_OLD_DO_NOT_USE` suffix) still exists on disk, and if
  so, whether IT has its own separate `data/quant_research.db` that's
  actually the one being written to (which would mean the "new" location's
  db and the "old" location's db have diverged, and the row counts we've been
  checking under `/Users/aman/Projects/Ai Trading Agent` may not reflect the
  authoritative live data at all).
- Why `logs/av_backfill_launchd.log` stopped updating on 2026-07-17 while
  `logs/av_backfill_state.jsonl` kept getting new entries through
  2026-07-24 — i.e., where is the currently-firing job's stdout actually
  going?

## What to fix

Once the live plist/wrapper location is confirmed:

- Update `scripts/av_backfill_wrapper.sh` in the tracked repo
  (`/Users/aman/Projects/Ai Trading Agent/scripts/av_backfill_wrapper.sh`) so
  `PROJECT_DIR` points at `/Users/aman/Projects/Ai Trading Agent`.
- If the live plist points somewhere else (e.g. still at the old Desktop
  path, or at an untracked copy of the wrapper), update the plist so it
  points at the tracked script under the new project root, and confirm
  `logs/av_backfill_launchd.log` starts receiving fresh entries on the next
  fire.
- If a second, divergent `quant_research.db` is found at the old Desktop
  location, do NOT merge or delete anything automatically — report exactly
  what's in it (row counts, most recent timestamps per source) so a human
  can decide how to reconcile it with the new location's database.

## Constraints

- Do not delete any files or directories without explicit confirmation.
- Do not run `git commit` — if changes are made to tracked files, just leave
  them staged/unstaged and describe what changed; commits in this repo are
  always done by the project owner from their own terminal.
- AlphaVantage's free-tier limit is 25 requests/day; the backfill script
  budgets 20 requests/day with 5 held as headroom (`AV_DAILY_BUDGET = 20` in
  `src/data/news_pipeline.py`). This is expected/by-design — do not treat
  `cap_hit_early: true` in the state log as a bug by itself.
