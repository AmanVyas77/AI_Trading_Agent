# Sprint 2 — Arratia Sentiment Smoothing (EWM, span=4)
Copy-paste each prompt into a fresh Cowork/coding-agent session, one at a time, in order.
Each prompt is self-contained.

Reference: Arratia (2021) — apply exponentially weighted moving average (span=4 quarters)
to per-ticker sentiment scores as a post-processing step before DB upsert, to dampen
noise in time-series sentiment signals.

Verified schemas from Sprint 0 recon (treat as fact, don't re-derive):
- sentiment_scores table: ticker TEXT, filing_date TEXT, period_of_report TEXT,
  finbert_score REAL, num_chunks INT, source TEXT — 3,679 rows
- lm_sentiment_scores table: ticker TEXT, filing_date TEXT, fiscal_year_end TEXT,
  negative_count INT, positive_count INT, uncertainty_count INT, litigious_count INT,
  total_words INT, lm_net_score REAL — 429 rows
- load_sentiment() returns: DataFrame with MultiIndex (ticker, date), column finbert_score
- load_lm_scores() returns: DataFrame with columns [ticker, quarter_end, lm_sentiment_score]

---

## PROMPT 1 of 5 — Read & Verify (no edits)

```
You are picking up the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Sprint 2
(Arratia sentiment smoothing), Prompt 1 of 5.

Sprint 0 wired Piotroski F, QMJ Safety/Payout, FinBERT sentiment, and LM
sentiment into the Phase 3 XGBoost ensemble and rebuilt the baseline
(2022 max drawdown -41.74%). Sprint 1 added the regime gate. Sprint 2
adds EWM smoothing (Arratia 2021, span=4) to FinBERT and LM sentiment
scores before they reach the ensemble — smoothing noisy time-series
sentiment signals is the goal.

This prompt is READ-ONLY reconnaissance — do not edit, create, or delete
any files.

Your tasks:
1. Read src/strategies/fundamental/sentiment_pipeline.py in full. For
   the FinBERT pipeline, record the exact signature and body of:
   - _upsert_score() — how finbert_score gets written to the DB.
   - load_sentiment() — how finbert_score gets read back.
   - run_sentiment_pipeline() — the top-level orchestrator.
   For the LM pipeline, record the exact signature and body of:
   - _upsert_lm_score() — how lm_net_score gets written to the DB.
   - load_lm_scores() — how lm_net_score gets read back.
   - run_lm_scoring_pipeline() — the top-level orchestrator.
2. Search the entire sentiment_pipeline.py for any existing use of
   "ewm", "smooth", "rolling", or "span" — confirm none exists.
3. Check the SQLite DB (data/quant_research.db) — confirm neither
   sentiment_scores nor lm_sentiment_scores has a column ending in
   "_smoothed" yet (run PRAGMA table_info(sentiment_scores) and
   PRAGMA table_info(lm_sentiment_scores)).
4. Search src/strategies/ for examples of idempotent ALTER TABLE
   (adding a column only if it doesn't already exist) used in any
   Session 1-4 migrations — record the exact pattern used so Sprint 2
   can follow the same convention.
5. Check src/strategies/ensemble/factor_export_fundamental.py — confirm
   it calls load_sentiment() and load_lm_scores() and uses the column
   names finbert_score and lm_sentiment_score respectively. (Sprint 0
   added these calls — confirm they're present.)

Output: written summary of all findings. Pay particular attention to
the exact parameters of _upsert_score() and _upsert_lm_score() (we need
to know where in those calls the smoothed value will be substituted), and
the idempotent ALTER TABLE pattern used elsewhere in the codebase. State
you're ready for Prompt 2 (implementation).
```

---

## PROMPT 2 of 5 — Implement EWM Smoothing (read-time, no DB changes)

```
You are continuing Sprint 2 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 2 of 5.
Prompt 1 confirmed no smoothing code exists in sentiment_pipeline.py and
no *_smoothed columns exist in either table. The recon also suggested
that read-time smoothing (applying EWM inside the load functions before
returning) is cleaner than write-time for this use case: no schema
migrations, no backfill, span is easy to tune by changing one constant.
We are using the read-time approach.

CRITICAL RULE: Do not edit config/settings.yaml or .env without explicitly
asking the user first. No config changes are needed for this sprint —
EWM_SPAN = 4 lives as a module-level constant in sentiment_pipeline.py.
Do not add it to settings.yaml unless the user explicitly asks.

Your task: modify src/strategies/fundamental/sentiment_pipeline.py only.
No DB schema changes, no ALTER TABLE, no backfill script.

────────────────────────────────────────────
PART A: Add EWM constant and helper
────────────────────────────────────────────
Near the top of sentiment_pipeline.py, add:
  EWM_SPAN = 4   # Arratia (2021): smooth over ~4 filing periods

Add a private helper _ewm_smooth(df, score_col, span=EWM_SPAN):
  - Input: df with at minimum columns ["ticker", "filing_date", score_col].
    filing_date is a date string or datetime column.
  - Sort by ["ticker", "filing_date"] to guarantee chronological order
    within each ticker before applying EWM.
  - Apply per-ticker EWM:
      df.groupby("ticker")[score_col].transform(
          lambda x: x.ewm(span=span, min_periods=1).mean()
      )
    min_periods=1 ensures tickers with fewer than span filings still get
    a smoothed value (their first filing's smoothed value = raw value).
  - Return the smoothed Series (same index as input df, not a DataFrame).
  - span=4 means ~4 consecutive filings (not calendar time) — a ticker
    with only 2 annual 10-Ks still gets smoothed values, just with less
    history to average over.

────────────────────────────────────────────
PART B: Apply smoothing in load_sentiment()
────────────────────────────────────────────
In load_sentiment() (lines ~528-591), after loading raw rows from
sentiment_scores into a DataFrame but BEFORE the forward-fill / reindex
step that converts per-filing data to daily frequency:
  1. Call _ewm_smooth(df, "finbert_score") to get a smoothed Series.
  2. Replace df["finbert_score"] with the smoothed values in-place.
  3. Continue with the existing forward-fill / reindex logic unchanged.

The returned DataFrame's column is still named finbert_score — callers
(including factor_export_fundamental.py line 241) see no change in
interface, only in values.

────────────────────────────────────────────
PART C: Apply smoothing in load_lm_scores()
────────────────────────────────────────────
In load_lm_scores() (lines ~1151-1235), after loading raw rows from
lm_sentiment_scores into a DataFrame but BEFORE the fiscal_year_end →
quarter_end rounding and column rename:
  1. Call _ewm_smooth(df, "lm_net_score") to get a smoothed Series.
  2. Replace df["lm_net_score"] with the smoothed values in-place.
  3. Continue with the existing rename (lm_net_score → lm_sentiment_score)
     and quarter_end alignment logic unchanged.

The returned DataFrame's column is still named lm_sentiment_score —
factor_export_fundamental.py (line 269) sees no interface change.

Output: the modified sentiment_pipeline.py showing the new EWM_SPAN
constant, _ewm_smooth() function, and the two modified load functions
(show only the changed/added sections, not the full file). State you're
ready for Prompt 3 (verify the smoothing integrates cleanly with the
factor export pipeline).
```

---

## PROMPT 3 of 5 — Integration Verification

```
You are continuing Sprint 2 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 3 of 5.
Prompt 2 (in a prior session) added EWM_SPAN=4 and _ewm_smooth() to
sentiment_pipeline.py, and modified load_sentiment() and load_lm_scores()
to apply EWM smoothing in-place before returning — no DB schema changes
were made. If you need to confirm this, read sentiment_pipeline.py and
check that _ewm_smooth exists and that both load functions call it.

CRITICAL RULE: Do not edit config/settings.yaml or .env without explicitly
asking the user first.

Your task:
1. Call load_sentiment() directly and compare a sample of smoothed vs
   raw values (query the raw finbert_score from sentiment_scores directly
   via SQL for the same ticker/dates). Confirm:
   (a) Smoothed values differ from raw by a measurable amount for
       tickers with multiple filings (EWM is actually being applied).
   (b) For a ticker with only one filing, smoothed == raw (min_periods=1).
   (c) No NaNs where raw values are non-NaN.

2. Call load_lm_scores() and do the same comparison against raw
   lm_net_score values from the DB.

3. Run `python -m src.strategies.ensemble.factor_export_fundamental`
   and confirm it completes without error. Load the output parquet and
   check that finbert_score and lm_sentiment_score columns are present
   with the same NaN rates as Sprint 0 (30.7% and 34.7% respectively).
   The values should now differ from the Sprint 0 baseline parquet
   (smoothing changed the actual numbers), but coverage should be unchanged.

4. Quick sanity check: for a ticker with a large one-quarter outlier in
   finbert_score history (inspect a few tickers from load_sentiment() to
   find one), confirm the smoothed series shows that outlier damped rather
   than passed through at full strength.

Output: results for all 4 tasks with explicit pass/fail. If task 1a fails
(smoothed == raw for multi-filing tickers), read sentiment_pipeline.py
again and check whether _ewm_smooth is actually being called in the load
functions — it may have been added as a helper but not wired in. Fix and
re-check before proceeding. State you're ready for Prompt 4 (smoke test).
```

---

## PROMPT 4 of 5 — Smoke Test

```
You are continuing Sprint 2 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 4 of 5.
Prior sessions modified sentiment_pipeline.py to apply EWM smoothing
(span=4) inside load_sentiment() and load_lm_scores() at read time —
no DB schema changes were made (no new columns, no backfill). If
uncertain, re-read sentiment_pipeline.py and confirm _ewm_smooth() is
called inside both load functions before proceeding.

This prompt is validation only — do not modify code unless you find a
genuine bug (note any fix explicitly).

Your tasks:

1. SYNTHETIC SERIES TEST:
   Build a small synthetic DataFrame for a single fake ticker with 8
   quarterly filings and one obvious outlier (e.g. scores:
   [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1]). Call _ewm_smooth()
   directly and print raw vs smoothed side by side. Confirm:
   (a) The outlier at position 4 is visibly damped (not passed through
       at full strength into subsequent values).
   (b) The first value's smoothed value equals its raw value
       (min_periods=1 guarantee).
   (c) Smoothed values converge back toward ~0.1 within ~4 periods
       after the outlier — consistent with span=4 half-life.

2. LOADER OUTPUT CHECK:
   Call load_sentiment() and load_lm_scores() directly. Confirm:
   (a) Returned columns are still named finbert_score and
       lm_sentiment_score respectively (no interface change).
   (b) Values differ measurably from the raw DB values for at least
       some tickers with multiple filings (query raw finbert_score /
       lm_net_score directly via SQL for the same ticker to compare).
   (c) No NaNs introduced where raw values were non-NaN.

3. FACTOR EXPORT CHECK:
   Run python -m src.strategies.ensemble.factor_export_fundamental and
   confirm it completes without error. Load the resulting parquet and
   verify finbert_score and lm_sentiment_score columns are present with
   the same NaN rates as Sprint 0 (30.7% and 34.7%) — coverage should
   be unchanged, only values differ.

Output: results for all 3 tasks with explicit pass/fail. If task 2b
fails (smoothed == raw for multi-filing tickers), the _ewm_smooth() call
is likely not actually wired into the load functions — re-read both
functions and fix before declaring the smoke test passed. State you're
ready for Prompt 5 (commit).
```

---

## PROMPT 5 of 5 — Commit + Update Memory

```
You are finishing Sprint 2 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 5 of 5.
The prior session's smoke test confirmed EWM smoothing is working:
synthetic outlier test passed, *_smoothed columns exist with zero NULLs,
loaders return smoothed values under the original column names, and
factor_export_fundamental.py runs cleanly with unchanged NaN rates.
If the smoke test did not pass, do not commit — stop and report.

CRITICAL RULE: Do not edit config/settings.yaml or .env without explicitly
asking the user first. Sprint 2 should have touched ONLY:
  - src/strategies/fundamental/sentiment_pipeline.py (modified)
Nothing else. If git status shows anything else changed (especially
settings.yaml, factor_export_fundamental.py, or feature_matrix.py),
stop and note it before committing.

Your tasks:
1. Run `git status` and `git diff --stat`. Confirm only
   sentiment_pipeline.py changed on the code side (plus any DB changes,
   which are not tracked by git). If settings.yaml or .env appear in
   the diff, DO NOT commit — stop and report.
2. Stage and commit with message:
   "Sprint 2: add Arratia (2021) EWM smoothing (span=4) to FinBERT and
   LM sentiment loaders (read-time, no DB schema changes)"
3. Update the project's persistent memory (memory/phase_progress.md).
   Add an entry noting:
   - Sprint 2 complete.
   - EWM_SPAN = 4 added to sentiment_pipeline.py (Arratia 2021).
   - Smoothing applied at read time inside load_sentiment() and
     load_lm_scores() — no DB schema changes, no backfill needed.
   - Downstream consumers (factor_export_fundamental.py → ensemble
     feature matrix) automatically get smoothed finbert_score and
     lm_sentiment_score with no further code changes.
   - Raw scores remain unchanged in the DB; smoothing is re-applied
     on every load call from the stored per-filing raw values.
4. Confirm the working tree is clean after the commit.

5. Print the git push command so the user can push to remote:
   Run `git remote -v` and `git branch --show-current`, then print the
   exact command — e.g. `git push origin main`. Do NOT run the push;
   print it for the user to execute.

Output: commit hash, memory update confirmation, the ready-to-run git
push command, and a statement that Sprint 2 is complete and Sprint 3
(proof run) can begin in a new session.
```
