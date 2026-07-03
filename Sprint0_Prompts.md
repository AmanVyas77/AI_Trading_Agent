# Sprint 0 — Baseline Rebuild + Ensemble Wiring Fix
Copy-paste each prompt below into a fresh Cowork/coding-agent session, one at a time, in order. Each prompt is self-contained.

Updated twice after real execution results:
1. Prompt 1's recon found `sentiment_scores` (FinBERT) empty — inserted a backfill prompt.
2. Prompt 2's backfill ran clean (54 min, 53/54 tickers) but returned 99.1% exactly-0.0 scores — almost certainly a text-truncation bug in `clean_filing_text()`, not real neutral sentiment. Inserted a diagnose-and-fix prompt before wiring. Sprint 0 is now 8 prompts.

---

## PROMPT 1 of 8 — Read & Verify (no edits) — COMPLETE

Already run. Confirmed exact column names (`piotroski_f`, `qmj_safety`,
`qmj_payout`, `finbert_score`, `lm_sentiment_score`), all 9 ensemble
pipeline scripts + their `python -m` CLI commands, and that
`sentiment_scores` (FinBERT) was empty while `lm_sentiment_scores` had 429
rows.

---

## PROMPT 2 of 8 — Backfill FinBERT Sentiment — COMPLETE (data quality issue found)

Already run. `python -m src.strategies.fundamental.sentiment_pipeline --cpu-only`
completed in ~54 minutes, scored 3,779/3,780 filings across 53/54 tickers
(CAMT had no 8-Ks in range), 3,688 rows landed in `sentiment_scores`. BUT:
3,655 of 3,688 rows (99.1%) scored exactly 0.0, only 4 positive and 29
negative. This is almost certainly `clean_filing_text()` truncating most
filings down to near-empty content at an early "Safe Harbor
Statement"/"Non-GAAP Financial Measures" marker, not genuine neutral
sentiment. Prompt 3 below fixes this before anything gets wired in.

---

## PROMPT 3 of 8 — Diagnose & Fix FinBERT Truncation, Re-backfill

```
You are continuing Sprint 0 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 3 of 8.
A prior session ran the FinBERT 8-K sentiment pipeline
(src/strategies/fundamental/sentiment_pipeline.py, run via
`python -m src.strategies.fundamental.sentiment_pipeline --cpu-only`) end
to end: 3,779/3,780 filings scored across 53/54 tickers, landing 3,688
rows in the sentiment_scores table. The result is suspicious: 3,655 of
3,688 rows (99.1%) have finbert_score exactly 0.0, with only 4 positive
and 29 negative. Real FinBERT inference on substantive financial text does
not produce exact 0.0 thousands of times in a row — this pattern strongly
suggests a text-truncation bug, not genuinely neutral filings.

Your task — diagnose first, then fix, then re-verify:

1. DIAGNOSE. Find clean_filing_text() (or equivalently named function) in
   src/strategies/fundamental/sentiment_pipeline.py. Read its full
   implementation. It likely searches for marker phrases like "Safe Harbor
   Statement" or "Non-GAAP Financial Measures" and truncates everything
   after the first match — confirm this exact behavior.
2. Pull 5-10 raw 8-K filings from disk (wherever this pipeline reads them
   from) for a few different tickers and run them through
   clean_filing_text() directly in a scratch script. For each one, print:
   raw text length (chars/words), cleaned text length, and the index/
   position in the raw text where truncation occurred. Manually inspect
   whether the truncation point is near the very beginning of the
   document (which would mean almost the entire filing is being discarded)
   or genuinely near the end (which would mean the function is working as
   intended and the 99.1% neutral rate reflects something else, e.g. a
   bug in how finbert_score itself is computed from FinBERT's
   positive/negative/neutral output probabilities — check that
   calculation too).
3. Based on what you find, fix the actual root cause. Likely candidates:
   - The marker search is matching too early (e.g. the phrase appears in
     a cover-page disclaimer near the top of many 8-Ks, not just in a
     trailing boilerplate section) — fix by only truncating after the
     LAST occurrence of these markers, or by requiring the marker to
     appear after some minimum fraction of the document, or by only
     stripping the specific paragraph containing the marker rather than
     everything that follows it.
   - A minimum-length safeguard is missing — if cleaned text falls below
     some reasonable word count (e.g. 50-100 words), that's a sign too
     much was stripped; in that case fall back to scoring more of the
     original text rather than scoring near-empty text.
   - Or: the truncation logic is fine, but finbert_score's calculation
     (however it converts FinBERT's 3-way output into a single score) has
     a separate bug causing it to default to 0.0 too often.
   Fix whichever of these (or something else you find) is the actual root
   cause. Don't guess — base the fix on what you observed in step 2.
4. After fixing, re-run the FinBERT pipeline on the full universe again
   (same command as before) to regenerate sentiment_scores with corrected
   logic. This will take real wall-clock time again (~1 hour on an 8GB
   Mac) — let it run to completion.
5. Re-check the sentiment distribution: query sentiment_scores and report
   the new count of positive / negative / exactly-zero rows. It should
   look meaningfully different from 99.1% zero. If it still looks
   degenerate, say so explicitly rather than declaring success.

CRITICAL RULE: Do not edit config/settings.yaml or .env without explicitly
asking the user first.

Output: what the root cause actually was (with the specific before/after
example from your diagnosis), the code fix you made, and the new
sentiment score distribution after re-running the backfill. State whether
you're confident finbert_score is now a meaningful signal, and if so, that
you're ready for Prompt 4 (wiring it into the ensemble along with
piotroski_f, qmj_safety, qmj_payout, and lm_sentiment_score).
```

---

## PROMPT 4 of 8 — Implement the Wiring

```
You are continuing Sprint 0 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 4 of 8.
Prior sessions confirmed:
- Exact column names: piotroski_f, qmj_safety, qmj_payout (from
  src/strategies/fundamental/xbrl_features.py's compute_piotroski_f(),
  compute_qmj_safety(), compute_qmj_payout()), finbert_score (from
  sentiment_pipeline.py's load_sentiment(), reading sentiment_scores.finbert_score),
  and lm_sentiment_score (from load_lm_scores(), reading
  lm_sentiment_scores.lm_net_score, renamed to lm_sentiment_score).
- sentiment_scores (FinBERT) was backfilled, found to be 99.1% degenerate
  zeros due to a text-truncation bug, then fixed and re-backfilled with a
  healthier score distribution (Prompt 3). lm_sentiment_scores already had
  429 rows and was fine.
- All 9 ensemble pipeline scripts are invoked via
  `python -m src.strategies.ensemble.<script_name>` (e.g.
  `python -m src.strategies.ensemble.model_trainer`).
If you can't confirm any of this, re-read the relevant files now before
proceeding — do not guess column names, and do not proceed if
sentiment_scores still looks degenerate (re-check the distribution first).

CRITICAL RULE: Do not edit config/settings.yaml or .env in this prompt, or
in any future prompt, without explicitly asking the user first — even if a
step below seems to call for it. This rule has been violated before and
caused real problems; do not assume it's fine "just this once." Editing
*.py files to add features or fix bugs is fine and expected. Note:
settings.yaml's fundamental_factors.weights block (piotroski_f_score,
finbert_sentiment, etc.) belongs to the separate Phase 2 hand-weighted
scorer (fundamental_scorer.py) and is NOT used by this ensemble — XGBoost
learns its own feature weights, so this wiring task needs no config
changes at all.

Background:
- The ensemble's feature set lives in src/strategies/ensemble/feature_matrix.py
  (QUANT_COLS, FUND_COLS, MACRO_COLS, FACTOR_COLS, OUTPUT_COLS) and is fed
  by src/strategies/ensemble/factor_export_fundamental.py
  (build_fundamental_scores() and its loader functions).
- FUND_COLS currently has 8 columns: gross_profitability, fcf_yield,
  revenue_acceleration, deferred_revenue_yoy, rd_intensity, sue_score,
  eps_revision_1m, eps_revision_3m. None are sentiment or Piotroski/QMJ.

Your task:
1. In src/strategies/ensemble/factor_export_fundamental.py, add loader
   function(s) that pull piotroski_f, qmj_safety, qmj_payout (reuse the
   existing compute functions in xbrl_features.py — do not re-implement
   the math) and finbert_score + lm_sentiment_score (reuse
   sentiment_pipeline.py's load_sentiment() and load_lm_scores()). Merge
   these onto the existing per-ticker-per-quarter fundamental scores frame
   inside build_fundamental_scores(), using a point-in-time-safe merge (a
   quarter's score must only use data available as of that quarter's
   actual filing date — no lookahead).
2. In src/strategies/ensemble/feature_matrix.py, extend FUND_COLS from 8
   to 13 entries by adding piotroski_f, qmj_safety, qmj_payout,
   finbert_score, lm_sentiment_score (use these exact names). Update
   FACTOR_COLS and OUTPUT_COLS so they pick up the new columns
   automatically if the existing code already derives them from
   FUND_COLS + QUANT_COLS, rather than hardcoding the new names twice.
3. Handle missing data gracefully: tickers/quarters without a value yet
   should get NaN rather than crashing the merge — the existing
   z-scoring step already has MIN_OBS_FOR_ZSCORE guards for this.
4. Don't touch QUANT_COLS, MACRO_COLS, or any other part of the pipeline
   beyond what's needed for this wiring.

Output: the modified files, plus a short note confirming exactly which
new columns were added and where each one's data comes from. State you're
ready for Prompt 5 (re-running the factor export pipeline).
```

---

## PROMPT 5 of 8 — Re-run Factor Export + Feature Matrix

```
You are continuing Sprint 0 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 5 of 8.
A prior session extended FUND_COLS in src/strategies/ensemble/feature_matrix.py
from 8 to 13 columns (added piotroski_f, qmj_safety, qmj_payout,
finbert_score, lm_sentiment_score) and wired their sources into
src/strategies/ensemble/factor_export_fundamental.py. Before that, a
truncation bug in the FinBERT pipeline was found and fixed, and
sentiment_scores was re-backfilled with a healthier distribution. If you
need to confirm any of this, read both .py files and check
SELECT COUNT(*) FROM sentiment_scores now before proceeding.

CRITICAL RULE: Do not edit config/settings.yaml or .env without explicitly
asking the user first.

Your task:
1. Run the quant factor export: `python -m src.strategies.ensemble.factor_export_quant`
2. Run the fundamental factor export: `python -m src.strategies.ensemble.factor_export_fundamental`
   — this should now produce the 5 new columns from Prompt 4.
3. Run the feature matrix builder: `python -m src.strategies.ensemble.feature_matrix`
   to regenerate data/processed/ensemble_feature_matrix.parquet.
4. Load the resulting parquet and report: total row count, date range,
   ticker count, and for each of the 5 new columns specifically — % of
   rows that are NaN, and min/max/mean of non-NaN values. Flag clearly if
   any new column is 100% NaN, or suspiciously degenerate (e.g. >95% of
   values identical) — either case means something needs fixing before
   continuing.
5. Do not proceed to model training in this prompt — that's Prompt 6.

Output: the coverage report from step 4. If any new column is fully or
mostly NaN, or degenerate, stop and explain what's likely wrong rather
than continuing. If coverage looks reasonable, state you're ready for
Prompt 6 (model training + backtest).
```

---

## PROMPT 6 of 8 — Re-run Model Training + Backtest, Save as Baseline

```
You are continuing Sprint 0 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 6 of 8.
A prior session regenerated data/processed/ensemble_feature_matrix.parquet
with 13 fundamental columns instead of 8, and confirmed reasonable data
coverage for the 5 new columns. If you need to confirm, re-load that
parquet now and check it has 13 FUND_COLS-derived columns before proceeding.

CRITICAL RULE: Do not edit config/settings.yaml or .env without explicitly
asking the user first.

IMPORTANT NAMING ISSUE: the backtest runner
(src/strategies/ensemble/backtest.py) always writes its output to these
fixed filenames regardless of context:
  backtests/results/ensemble_sprint3_train_equity.csv
  backtests/results/ensemble_sprint3_test_equity.csv
  backtests/results/ensemble_sprint3_stats.csv
This is legacy naming unrelated to "Sprint 0/1/2/3" in the current plan.
You MUST copy these three files to baseline-namespaced filenames
immediately after the run:
  ensemble_baseline_train_equity.csv
  ensemble_baseline_test_equity.csv
  ensemble_baseline_stats.csv
Also copy data/processed/ensemble_feature_matrix.parquet to
data/processed/ensemble_feature_matrix_baseline.parquet so a later proof
run (Sprint 3) can diff against this exact baseline without it being
overwritten.

Your task:
1. Re-run the full pipeline in order:
   `python -m src.strategies.ensemble.target_builder`
   `python -m src.strategies.ensemble.model_trainer`
   `python -m src.strategies.ensemble.score_generator`
   `python -m src.strategies.ensemble.portfolio_builder`
   `python -m src.strategies.ensemble.backtest`
2. Immediately after the backtest completes, perform the file-copy step
   described above, before anything else touches backtests/results/.
3. Report the headline backtest metrics from ensemble_baseline_stats.csv:
   Total Return, CAGR, Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar
   Ratio — for both train and test periods.
4. Specifically isolate and report the max drawdown during 2022 (peak-to-
   trough within the 2022 subset of the equity curve, not just the
   full-period max drawdown).

Output: the metrics from steps 3 and 4, confirmation the baseline files
were copied to the new filenames, and the full paths to all baseline
files created. State you're ready for Prompt 7 (smoke test).
```

---

## PROMPT 7 of 8 — Smoke Test + Save Baseline Metrics

```
You are continuing Sprint 0 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 7 of 8.
A prior session re-ran the full ensemble pipeline with the new 13-column
fundamental feature set and saved baseline outputs to
ensemble_baseline_{train,test}_equity.csv, ensemble_baseline_stats.csv, and
ensemble_feature_matrix_baseline.parquet. If these files don't exist yet,
stop and say so rather than re-running the pipeline yourself.

Your task — verification only, this is a smoke test, not new development:
1. Load data/processed/ensemble_feature_matrix_baseline.parquet and
   re-confirm the 5 new fundamental columns (piotroski_f, qmj_safety,
   qmj_payout, finbert_score, lm_sentiment_score) are present, not
   entirely NaN, and not degenerate (check value distribution, not just
   NaN rate — finbert_score specifically had a truncation bug earlier in
   this sprint, so re-verify it actually has variation now).
2. Load the trained XGBoost model (or re-run model_trainer.py if the
   model object wasn't persisted) and inspect feature_importances_ (or
   the equivalent attribute). Confirm at least one of the 5 new columns
   has a non-zero importance — doesn't need to be high, just non-zero,
   confirming the model is actually using the new data.
3. Load backtests/results/ensemble_baseline_stats.csv and confirm every
   metric (Total Return, CAGR, Sharpe Ratio, Sortino Ratio, Max Drawdown,
   Calmar Ratio) is a finite number (not NaN, not inf) for both train and
   test periods.
4. Write a new file backtests/results/baseline_metrics.json containing:
   the train and test metrics from step 3, the 2022-specific max drawdown
   number from Prompt 6, the date this baseline was generated, and the
   HEAD git commit hash at the time of this run. This is the reference
   point Sprint 3's proof run will diff against later.

Output: pass/fail for each of the 4 checks, the contents of the new
baseline_metrics.json, and an explicit statement of whether Sprint 0 is
ready to be committed (Prompt 8) or whether something needs fixing first.
```

---

## PROMPT 8 of 8 — Commit + Update Memory

```
You are finishing Sprint 0 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 8 of 8,
the final step. A prior session confirmed the baseline rebuild passed its
smoke test and produced backtests/results/baseline_metrics.json. If that
file doesn't exist or the prior smoke test failed, stop and say so rather
than committing.

CRITICAL RULE: Do not edit config/settings.yaml or .env without explicitly
asking the user first. This sprint should not have required any config
changes — if you find yourself needing one, stop and ask before proceeding.

Your task:
1. Run `git status` and `git diff --stat` to review everything changed
   during Sprint 0 (expected: sentiment_pipeline.py's text-cleaning fix,
   factor_export_fundamental.py, feature_matrix.py, plus new data/result
   files — NOT config/settings.yaml or .env).
2. Stage and commit the code changes
   (src/strategies/fundamental/sentiment_pipeline.py,
   src/strategies/ensemble/factor_export_fundamental.py,
   src/strategies/ensemble/feature_matrix.py) and the new baseline data
   artifacts (ensemble_feature_matrix_baseline.parquet,
   ensemble_baseline_{train,test}_equity.csv, ensemble_baseline_stats.csv,
   baseline_metrics.json) with a clear commit message describing this as
   "Sprint 0: fix FinBERT text-truncation bug, backfill sentiment, wire
   sentiment + Piotroski/QMJ into ensemble feature set, rebuild baseline".
3. Update the project's persistent memory: in particular,
   memory/phase_progress.md (or wherever Session/Sprint history is
   tracked) — add an entry noting Sprint 0 is complete, summarizing what
   was wired in (including the FinBERT truncation bug and its fix, since
   that's exactly the kind of non-obvious thing future sessions need to
   know about), and recording the baseline metrics (CAGR, Sharpe, full max
   drawdown, 2022-specific max drawdown) so future sessions (specifically
   Sprint 3's proof run) have this as ground truth without needing to
   re-derive it.
4. Confirm the working tree is clean after the commit (no uncommitted
   changes related to this sprint left behind).

Output: the commit hash, a summary of what was committed, and confirmation
that memory was updated with the baseline metrics. State clearly that
Sprint 0 is complete and Sprint 1 (regime gate) can begin in a new session.
```
