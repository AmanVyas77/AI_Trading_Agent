# Sprint 7 — Data Freshness + 2025-26 True-Holdout Validation (FROZEN model)
Copy-paste each prompt into a fresh Cowork/coding-agent session, one at a time, in order.
Each prompt is self-contained.

Sprint 7 does two things that share one plumbing effort:
1. Brings every data feed current (prices, macro, FinBERT backlog, TimesFM month-ends) —
   this is also the prerequisite work for the Sprint 8 live paper-trading pipeline.
2. Runs the FROZEN Sprint 5 production model over 2025-01 → 2026-06 as a TRUE
   out-of-sample holdout. Every design decision in Sprints 0-6 used data ≤ 2024-11,
   so this window is genuinely unseen. It is the strongest validation evidence this
   project can produce.

⚠️ BURN-ONCE RULE (the whole point of this sprint — read twice):
The 2025-26 window is a one-shot holdout. ONE run, ONE verdict against the
pre-committed rule below. NO iteration against holdout results: no retraining, no
parameter changes, no feature changes, no threshold tweaks in response to what the
holdout shows. Data BUGS found mid-run (wrong dates, missing rows, join errors) may be
fixed and the run repeated; anything that changes model behavior may not. If the verdict
is FAIL, that is recorded as a finding — the response happens in a FUTURE sprint, not
by re-running this one.

PRE-COMMITTED VERDICT RULE (decided 2026-07-04, before any holdout computation):
  PASS  = holdout Sharpe (2025-01-01 → 2026-06-30, daily, mean/std·√252) > 0.600
          (the Sprint 0 full-period baseline)
  Everything else = FAIL.
  Informational only (report, but NOT part of the rule): comparison vs the 2023-24
  test Sharpe 0.990, vs SPY buy-and-hold Sharpe over the same window, holdout CAGR,
  max DD, monthly breadth, regime-label distribution.

Verified context (treat as fact, do not re-derive; verified 2026-07-03/04):
- Repo: TWO NESTED GIT REPOS. Authoritative = INNER repo rooted at "Ai Trading Agent/".
  Verify with git rev-parse --show-toplevel (must end "Ai Trading Agent") before ANY
  git command. NEVER push/commit/PR from the outer Stock_Project/ repo.
  Chain: 65bc3bb (Sprint 5) → 5c367c6 (catch-up) → 13a2b6c (Sprint 6 REFUTED).
  Working tree may also carry uncommitted Dashboard-v2 files (src/dashboard/index_v2.html,
  dashboard_v2.js, export_data.py edits, memory/future_ideas.md edits, Sprint*_Prompts.md)
  — leave them alone; they are a separate pending commit.
- Production model: models/ensemble_models.pkl = Sprint 5 expanding-window walk-forward,
  78 folds, last fold test month 2024-11, last fold effective_train_end 2024-07-31.
  Backup exists at models/ensemble_models_sprint5.pkl (byte-identical, 4,120,378 B).
  THE MODEL IS NOT RETRAINED IN THIS SPRINT — model_trainer is not run at all.
- Data freshness (as of 2026-07-03): prices table → 2026-05-08; macro_series → 2026-05
  (cpi 2026-03, monthly cadence); lm_sentiment_scores → 2026-06-05;
  eps_revisions → 2026-04-03; sentiment_scores (FinBERT 8-K) → **2024-12-27 (STALE —
  18 months behind; this backlog is Prompt 2's main job)**.
- settings.yaml TIMELINE clamps: train_start 2015-01-01, test_end 2024-12-31,
  live_start 2025-01-01. Several loaders hardcode TIMELINE["test_end"] as their end
  date — the holdout needs end-date overrides. CRITICAL RULE: never edit
  config/settings.yaml or .env. All overrides are module-level constants, function
  parameters, or CLI args.
- score_generator.py CANNOT score the holdout: it scores only months present in
  ensemble_labeled.parquet (labels need forward returns). The holdout months are
  scored by a NEW scorer (src/live/scorer.py) that reads the FEATURE MATRIX directly
  and applies the LAST fold's model. This scorer is deliberately dual-purpose: it is
  also the live-mode scorer Sprint 8 (paper trading) needs.
- factor_export_quant.build_quant_scores(start, end) and
  feature_matrix.build_feature_matrix(start, end) already accept date overrides via
  CLI (--start/--end). factor_export_quant step 3.5 auto-runs TimesFM for every
  month-end in range (~18 new batched calls for 2025-01→2026-06, minutes on CPU;
  full-range re-run ~138 calls ≈ 5 min — acceptable).
- TimesFM caveat for the writeup: the pretrained checkpoint's training-data vintage is
  a possible subtle leak channel for 2025 — acknowledge in verdict notes, don't overclaim.
- Machine: 8 GB Mac. FinBERT runs CPU-batched; Ollama is FORBIDDEN on this machine
  (kernel panics). Use .venv/bin/python for all project commands.

Pipeline commands (python -m …): src.data.quant_pipeline |
src.strategies.fundamental.sentiment_pipeline | src.strategies.ensemble.factor_export_quant |
src.strategies.ensemble.factor_export_fundamental | src.strategies.ensemble.feature_matrix

---

## PROMPT 1 of 5 — Recon (read-only)

```
You are picking up the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Sprint 7
(data freshness + 2025-26 true-holdout validation), Prompt 1 of 5.
READ-ONLY — no edits, no installs, no pipeline runs.

REPO GUARD: cd to the project root; git rev-parse --show-toplevel must
end in "Ai Trading Agent" (inner repo). git log --oneline -3 should show
13a2b6c (Sprint 6) → 5c367c6 → 65bc3bb. Uncommitted Dashboard-v2 files
in the working tree are expected — list them but leave them alone.

Use .venv/bin/python throughout.

1. Data freshness audit — query data/quant_research.db and report max
   dates + row counts for: prices (per a few tickers AND overall),
   macro_series (per series), sentiment_scores (FinBERT), 
   lm_sentiment_scores, eps_revisions, edgar_10k_filings. Confirm or
   correct: prices → 2026-05-08, FinBERT → 2024-12-27 (stale).

2. FinBERT backlog sizing — count 8-K filings that would need collecting
   and scoring for 2025-01 → today across the 54-ticker universe.
   Inspect src/strategies/fundamental/sentiment_pipeline.py: identify
   the collection entry point (form types, date filtering), the scoring
   entry point, whether it processes only unscored rows (incremental)
   or everything, and estimate runtime on CPU. Report the plan — do not
   run it.

3. TIMELINE clamp inventory — grep for TIMELINE usage across
   src/strategies/ensemble/*.py and src/data/*.py. For each file, record
   whether the holdout can already be reached via function/CLI args
   (factor_export_quant, feature_matrix — expected YES) or whether the
   loader hardcodes TIMELINE["test_end"] and needs an override
   (portfolio_builder._load_prices_wide ~line 99, backtest._load_prices
   ~line 68, others you find). Quote exact lines.

4. Frozen model check — load models/ensemble_models.pkl: confirm 78
   folds, last fold date 2024-11-30, effective_train_end 2024-07-31,
   feature_names length 23 and the exact ordered list (the holdout
   scorer must reproduce this order exactly). Also confirm the
   models/ensemble_models_sprint5.pkl backup still exists and matches.

5. Feature-matrix readiness — read feature_matrix.py QUANT_COLS /
   FUND_COLS / MACRO_COLS and the NaN handling (fillna policy is in
   model_trainer._prepare_xy: fillna(0.0)). Confirm
   factor_export_fundamental has (or lacks) date-range args — quote its
   CLI/main signature.

6. Regime gate range check — get_historical_regime_multipliers(start,end)
   takes explicit dates (regime_gate.py ~line 99): confirm it will span
   2025-2026 given macro_series content, and note the CPI z-score
   window implications in feature_matrix._load_macro (36-month rolling
   — verify enough macro history exists so 2025-26 rows are not NaN).

Output: freshness table, backlog size + runtime estimate, clamp
inventory with line numbers, the exact 23-feature ordered list, and any
blocker that would prevent Prompt 2/3 from proceeding. State you're
ready for Prompt 2.
```

---

## PROMPT 2 of 5 — Data refresh (prices, macro, FinBERT backlog)

```
You are continuing Sprint 7 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. Prompt 2 of 5.
Prompt 1 (prior session) audited freshness: prices → 2026-05-08, macro
→ 2026-05, FinBERT sentiment_scores → 2024-12-27 (18-month backlog).
REPO GUARD first (inner repo — toplevel ends "Ai Trading Agent").
Use .venv/bin/python. No code edits in this prompt except where a
pipeline hard-fails and needs a minimal fix (document any such fix).

CRITICAL RULE: never edit config/settings.yaml or .env.

1. Prices + macro refresh:
   .venv/bin/python -m src.data.quant_pipeline
   Then verify: prices max date is within a few trading days of today
   (2026-07-04); every macro series advanced (vix/fed_funds/usd/hy
   ~daily; cpi/industrial_production monthly cadence is normal).
   Report before/after max dates per series.

2. FinBERT 8-K backlog (the long pole — budget accordingly):
   Run the collection pipeline for the missing window per Prompt 1's
   plan, then the scoring pipeline. Process incrementally (only
   unscored rows). On this 8 GB machine: keep FinBERT batch sizes
   small, run tickers in chunks, and checkpoint (the pipeline upserts —
   re-running after an interruption must be safe; verify it is before
   relying on it). If total runtime is projected over ~2 hours, do the
   2025-01→2026-07 window in two chunks and report progress after each.
   After scoring: report sentiment_scores max filing_date, rows added,
   and per-year coverage counts (2025, 2026) vs 2024 as a sanity ratio.

3. LM 10-K refresh (should be near-current already — 2026-06-05):
   run its pipeline only if Prompt 1 found missing filings; otherwise
   just confirm freshness.

4. XLK cache: data/raw/xlk_monthly.csv currently ends ~2025-01. The
   holdout backtest benchmarks against SPY (downloaded on the fly) and
   labels are NOT needed for holdout scoring, so XLK is NOT a blocker —
   but refresh the cache anyway for future use: delete the cache file
   and let target_builder._load_xlk_monthly re-download on next use, OR
   extend it manually via yfinance to today. Verify the new range.

5. Final freshness table (same format as Prompt 1) proving all feeds
   are current. Also run: git status --short — confirm NOTHING tracked
   changed in this prompt beyond (possibly) a documented minimal fix;
   the DB and caches are gitignored.

Output: before/after freshness table, FinBERT rows added + coverage
ratio, any pipeline fixes made (exact diff), runtime notes for the
Sprint 8 live schedule. State you're ready for Prompt 3.
```

---

## PROMPT 3 of 5 — Holdout plumbing: feature matrix to 2026 + src/live/scorer.py

```
You are continuing Sprint 7 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. Prompt 3 of 5.
Prompt 2 (prior session) brought all data feeds current. REPO GUARD
first (inner repo). Use .venv/bin/python.

CRITICAL RULES: never edit config/settings.yaml or .env. Do NOT run or
modify model_trainer.py — the model stays frozen. Do NOT touch the
tracked Sprint 5/6 result files.

PART A — Rebuild factors + feature matrix through the holdout end:
1. .venv/bin/python -m src.strategies.ensemble.factor_export_quant \
     --start 2015-01-01 --end 2026-06-30
   (TimesFM step runs for every month-end in range — ~138 batched calls,
   ≈5 min CPU. Watch for per-month "forecast OK" logs; any failed month
   yields NaN and must be reported.)
2. Fundamental export: use the date-range mechanism Prompt 1 identified
   for factor_export_fundamental (add a minimal --end passthrough ONLY
   if Prompt 1 confirmed it lacks one — module-level default, not yaml).
3. .venv/bin/python -m src.strategies.ensemble.feature_matrix \
     --start 2015-01-01 --end 2026-06-30
4. Verify the new ensemble_feature_matrix.parquet: months through
   2026-06-30, 23 feature columns in the exact order Prompt 1 recorded,
   and a NaN report for the 2025-01→2026-06 rows per column. FinBERT
   and LM columns must show real (non-ffilled-from-2024) values in 2025+
   — spot-check 3 tickers. Macro columns must be non-NaN (36-month
   z-score window has ample history).

PART B — Create src/live/ (new module, dual-purpose: holdout now,
paper-trading in Sprint 8):
- src/live/__init__.py (empty)
- src/live/scorer.py with:
    FROZEN_MODEL_PATH = models/ensemble_models.pkl  (module constant)
    def score_months(start, end, model_path=None) -> pd.DataFrame
  Logic: load the pickle, take the LAST fold's model + feature_names
  (this is the production model — trained through 2024-07-31); load
  ensemble_feature_matrix.parquet; select rows in [start, end]; build X
  with EXACTLY the pickle's feature_names order; fillna(0.0) (mirror
  model_trainer._prepare_xy); predict_proba[:,1] → ensemble_score;
  return/save [date, ticker, ensemble_score] to
  data/processed/holdout_scores.parquet. Log the model's
  effective_train_end and n_features at runtime as an audit line.
- Do NOT modify score_generator.py.

PART C — Holdout run wiring (overrides, not yaml):
- portfolio_builder.build_portfolio_weights already accepts injected
  scores_df and prices. backtest.run_backtest accepts weights_df. The
  clamped internal loaders must be bypassed by INJECTION, not edited,
  wherever possible. Write scripts/run_holdout.py that:
  1. scorer.score_months("2025-01-01", "2026-06-30")
  2. loads daily prices 2025-01-01→2026-06-30 directly from the DB
     (own query — do not use the clamped loaders)
  3. build_portfolio_weights(scores_df=…, prices=…) — regime gate
     applies automatically via get_historical_regime_multipliers with
     the weights' own date range (verify its log shows 2025-26 months)
  4. runs a vectorbt backtest over the holdout window only (mirror
     backtest.py's from_orders call + _compute_metrics; SPY benchmark
     via yfinance for the same window)
  5. writes backtests/results/holdout_2025_26_equity.csv (+ SPY column)
     — computation only, NO verdict logic in this prompt.
- If any clamped loader cannot be bypassed by injection, make the
  minimal module-level override edit and document the diff.

PART D — Dry-run sanity (NOT the verdict run):
- Run scripts/run_holdout.py end-to-end once to prove plumbing works.
- Report ONLY plumbing facts: months scored (should be 18: 2025-01 →
  2026-06), score distribution summary (min/mean/max — sanity vs the
  0.45-0.70 historical range), monthly breadth counts above 0.52,
  regime labels for 2025-26 months, equity CSV row count and date range.
- ⚠️ Do NOT compute or report Sharpe/CAGR/DD numbers in this prompt.
  The verdict computation belongs to Prompt 4 ONLY (burn-once
  discipline: the metric is computed once, against the pre-committed
  rule, in a session that has the rule in front of it).

Output: Part A NaN report, the scorer audit line, git diff --stat (new
files + any documented minimal overrides), plumbing facts from Part D.
State you're ready for Prompt 4 (the one-shot verdict run).
```

---

## PROMPT 4 of 5 — The one-shot holdout verdict

```
You are continuing Sprint 7 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. Prompt 4 of 5.
Prompt 3 (prior session) verified the holdout plumbing end-to-end
(18 months scored, equity CSV written) WITHOUT computing performance
metrics. This session computes them ONCE and applies the verdict.
REPO GUARD first (inner repo). Use .venv/bin/python.

PRE-COMMITTED RULE (fixed 2026-07-04, before any holdout metric was
ever computed — apply EXACTLY, no reinterpretation):
  PASS = holdout Sharpe (2025-01-01 → 2026-06-30, daily returns,
         mean/std·√252, computed from the strategy equity curve) > 0.600
  Anything else = FAIL.
This window is BURN-ONCE. Whatever the numbers say, there is no
re-running with changes. If you find a DATA bug (wrong dates, broken
join), fix the bug, regenerate, and note it; nothing that changes model
behavior may be touched.

1. Re-run scripts/run_holdout.py fresh (deterministic inputs — confirm
   the equity CSV is byte-stable vs Prompt 3's, or explain any diff;
   yfinance SPY re-download may differ trivially, that's acceptable).

2. Compute programmatically from the equity CSV (never from printed
   logs): holdout Sharpe, CAGR, Sortino, max DD, total return; the same
   for the SPY column. State the verdict per the rule above.

3. Informational context (report, NOT part of the rule):
   - vs the 2023-24 test Sharpe 0.990 and Sprint 0 test Sharpe 0.783
   - monthly breadth 2025-26 (names > 0.52) vs the 24.6 production mean
   - regime-label mix 2025-26 and days at 0.5×/1.0×/1.2×
   - month-by-month strategy vs SPY table
   - TimesFM caveat: note that the pretrained checkpoint's data vintage
     is a possible subtle leak channel for 2025 — acknowledge plainly.

4. Write backtests/results/sprint7_results.json:
   { generated_at, sprint: "Sprint 7 — 2025-26 true-holdout validation
     (frozen Sprint 5 model)", pre_committed_rule: "...verbatim...",
     holdout: {sharpe, cagr, sortino, max_dd, total_return, start, end},
     spy_holdout: {…}, informational: {breadth_mean, months_empty,
     regime_mix, vs_test_sharpe_0990: "…"}, model_provenance:
     {pickle: "ensemble_models.pkl", folds: 78, effective_train_end:
     "2024-07-31", scorer: "src/live/scorer.py last-fold model"},
     caveats: ["TimesFM pretraining vintage …", "18-month window —
     limited statistical power"], verdict: "PASS"|"FAIL",
     verdict_notes: "3-6 sentences: the number, the rule, what drove
     it (breadth? regime? selection?), and what it does and does not
     prove given the window length." }

5. Whatever the verdict: NO code or model changes in response. State
   the verdict explicitly and that you're ready for Prompt 5 (commit).
```

---

## PROMPT 5 of 5 — Commit (single branch — nothing to revert)

```
You are finishing Sprint 7 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. Prompt 5 of 5.
Prompt 4 wrote backtests/results/sprint7_results.json with a verdict.
Unlike Sprint 6 there is no revert branch: the model was never changed,
so PASS and FAIL commit the same artifacts — the verdict is a recorded
finding either way.

REPO GUARD: inner repo only (toplevel ends "Ai Trading Agent");
git log should show 13a2b6c at or near HEAD. NEVER touch the outer
Stock_Project repo. If push is rejected non-fast-forward: STOP, do not
force-push.

1. Stage EXACTLY the Sprint 7 deliverables:
   - src/live/__init__.py, src/live/scorer.py, scripts/run_holdout.py
   - any documented minimal override diffs from Prompt 3
   - backtests/results/sprint7_results.json
   - git add -f backtests/results/holdout_2025_26_equity.csv
     (force past *.csv gitignore, same as prior sprints)
   - memory/phase_progress.md + memory/future_ideas.md updates (below)
   Do NOT stage: config/settings.yaml (must show no diff), models/*.pkl,
   data/**, and the separate pending Dashboard-v2 files
   (src/dashboard/*, Sprint*_Prompts.md) — leave those for their own
   commit unless the user has said otherwise.

2. memory/phase_progress.md: append "## Sprint 7 — 2025-26 true-holdout
   validation" in the established format: what ran (frozen model,
   src/live/scorer.py), the pre-committed rule verbatim, the numbers
   table (holdout vs SPY vs the 2023-24 test period), verdict verbatim
   from the JSON, and the burn-once note ("this window is now spent —
   no future experiment may tune against 2025-26").

3. memory/future_ideas.md: update the "Sprint 7 candidate" entry to an
   outcome-log entry (VALIDATED or FAILED-VALIDATION) with key numbers;
   note that Sprint 8 (Option B) inherits src/live/scorer.py and the
   now-current data feeds; if FAIL, add explicit guidance that the
   response is a future investigation sprint, NOT a holdout re-run.

4. Commit message:
   PASS: "Sprint 7: 2025-26 true holdout PASS — frozen model validated
          (sprint7_results.json)"
   FAIL: "Sprint 7: 2025-26 true holdout FAIL — recorded, window spent
          (sprint7_results.json)"
   Push origin main. Report git log -1 --stat + push confirmation +
   remaining untracked/uncommitted list (Dashboard-v2 files expected).
```

---

## Usage notes

- One prompt per fresh session, in order. Prompt 2 is the long one (FinBERT
  backlog on CPU) — it's chunked and resumable by design; it can be split across
  two sessions if needed, finishing step 2 before moving on.
- The deliberate firewall: Prompt 3 proves plumbing but is FORBIDDEN from computing
  performance metrics; Prompt 4 computes them exactly once with the rule in view.
  Don't let a session blur that line — it's what makes the holdout believable.
- If Prompt 1 finds a blocker (e.g., EDGAR rate limits, missing 2025 filings),
  bring it back to the knowledge-base chat before proceeding.
- After Sprint 7, Sprint 8 (Option B paper trading) starts with src/live/scorer.py,
  current data feeds, and a validated (or honestly flagged) production model.
