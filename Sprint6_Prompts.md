# Sprint 6 — Rolling 3-Year XGBoost Training Window (Model Refresh)
Copy-paste each prompt into a fresh Cowork/coding-agent session, one at a time, in order.
Each prompt is self-contained.

Sprint 6 changes the XGBoost walk-forward training window from anchored-expanding to a
rolling 3-year (36-month) trailing window. One module constant + a two-line trim in
`model_trainer.py`. No feature changes, no gate changes, no TimesFM rerun.

Motivation (CORRECTED from the future_ideas.md draft — treat this framing as fact):
The walk-forward is EXPANDING, so late test folds already train on 2023-24 data (the
2024-11 fold trains through 2024-07). The problem is DILUTION, not absence: late folds
are ~85% pre-2023 data. A rolling 36-month window makes post-2022 data ~60-70% of the
window for late folds — at the cost of cutting training rows ~50% on average and ~67%
on the final fold. Prior probability of improving test Sharpe is ~35%. This sprint is a
cheap, fully reversible falsification test: FAIL is itself a useful finding (old-regime
data still carries weight → green-lights the live pipeline as Phase 5 instead).

Verified context (treat as fact, do not re-derive):
- TWO NESTED GIT REPOS exist (resolved 2026-07-03): the AUTHORITATIVE repo is the INNER
  one rooted at "Ai Trading Agent/" — HEAD 65bc3bb, full Sprint 3→4→5 chain, tracks
  src/strategies/ensemble/model_trainer.py. The OUTER repo at "Stock_Project/" (HEAD
  70c3dff, 2 commits) is a stale snapshot pointing at the SAME GitHub remote.
  RULES FOR EVERY SESSION:
  (a) Run ALL git commands from inside "Ai Trading Agent/" and verify first with
      git rev-parse --show-toplevel → must end in "Ai Trading Agent". If it ends in
      "Stock_Project", you are in the wrong repo — cd and re-verify.
  (b) NEVER push, commit, or open a PR from the outer Stock_Project repo. Its history
      is unrelated to origin/main's sprint chain; pushing it could clobber the remote.
  (c) Sprint 5 head = 65bc3bb (inner). Any 70c3dff sighting means you're in the outer repo.
- Inner-repo working tree (verified): modified memory/*.md; untracked Sprint*_Prompts.md,
  Phase5_Analysis_Prompt.md, data/chroma_db/, data/stock_data.db. config/settings.yaml,
  .gitignore, requirements.txt, quant_factors.py are CLEAN in the inner repo. The
  PRE-STEP below commits the notes/memory drift as the Sprint 6 revert anchor.
- Sprint 5 comparison target, from backtests/results/sprint5_results.json:
    full CAGR 13.5977% | full Sharpe 0.6818 | 2022 max DD -29.7815%
    test CAGR 18.2699% | test Sharpe 0.9898 | test max DD -21.3474%
    TimesFM deflated t-stat 30.48 | all 23 features survive DSR (threshold 1.0)
- Sprint 0 baseline: full CAGR 14.78%, full Sharpe 0.600, test CAGR 13.81%, test Sharpe
  0.783, 2022 max DD -41.74%.
  NOTE: full-period baseline Sharpe is 0.600, NOT 0.721 — the 0.721 in
  baseline_metrics.json is the train-period Sharpe only. Do not use 0.721 anywhere.
- Fold arithmetic (verified programmatically against data/processed/ensemble_labeled.parquet):
    117 unique months (2015-03-31 → 2024-11-30), 5,974 rows, ~51 rows/month
    MIN_TRAIN_MONTHS = 36 (target_builder.py:56), PURGE_MONTHS = 3 (model_trainer.py:68),
    EVAL_MONTHS = 6 (model_trainer.py:63) → 78 folds
    Expanding train rows: 1,705 (fold 1) → 5,762 (fold 78), mean 3,725
    Rolling-36 train rows: ~1,869 mean, ~1,908 final fold (-49.8% mean, -66.9% final)
- The fold generator is walk_forward_folds() in target_builder.py:297-352, NOT in
  model_trainer.py. The purge shifts train_end back (train_end = unique_months[i-1-purge]);
  it does NOT shrink a rolling window. Early folds have exactly 36 train months, so the
  rolling trim is a no-op on fold 1 — this is the key verification invariant.
- The DSR check (model_trainer.py:98-129, flagged at :285) is REPORTING-ONLY. Rolling
  windows will raise cross-fold importance variance → all deflated-t values will drop.
  Expect the "all features survive" log to possibly become a warning list. Not a failure.
- Pipeline for this sprint (python -m src.strategies.ensemble.<name>):
  model_trainer → score_generator → portfolio_builder → backtest
  Do NOT rerun factor_export_quant / factor_export_fundamental / feature_matrix /
  target_builder — features and labels are unchanged; rerunning wastes time (TimesFM ~4 min)
  and risks perturbing the frozen inputs.
- PASS rule (strict, decided in advance): PASS requires ALL THREE of
    test Sharpe  > 0.9898
    test CAGR    > 18.2699%
    2022 max DD  >= -29.7815% (i.e. no worse / less negative)
  Anything else, including near-ties, is FAIL → revert WINDOW_YEARS to None and commit
  the reverted code + results JSON as the sprint record (same pattern as Sprints 3/4).

CRITICAL RULE (all prompts): Never edit config/settings.yaml or .env. settings.yaml
contains an UNUSED ml.train_window_years key — leave it alone; the new constant lives
in model_trainer.py as a module-level value (same pattern as timesfm_factor.py).

---

## PROMPT 1 of 5 — Recon (no edits)

```
You are picking up the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Sprint 6
(rolling 3-year XGBoost training window), Prompt 1 of 5.

Context: Sprints 0-5 are complete. Sprint 5 (commit 65bc3bb) added a
TimesFM 1M-return forecast as the 23rd ensemble feature and PASSED
(test Sharpe 0.990, test CAGR 18.27%). Sprint 6 will change the
walk-forward training window from anchored-expanding to rolling 36
months, implemented entirely in model_trainer.py. This prompt is
READ-ONLY — do not edit or install anything.

Use the project venv for all python: .venv/bin/python

Your tasks:
1. Git state: from the INNER repo root ("Ai Trading Agent/" — verify
   with git rev-parse --show-toplevel), run git log --oneline -5 and
   git status --short. Confirm HEAD is at or after 65bc3bb ("Sprint 5").
   List any untracked or modified files.
   [COMPLETED 2026-07-03 — first run hit the outer Stock_Project repo
   (HEAD 70c3dff) and false-flagged the ensemble tree as untracked;
   resolved: inner repo at 65bc3bb is authoritative. See header rules.]

2. Read src/strategies/ensemble/target_builder.py lines 295-352
   (walk_forward_folds) and src/strategies/ensemble/model_trainer.py in
   full. Confirm and record:
   (a) The expanding-window line: train_df = df[df["date"] <= train_end]
       (target_builder.py ~line 346).
   (b) Where model_trainer consumes the folds (the for-loop, ~line 191)
       and where PURGE_MONTHS (~line 68) and EVAL_MONTHS (~line 63) are
       defined.
   (c) The results-dict append (~line 227-239) where per-fold metadata
       is stored — Sprint 6 adds a "window_start" key here.
   (d) Confirm there is NO existing rolling-window logic anywhere in
       model_trainer.py (grep for WINDOW, rolling, DateOffset).

3. Fold arithmetic against live data. With .venv/bin/python, load
   data/processed/ensemble_labeled.parquet and verify:
   (a) unique months == 117, range 2015-03-31 → 2024-11-30, rows == 5974
   (b) implied folds == 117 - 36 - 3 == 78
   (c) per-fold expanding train-row counts: first ~1705, last ~5762
   If any of these differ, STOP and report — the parquet has changed
   since this prompt was written and the Sprint 6 thresholds need review.

4. Baseline model snapshot. Load models/ensemble_models.pkl with pickle:
   (a) Confirm len == 78 (folds).
   (b) Print results[0]["deflated_tstats"] sorted ascending. Record the
       5 features with the LOWEST deflated t-stats — these are nearest
       the DSR floor (1.0) and most likely to flip under rolling windows.
   (c) Print the train_size of fold 1, fold 40, fold 78.

5. Confirm the comparison target: load
   backtests/results/sprint5_results.json and print sprint5_timesfm
   (all fields). These are the numbers Prompt 4 benchmarks against.

6. Check that backtests/results/ contains the Sprint 5 equity CSVs
   (ensemble_timesfm_train_equity.csv, ensemble_timesfm_test_equity.csv,
   ensemble_timesfm_stats.csv). Report which exist.

Output: full written summary for all 6 tasks, quoting exact line numbers
for 2(a)-(c). State you're ready for Prompt 2 (implement).
```

---

## PRE-STEP (run after Prompt 1, before Prompt 2) — Catch-up commit in the INNER repo

STATUS: REQUIRED. Commits the memory/notes drift so Sprint 6 starts from a clean,
pushed anchor. (Prompt 1's original "untracked ensemble tree" finding was an artifact
of running git in the outer Stock_Project repo — resolved; see header rules.)

```
You are doing a housekeeping step for the "AI Trading Agent" project,
between Prompts 1 and 2 of Sprint 6. Make NO code changes in this session.

REPO GUARD (do this first):
  cd "/Users/aman/Desktop/Stock_Project/Ai Trading Agent"
  git rev-parse --show-toplevel
→ must end in "Ai Trading Agent". If it ends in "Stock_Project" you are
in the WRONG (outer) repo — never commit or push from there. All git
commands below run from the inner repo root.

1. Baseline: git log --oneline -3 (expect HEAD 65bc3bb, "Sprint 5…")
   and git status --short. Confirm config/settings.yaml, .gitignore,
   requirements.txt, and src/strategies/quant/quant_factors.py are all
   CLEAN. If settings.yaml shows modified here, STOP and report.

2. Stage the catch-up set (notes + memory drift + result JSONs):
     git add memory/ Sprint0_Prompts.md Sprint1_Prompts.md \
       Sprint2_Prompts.md Sprint3_Prompts.md Sprint4_Prompts.md \
       Sprint5_Prompts.md Sprint6_Prompts.md PHASE4_REVIEW_PROMPT.md \
       Phase5_Analysis_Prompt.md backtests/results/*.json

3. Do NOT stage: data/chroma_db/, data/stock_data.db, anything else
   under data/, models/*.pkl, .env, config/settings.yaml, .DS_Store.
   Verify with git status --short that none of these are staged.

4. Sanity-check: git diff --cached --stat, and confirm
   src/strategies/ensemble/model_trainer.py is ALREADY TRACKED
   (git ls-files | grep model_trainer.py → non-empty). It should not
   appear in this commit (unchanged).

5. Commit and push:
     git commit -m "Catch-up: sprint notes + memory drift before Sprint 6"
     git push origin main
   If the push is rejected non-fast-forward, STOP and report — do NOT
   force-push (a rejected push may mean the outer repo contaminated the
   remote; that needs manual review).

6. Report: the commit hash (Sprint 6 REVERT ANCHOR — quote it
   prominently), git log --oneline -3, and the remaining untracked list
   (expected: only the deliberately excluded data/ paths).

Output: the revert-anchor hash and the repo-guard confirmation. State
you're ready for Prompt 2.
```

---

## PROMPT 2 of 5 — Implement (WINDOW_YEARS + backup)

```
You are continuing Sprint 6 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 2 of 5.
Prompt 1 (prior session) confirmed: 117 months / 78 folds in
ensemble_labeled.parquet, models/ensemble_models.pkl holds the Sprint 5
expanding-window models, and walk_forward_folds() in target_builder.py
yields anchored-expanding train sets. A PRE-STEP session then made a
catch-up commit ("Catch-up: sprint notes + memory drift…").

REPO GUARD (do this first): cd into the project root and run
git rev-parse --show-toplevel → must end in "Ai Trading Agent" (the
inner, authoritative repo — HEAD chain includes 65bc3bb "Sprint 5").
There is a stale OUTER repo at Stock_Project/ — never run git there.
Then verify the PRE-STEP catch-up commit exists (git log --oneline -3)
and the working tree is clean apart from data/ untracked paths. If not,
STOP and run the PRE-STEP first.

CRITICAL RULE: Do not edit config/settings.yaml or .env. settings.yaml
has an unused ml.train_window_years key — LEAVE IT ALONE. The new
constant lives in model_trainer.py as a module-level value. Do not edit
target_builder.py either — the entire change is in model_trainer.py.

────────────────────────────────────────────
PART A: Back up the Sprint 5 model pickle
────────────────────────────────────────────
Before anything else:
  cp models/ensemble_models.pkl models/ensemble_models_sprint5.pkl
This is the revert path if Sprint 6 FAILs — restoring the pickle is
instant vs re-running expanding training. Verify the copy exists and
byte-sizes match.

────────────────────────────────────────────
PART B: Edit src/strategies/ensemble/model_trainer.py
────────────────────────────────────────────
Change 1 — module-level constant, immediately after EVAL_MONTHS = 6
(~line 63):

  # Sprint 6: rolling training window in years. None = anchored expanding
  # (Sprint 0-5 behavior). Trailing window is measured from each fold's
  # (already purge-shifted) train_end, so every fold keeps up to
  # WINDOW_YEARS*12 train months and early folds are unaffected.
  WINDOW_YEARS = 3

Change 2 — inside train_walk_forward(), in the fold loop. Current code
(~lines 191-193):

    for train_df, test_df in walk_forward_folds(labeled_df, purge_months=PURGE_MONTHS):
        fold_num += 1
        test_date = test_df["date"].iloc[0]

Insert directly after the test_date line:

        window_start = None
        if WINDOW_YEARS is not None:
            window_start = train_df["date"].max() - pd.DateOffset(years=WINDOW_YEARS)
            train_df = train_df[train_df["date"] > window_start]

Change 3 — in the results.append({...}) dict (~lines 227-239), add one
key after "effective_train_end":

            "window_start": window_start.date() if window_start is not None else None,

Change 4 — update the module docstring's one-line description to mention
the rolling window, and update the log line at ~line 189-190 (the
"Expanding window" comment) to reflect that a trailing trim is applied
when WINDOW_YEARS is set.

Nothing else changes. XGB_PARAMS, PURGE_MONTHS, EVAL_MONTHS, the eval
split, the DSR audit — all untouched.

────────────────────────────────────────────
PART C: Sanity checks (no training run yet)
────────────────────────────────────────────
1. .venv/bin/python -c "from src.strategies.ensemble import model_trainer; print(model_trainer.WINDOW_YEARS)"
   → must print 3.
2. git diff --stat → must show exactly ONE modified file:
   src/strategies/ensemble/model_trainer.py. Any other modified file
   (including config/settings.yaml, which is clean in this repo) →
   STOP and investigate.
3. git diff src/strategies/ensemble/model_trainer.py → paste the full
   diff in your output.

Output: the diff, confirmation of the pickle backup, and the two sanity
check results. Do NOT run training — that is Prompt 3. State you're
ready for Prompt 3 (verify + retrain).
```

---

## PROMPT 3 of 5 — Verify invariants + retrain

```
You are continuing Sprint 6 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 3 of 5.
Prompt 2 (prior session) added WINDOW_YEARS = 3 and a trailing-window
trim to src/strategies/ensemble/model_trainer.py, and backed up the
Sprint 5 models to models/ensemble_models_sprint5.pkl. Verify both are
true before proceeding (git diff should show model_trainer.py modified;
the backup pickle should exist). Use .venv/bin/python throughout.

────────────────────────────────────────────
PART A: Pre-training invariants (must ALL pass)
────────────────────────────────────────────
Write a throwaway script (do not commit it) that imports
walk_forward_folds from src.strategies.ensemble.target_builder, loads
data/processed/ensemble_labeled.parquet, and replicates the
model_trainer loop's trim logic. Assert:

1. FOLD COUNT UNCHANGED: iterating walk_forward_folds(df, purge_months=3)
   yields exactly 78 folds (the trim happens after yielding, so the
   count cannot change — verify anyway).
2. FOLD 1 IS A NO-OP: for the first fold, the trimmed train set equals
   the untrimmed train set exactly (same shape, same date range).
   Fold 1 has exactly 36 train months; a 3-year trim must remove nothing.
   If fold 1 differs, the window is anchored wrong (e.g. measured from
   test_month instead of train_end) — STOP and fix before training.
3. WINDOW CAP: for every fold, the trimmed train set spans <= 37 unique
   month-ends (36 expected; 37 tolerated for calendar-offset edge cases
   of pd.DateOffset). Report the actual max.
4. ROW COUNTS: trimmed train rows plateau near ~1,850-1,910 for late
   folds; fold 78 should be ~1,908 (vs 5,762 untrimmed). Report fold 1,
   40, 78 trimmed row counts.
5. LAST-FOLD WINDOW START: report the final fold's window_start — it
   should be ~2021-07/2021-08, which still CONTAINS the 2022 bear market.
   Record this value; Prompt 4 writes it into the results JSON (it is
   the canary for future re-runs: a 2026 data refresh would push
   window_start past 2022 and silently drop the bear market).

────────────────────────────────────────────
PART B: Retrain
────────────────────────────────────────────
Run: .venv/bin/python -m src.strategies.ensemble.model_trainer
(This overwrites models/ensemble_models.pkl — the Sprint 5 backup from
Prompt 2 is the safety net.)

From the run output and the new pickle, record:
1. Fold count (must be 78) and the per-fold summary table.
2. Train sizes: confirm they plateau (~1,550-1,650 fit rows after the
   6-month eval carve-out) instead of growing to ~5,400.
3. Eval AUC mean/std vs Sprint 5's (compute Sprint 5's from
   models/ensemble_models_sprint5.pkl for an exact comparison — do not
   eyeball old logs).
4. The DSR report: which features (if any) now fall below deflated-t
   1.0. Compare against the 5 lowest-t features recorded in Prompt 1.
   A longer warning list is EXPECTED (rolling windows raise cross-fold
   importance variance) and is reporting-only — note it, don't fix it.
5. Confirm every results entry has a non-null "window_start".

Output: all Part A assertion results, the Part B comparisons, and an
explicit statement that models/ensemble_models.pkl now holds ROLLING
models while models/ensemble_models_sprint5.pkl holds the EXPANDING
backup. State you're ready for Prompt 4 (backtest + verdict).
```

---

## PROMPT 4 of 5 — Backtest + verdict (sprint6_results.json)

```
You are continuing Sprint 6 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 4 of 5.
Prompt 3 (prior session) retrained the 78 walk-forward XGBoost models
with a rolling 36-month training window; models/ensemble_models.pkl now
holds the rolling models (Sprint 5 expanding backup at
models/ensemble_models_sprint5.pkl). Use .venv/bin/python throughout.

Run the downstream pipeline IN THIS ORDER (do NOT rerun factor exports,
feature_matrix, or target_builder — features/labels are unchanged):
  .venv/bin/python -m src.strategies.ensemble.score_generator
  .venv/bin/python -m src.strategies.ensemble.portfolio_builder
  .venv/bin/python -m src.strategies.ensemble.backtest

BREADTH DIAGNOSTIC (added after Prompt 3 — smaller train sets compress
predicted probabilities toward 0.5, so fewer names may clear
portfolio_builder's MIN_SCORE=0.52 threshold; a breadth change is a
CONFOUND, not ranking skill):
- After score_generator runs, compare per-month counts of tickers with
  ensemble_score > 0.52 between the new ensemble_scores.parquet and the
  Sprint 5 run (portfolio_builder's log prints "above 0.52: NN" per
  month — also recoverable by scoring with the backup pickle if needed).
- Report: mean monthly count above threshold (rolling vs Sprint 5), any
  months with < 10 selected, any months with 0 selected. Include these
  in fold_diagnostics as "mean_names_above_threshold", "months_below_10",
  "months_empty". If breadth collapsed (mean count down > 30%), say so
  prominently in verdict_notes — the Sharpe comparison is then partly a
  concentration effect and the verdict interpretation must note it.

Notes:
- backtest.py writes FIXED filenames (ensemble_sprint3_train_equity.csv,
  ensemble_sprint3_test_equity.csv, ensemble_sprint3_stats.csv in
  backtests/results/). After it finishes, COPY them to
  ensemble_rolling_train_equity.csv / ensemble_rolling_test_equity.csv /
  ensemble_rolling_stats.csv so the Sprint 6 artifacts are preserved.
- The regime gate inside portfolio_builder is the Sprint 4 graded gate —
  unchanged this sprint; expect the same multiplier day-counts as Sprint 5.

METRICS — COMPUTE, DO NOT TRUST PRINTED OUTPUT:
Recompute CAGR / Sharpe / max DD directly from the saved equity CSVs
with your own script (project rule: never claim a checked number without
computing it programmatically). Definitions match backtest.py
_compute_metrics: CAGR from endpoint ratio annualized by days/365.25;
Sharpe = daily mean/std * sqrt(252); max DD from running max. 2022 max
DD = max drawdown of the strategy equity restricted to 2022-01-01 →
2022-12-31. Test period = 2023-01-01 → test end.

VERDICT — STRICT RULE, DECIDED IN ADVANCE (no reinterpretation):
PASS requires ALL THREE:
  (1) test Sharpe  > 0.9898
  (2) test CAGR    > 18.2699%
  (3) 2022 max DD  >= -29.7815% (no worse, i.e. less negative or equal)
Anything else — including near-ties — is FAIL. Prior probability of PASS
was assessed at ~35% before running; a FAIL is an informative outcome
(pre-2020 data still carries weight), not a mistake.

Write backtests/results/sprint6_results.json matching the
sprint5_results.json schema, with:
  - "sprint": "Sprint 6 — Rolling 36-month XGBoost training window"
  - "sprint0_baseline": copy from sprint5_results.json (full Sharpe 0.600
    — never 0.721)
  - "sprint5_timesfm": copy verbatim from sprint5_results.json
  - "sprint6_rolling": full/train/test CAGR, Sharpe, Sortino, max DDs,
    2022 max DD — all recomputed programmatically
  - "deltas_vs_sprint5": per-metric deltas
  - "fold_diagnostics": {"n_folds": 78, "mean_train_rows": ...,
    "final_fold_train_rows": ..., "last_fold_window_start": "...",
    "eval_auc_mean": ..., "eval_auc_std": ...,
    "dsr_below_threshold": [list of features], "timesfm_deflated_t": ...}
  - "checks": the three PASS-rule booleans plus the composite verdict
  - "verdict": "PASS" or "FAIL"
  - "verdict_notes": 3-6 sentences explaining WHERE the difference came
    from (which test months diverged vs Sprint 5 — compare the two test
    equity curves month-by-month: ensemble_rolling_test_equity.csv vs
    ensemble_timesfm_test_equity.csv)

Output: the three-way comparison table (Sprint 0 / Sprint 5 / Sprint 6),
the three check results, the verdict, and the month-by-month divergence
summary. State the verdict explicitly and that you're ready for
Prompt 5 (commit — which has separate PASS/FAIL branches).
```

---

## PROMPT 5 of 5 — Commit (PASS and FAIL branches)

```
You are finishing Sprint 6 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 5 of 5.
Prompt 4 (prior session) wrote backtests/results/sprint6_results.json
with a verdict. Read that file FIRST — the verdict decides which branch
below applies. Use .venv/bin/python where needed.

COMMON RULES (both branches):
- REPO GUARD first: cd "/Users/aman/Desktop/Stock_Project/Ai Trading
  Agent"; git rev-parse --show-toplevel must end in "Ai Trading Agent".
  Never run git from the outer Stock_Project repo; never push or PR it.
- The PRE-STEP catch-up commit ("Catch-up: sprint notes + memory
  drift…") is the revert anchor — confirm it exists in git log before
  committing anything.
- Commit ONLY Sprint 6 files. git status will still show deliberately
  excluded untracked paths (data/chroma_db/, data/stock_data.db) — do
  NOT add them.
- config/settings.yaml must show NO diff in this repo — if it does, a
  Sprint 6 session edited it (violates the project rule): STOP.
- Never commit: models/*.pkl, data/processed/*.parquet, .env,
  config/settings.yaml.
- If git push is rejected non-fast-forward, STOP and report — do NOT
  force-push (possible outer-repo contamination of the remote).
- Force-add the three ensemble_rolling_*.csv equity/stats files despite
  the *.csv gitignore (same as Sprint 5 did for ensemble_timesfm_*.csv):
  git add -f backtests/results/ensemble_rolling_*.csv
- Update memory/phase_progress.md: append a "## Sprint 6 — Rolling
  36-month training window" section in the same format as the Sprint 5
  entry (change description, three-way table Sprint 0 / Sprint 5 /
  Sprint 6, checks, verbatim verdict from the JSON).
- Update memory/future_ideas.md: move "Model refresh — rolling 3-year
  XGBoost retrain" out of the open list into a VALIDATED or REFUTED
  outcome-log entry (mirroring the Sprint 5 TimesFM outcome-log format),
  with the key numbers.
- Push to origin main after committing.

────────────────────────────────────────────
IF VERDICT == PASS
────────────────────────────────────────────
1. Keep WINDOW_YEARS = 3 in model_trainer.py.
2. Commit: modified model_trainer.py, sprint6_results.json, the
   force-added ensemble_rolling_*.csv, and the two memory/*.md updates
   (memory/ is tracked as of the PRE-STEP catch-up commit).
   Message: "Sprint 6: rolling 36-month XGBoost training window (PASS)"
3. In future_ideas.md, note the follow-up: last_fold_window_start from
   the JSON is ~2021-07; a 2026 data refresh will push the window past
   the 2022 bear market — re-validate before trusting a refreshed model.

────────────────────────────────────────────
IF VERDICT == FAIL
────────────────────────────────────────────
1. Revert the constant: WINDOW_YEARS = None in model_trainer.py. Leave
   the trim code and "window_start" bookkeeping in place (dormant when
   None) — the mechanism is validated, only the setting is rejected.
2. Restore the production models:
   cp models/ensemble_models_sprint5.pkl models/ensemble_models.pkl
   Then rerun .venv/bin/python -m src.strategies.ensemble.score_generator
   and portfolio_builder so data/processed/ scores + weights revert to
   Sprint 5 state. Verify: regenerated ensemble_scores.parquet should
   reproduce Sprint 5 numbers if backtest is rerun (spot-check by
   rerunning backtest and comparing test Sharpe to 0.9898 within ±0.001).
3. Commit: model_trainer.py (with WINDOW_YEARS = None + dormant
   mechanism), sprint6_results.json, force-added ensemble_rolling_*.csv,
   memory updates (same tracked-file caveat as PASS branch).
   Message: "Sprint 6: rolling 36-month window REFUTED — revert to
   expanding (records in sprint6_results.json)"
4. In future_ideas.md's refuted-log entry, record the ACTUAL finding
   (updated after Prompt 4 — do not use the older "dilution is not the
   binding constraint" framing):
   - Verdict FAIL was driven by BREADTH STARVATION, not proven ranking
     failure: rolling models compress scores toward 0.5, so the fixed
     MIN_SCORE=0.52 gate left 14 of ~23 test months with ZERO holdings
     (cash), forfeiting the 2023-24 rally (test CAGR 14.81% vs 18.27%).
     Test Sharpe 1.041 "beat" Sprint 5 only as a zero-vol cash artifact.
   - Therefore: the rolling-window hypothesis is NOT cleanly falsified;
     the tested configuration (rolling + fixed absolute threshold) is
     rejected. What WAS confirmed: 2022 DD improved to -26.23% and test
     max DD to -12.52% (cash months also dodged drawdowns).
   - Add a new open idea: "Rank-based selection under rolling windows"
     (top-N by score with no absolute floor, or per-month score
     z-scoring) — WITH an explicit multiple-testing warning: Sprint 6
     counts as trial 1; any variant needs its own pre-committed PASS
     rule; do not iterate variants until one passes. Priority: BELOW
     the live paper-trading pipeline (Option B), whose prerequisites
     remain the FinBERT backlog (stale since 2024-12-27), the
     regime-gate LIMIT-7/staleness fixes, and a live scorer for
     unlabeled current months.

Output: the branch taken, the exact git log -1 --stat of the commit, the
push confirmation, and (FAIL branch only) the spot-check result
confirming Sprint 5 state was restored.
```

---

## Usage notes

- One prompt per fresh session, in order. Each restates the facts the next
  session needs, so no output-pasting between steps is required — but if a
  prompt's verification numbers disagree with what it says to expect, stop
  and bring the discrepancy back here before continuing.
- Most likely to need iteration: Prompt 3 Part A assertion 2 (window
  anchoring) and Prompt 4's month-by-month divergence analysis.
- Prompt 2's pickle backup is what makes the FAIL branch cheap. If a session
  skips it, do it manually before running Prompt 3.
