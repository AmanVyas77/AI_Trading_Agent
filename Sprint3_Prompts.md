# Sprint 3 — Proof Run (Regime Gate + Sentiment Smoothing)
Copy-paste each prompt into a fresh Cowork/coding-agent session, one at a time, in order.
Each prompt is self-contained.

Sprint 3 re-runs the full ensemble pipeline with both Sprint 1 (regime gate) and Sprint 2
(EWM sentiment smoothing) active, then diffs every metric against the Sprint 0 baseline.
The primary target: improve the 2022 max drawdown from -41.74% without gutting CAGR.

Verified context (treat as fact, don't re-derive):
- Baseline on record: backtests/results/baseline_metrics.json
  CAGR 14.78% (train 15.17%, test 13.81%), Sharpe 0.721, 2022 max drawdown -41.74%
- backtest.py always writes to ensemble_sprint3_*.csv — copy to gated_* names immediately
  after the run (same pattern Sprint 0 used for baseline_* names)
- Pipeline CLI commands (all python -m src.strategies.ensemble.<name>):
  factor_export_quant | factor_export_fundamental | feature_matrix |
  target_builder | model_trainer | score_generator | portfolio_builder | backtest

---

## PROMPT 1 of 5 — Verify State (no edits)

```
You are picking up the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Sprint 3
(proof run), Prompt 1 of 5. This sprint re-runs the full ensemble pipeline
with all Sprint 1 and Sprint 2 changes active, then diffs the results
against the Sprint 0 baseline to quantify the improvement.

This prompt is READ-ONLY — do not edit any files.

Sprint history to verify:
- Sprint 0: wired Piotroski F, QMJ Safety/Payout, FinBERT sentiment, LM
  sentiment into the ensemble; rebuilt baseline. Baseline on record at
  backtests/results/baseline_metrics.json (2022 max drawdown: -41.74%).
- Sprint 1: created src/strategies/ensemble/regime_gate.py and wired
  get_historical_regime_multipliers() into portfolio_builder.py to scale
  weights 0.5x/1.0x/1.2x based on VIX/yield-curve regime.
- Sprint 2: added EWM_SPAN=4 and _ewm_smooth() to
  src/strategies/fundamental/sentiment_pipeline.py; modified
  load_sentiment() and load_lm_scores() to apply smoothing at read time.

Your tasks:
1. Run `git log --oneline -6` and confirm the three Sprint commits are
   present (Sprint 0 baseline rebuild, Sprint 1 regime gate, Sprint 2
   EWM smoothing).
2. Confirm src/strategies/ensemble/regime_gate.py exists and has both
   get_historical_regime_multipliers() and get_live_regime_signal().
3. Confirm portfolio_builder.py imports from regime_gate and calls
   get_historical_regime_multipliers() inside build_portfolio_weights().
4. Confirm sentiment_pipeline.py has EWM_SPAN=4, _ewm_smooth(), and
   that both load_sentiment() and load_lm_scores() call _ewm_smooth()
   before returning.
5. Load backtests/results/baseline_metrics.json and print its contents
   — these are the numbers Sprint 3 will diff against.
6. Confirm backtests/results/ensemble_baseline_train_equity.csv,
   ensemble_baseline_test_equity.csv, and ensemble_baseline_stats.csv
   all exist (Sprint 0 saved these so the proof run can't overwrite them).

Output: confirmation (pass/fail) for each of the 6 tasks. If any Sprint
commit is missing or any file is absent, stop and report — do not proceed
to the pipeline re-run until state is confirmed clean. State you're ready
for Prompt 2 (full pipeline re-run).
```

---

## PROMPT 2 of 5 — Full Pipeline Re-run

```
You are continuing Sprint 3 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 2 of 5.
Prompt 1 (in a prior session) confirmed Sprint 1 (regime gate) and Sprint 2
(EWM smoothing) are both committed and wired correctly. The Sprint 0
baseline is on record at backtests/results/baseline_metrics.json
(2022 max drawdown: -41.74%, full-period CAGR: 14.78%, Sharpe: 0.721).

CRITICAL RULE: Do not edit config/settings.yaml or .env without explicitly
asking the user first.

IMPORTANT — FILE NAMING: backtest.py always writes to these fixed names:
  backtests/results/ensemble_sprint3_train_equity.csv
  backtests/results/ensemble_sprint3_test_equity.csv
  backtests/results/ensemble_sprint3_stats.csv
You MUST copy these to gated-namespaced filenames immediately after the
backtest completes, before anything else can overwrite them:
  ensemble_gated_train_equity.csv
  ensemble_gated_test_equity.csv
  ensemble_gated_stats.csv

Why this run is different from Sprint 0:
- Sprint 2's EWM smoothing changed the values returned by load_sentiment()
  and load_lm_scores() — the feature matrix needs to be rebuilt so the
  model retrains on smoothed sentiment rather than Sprint 0's raw values.
- Sprint 1's regime gate is wired into portfolio_builder.py — it will
  automatically scale weights during the backtest without any extra steps.
- The quant factor export (factor_export_quant) is UNCHANGED since Sprint 0
  and does not need to be re-run.

Your task — run the pipeline in this exact order:
1. `python -m src.strategies.ensemble.factor_export_fundamental`
   (picks up smoothed sentiment via the updated load functions)
2. `python -m src.strategies.ensemble.feature_matrix`
   (rebuilds ensemble_feature_matrix.parquet with smoothed sentiment values)
3. `python -m src.strategies.ensemble.target_builder`
4. `python -m src.strategies.ensemble.model_trainer`
   (retrains XGBoost on the smoothed feature matrix — this is the slowest
   step, expect several minutes for walk-forward training)
5. `python -m src.strategies.ensemble.score_generator`
6. `python -m src.strategies.ensemble.portfolio_builder`
   (regime gate now active — weights scaled by 0.5/1.0/1.2 per month)
7. `python -m src.strategies.ensemble.backtest`
8. Immediately copy outputs:
   cp backtests/results/ensemble_sprint3_train_equity.csv
      backtests/results/ensemble_gated_train_equity.csv
   cp backtests/results/ensemble_sprint3_test_equity.csv
      backtests/results/ensemble_gated_test_equity.csv
   cp backtests/results/ensemble_sprint3_stats.csv
      backtests/results/ensemble_gated_stats.csv
   cp data/processed/ensemble_feature_matrix.parquet
      data/processed/ensemble_feature_matrix_gated.parquet

Report the headline metrics from ensemble_gated_stats.csv:
Total Return, CAGR, Sharpe, Sortino, Max Drawdown, Calmar — for both
train and test periods. Also isolate and report the 2022-specific max
drawdown (peak-to-trough within the 2022 equity curve subset).

Do not build the diff table yet — that's Prompt 3. State you're ready
for Prompt 3 once the run completes and files are copied.
```

---

## PROMPT 3 of 5 — Before/After Diff

```
You are continuing Sprint 3 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 3 of 5.
Prompt 2 (in a prior session) re-ran the full pipeline with both Sprint 1
(regime gate) and Sprint 2 (EWM smoothing) active, and saved results to
ensemble_gated_*.csv. If these files don't exist, stop and say so rather
than re-running the pipeline.

Your task: build a complete before/after comparison.

Load both:
  - backtests/results/baseline_metrics.json  (Sprint 0 baseline)
  - backtests/results/ensemble_gated_stats.csv  (Sprint 3 gated run)

Produce the following diff table covering BOTH train and test periods:

Metric              | Baseline | Gated  | Delta  | Direction
--------------------|----------|--------|--------|----------
Total Return (full) |          |        |        |
CAGR (full)         |          |        |        |
Sharpe (full)       |          |        |        |
Sortino (full)      |          |        |        |
Max Drawdown (full) |          |        |        |
Calmar (full)       |          |        |        |
2022 Max Drawdown   | -41.74%  |        |        |
2022 Full-Year Ret  | -38.01%  |        |        |

Delta = Gated minus Baseline (e.g. Drawdown: -20% - (-41.74%) = +21.74pp
improvement; show as positive for improvements).

Then write this diff as a structured file:
  backtests/results/sprint3_proof_results.json
containing: baseline metrics, gated metrics, all deltas, and a
"verdict_notes" field you'll fill in Prompt 4.

Output: the filled-in diff table and confirmation that
sprint3_proof_results.json was written. State you're ready for Prompt 4
(tradeoff validation).
```

---

## PROMPT 4 of 5 — Tradeoff Validation + Verdict

```
You are continuing Sprint 3 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 4 of 5.
Prompt 3 (in a prior session) built the before/after diff and wrote
backtests/results/sprint3_proof_results.json. Load that file now — the
numbers in it drive the verdict in this prompt.

Your task: validate the tradeoff and write the verdict.

1. PRIMARY CHECK — did the regime gate improve the 2022 drawdown?
   The baseline 2022 max drawdown was -41.74%. The regime gate had
   10/12 months of 2022 at 0.5x multiplier (confirmed in Sprint 1's
   smoke test). A meaningful improvement is at least 10 percentage
   points better (i.e. gated 2022 drawdown better than -31.74%).
   State explicitly: PASS or FAIL with the actual number.

2. TRADEOFF CHECK — at what cost to full-period returns?
   The baseline full-period CAGR was 14.78%. Evaluate:
   (a) If gated CAGR is within 3pp of baseline (>11.78%): acceptable
       tradeoff — drawdown improved without gutting returns.
   (b) If gated CAGR dropped more than 3pp below baseline: the regime
       gate is too aggressive — it's protecting against drawdown by
       simply sitting out too much of the market, not by making smarter
       bets. Note this as a concern.
   Also check whether Sharpe ratio improved (risk-adjusted returns got
   better) — a better Sharpe with a lower CAGR is a sign the gate is
   working correctly (less risk per unit of return), while a lower CAGR
   with no Sharpe improvement suggests it's just reducing exposure
   without adding intelligence.

3. RISK-ON UPSIDE CHECK — did the 1.2x risk-on multiplier contribute
   positively? Check the test period (2023-2024, which was largely
   risk-on per the live signal in Sprint 1's smoke test). If the gated
   test CAGR or Sharpe improved vs baseline test period, that's evidence
   the 1.2x multiplier is adding value, not just the 0.5x protection.

4. Write a plain-English verdict (3-5 sentences) summarising:
   - Whether the primary objective was achieved (2022 drawdown reduced)
   - Whether the tradeoff is acceptable (CAGR cost vs drawdown benefit)
   - Whether the Sharpe improved (the cleanest sign of net benefit)
   - Whether Sprint 4 (TimesFM momentum factor) should be gated open
     based on these results (per the original plan: only proceed with
     Phase 5 if the regime gate proved out)

5. Update the "verdict_notes" field in sprint3_proof_results.json with
   the plain-English verdict from step 4.

Output: answers to checks 1-3 with explicit PASS/FAIL/NOTE labels, the
plain-English verdict, and confirmation that sprint3_proof_results.json
was updated. State you're ready for Prompt 5 (commit).
```

---

## PROMPT 5 of 5 — Commit + Update Memory

```
You are finishing Sprint 3 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 5 of 5.
Prompt 4 (in a prior session) completed the tradeoff validation and wrote
the verdict into backtests/results/sprint3_proof_results.json. If that
file doesn't exist or is missing the verdict_notes field, stop and report
rather than committing.

CRITICAL RULE: Do not edit config/settings.yaml or .env without explicitly
asking the user first. Sprint 3 should have produced only new result files
(ensemble_gated_*.csv, ensemble_feature_matrix_gated.parquet,
sprint3_proof_results.json) — no code changes. If git status shows code
changes, stop and report before committing.

Your tasks:
1. Run `git status` and `git diff --stat`. Confirm only new result/data
   files appear (ensemble_gated_*.csv, ensemble_feature_matrix_gated.parquet,
   sprint3_proof_results.json). No .py files should have changed.
2. Stage and commit the result files with message:
   "Sprint 3: proof run with regime gate + EWM smoothing; results in
   sprint3_proof_results.json"
3. Update the project's persistent memory (memory/phase_progress.md).
   Add an entry noting:
   - Sprint 3 complete. Include the actual numbers: baseline vs gated
     for 2022 max drawdown, full-period CAGR, and Sharpe.
   - Whether the primary objective was achieved (2022 drawdown improved).
   - The plain-English verdict from sprint3_proof_results.json verbatim.
   - Whether Sprint 4 (TimesFM) is gated open or closed based on the
     results (per the project plan: only proceed with Phase 5 if the
     regime gate genuinely improved the drawdown without gutting returns).
   - The full sprint chain (Sprint 0-3) is now complete. Phase 4b is
     done.
4. Confirm the working tree is clean after the commit.

5. Print the git push command so the user can push to remote:
   Run `git remote -v` and `git branch --show-current`, then print the
   exact command — e.g. `git push origin main`. Do NOT run the push;
   print it for the user to execute.

Output: commit hash, memory update confirmation, the final verdict on
Sprint 4's gate status, the ready-to-run git push command, and a
statement that the full Sprint 0-3 plan is complete.
```
