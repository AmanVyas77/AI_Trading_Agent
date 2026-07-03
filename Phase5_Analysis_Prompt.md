# Phase 5 Decision Analysis — AI Trading Agent
Copy-paste this entire prompt into a new session. This is a READ + ANALYSE task only —
do not edit any files, do not run the pipeline. The goal is a thorough technical
assessment of two competing Phase 5 directions and a concrete recommendation.

---

## Project overview

Project: "AI Trading Agent" at `/Users/aman/Desktop/Stock_Project/Ai Trading Agent`
Universe: ~54 US tech stocks (2015–2024 daily price history in SQLite)
Stack: Python 3.11 · XGBoost ensemble · vectorbt backtests · TimesFM 2.5 (CPU)
Git remote: https://github.com/AmanVyas77/AI_Trading_Agent.git (branch: main)

The system is a 4-phase quant + fundamental + ensemble + RAG platform:
- Phase 1–2: quant factor export (momentum, vol, Piotroski F, QMJ) + fundamental
  scores (FinBERT / LM sentiment from SEC filings)
- Phase 3: XGBoost walk-forward ensemble that ranks stocks monthly and builds
  a long-only portfolio, backtested with vectorbt
- Phase 4a: RAG pipeline (ChromaDB + LLM) for earnings/macro context retrieval
- Phase 4b: regime gate (VIX / yield-curve rule-based multipliers) + EWM sentiment
  smoothing + TimesFM 1M forward-return factor — COMPLETE as of Sprint 5

CRITICAL RULE (applies to every file you touch or analyse):
Never propose editing config/settings.yaml or .env. All new constants must live
as module-level values in the relevant .py file. Code files (*.py) are fine to discuss
or propose edits to.

---

## Verified sprint history — treat these numbers as facts, do not re-derive

Sprint 0 baseline (commit c0f9c65):
  CAGR 14.78% | Sharpe 0.600 (full-period, from ensemble_baseline_stats.csv)
  Test CAGR 13.81% | Test Sharpe 0.783 | 2022 max DD −41.74%
  NOTE: baseline_metrics.json shows Sharpe 0.721 — that is the TRAIN-period Sharpe only.
  The correct full-period baseline Sharpe is 0.600.

Sprint 1 (commit b6256e4): regime_gate.py added. Rule-based gate for backtest
  (VIX/yield-curve → 0.5×/1.0×/1.2× multipliers), RAG blend for live mode.

Sprint 2 (commit 06c4b12): Arratia (2021) EWM smoothing (span=4) added to
  FinBERT and LM sentiment loaders at read time.

Sprint 3 (commit ee3f25e): proof run with Sprints 1+2 active.
  OR gate (VIX>25 OR spread<0) was too loose — 18/19 test-period months tripped
  RISK_OFF due to persistent 2023-24 yield-curve inversion.

Sprint 4 (commit 4036e74): graded gate recalibration.
  risk_off = (VIX > 25) OR (VIX > 20 AND spread < 0)
  Full-period CAGR 9.80% | Sharpe 0.530 | Test CAGR 6.77% | Test Sharpe 0.478
  2022 max DD −34.23% | Verdict: FAIL on all primary checks (gate improved
  2022 protection but gutted test-period CAGR by −7.04pp vs baseline)

Sprint 5 (commit 65bc3bb): TimesFM 1M forward-return factor added as 23rd feature.
  Installed: timesfm 2.0.2 (TimesFM 2.5 architecture, NOT 1.x API)
  HF_REPO: google/timesfm-2.5-200m-pytorch | CONTEXT_LEN=252 | HORIZON_LEN=21 | MIN_CONTEXT=63
  Batched by month-end (~120 calls for full 2015–2024 window, ~4 min on CPU)
  Full-period CAGR 13.60% | Sharpe 0.682 | Test CAGR 18.27% | Test Sharpe 0.990
  2022 max DD −29.78% | Test-CAGR recovery: 163.35% of the 7.04pp Sprint-4 gap
  TimesFM deflated t-stat: 30.48 (DSR threshold = 1.0) — signal is real
  XGBoost gain rank: 17/23 (middling in tree splits, statistically significant)
  Verdict: PASS. Phase 4b complete.

Current local git state (as of 2026-07-03):
  main: scaffold → Sprint 1 → Sprint 2 → Sprint 3 → Sprint 4 → Sprint 5 (65bc3bb, pushed)
  Working tree: memory/*.md + Sprint*_Prompts.md + data/chroma_db + data/stock_data.db
  still untracked (catch-up commit pending, not part of this analysis)

---

## Key files to read (prioritised)

Read these files IN FULL before forming any opinion. The analysis lives in the code,
not in this prompt.

TIER 1 — must read:
  src/strategies/ensemble/model_trainer.py
    → This is the most important file for Option A. You need to understand the exact
      walk-forward training loop: how folds are defined, where the train window starts
      and ends, how purge gaps are applied (AlgoXpert Session 4 controls), and what
      the deflated-t DSR check does. The function to focus on is train_walk_forward()
      or equivalent. Record the exact line numbers and logic.

  src/strategies/ensemble/regime_gate.py
    → Critical for Option B. get_live_regime_signal() is the live-mode RAG/LLM blend
      that has never been used in production. Understand how it calls LLMClient and
      what macro data it loads. Note any gaps or failure modes.

  src/strategies/quant/timesfm_factor.py
    → New Sprint 5 module. Understand the _load_model() singleton, the per-month-end
      batch loop, the normalisation strategy, and what happens when a batch fails.

  backtests/results/sprint5_results.json
    → Ground-truth numbers for the current model. Read the full JSON.

TIER 2 — read for context:
  src/strategies/ensemble/factor_export_quant.py
    → How TimesFM is wired into the factor pipeline (step 3.5, merged after resample).
  src/strategies/ensemble/feature_matrix.py
    → QUANT_COLS list (now 7 columns including timesfm_pred_return_1m).
  src/strategies/ensemble/portfolio_builder.py
    → How regime multipliers are applied to daily weights (no renormalisation).
  src/strategies/ensemble/backtest.py
    → What it writes (ensemble_sprint3_*.csv fixed names) and what stats it computes.
  memory/phase_progress.md
    → Full sprint narrative with all commit hashes and three-way comparison tables.
  memory/future_ideas.md
    → Draft Sprint 6 path for Option A and prerequisites list for Option B.

TIER 3 — skim if time permits:
  src/strategies/ensemble/score_generator.py
  src/strategies/ensemble/regime_analysis.py
  src/strategies/fundamental/sentiment_pipeline.py (EWM smoothing is here)

---

## The two Phase 5 options under evaluation

### Option A: Rolling 3-year XGBoost retrain (model refresh)

Core change: modify model_trainer.py to use a trailing N-year training window
instead of the current expanding window. Each walk-forward fold drops data older
than WINDOW_YEARS (suggested = 3) before fitting. This is a module-level constant
change only — no settings.yaml edit.

Hypothesis: the current XGBoost is trained on 2018–2022 data and has never seen the
2023–24 AI-boom regime in-sample. Even though TimesFM partially compensates (163%
CAGR recovery), the base model's feature weights were calibrated against pre-AI-boom
return distributions. A rolling 3-year window would include 2022–2024 in the training
folds, giving XGBoost exposure to the current regime.

Known risks already identified (verify in code):
  1. Smaller sample size per fold — purge gaps become proportionally more expensive
  2. Early folds (2018–2020) may not have 3 years of data yet, requiring a fallback
  3. Risk of overfitting to the AI-boom regime — if 2025 rotates to value, the model
     has almost no exposure to non-momentum regimes
  4. The 2022 bear market sits at the edge of the 3-year window; a 2026 re-run of
     this code might drop 2022 data entirely

### Option B: Live paper-trading pipeline

Core change: build a new src/live/ module with a scheduled monthly rebalance harness
that reads the latest score/weight files and posts paper orders. Uses portfolio_builder
outputs directly — no new alpha logic.

Prerequisites already documented in memory/future_ideas.md:
  - Model refresh (Option A) should ideally come first
  - Execution venue + broker API credentials in .env
  - Live data feeds: prices (daily), macro (VIX, yield curve, CPI, fed funds), SEC
    filings (for the sentiment pipeline to stay current)
  - get_live_regime_signal() in regime_gate.py needs validation (has never run in
    production; LLMClient calls may fail silently)
  - TimesFM needs a monthly scheduler; model weights need to be cached at a stable path

---

## What has already been discussed (do not repeat, build on it)

The following tradeoffs have been identified. Your job is to verify them in the code
and either confirm, refine, or refute them with specific line-number references:

1. Walk-forward purge gap cost: with a 3-year rolling window, the purge gap eats
   a larger fraction of effective training data than with an expanding window.
   → Verify: how many months is the purge gap in model_trainer.py? What % of a
   36-month window does it consume vs the current expanding window?

2. Early-fold fallback: the first walk-forward fold may not have 36 months of
   history. Is there any minimum-window guard in the current code, and would a
   rolling window break it?

3. Live regime gate gap: get_live_regime_signal() exists in regime_gate.py but
   has never been called in any backtest or smoke test. What does it actually do —
   what macro data does it load, what LLM call does it make, and what are the
   realistic failure modes?

4. Data freshness for live pipeline: the prices table in data/quant_research.db
   goes to 2026-05-08. What would need to change to keep it current on a monthly
   cadence? Is there any existing ingestion script, or would it need to be built
   from scratch?

5. TimesFM on a live schedule: compute_timesfm_predictions() in timesfm_factor.py
   currently takes a price_df and a list of month_ends as inputs — it's designed for
   batch historical runs. What would need to change to run it for a single month_end
   in a live context?

---

## Your tasks

1. READ all Tier 1 files in full. Read Tier 2 files. Do not skip model_trainer.py —
   the walk-forward fold logic is the crux of the Option A assessment.

2. For Option A, answer precisely:
   (a) What is the exact current walk-forward logic (window start, window end, purge
       gap, number of folds)? Quote the relevant code.
   (b) What is the minimal code change to implement a rolling WINDOW_YEARS=3 window?
       Show the before/after diff (pseudocode is fine, exact lines preferred).
   (c) How many effective training rows per fold would be lost compared to the current
       expanding window, and how does that interact with the DSR deflated-t check?
   (d) What is your honest probability that a rolling 3-year window improves the
       Sprint 5 test Sharpe (0.990) further vs degrades it? Reason from the code,
       not from general ML knowledge.

3. For Option B, answer precisely:
   (a) What is the minimum viable set of components needed for a paper-trading loop?
       Map each component to an existing file or a gap that needs to be built.
   (b) Is get_live_regime_signal() in regime_gate.py actually usable today, or does
       it have silent failure modes that would make live signals unreliable?
   (c) What is the realistic timeline estimate for a working paper-trading pipeline
       given the current codebase, assuming no new alpha work (just infrastructure)?
   (d) What is the single highest-risk dependency for Option B that could block the
       whole effort?

4. Head-to-head recommendation:
   State clearly which option you recommend for Phase 5, and why, anchored to what
   you found in the code — not to general principles. If the answer depends on a
   specific finding (e.g. the purge gap is large enough to make rolling windows
   harmful), state the condition and what it would take to change your recommendation.

5. If recommending Option A, propose the exact Sprint 6 prompt structure (like the
   Sprint 1–5 prompts in this project: Prompt 1 = recon, Prompt 2 = implement,
   Prompt 3 = verify, Prompt 4 = backtest + verdict, Prompt 5 = commit). Keep it
   brief — just the outline, not the full copy-pasteable text.

6. If recommending Option B, propose the minimal viable architecture for src/live/
   and identify which existing modules it reuses vs what needs to be written new.

---

## Output format

- Lead with a one-paragraph executive summary of your recommendation.
- Then answer tasks 2–6 in order, with clear section headers.
- Use specific file:line references wherever possible.
- Do not hedge with "it depends" without immediately stating what it depends on and
  how to resolve the dependency.
- Do not propose changes to settings.yaml or .env under any circumstances.
