# Sprint 5 — TimesFM Forward-Momentum Factor
Copy-paste each prompt into a fresh Cowork/coding-agent session, one at a time, in order.
Each prompt is self-contained.

Sprint 5 adds Google Research's TimesFM (200M-param time series foundation model) as a new
quant factor: `timesfm_pred_return_1m`. It AUGMENTS the existing 1M/3M/6M/12M backward-
looking momentum lookbacks — not replacing any. XGBoost retrains on all factors together and
discovers the optimal weight for TimesFM vs existing signals.

Motivation: Sprint 4 revealed that restoring full exposure in 2023-24 worsened test Sharpe
(0.596 → 0.478), meaning the XGBoost model trained on 2018-2022 data is not selecting the
AI-boom winners (NVDA/MSFT/META/AVGO). TimesFM forward-momentum should naturally overweight
these names in 2023-24 since their price trajectories are structurally different from the
pre-AI-boom period that dominates the training window.

Verified context (treat as fact, don't re-derive):
- Sprint 0 baseline: CAGR 14.78%, Sharpe 0.600 (full-period), test CAGR 13.81%, test Sharpe
  0.783, 2022 max DD -41.74%
  NOTE: full-period baseline Sharpe is 0.600, NOT 0.721 — the 0.721 figure in
  baseline_metrics.json is the train-period Sharpe only.
- Sprint 4 graded gate: CAGR 9.80%, Sharpe 0.530, test CAGR 6.77%, test Sharpe 0.478,
  2022 max DD -34.23%
- Test-CAGR gap to close: 13.81% - 6.77% = 7.04pp
- Recovery thresholds (from project plan):
    >= 50% recovery (test CAGR >= 10.29%) → PASS: keep graded gate + TimesFM
    20-49% recovery (test CAGR 8.18-10.29%) → PARTIAL: note but proceed
    <  20% recovery (test CAGR < 8.18%)  → try softer multipliers next
- Graded gate is committed and active in regime_gate.py (Sprint 4, commit 4036e74).
  No gate changes in Sprint 5.
- Pipeline commands (python -m src.strategies.ensemble.<name>):
  factor_export_quant | factor_export_fundamental | feature_matrix |
  target_builder | model_trainer | score_generator | portfolio_builder | backtest

---

## PROMPT 1 of 5 — Recon (no edits)

```
You are picking up the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Sprint 5
(TimesFM momentum factor), Prompt 1 of 5.

Context: Sprints 0-4 are complete. The ensemble now has a graded regime
gate (VIX>25 OR (VIX>20 AND spread<0) → 0.5x; else 1.0x/1.2x) and EWM
sentiment smoothing. The test-period Sharpe (0.478) is below the Sprint 0
baseline (0.600 full-period) because the XGBoost model trained on 2018-2022
data misses the 2023-24 AI-boom winners. Sprint 5 adds TimesFM as a new
quant factor to address this.

This prompt is READ-ONLY — do not edit or install anything.

Your tasks:
1. Read src/strategies/ensemble/factor_export_quant.py in full. Record:
   (a) Every quant factor it currently computes (names, lookback windows).
   (b) The price data source — which DB table, parquet file, or function
       it reads closing prices from. Copy the exact query/load call.
   (c) The output format — what columns does the exported parquet/CSV
       contain? Index structure (ticker, date)? Column names?
   (d) The exact CLI entry point (if __name__ == "__main__": block or
       python -m invocation target).

2. Read src/strategies/ensemble/feature_matrix.py. Record:
   (a) The variable/list that holds quant factor column names (e.g.
       QUANT_COLS, FEATURE_COLS, or equivalent). Copy it verbatim.
   (b) How quant factor data is merged into the feature matrix — is it
       a direct merge, a pivot, or something else?
   (c) Confirm where a new quant factor column should be added to be
       picked up automatically by the feature matrix build step.

3. Check the price data directly. Using the source identified in task 1b,
   query or load the price table and report:
   (a) Tickers present (count and first/last 5 alphabetically).
   (b) Date range (min and max date).
   (c) Column names (confirm a closing price column exists; record its
       exact name — "close", "adj_close", "Close", etc.).
   (d) Row count and any obvious gaps (e.g. missing tickers for certain
       date ranges).

4. Check TimesFM install status:
   Run: python -c "import timesfm; print(timesfm.__version__)"
   If installed, print the version. If ImportError, note it — do not
   install yet (Prompt 2 handles installation).

5. Estimate inference load:
   Count unique (ticker, month_end) combinations in the price data that
   fall within the backtest window (2015-01-01 to 2024-12-31). Each
   combination requires one TimesFM inference call. Report the total
   count — this determines whether we need batching or parallel inference.

6. Check available RAM:
   python -c "import psutil; m=psutil.virtual_memory();
   print(f'Total: {m.total/1e9:.1f}GB, Available: {m.available/1e9:.1f}GB')"
   TimesFM (200M params, PyTorch CPU) needs ~1.5-2GB at runtime. Report
   whether available RAM is sufficient.

7. Load backtests/results/sprint4_recal_results.json and print the
   sprint4_and_gate test_cagr and test_sharpe fields to confirm the
   Sprint 4 numbers this sprint is benchmarking against.

Output: full written summary for all 7 tasks. Pay particular attention
to the exact closing price column name (task 3c) and the quant factor
column list (task 2a) — these are the exact identifiers Prompt 2 will
use. State you're ready for Prompt 2 (install + implement).
```

---

## PROMPT 2 of 5 — Install + Implement timesfm_factor.py

```
You are continuing Sprint 5 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 2 of 5.
Prompt 1 (in a prior session) confirmed:
- The price data source, closing price column name, and date range
  (re-read that output now if you don't have it — do not guess column names).
- The quant factor column list in feature_matrix.py.
- TimesFM install status and available RAM.

CRITICAL RULE: Do not edit config/settings.yaml or .env without
explicitly asking the user first. The TimesFM context length, horizon,
and related constants live as module-level constants in timesfm_factor.py
only — not in settings.yaml.

Verified facts from Prompt 1 (treat as given, do not re-derive):
- Price column to use: adj_close (NOT "close")
- Price source: SQLite table "prices" in data/quant_research.db
- 54 tickers, 6,407 (ticker × month_end) pairs in backtest window
- .venv location: Ai Trading Agent/.venv (Python 3.11.7)
- Available RAM: ~1.9 GB free — tight, use CPU single-thread only
- TimesFM not yet installed in the .venv

────────────────────────────────────────────
PART A: Install TimesFM into the project venv
────────────────────────────────────────────
Install into the project virtual environment (NOT system pip):
  cd "/Users/aman/Desktop/Stock_Project/Ai Trading Agent"
  .venv/bin/pip install timesfm

After install, verify with:
  .venv/bin/python -c "import timesfm; print(timesfm.__version__)"

If pip install fails with a dependency conflict, try:
  .venv/bin/pip install "timesfm[torch]"

Do NOT use pip install --break-system-packages — the project uses a .venv
and we must install there to avoid polluting system packages.

Note: the first time timesfm.TimesFm(...) is instantiated it downloads
~800MB of model weights from HuggingFace. This only happens once;
subsequent runs load from the local HuggingFace cache (~/.cache/huggingface).

────────────────────────────────────────────
PART B: Create src/strategies/quant/timesfm_factor.py
────────────────────────────────────────────
Create a new file at that path. Contents:

Module-level constants (NOT in settings.yaml):
  CONTEXT_LEN = 252   # trading days of adj_close history per inference
  HORIZON_LEN = 21    # forecast 21 days ahead (~1 calendar month)
  MIN_CONTEXT = 63    # minimum non-NaN days required; below this → NaN
  HF_REPO     = "google/timesfm-1.0-200m-pytorch"

Public function: compute_timesfm_predictions(price_df, month_ends)
  Inputs:
    price_df:   Wide DataFrame — DatetimeIndex × ticker columns,
                values = daily adj_close prices (from the prices SQLite
                table, pivoted on adj_close). This is the exact format
                factor_export_quant.py already produces internally.
    month_ends: list or pd.DatetimeIndex of month-end dates.

  BATCHING STRATEGY — batch by month-end, not by ticker:
    Prompt 1 found 6,407 (ticker × month_end) pairs. Running one
    inference call per pair would take 1-4 hours. TimesFM's forecast()
    accepts a LIST of arrays, so we pass ALL tickers for a given month-end
    in a single call. This reduces ~6,407 calls → ~120 calls (one per
    month-end) and cuts runtime from hours to ~2-5 minutes.

  Logic:
    1. Check the installed TimesFM API before coding. Import timesfm
       and run help(timesfm.TimesFm) to confirm the exact constructor
       signature. Reference (verify and adjust to actual installed API):
         import torch
         torch.set_num_threads(2)   # RAM-conservative on 8GB Mac

         tfm = timesfm.TimesFm(
             hparams=timesfm.TimesFmHparams(
                 backend="cpu",
                 horizon_len=HORIZON_LEN,
                 input_patch_len=32,
                 output_patch_len=128,
             ),
             checkpoint=timesfm.TimesFmCheckpoint(
                 huggingface_repo_id=HF_REPO,
             ),
         )
       If constructor kwargs differ from the reference, use the correct
       API — do not force kwargs that don't exist.

    2. For each month_end in month_ends:
       (a) For each ticker, extract adj_close prices up to and including
           month_end: price_df.loc[:month_end, ticker].dropna()
           Take the last CONTEXT_LEN values (or all available if fewer).
       (b) Filter: tickers with fewer than MIN_CONTEXT (63) non-NaN rows
           are excluded from this batch and get NaN in the output.
       (c) For each eligible ticker, NORMALIZE the series:
             series = series / series.iloc[0]
           (Dividing by first value puts prices on a common scale near 1.0,
           which improves TimesFM stability on financial data.)
       (d) Run ONE batch call for all eligible tickers this month_end:
             inputs = [series.values for series in eligible_series]
             freqs  = [0] * len(inputs)   # 0 = high-frequency / daily
             predictions, quantile_preds = tfm.forecast(inputs, freq=freqs)
           quantile_preds is a list of arrays shaped (HORIZON_LEN, n_quantiles).
           The 0.5 quantile (median) is at quantile index corresponding
           to p=0.5 — check tfm.quantiles attribute to find the right index.
       (e) For each ticker in the batch, extract the LAST horizon step's
           median prediction (index HORIZON_LEN-1):
             pred_normalized = quantile_preds[i][HORIZON_LEN-1, median_idx]
           Denormalize back to price space:
             pred_price = pred_normalized * last_actual_price
           Compute predicted return:
             pred_return = (pred_price - last_actual_price) / last_actual_price
       (f) Store: (ticker, month_end) → pred_return for eligible tickers;
           NaN for filtered tickers.

    3. Return a DataFrame with columns ["ticker", "date",
       "timesfm_pred_return_1m"] where "date" = month_end.
       Include ALL tickers for ALL month_ends (NaN where excluded).
       Sort by ["date", "ticker"].

  Error handling:
    - If the batch forecast call raises an exception for a month_end,
      log a WARNING with the month_end date and produce NaN for all
      tickers that month — do not crash the run.
    - If the model fails to load (HuggingFace unreachable or download
      fails), raise immediately with a clear message rather than
      silently returning all NaN.
    - Wrap the model instantiation in a try/except that catches
      HuggingFace auth errors and version mismatch errors separately.

Output: the full timesfm_factor.py file. After creating it, run a quick
sanity check:
  python -c "
  from src.strategies.quant.timesfm_factor import compute_timesfm_predictions
  print('Import OK')
  "
If the import fails, fix before declaring Prompt 2 complete. State you're
ready for Prompt 3 (wire into pipeline + run factor export).
```

---

## PROMPT 3 of 5 — Wire into Pipeline + Factor Export

```
You are continuing Sprint 5 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 3 of 5.
Prompt 2 (in a prior session) created src/strategies/quant/timesfm_factor.py
with compute_timesfm_predictions(). If uncertain, import the function and
confirm it loads without error before proceeding.

CRITICAL RULE: Do not edit config/settings.yaml or .env without
explicitly asking the user first. No config changes are needed —
CONTEXT_LEN, HORIZON_LEN, MIN_CONTEXT live in timesfm_factor.py only.

IMPORTANT — installed API: timesfm 2.0.2 (TimesFM 2.5 architecture)
was installed in Prompt 2. The module already uses the correct 2.5 API
(TimesFM_2p5_200M_torch.from_pretrained + .compile(ForecastConfig);
forecast(horizon=N, inputs=list) with no freq parameter; median at
quantile index 4). Do not attempt to use the 1.x API (TimesFmHparams /
TimesFmCheckpoint) — it does not exist in the installed package.

────────────────────────────────────────────
PART A: Wire into factor_export_quant.py
────────────────────────────────────────────
Modify src/strategies/ensemble/factor_export_quant.py to:
1. Import compute_timesfm_predictions from timesfm_factor:
     from src.strategies.quant.timesfm_factor import (
         compute_timesfm_predictions
     )

2. After the existing momentum factor computations (not replacing them),
   add a block that:
   (a) Loads the closing price panel (same source as existing factors —
       confirm from Prompt 1's reconnaissance output; do not re-guess).
   (b) Determines the set of month-end dates covered by the backtest
       window (use the same month_ends already computed elsewhere in
       the file, or derive them from the price data date range).
   (c) Calls compute_timesfm_predictions(price_df, month_ends) to
       produce a DataFrame with [ticker, date, timesfm_pred_return_1m].
   (d) Merges the TimesFM factor onto the main quant factor output
       using the same merge pattern used for other quant factors
       (confirm the exact join keys and method from Prompt 1).

────────────────────────────────────────────
PART B: Add to QUANT_COLS in feature_matrix.py
────────────────────────────────────────────
In src/strategies/ensemble/feature_matrix.py, add
"timesfm_pred_return_1m" to the quant factor column list (the exact
variable name confirmed in Prompt 1 — QUANT_COLS, FEATURE_COLS, etc.).
Insert it at the END of the list so existing column ordering is
preserved for any positional references elsewhere.

────────────────────────────────────────────
PART C: Smoke Test
────────────────────────────────────────────
Before running the full factor export, run a targeted smoke test:
  python -c "
  import pandas as pd
  from src.strategies.quant.timesfm_factor import compute_timesfm_predictions
  # Load 3 tickers, 12 months — confirmed price source from Prompt 1
  # [agent: use actual price loading code, not pseudocode]
  # Run predictions for those 3 tickers, 12 month-ends
  # Print shape, NaN count, value range
  result = compute_timesfm_predictions(price_df_3tickers, month_ends_12)
  print(result.shape)
  print(result['timesfm_pred_return_1m'].describe())
  print(result.head(12))
  "
  Expected: shape (36, 3), values roughly in [-0.30, +0.30] for monthly
  predicted returns. If values are outside [-0.50, +0.50], the
  normalization or denormalization step has a bug — check and fix.

────────────────────────────────────────────
PART D: Full Factor Export
────────────────────────────────────────────
Once smoke test passes, run:
  python -m src.strategies.ensemble.factor_export_quant

Note: the first run will download TimesFM model weights (~800MB) from
HuggingFace if not already cached (~2-5 min on a fast connection).
Inference runs in ~120 batch calls (one per month-end, all 54 tickers
per call) — expect 5-15 minutes total including the download.

After it completes:
- Load the output parquet and confirm timesfm_pred_return_1m is present.
- Report NaN rate for timesfm_pred_return_1m (expected: 0-10% for early
  dates with insufficient price history; near 0% for dates after 2016).
- Report the NaN rates for existing quant factors — they should be
  UNCHANGED vs Sprint 4 (the new factor must not affect other columns).

Output: the modified sections of factor_export_quant.py and
feature_matrix.py (show only changed/added lines), smoke test results
(shape, value range), and factor export completion report with NaN rates.
State you're ready for Prompt 4 (full retrain + backtest + verdict).
```

---

## PROMPT 4 of 5 — Full Retrain + Backtest + Verdict

```
You are continuing Sprint 5 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 4 of 5.
Prompt 3 (in a prior session) completed factor export with the new
timesfm_pred_return_1m column. If uncertain, load the quant factor
parquet and confirm the column is present before running anything.

CRITICAL RULE: Do not edit config/settings.yaml or .env without
explicitly asking the user first.

IMPORTANT — FILE NAMING: backtest.py always writes to fixed names:
  backtests/results/ensemble_sprint3_train_equity.csv
  backtests/results/ensemble_sprint3_test_equity.csv
  backtests/results/ensemble_sprint3_stats.csv
Copy to TimesFM-namespaced files IMMEDIATELY after the backtest:
  ensemble_timesfm_train_equity.csv
  ensemble_timesfm_test_equity.csv
  ensemble_timesfm_stats.csv

Why we skip factor_export_fundamental:
  Fundamental factors (Piotroski, QMJ, FinBERT, LM, EWM smoothing) are
  unchanged since Sprint 2. Only the quant factor parquet changed
  (new timesfm_pred_return_1m column). Start from feature_matrix onward.

Run the pipeline in this exact order:
1. `python -m src.strategies.ensemble.feature_matrix`
   (rebuilds the feature matrix adding timesfm_pred_return_1m; the
   fundamentals parquet from Sprint 2 is merged in automatically)
2. `python -m src.strategies.ensemble.target_builder`
3. `python -m src.strategies.ensemble.model_trainer`
   (walk-forward XGBoost retrains on the expanded feature set — expect
   several minutes; this is the slowest step)
4. `python -m src.strategies.ensemble.score_generator`
5. `python -m src.strategies.ensemble.portfolio_builder`
   (graded gate already active — no changes needed)
6. `python -m src.strategies.ensemble.backtest`
7. Immediately copy:
   cp backtests/results/ensemble_sprint3_train_equity.csv \
      backtests/results/ensemble_timesfm_train_equity.csv
   cp backtests/results/ensemble_sprint3_test_equity.csv \
      backtests/results/ensemble_timesfm_test_equity.csv
   cp backtests/results/ensemble_sprint3_stats.csv \
      backtests/results/ensemble_timesfm_stats.csv

────────────────────────────────────────────
THREE-WAY DIFF TABLE
────────────────────────────────────────────
Build the comparison (fill in Sprint 5 column from ensemble_timesfm_stats.csv;
Sprint 0 and Sprint 4 numbers are verified facts — treat as given):

Metric              | Sprint 0 (Baseline) | Sprint 4 (Graded) | Sprint 5 (+ TimesFM) | Δ S5 vs S0 | Δ S5 vs S4
--------------------|---------------------|-------------------|----------------------|------------|----------
Total Return (full) |         +144.97%    |       +83.57%     |                      |            |
CAGR (full)         |          +14.78%    |        +9.80%     |                      |            |
Sharpe (full)       |           0.600     |        0.530      |                      |            |
Sortino (full)      |           0.748     |        0.696      |                      |            |
Max Drawdown (full) |          -43.79%    |       -37.13%     |                      |            |
Calmar (full)       |           0.338     |        0.264      |                      |            |
Train CAGR          |          +15.17%    |       +11.19%     |                      |            |
Train Sharpe        |           0.578     |        0.554      |                      |            |
2022 Max Drawdown   |          -41.74%    |       -34.23%     |                      |            |
Test CAGR           |          +13.81%    |        +6.77%     |                      |            |
Test Sharpe         |           0.783     |        0.478      |                      |            |
Test Max Drawdown   |          -15.57%    |       -14.25%     |                      |            |

────────────────────────────────────────────
VERDICT CHECKS
────────────────────────────────────────────
1. TEST-CAGR RECOVERY CHECK (primary):
   Gap = Sprint 0 test CAGR (13.81%) − Sprint 4 test CAGR (6.77%) = 7.04pp
   Recovery = (Sprint 5 test CAGR − 6.77%) / 7.04pp × 100%
   State:
     PASS     → recovery >= 50% (Sprint 5 test CAGR >= 10.29%)
     PARTIAL  → recovery 20-49% (Sprint 5 test CAGR 8.18-10.28%)
     FAIL     → recovery < 20% (Sprint 5 test CAGR < 8.18%)

2. SHARPE CHECK:
   Did full-period Sharpe improve vs Sprint 4 (0.530)? State: YES or NO.
   Did it clear the Sprint 0 full-period baseline (0.600)? State: YES or NO.

3. 2022 PROTECTION CHECK:
   Confirm 2022 max DD is no worse than Sprint 4's -34.23%.
   If it worsened (e.g. model learned to be more aggressive in 2022),
   state the delta. State: PRESERVED or WORSENED.

4. FEATURE IMPORTANCE CHECK:
   After model_trainer runs, check the deflated t-stat output (logged
   as WARNING in model_trainer). Report whether timesfm_pred_return_1m
   has a deflated_t above the DSR_THRESHOLD (default 1.0) — i.e. does
   TimesFM contribute meaningful signal per the AlgoXpert overfitting
   controls from Session 4?

────────────────────────────────────────────
SPRINT 5 GATE DECISION
────────────────────────────────────────────
Based on the four checks, state:
  "Sprint 5 verdict: PASS — TimesFM recovers [X]% of test-CAGR gap;
   Phase 4b complete; suggest Phase 5 (model refresh or additional factors)"
OR
  "Sprint 5 verdict: PARTIAL — TimesFM recovers [X]% of gap; recommend
   [next step: softer gate multipliers / model retraining on more recent
   data / additional factors]"
OR
  "Sprint 5 verdict: FAIL — TimesFM recovers <20% of gap; XGBoost model
   itself needs refreshing with post-2022 training data before further
   factor addition will help"

Write everything to backtests/results/sprint5_results.json:
{
  "sprint0_baseline": { "cagr_full": 14.78, "sharpe_full": 0.600,
    "max_dd_2022": -41.74, "cagr_test": 13.81, "sharpe_test": 0.783 },
  "sprint4_graded_gate": { "cagr_full": 9.80, "sharpe_full": 0.530,
    "max_dd_2022": -34.23, "cagr_test": 6.77, "sharpe_test": 0.478 },
  "sprint5_timesfm": { "cagr_full": <actual>, "sharpe_full": <actual>,
    "max_dd_2022": <actual>, "cagr_test": <actual>, "sharpe_test": <actual> },
  "deltas_vs_baseline": { ... },
  "deltas_vs_sprint4": { ... },
  "test_cagr_recovery_pct": <float>,
  "checks": {
    "test_cagr_recovery": "PASS" | "PARTIAL" | "FAIL",
    "sharpe_vs_sprint4": "IMPROVED" | "FLAT" | "WORSE",
    "sharpe_vs_baseline": "ABOVE" | "BELOW",
    "dd_2022_preserved": true | false,
    "timesfm_deflated_t": <float or null>
  },
  "verdict": "PASS" | "PARTIAL" | "FAIL",
  "verdict_notes": "<plain-English summary>"
}

Output: the three-way diff table, all four check results, the sprint
verdict, and confirmation that sprint5_results.json was written. State
you're ready for Prompt 5 (commit).
```

---

## PROMPT 5 of 5 — Commit + Update Memory

```
You are finishing Sprint 5 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 5 of 5.
Prompt 4 (in a prior session) ran the full retrain + backtest and wrote
backtests/results/sprint5_results.json. Load that file now and read the
verdict and verdict_notes fields before doing anything else. If the file
is missing or verdict is empty, stop and report.

CRITICAL RULE: Do not edit config/settings.yaml or .env without
explicitly asking the user first. Sprint 5 should have touched:
  New file:
    src/strategies/quant/timesfm_factor.py
  Modified files:
    src/strategies/ensemble/factor_export_quant.py
    src/strategies/ensemble/feature_matrix.py
  New result files:
    backtests/results/ensemble_timesfm_train_equity.csv
    backtests/results/ensemble_timesfm_test_equity.csv
    backtests/results/ensemble_timesfm_stats.csv
    backtests/results/sprint5_results.json
If git status shows settings.yaml, .env, or any *_pipeline.py changed,
stop and report before committing.

Your tasks:
1. Run `git status` and `git diff --stat`. Confirm only the three
   expected .py files changed plus new result files.

2. Run `git diff src/strategies/ensemble/feature_matrix.py` and confirm
   the only change is "timesfm_pred_return_1m" appended to the quant
   column list — no other logic changed.

3. Stage and commit with message:
   "Sprint 5: add TimesFM 1M forward-return factor (timesfm_pred_return_1m);
   retrain ensemble with graded gate + TimesFM; verdict: [PASS/PARTIAL/FAIL]
   ([X]% test-CAGR recovery); results in sprint5_results.json"
   Replace [PASS/PARTIAL/FAIL] and [X]% with actual values from
   sprint5_results.json.

4. Update the project's persistent memory (memory/phase_progress.md).
   Add a Sprint 5 entry noting:
   - Sprint 5 complete.
   - New factor: timesfm_pred_return_1m (CONTEXT_LEN=252 days,
     HORIZON_LEN=21 days, backend=cpu). Augments existing 1M/3M/6M/12M
     momentum lookbacks — does not replace any.
   - TimesFM deflated t-stat from Session 4's AlgoXpert DSR check:
     [report value from sprint5_results.json checks.timesfm_deflated_t].
   - Three-way comparison: Sprint 0 vs Sprint 4 vs Sprint 5 — include
     full-period CAGR, Sharpe, 2022 max DD, test CAGR, test Sharpe.
   - Test-CAGR recovery: [X]% of the 7.04pp gap.
   - The verdict_notes from sprint5_results.json verbatim.
   - What Sprint 6 should be based on the verdict:
       PASS    → Phase 4b complete; Phase 5 = model refresh with
                 post-2022 training data OR live trading pipeline
       PARTIAL → try softer gate multipliers (0.7/1.0/1.1) to squeeze
                 more CAGR recovery, then reassess
       FAIL    → XGBoost model staleness is the root problem; Phase 5 =
                 retrain with rolling 3-year window (drop pre-2020 data)
                 before adding more factors

5. Update memory/future_ideas.md with Sprint 5 findings:
   - Record TimesFM's actual contribution (recovery %) and deflated t-stat.
   - If PASS: note TimesFM is validated, remove from "to-try" list,
     add "model refresh" as next candidate.
   - If PARTIAL or FAIL: note TimesFM marginal; root problem may be
     model staleness (XGBoost trained on 2018-2022 without AI-boom data).
     Most impactful next step = rolling window retrain (drop pre-2020,
     add 2023-24 to training set) rather than more factors.

6. Confirm the working tree is clean after the commit.

7. Print the git push commands so the user can push to remote:
   Run `git remote -v` to confirm the remote name (typically "origin")
   and URL, then print the exact command(s) to push:
     git push origin main
   If the branch name differs from "main" (check with `git branch
   --show-current`), use the actual branch name. Do NOT run the push
   yourself — just print the command(s) for the user to execute.

Output: commit hash, git diff summary confirming only expected files
changed, memory update confirmations for both files, the final verdict
with a clear statement of what Sprint 6 / Phase 5 should be, and the
ready-to-run git push command.
```
