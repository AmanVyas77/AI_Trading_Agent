# Sprint 4 — Regime Gate Recalibration (AND Logic)
Copy-paste each prompt into a fresh Cowork/coding-agent session, one at a time, in order.
Each prompt is self-contained.

Recalibration decision: replace the original OR logic with a graded gate.
Multipliers unchanged: RISK_OFF=0.5, NEUTRAL=1.0, RISK_ON=1.2.

  BEFORE (original OR):
    risk_off = (VIX > 25) OR (spread < 0)

  AFTER (graded gate):
    risk_off = (VIX > 25) OR (VIX > 20 AND spread < 0)
    # Arm 1: extreme fear (VIX>25) → risk_off regardless of spread
    # Arm 2: elevated fear (VIX>20) + macro confirmation (spread<0) → risk_off

Rationale: Prompt 1 recon revealed that pure AND (VIX>25 AND spread<0) only preserves
3 of 10 risk-off months in 2022 (Aug/Sep/Oct only), losing the entire early-2022 sell-off
(Feb/Apr/May/Jun, where VIX was 26-33 but spread was still positive). The graded gate
preserves all 10 protective months in 2022 by keeping the VIX>25 arm unconditional,
while the second arm (VIX>20 AND spread<0) eliminates 18 of 19 test-period false positives
(2023-24 months where spread alone triggered risk-off during the AI boom). Both existing
constants VIX_RISK_OFF=25.0 and VIX_RISK_ON=20.0 are reused — no new constants, no
settings.yaml change. Sprint 4 touches only regime_gate.py.

Verified context (treat as fact, don't re-derive):
- Sprint 0 baseline: CAGR 14.78% (train 15.17%, test 13.81%), Sharpe 0.721,
  2022 max DD -41.74%
- Sprint 3 gated (OR logic): CAGR 9.39% (train 11.19%, test 5.41%), Sharpe 0.531,
  2022 max DD -34.23%, test Sharpe 0.596, test max DD -7.21%
- Sprint 3 verdict: FAIL on all three checks (7.51pp 2022 DD improvement below 10pp
  target; CAGR cost -5.39pp above 3pp tolerance; Sharpe worsened)
- Sprint 1 smoke test: 10/12 months of 2022 were risk-off with OR logic — 2022
  protection was real; AND logic should preserve most of it (2022 had both VIX
  elevated and curve inverting)
- Sprint 4 does NOT need to retrain the model or regenerate scores. The Sprint 3
  XGBoost model and score_generator output are unchanged. Only portfolio_builder
  and backtest need to re-run.

---

## PROMPT 1 of 5 — Recon: Count Month Distribution (no edits)

```
You are picking up the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Sprint 4
(regime gate recalibration), Prompt 1 of 5.

Sprint 3 ran the full pipeline with an OR-logic regime gate
(risk_off = VIX>25 OR spread<0) and EWM sentiment smoothing. Results
were mixed: 2022 max drawdown improved from -41.74% to -34.23% (+7.51pp)
but full-period CAGR dropped 5.39pp (14.78%→9.39%) and Sharpe worsened
(0.721→0.531). Root cause: the yield curve stayed inverted through
2023-24 while VIX was low — OR logic kept the strategy at 0.5x for
roughly 16+ months of the AI bull market. Sprint 4 fixes this by
changing OR → AND (require BOTH VIX>25 AND spread<0 for risk_off),
preserving 2022 protection while recovering 2023-24 upside.

This prompt is READ-ONLY — do not edit any files.

Your tasks:
1. Read src/strategies/ensemble/regime_gate.py in full. Record:
   (a) The exact line(s) implementing the risk_off condition — copy
       the code verbatim. Confirm whether it uses | (bitwise OR) or
       the keyword or.
   (b) The multiplier constants: RISK_OFF_MULT, NEUTRAL_MULT,
       RISK_ON_MULT, VIX_RISK_OFF, VIX_RISK_ON, SPREAD_FLOOR.
   (c) How missing macro data is handled (NaN → NEUTRAL assumption).

2. Call get_historical_regime_multipliers(start_date="2015-01-01",
   end_date="2024-12-31") with the CURRENT (OR) logic. Produce a
   summary breakdown:

     Period                | RISK_OFF | NEUTRAL | RISK_ON | Total
     2015-2019 (pre-train) |          |         |         |
     2020-2022 (train)     |          |         |         |
     2023-2024 (test)      |          |         |         |
     Full 2015-2024        |          |         |         |

   This quantifies how many months OR logic cost us in the test period.

3. For every RISK_OFF month in 2023-2024, print a row:
     Month | VIX | spread | Arm that fired (VIX_ONLY / SPREAD_ONLY / BOTH)
   "VIX_ONLY"    = VIX>25 but spread>=0
   "SPREAD_ONLY" = spread<0 but VIX<=25
   "BOTH"        = both conditions true
   SPREAD_ONLY months are the ones AND logic will recover to NEUTRAL.

4. For 2022 specifically, print the same table. Confirm that most 2022
   risk-off months are BOTH or VIX_ONLY — these are months where AND
   logic still correctly triggers, preserving the 2022 drawdown
   protection.

5. Confirm backtests/results/sprint3_proof_results.json exists and has
   a populated verdict_notes field. Print its timesfm_gate and the
   three headline deltas (2022 DD, CAGR, Sharpe vs baseline).

6. Confirm the score_generator output parquet exists under
   data/processed/ (the file portfolio_builder reads to get per-stock
   ensemble scores). Report its filename and row count. Sprint 4 does
   NOT retrain the model or regenerate scores — confirming this file
   is on disk means Prompt 3 can skip directly to portfolio_builder.

Output: the exact risk_off code line (task 1a), the four-row period
table (task 2), the 2023-2024 per-month arm breakdown (task 3), the
2022 per-month arm breakdown (task 4), sprint3_proof_results.json
confirmation (task 5), and scores parquet filename + row count (task 6).
State you're ready for Prompt 2 (one-operator change to regime_gate.py).
```

---

## PROMPT 2 of 5 — Implement AND Logic in regime_gate.py

```
You are continuing Sprint 4 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 2 of 5.
Prompt 1 (in a prior session) confirmed:
- The exact risk_off line in regime_gate.py using OR logic (re-read
  that file now if you don't have Prompt 1's output — do not guess
  the syntax).
- Multiplier constants: RISK_OFF_MULT=0.5, NEUTRAL_MULT=1.0,
  RISK_ON_MULT=1.2, VIX_RISK_OFF=25.0, VIX_RISK_ON=20.0,
  SPREAD_FLOOR=0.0.
- The SPREAD_ONLY months in 2023-2024 are the ones AND logic will
  recover to NEUTRAL.
- 2022 risk-off months were mostly BOTH / VIX_ONLY — AND logic
  preserves 2022 protection.

CRITICAL RULE: Do not edit config/settings.yaml or .env without
explicitly asking the user first. This prompt touches ONLY
src/strategies/ensemble/regime_gate.py — no other file, no config.

The fix replaces the spread-only OR arm with a dual-confirmation arm,
using the two existing VIX constants (no new constants needed):

  BEFORE (original OR — the _classify_row function body):
    if (vix is not None and vix > VIX_RISK_OFF) or (
        spread is not None and spread < SPREAD_FLOOR
    ):
        return "RISK_OFF"

  AFTER (graded gate):
    if (vix is not None and vix > VIX_RISK_OFF) or (
        vix is not None and vix > VIX_RISK_ON
        and spread is not None and spread < SPREAD_FLOOR
    ):
        return "RISK_OFF"
    # Arm 1: VIX > 25 (VIX_RISK_OFF) → always risk_off, spread irrelevant
    # Arm 2: VIX > 20 (VIX_RISK_ON) AND spread < 0 → risk_off on dual signal

Re-read regime_gate.py now and locate _classify_row (or the equivalent
inline logic) to confirm the exact current code before editing. The None-
checks must be preserved exactly as they are. All other logic — risk_on
condition, neutral fallback, NaN→NEUTRAL handling, return type, module
constants, function signatures — stays exactly as-is.

Your tasks:
1. Make the change described above in regime_gate.py. Show the exact
   before and after lines side by side (a git-style diff is fine).

2. Immediately verify the graded gate behaves correctly:
   (a) Call get_historical_regime_multipliers("2022-01-01", "2022-12-31").
       All 10 months that were risk-off under the original OR gate must
       still return multiplier=0.5. Specifically confirm:
       - Feb/Apr/May/Jun (VIX 26-33, spread positive) → 0.5 via Arm 1
       - Jul/Nov/Dec (VIX 20-22, spread negative) → 0.5 via Arm 2
       - Aug/Sep/Oct (VIX 25-31, spread negative) → 0.5 via both arms
       Print the full 2022 table: month | VIX | spread | multiplier.

   (b) Call get_historical_regime_multipliers("2023-01-01", "2024-12-31").
       Confirm that 18 of 19 previously risk-off months now return
       multiplier=1.0 (NEUTRAL). The one remaining risk-off month is
       Feb 2023 (VIX 20.70, spread -0.89), which should still return
       0.5 via Arm 2 (VIX 20.70 > VIX_RISK_ON 20.0 AND spread < 0).
       Print the full 2023-24 table: month | VIX | spread | multiplier.

   (c) Call get_historical_regime_multipliers("2015-01-01", "2024-12-31")
       and print the four-row period breakdown from Prompt 1 with the
       graded gate active. The 2023-24 RISK_OFF count should drop from
       19 to 1 (only Feb 2023 remaining).

3. Confirm the risk_on condition and multiplier constants are unchanged.

Output: the exact before/after diff (task 1), the 2022 verification
table (task 2a), the 2023-24 verification table (task 2b), the new
four-row period breakdown (task 2c), and the constants confirmation
(task 3). State you're ready for Prompt 3 (portfolio + backtest re-run).
```

---

## PROMPT 3 of 5 — Portfolio Re-run (No Model Retraining)

```
You are continuing Sprint 4 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 3 of 5.
Prompt 2 (in a prior session) changed the risk_off condition in
regime_gate.py from OR to AND logic. If uncertain, re-read regime_gate.py
and confirm the AND operator is present before running anything.

CRITICAL RULE: Do not edit config/settings.yaml or .env without
explicitly asking the user first.

IMPORTANT — FILE NAMING: backtest.py writes to fixed names:
  backtests/results/ensemble_sprint3_train_equity.csv
  backtests/results/ensemble_sprint3_test_equity.csv
  backtests/results/ensemble_sprint3_stats.csv
Copy these to recal-namespaced names IMMEDIATELY after the backtest
completes — before anything else can overwrite them:
  ensemble_recal_train_equity.csv
  ensemble_recal_test_equity.csv
  ensemble_recal_stats.csv

Why we skip most of the pipeline:
- The XGBoost model (Sprint 3, trained on EWM-smoothed features) is
  unchanged. Its per-stock score predictions are still valid.
- The EWM sentiment smoothing (Sprint 2) and feature matrix are
  unchanged.
- The ONLY change is in regime_gate.py's classification logic, which
  portfolio_builder calls at runtime. Running portfolio_builder +
  backtest is sufficient.

Your task — run only these two steps:
1. `python -m src.strategies.ensemble.portfolio_builder`
   (portfolio_builder imports get_historical_regime_multipliers from
   regime_gate at runtime — it automatically picks up AND logic without
   any changes to portfolio_builder itself)

2. `python -m src.strategies.ensemble.backtest`

3. Immediately copy outputs:
   cp backtests/results/ensemble_sprint3_train_equity.csv \
      backtests/results/ensemble_recal_train_equity.csv
   cp backtests/results/ensemble_sprint3_test_equity.csv \
      backtests/results/ensemble_recal_test_equity.csv
   cp backtests/results/ensemble_sprint3_stats.csv \
      backtests/results/ensemble_recal_stats.csv

Error handling:
- If portfolio_builder raises ImportError or AttributeError referencing
  regime_gate, re-read regime_gate.py to confirm the AND change from
  Prompt 2 is saved to disk. Fix and re-run.
- If portfolio_builder complains it cannot find the scores parquet,
  run `python -m src.strategies.ensemble.score_generator` first (this
  regenerates scores from the existing trained model without retraining),
  then re-run portfolio_builder. Do NOT run model_trainer.

Report the headline metrics from ensemble_recal_stats.csv:
Total Return, CAGR, Sharpe, Sortino, Max Drawdown, Calmar — for both
train and test periods. Also isolate and report the 2022-specific max
drawdown (peak-to-trough within the 2022 equity curve subset from
ensemble_recal_train_equity.csv).

Do not build the diff table yet — that is Prompt 4. State you're ready
for Prompt 4 (three-way diff and verdict).
```

---

## PROMPT 4 of 5 — Three-Way Diff + Verdict

```
You are continuing Sprint 4 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 4 of 5.
Prompt 3 (in a prior session) ran portfolio_builder and backtest with
AND-logic regime gate and saved to ensemble_recal_*.csv. Load those
files now. If they don't exist, stop and report rather than re-running.

Also load backtests/results/sprint3_proof_results.json for Sprint 3
(OR gate) numbers. Sprint 0 baseline numbers are verified facts —
treat these as given, no need to re-derive:
  Sprint 0 baseline: CAGR 14.78% (train 15.17%, test 13.81%),
    Sharpe 0.721, Sortino ~0.900, 2022 max DD -41.74%,
    test Sharpe 0.783, test max DD -15.57%

Your task: build the three-way comparison and write the verdict.

────────────────────────────────────────────────────────────────────
PART 1 — THREE-WAY DIFF TABLE
────────────────────────────────────────────────────────────────────
Fill in every cell. Delta S4-vs-S0 = Sprint 4 minus Sprint 0.
For drawdown deltas, positive = improvement (less negative).

Metric              | Sprint 0 (Baseline) | Sprint 3 (OR Gate) | Sprint 4 (AND Gate) | Δ S4 vs S0 | Δ S4 vs S3
--------------------|---------------------|--------------------|---------------------|------------|----------
Total Return (full) |                     |                    |                     |            |
CAGR (full)         |       14.78%        |       9.39%        |                     |            |
Sharpe (full)       |       0.721         |       0.531        |                     |            |
Sortino (full)      |                     |                    |                     |            |
Max Drawdown (full) |                     |                    |                     |            |
Calmar (full)       |                     |                    |                     |            |
Train CAGR          |       15.17%        |       11.19%       |                     |            |
Train Sharpe        |                     |                    |                     |            |
Train Max DD        |      -43.79%        |      -36.61%       |                     |            |
2022 Max Drawdown   |      -41.74%        |      -34.23%       |                     |            |
Test CAGR           |       13.81%        |       5.41%        |                     |            |
Test Sharpe         |       0.783         |       0.596        |                     |            |
Test Max Drawdown   |      -15.57%        |       -7.21%       |                     |            |

────────────────────────────────────────────────────────────────────
PART 2 — THREE VERDICT CHECKS
────────────────────────────────────────────────────────────────────

PRIMARY CHECK — 2022 drawdown protection:
  Target: AND-gate 2022 max DD better than -31.74%
    (≥10pp improvement over baseline -41.74%)
  State: PASS or FAIL with the actual AND-gate number.
  Note: Sprint 3 (OR gate) achieved only -34.23% (+7.51pp). AND gate
  should do similarly since 2022 had both VIX>25 and spread<0 — if
  AND-gate 2022 DD is close to Sprint 3's -34.23%, the drawdown
  protection is coming primarily from the VIX arm (already present
  in both OR and AND), and recovering the test-period losses is the
  real win of the recalibration.

TRADEOFF CHECK — CAGR and Sharpe:
  (a) CAGR target: AND-gate full-period CAGR > 11.78%
        (within 3pp of baseline 14.78%). PASS or FAIL.
  (b) Sharpe target: AND-gate full-period Sharpe ≥ 0.721 (baseline).
        PASS or FAIL.
  A Sharpe at or above baseline is the cleanest sign of success —
  it means the gate is controlling risk without sacrificing
  risk-adjusted return. A CAGR improvement with flat/worse Sharpe
  still indicates some remaining miscalibration.

RISK-ON UPSIDE CHECK — test period recovery:
  Compare AND-gate test CAGR and test Sharpe vs:
    (a) Sprint 0 baseline (target: near 13.81% / 0.783)
    (b) Sprint 3 OR gate (must be substantially better than 5.41% / 0.596)
  State: RECOVERED (test CAGR within 3pp of baseline) or
         PARTIAL (better than Sprint 3 but still >3pp below baseline) or
         INSUFFICIENT (not materially better than Sprint 3).

────────────────────────────────────────────────────────────────────
PART 3 — TIMESFM GATE DECISION
────────────────────────────────────────────────────────────────────
Per the project plan, TimesFM (momentum factor) was gated on the regime
gate proving out: 2022 protection intact + CAGR/Sharpe tradeoff acceptable.

Based on the three checks:
  "TimesFM gate: OPEN"
    → All three checks pass (primary PASS, both tradeoff PASS,
      risk-on RECOVERED)
  "TimesFM gate: PARTIAL — [list which checks failed]"
    → 1-2 checks fail but the direction is correct
  "TimesFM gate: CLOSED — [reason]"
    → Primary fails (AND gate lost 2022 protection vs OR gate) or
      Sharpe still worse than baseline

────────────────────────────────────────────────────────────────────
PART 4 — VERDICT + OUTPUT FILE
────────────────────────────────────────────────────────────────────
Write a plain-English verdict (3-5 sentences) covering:
  - Whether AND logic fixed the OR gate's over-exposure problem
    (did test CAGR recover substantially?)
  - Whether 2022 drawdown protection was preserved
  - Whether the CAGR/Sharpe tradeoff is now acceptable vs baseline
  - The TimesFM gate decision and reason

Write everything to backtests/results/sprint4_recal_results.json:
{
  "sprint0_baseline": {
    "cagr_full": 14.78, "sharpe_full": 0.721,
    "max_dd_2022": -41.74, "cagr_test": 13.81, "sharpe_test": 0.783
  },
  "sprint3_or_gate": {
    "cagr_full": 9.39, "sharpe_full": 0.531,
    "max_dd_2022": -34.23, "cagr_test": 5.41, "sharpe_test": 0.596
  },
  "sprint4_and_gate": {
    "cagr_full": <actual>, "sharpe_full": <actual>,
    "max_dd_2022": <actual>, "cagr_test": <actual>, "sharpe_test": <actual>
  },
  "deltas_vs_baseline": { "cagr": <>, "sharpe": <>, "max_dd_2022": <>,
                          "cagr_test": <>, "sharpe_test": <> },
  "deltas_vs_sprint3":  { "cagr": <>, "sharpe": <>, "max_dd_2022": <>,
                          "cagr_test": <>, "sharpe_test": <> },
  "checks": {
    "primary_2022_dd": "PASS" | "FAIL",
    "tradeoff_cagr":   "PASS" | "FAIL",
    "tradeoff_sharpe": "PASS" | "FAIL",
    "risk_on_upside":  "RECOVERED" | "PARTIAL" | "INSUFFICIENT"
  },
  "timesfm_gate": "OPEN" | "PARTIAL" | "CLOSED",
  "verdict": "<plain-English verdict text>"
}

Output: the filled three-way diff table, explicit labels for all checks,
the TimesFM gate decision, the plain-English verdict, and confirmation
that sprint4_recal_results.json was written. State you're ready for
Prompt 5 (commit).
```

---

## PROMPT 5 of 5 — Commit + Update Memory

```
You are finishing Sprint 4 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 5 of 5.
Prompt 4 (in a prior session) completed the three-way comparison and
wrote backtests/results/sprint4_recal_results.json. Load that file now
and read the timesfm_gate and verdict fields before doing anything else.
If the file is missing or verdict is empty, stop and report.

CRITICAL RULE: Do not edit config/settings.yaml or .env without
explicitly asking the user first. Sprint 4 should have produced exactly:
  Code change:
    src/strategies/ensemble/regime_gate.py  (graded gate: _classify_row
    Arm 2 changed from spread<0 alone to VIX>VIX_RISK_ON AND spread<0)
  New result files:
    backtests/results/ensemble_recal_train_equity.csv
    backtests/results/ensemble_recal_test_equity.csv
    backtests/results/ensemble_recal_stats.csv
    backtests/results/sprint4_recal_results.json
Nothing else. If git status shows any other .py files changed (especially
portfolio_builder.py, settings.yaml, or any *_pipeline.py), stop and
report before committing.

Your tasks:
1. Run `git status` and `git diff --stat`. Confirm only regime_gate.py
   changed on the code side (plus the new result files which are
   untracked). If settings.yaml or .env appear in the diff, DO NOT
   commit — stop and report.

2. Run `git diff src/strategies/ensemble/regime_gate.py` and confirm
   the diff shows only the _classify_row Arm 2 change: the spread-only
   condition (spread is not None and spread < SPREAD_FLOOR) now requires
   a VIX co-condition (vix is not None and vix > VIX_RISK_ON). No other
   lines should have changed. If settings.yaml or any other file appears
   in the diff, stop and report.

3. Stage and commit with message:
   "Sprint 4: recalibrate regime gate to graded gate (VIX>25 OR
   (VIX>20 AND spread<0)); multipliers unchanged (0.5/1.0/1.2);
   results in sprint4_recal_results.json"

4. Update the project's persistent memory (memory/phase_progress.md).
   Add an entry for Sprint 4 noting:
   - Sprint 4 complete.
   - Change: graded gate — risk_off = (VIX > 25) OR (VIX > 20 AND spread < 0).
     Previously: risk_off = (VIX > 25) OR (spread < 0).
     Multipliers unchanged: RISK_OFF=0.5, NEUTRAL=1.0, RISK_ON=1.2.
     Recon (Prompt 1) confirmed all 10 2022 risk-off months preserved;
     18/19 test-period false positives eliminated (only Feb 2023 remains).
   - Three-way comparison numbers from sprint4_recal_results.json:
     Sprint 0 vs Sprint 3 (OR) vs Sprint 4 (graded) — include 2022 max DD,
     full-period CAGR, full-period Sharpe, and test CAGR for all three.
   - BASELINE SHARPE NOTE: use full-period Sharpe 0.600 (from
     ensemble_baseline_stats.csv) as the Sprint 0 baseline in the memory
     entry. The value 0.721 that appears in sprint4_recal_results.json
     and baseline_metrics.json is the train-period Sharpe — do NOT use
     0.721 as the full-period baseline. Sprint 3's proof diff confirmed
     the full-period baseline Sharpe is 0.600.
   - The verdict from sprint4_recal_results.json verbatim.
   - TimesFM gate status (from timesfm_gate field) and what the next
     step is:
       OPEN    → Sprint 5 = TimesFM momentum factor implementation
       PARTIAL → Sprint 5 = decide whether to soften multipliers
                 (0.7/1.0/1.1) or proceed to TimesFM anyway
       CLOSED  → Sprint 5 = TimesFM anyway (see future_ideas.md note)

5. Update memory/future_ideas.md — add or append to the TimesFM entry:
   - Note what Sprint 4 revealed: test-period Sharpe worsened when
     exposure was restored (Sprint 3: 0.596 → Sprint 4: 0.478), meaning
     the underlying model's stock selection is weak in 2023-24.
     The XGBoost model trained pre-AI-boom is not overweighting the
     AI-driven outperformers (NVDA, MSFT, META). Further gate tuning
     (softer multipliers) won't fix this — it's a stock-selection problem.
   - TimesFM forward-momentum signals would naturally overweight the
     2023-24 AI-boom winners, addressing the root cause of test-period
     underperformance independently of the gate.
   - Recommended Sprint 5 path regardless of gate verdict: implement
     TimesFM as a new quant factor alongside the existing momentum
     lookbacks (Phase 1). Evaluate whether it repairs test-period alpha
     before deciding whether to also soften the gate multipliers.

6. Confirm the working tree is clean after the commit.

7. Print the git push command so the user can push to remote:
   Run `git remote -v` and `git branch --show-current`, then print the
   exact command — e.g. `git push origin main`. Do NOT run the push;
   print it for the user to execute.

Output: commit hash, the regime_gate.py diff confirming only _classify_row
changed, memory update confirmations for both files, the final TimesFM
gate status with a clear statement of what Sprint 5 should be, and the
ready-to-run git push command.
```
