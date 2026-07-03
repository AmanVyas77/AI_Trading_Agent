# Sprint 1 — Regime Gate (Rule-Based Backtest / RAG Live Mode)
Copy-paste each prompt into a fresh Cowork/coding-agent session, one at a time, in order.
Each prompt is self-contained.

Design decision (already resolved): rule-based ONLY for the historical backtest
(to avoid lookahead bias from applying present-day RAG context to historical periods);
RAG/LLM blending reserved for live/present-day queries only.

Baseline on record (from Sprint 0): CAGR 14.78%, Sharpe 0.721, 2022 max drawdown -41.74%.
Sprint 1's regime gate should reduce that 2022 drawdown significantly by cutting exposure
during the rate-hike / VIX-spike risk-off environment.

---

## PROMPT 1 of 5 — Read & Verify (no edits)

```
You are picking up the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Sprint 1
(regime gate), Prompt 1 of 5. Sprint 0 is complete — it wired Piotroski F,
QMJ Safety/Payout, FinBERT sentiment, and LM sentiment into the Phase 3
XGBoost ensemble and rebuilt the baseline (CAGR 14.78%, Sharpe 0.721,
2022 max drawdown -41.74%). The baseline is recorded in
backtests/results/baseline_metrics.json.

Sprint 1 creates src/strategies/ensemble/regime_gate.py (which does NOT
exist yet) and wires it into src/strategies/ensemble/portfolio_builder.py
to scale position weights down during risk-off macro regimes.

This prompt is READ-ONLY reconnaissance — do not edit, create, or delete
any files.

Your tasks:
1. Confirm src/strategies/ensemble/regime_gate.py does NOT exist.
2. Read src/strategies/ensemble/regime_analysis.py in full. Record:
   - The exact signature of _classify_regimes() — what argument(s) it
     takes, what shape/columns the input DataFrame must have, and exactly
     what it returns (a dict of boolean Series? a single Series with regime
     labels? something else?).
   - The exact signature and data source of _load_raw_macro() or any
     equivalent macro-loading function in that file — where does it read
     macro data from (DB table? parquet file? FRED CSV?) and what columns
     does it return?
   - Any module-level constants (VIX thresholds, spread thresholds, etc.)
     used by _classify_regimes().
3. Read src/strategies/ensemble/portfolio_builder.py in full. Record:
   - The call chain: build_portfolio_weights() → what it calls → where
     the final daily_weights DataFrame is returned.
   - The exact shape of that daily_weights DataFrame (dates × tickers?
     tickers × dates? what dtype?).
   - Whether build_portfolio_weights() takes any date range arguments or
     derives dates internally.
4. Read src/rag/query/llm_client.py — confirm the LLMClient constructor
   signature and the generate() method signature (for live mode use).
5. Read src/rag/query/context_builder.py — specifically _macro_context()
   and _load_macro() to understand how the RAG layer currently loads and
   classifies macro data (this is the pattern regime_gate.py's live mode
   will mirror).
6. Read config/settings.yaml — confirm there is NO existing regime or
   regime_gate config block.
7. Load backtests/results/baseline_metrics.json and confirm the 2022
   max drawdown is recorded there.

Output: a written summary covering all findings from tasks 1-7. Pay
particular attention to the exact return type of _classify_regimes() and
the shape of portfolio_builder's daily_weights — these determine how
Prompt 2 and 3 implement the gate. State you're ready for Prompt 2
(implementation of regime_gate.py).
```

---

## PROMPT 2 of 5 — Implement regime_gate.py

```
You are continuing Sprint 1 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 2 of 5.
Prompt 1 (in a prior session) verified that src/strategies/ensemble/
regime_gate.py does NOT yet exist and confirmed:
- The exact signature of _classify_regimes() and _load_raw_macro() in
  regime_analysis.py (re-read that file now if you don't have the output
  from Prompt 1 — do not guess at signatures).
- The shape of portfolio_builder.py's daily_weights DataFrame.
- The LLMClient.generate(prompt) API.
- settings.yaml has no existing regime config block.

CRITICAL RULE: Do not edit config/settings.yaml or .env in this prompt
or any future Sprint 1 prompt without explicitly asking the user first.
The multiplier values (0.5 / 1.0 / 1.2) and VIX/spread thresholds live
as module-level constants in regime_gate.py itself — no settings.yaml
change is needed. If you find yourself wanting to move them to config,
ask before doing so.

Your task: create src/strategies/ensemble/regime_gate.py with two
public functions.

────────────────────────────────────────────
FUNCTION 1: get_historical_regime_multipliers(start_date, end_date)
────────────────────────────────────────────
Purpose: produce a monthly pd.Series of regime multipliers for the
historical backtest. PURELY rule-based — no LLM calls, no RAG.

Logic:
- Load a monthly macro panel covering [start_date, end_date]. Reuse
  regime_analysis._load_raw_macro() (import it directly if it's
  importable and its return shape matches what _classify_regimes()
  expects; otherwise replicate its DB/file query). The columns you need
  are at minimum: vix, yield_spread_10y2y.
- For each month-end row, classify using these exact thresholds
  (matching regime_analysis._classify_regimes — DO NOT use
  context_builder.py's different inline thresholds, which have VIX<18
  and spread>0.5):
    risk_off  = (vix > 25) OR (yield_spread_10y2y < 0)
    risk_on   = (vix < 20) AND (yield_spread_10y2y > 0)
    neutral   = everything else
  Priority rule when both conditions are met: risk_off beats risk_on
  (capital preservation takes priority).
- Map to multiplier: risk_off → 0.5, neutral → 1.0, risk_on → 1.2.
- If VIX or yield_spread is missing for a month, default to neutral
  (1.0) rather than crashing.
- Return a pd.Series with a DatetimeIndex of month-end dates and float
  values in {0.5, 1.0, 1.2}.

────────────────────────────────────────────
FUNCTION 2: get_live_regime_signal()
────────────────────────────────────────────
Purpose: produce a regime multiplier for PRESENT-DAY use, blending
the rule-based signal with an LLM assessment. This function is NEVER
called inside the historical backtest loop — only for live/forward use.

Logic:
- Load the most recent macro snapshot from the DB using the same
  _load_macro() pattern as context_builder.py (read from the
  macro_series table, take the most recent value per series).
- Apply the same rule-based classification as Function 1 to get
  rule_signal ("RISK_ON" / "RISK_OFF" / "NEUTRAL").
- Build a prompt for LLMClient that:
  (a) presents the current VIX, yield_spread_10y2y, fed_funds_rate,
      and CPI values from the macro snapshot.
  (b) asks: classify the current macro regime as exactly one of:
      RISK_ON, RISK_OFF, or NEUTRAL. Start your response with exactly
      that label on its own line, then explain briefly.
- Call LLMClient().generate(prompt) to get llm_response. Parse the
  first non-empty line for the label (RISK_ON / RISK_OFF / NEUTRAL).
  If parsing fails or the response is unexpected, default to NEUTRAL.
- Blend: if rule_signal == llm_signal → use that signal's multiplier;
  if they disagree → fall back to NEUTRAL (1.0) as a conservative
  default. Do not apply a 0.5 or 1.2 multiplier when the two signals
  contradict each other.
- Return a dict:
  {
    "multiplier": float,       # 0.5 / 1.0 / 1.2
    "rule_signal": str,        # "RISK_ON" / "RISK_OFF" / "NEUTRAL"
    "llm_signal": str,         # same enum, parsed from LLM response
    "reasoning": str,          # raw LLM response text
    "macro_snapshot": dict     # the raw macro values used
  }

────────────────────────────────────────────
Module-level constants (top of file):
  RISK_OFF_MULT = 0.5
  NEUTRAL_MULT  = 1.0
  RISK_ON_MULT  = 1.2
  VIX_RISK_OFF  = 25.0   # VIX above this → risk_off
  VIX_RISK_ON   = 20.0   # VIX below this (and spread > 0) → risk_on
  SPREAD_FLOOR  = 0.0    # yield_spread below this → risk_off

Output: the full regime_gate.py file. Include a brief module docstring
explaining the two-mode design (rule-based historical vs. RAG live).
State you're ready for Prompt 3 (wiring into portfolio_builder.py).
```

---

## PROMPT 3 of 5 — Wire into portfolio_builder.py

```
You are continuing Sprint 1 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 3 of 5.
Prompt 2 (in a prior session) created src/strategies/ensemble/regime_gate.py
with get_historical_regime_multipliers() (rule-based monthly series) and
get_live_regime_signal() (rule + RAG live blend). If you need to confirm
this file exists and has the right functions, read it now before proceeding.

CRITICAL RULE: Do not edit config/settings.yaml or .env without explicitly
asking the user first. No settings.yaml change is needed for this wiring —
do not add a regime_gate block or move the multiplier constants to config
unless the user explicitly approves. Code files (*.py) are fine to edit.

Your task: modify src/strategies/ensemble/portfolio_builder.py to apply
the regime multiplier to daily portfolio weights.

Implementation:
- In build_portfolio_weights() (or the function that produces and returns
  the final daily_weights DataFrame — confirm the exact function name from
  Prompt 1's reconnaissance), add the following logic AFTER the existing
  code that builds daily_weights and BEFORE the function returns:

  1. Call get_historical_regime_multipliers(
         start_date=daily_weights.index.min(),
         end_date=daily_weights.index.max()
     ) to get a monthly multiplier pd.Series.

  2. Reindex the monthly multiplier Series to the daily_weights DatetimeIndex
     using method='ffill' (carry each month's multiplier forward across all
     trading days in that month).

  3. Multiply daily_weights by the aligned daily multiplier:
       daily_weights = daily_weights.multiply(daily_mults, axis=0)
     This applies the same scalar multiplier to ALL tickers on each day
     (e.g. on a risk_off day the entire portfolio scales to 50% of its
     normal weights, not individual tickers selectively).

  4. Do NOT renormalize the weights back to sum=1 after scaling. The
     reduction in total portfolio exposure IS the regime gate's mechanism
     for risk control. On risk_off days the total allocation should drop to
     ~50% of normal; on risk_on days it rises to ~120%.

- Add the import for regime_gate at the top of portfolio_builder.py:
    from src.strategies.ensemble.regime_gate import (
        get_historical_regime_multipliers
    )

- Do not import get_live_regime_signal() here — that function is for
  live/interactive use only, not the backtest pipeline.

- Do not change any other logic in portfolio_builder.py (TOP_N, MIN_SCORE,
  _select_monthly_holdings, _build_daily_weights, etc. all stay as-is).

Edge cases to handle:
- If get_historical_regime_multipliers() returns a Series that doesn't
  perfectly cover the daily_weights date range (e.g. a few days at the
  start/end are outside the monthly range), ffill should handle the
  interior; for any remaining NaN after ffill (at the very start before
  the first multiplier date), fill with NEUTRAL_MULT (1.0) rather than
  dropping rows.

Output: the modified portfolio_builder.py (show the full diff or the
modified function). State you're ready for Prompt 4 (smoke test against
the 2022 window).
```

---

## PROMPT 4 of 5 — Smoke Test (2022 Validation)

```
You are continuing Sprint 1 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 4 of 5.
Prior sessions created src/strategies/ensemble/regime_gate.py and wired
get_historical_regime_multipliers() into portfolio_builder.py's weight
construction. If you can't confirm this was done, re-read both files now
before proceeding.

This prompt is validation only — do not modify any files unless a clear
bug is found (and if you do fix something, note it explicitly in your output).

The baseline 2022 max drawdown (from Sprint 0, before the regime gate) was
-41.74% (full-year 2022 return was -38.01%). The regime gate should reduce
this by cutting exposure during the rate-hike / VIX-spike window.

Your tasks:

1. MONTHLY MULTIPLIER TABLE (2022):
   Call get_historical_regime_multipliers(start_date="2022-01-01",
   end_date="2022-12-31") and print a table: month | VIX | yield_spread |
   multiplier. Confirm that the months where VIX > 25 OR yield spread < 0
   correctly get multiplier=0.5. (In 2022, VIX was frequently above 25
   and the yield curve inverted — expect substantial coverage at 0.5.)

2. PORTFOLIO WEIGHT SANITY CHECK:
   Run portfolio_builder.py's build_portfolio_weights() (just portfolio
   construction — do NOT run the full backtest pipeline). For a risk-off
   month in 2022 and a risk-on month from 2021 or early 2023, compare the
   sum of daily_weights across all tickers for each period. The 2022 risk-off
   period's total weight sum should be ~50% of the normal-period sum. Report
   both numbers.

3. LIVE SIGNAL CHECK (best-effort):
   Call get_live_regime_signal() once for today's conditions. Confirm it
   returns a valid dict with keys: multiplier, rule_signal, llm_signal,
   reasoning, macro_snapshot. If the LLM call fails (e.g. missing API key,
   no network access), that is acceptable — report the error and confirm
   the rule_signal and macro_snapshot portions still work correctly.

4. REGRESSION CHECK:
   Confirm that portfolio_builder.py still runs without errors on the full
   date range (2015-2024). The regime gate should not have changed the
   number of tickers held, the monthly rebalance logic, or anything except
   the daily weight magnitudes. A quick shape/date-range check of the
   output weights is sufficient — no need to run the full backtest.

Output: the 2022 monthly multiplier table from task 1, the weight-sum
comparison from task 2, and pass/fail for tasks 3 and 4. If any task
fails unexpectedly (e.g. wrong number of risk-off months in 2022, weight
sums not showing the expected reduction), diagnose and fix before
declaring the smoke test passed. State you're ready for Prompt 5 (commit).
```

---

## PROMPT 5 of 5 — Commit + Update Memory

```
You are finishing Sprint 1 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Prompt 5 of 5.
The prior session's smoke test confirmed the regime gate is working: the
2022 monthly multiplier table showed appropriate risk-off periods, and the
portfolio weight sums showed the expected ~50% reduction during those months.
If you're not sure the smoke test passed, check git log and read the most
recent state of regime_gate.py and portfolio_builder.py before committing.

CRITICAL RULE: Do not edit config/settings.yaml or .env without explicitly
asking the user first. Sprint 1 should have touched only:
  - src/strategies/ensemble/regime_gate.py (new file)
  - src/strategies/ensemble/portfolio_builder.py (modified)
Nothing else. If git status shows anything else changed, stop and note it
before committing.

Your tasks:
1. Run `git status` and `git diff --stat`. Confirm only the two expected
   files changed (regime_gate.py new, portfolio_builder.py modified).
   If settings.yaml or .env appear in the diff, DO NOT commit — stop and
   report immediately.
2. Stage and commit with message:
   "Sprint 1: add regime_gate.py (rule-based VIX/yield-curve gate for
   historical backtest, RAG blend for live mode), wire into
   portfolio_builder.build_portfolio_weights()"
3. Update the project's persistent memory (memory/phase_progress.md or
   wherever Sprint history is tracked). Add an entry noting:
   - Sprint 1 complete.
   - regime_gate.py created with two modes:
     get_historical_regime_multipliers() (rule-based, safe for backtest)
     and get_live_regime_signal() (rule + RAG blend, live only).
   - Design decision recorded: rule-based-only for backtest was a
     deliberate choice to avoid lookahead bias — RAG queries reflect
     present-day context and cannot faithfully reconstruct historical
     macro narratives.
   - Multiplier mapping: risk_off=0.5, neutral=1.0, risk_on=1.2.
   - portfolio_builder now scales daily_weights by this multiplier
     without renormalizing (exposure reduction is intentional).
   - Sprint 3 (proof run) will quantify the 2022 drawdown improvement.
4. Confirm the working tree is clean after the commit.

5. Print the git push command so the user can push to remote:
   Run `git remote -v` and `git branch --show-current`, then print the
   exact command — e.g. `git push origin main`. Do NOT run the push;
   print it for the user to execute.

Output: the commit hash, confirmation that memory was updated, the
ready-to-run git push command, and a statement that Sprint 1 is complete
and Sprint 2 (Arratia sentiment smoothing) can begin in a new session.
```
