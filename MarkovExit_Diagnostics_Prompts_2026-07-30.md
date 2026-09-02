# Markov Exit Layer — DIAGNOSTIC BATCH (post-REV 4 second opinion)
Date: 2026-07-30 (context refreshed 2026-08-03). Branch: `prototype/markov-exit-layer`,
HEAD = **efc1831** (REV 4 committed and pushed to origin).

Copy-paste each prompt into a fresh Opus session, one at a time, in order. Each prompt is
self-contained. Prompt 1 is a HARD GATE — if it fails, STOP and report to Aman; do not run
Prompts 2-4.

## WHY THIS BATCH EXISTS

REV 4's two fixes (K-scaling, regime gate) were bit-for-bit inert: experimental Sharpe
+1.0783 vs baseline +1.5871 in both REV 3 and REV 4. Before deciding whether to keep,
retune, or shelve the exit layer, four things need to be established that the REV 3/REV 4
diagnostic backtest never measured. This batch measures them. **It writes no new model
code and adds no new signal** — it instruments what already exists.

Three findings from a read of `scripts/backtest_exit_layer.py` on 2026-07-30 that reframe
the problem and are pre-supplied here so Opus does not re-derive them:

1. **There is no cross-month exclusion state.** Line ~222: `exp_ret = 0.0 if
   review.action == "SELL" else ret_full`, evaluated fresh inside the per-month loop with
   no carry-over set. A SELL zeroes that ticker's contribution for that month only; next
   month it is back in the book if the ensemble still ranks it top-N. The docstring's
   "do NOT re-buy on the ensemble signal alone" (line 11) does **not** describe the code.
   Consequence: time-out-of-position is capped at one month, so the −0.5089 is **not**
   lockout opportunity cost. A "buy-back / re-entry gate" cannot recover it. That design
   direction is closed by the code as written — Prompt 2 confirms this empirically.

2. **Exits go to cash at weight 1/n and are not redistributed** (line ~253:
   `exp_m = float(np.sum(exp_rets) / n)`). In an up-drifting tape this is a mechanical
   drag that exists independently of whether the exit signal is any good. Prompt 3
   isolates it.

3. **`_annualized_sharpe` (line 136) is `mean/sd × √12` with no risk-free subtraction**,
   and the backtest applies no regime multiplier, no transaction costs, no slippage, and
   equal weights. Sprint 7's holdout harness almost certainly differs on several of these.
   That is the leading hypothesis for the +1.5871 vs 1.016 gap. Prompt 1 settles it.

## PRE-COMMITTED DECISION RULE (fixed before any code runs)

Written down now so the result cannot be rationalised afterwards.

- **Prompt 1 gate.** If the baseline/Sprint-7 gap cannot be reconciled to within 0.05
  Sharpe by an explicit, itemised bridge, then STOP. Every number produced by
  `backtest_exit_layer.py` is unreliable and the REV 3/REV 4 conclusions — in both
  directions — are withdrawn. No further prompts run until the harness is fixed.
- **KEEP the layer** only if ALL THREE hold:
  (a) SELL-flagged positions have a *negative* mean `ret_full` (the layer cuts losers on
      average), AND
  (b) the paired monthly difference is significant at the 0.05 level by BOTH sign test and
      paired t-test, in the layer's favour, AND
  (c) the redistribute-to-survivors variant still beats baseline on Sharpe **or** cuts max
      drawdown by ≥ 3 percentage points.
- **SHELVE the layer** (freeze branch, no merge, no production wiring) if (a) fails —
  i.e. if SELL-flagged positions went up on average. In that case the exit signal is
  anti-predictive on this window and no amount of gating, thresholding, or re-entry logic
  is worth building on top of it.
- **INCONCLUSIVE** (also → shelve, but note it) if (a) holds but (b) fails: right sign,
  no power. Revisit only with genuine bear data.

`hard_stop` is out of scope for the KEEP/SHELVE call — it fired 0/17 and Prompt 3's Case
census will most likely show why.

## VERIFIED CONTEXT (established 2026-07-30 — do not re-derive)

- **REPO PATH — read this before anything else.** As of 2026-08-03 the repo is at
  **`/Users/aman/dev/Ai Trading Agent`**. If your shell starts anywhere else, `cd` there
  first.

  **DEAD paths. Empty directories still exist at some of these. Never work in them:**
  `~/Projects/Ai Trading Agent`, `~/Desktop/Ai Trading Agent`,
  `~/Desktop/Ai Trading Agent OLD-DO-NOT-USE`, `~/Desktop/Stock_Project/Ai Trading Agent`.

  Guard, and ALL FOUR must pass: (1) `basename "$PWD"` == `Ai Trading Agent`;
  (2) `git rev-parse --show-toplevel` == `$PWD`; (3) `models/ensemble_models.pkl` exists
  with md5 `296e589f4da205eb1d171c2121d90f82`; (4) `scripts/backtest_exit_layer.py` and
  `src/exit/zhang_optimal.py` both exist.

  **If the guard fails, the overwhelmingly likely cause is that you are in one of the dead
  directories — NOT that data was lost.** `basename` alone is not sufficient to identify
  the repo, because every dead path has the same basename. Before reporting any kind of
  loss or damage, `cd /Users/aman/dev/Ai Trading Agent` and re-run all four checks. Only
  if the guard fails THERE is something actually wrong, and in that case stop and tell
  Aman — do not attempt recovery yourself. He has the repo on GitHub
  (`AmanVyas77/AI_Trading_Agent`), a `git bundle`, and file-level backups.
- **I/O hazard RESOLVED 2026-08-03.** The repo previously lived under `~/Desktop/` inside
  iCloud's synced Desktop tree and threw `OSError [Errno 35] Resource deadlock avoided` on
  cold files (it once crashed `git status` with a bus error). It has been MOVED OUT of the
  synced tree. If any read ever returns Errno 35 again, STOP and tell Aman — do not retry
  in a loop and do not work around it by reconstructing the file from memory.
- **Environment rebuilt 2026-08-03 and verified.** `.venv` was recreated and pinned from
  `requirements.lock.txt` — NOT from `requirements.txt`, whose `>=` pins pull pandas 3.x
  and sklearn 1.9 and silently break the frozen pickle. Confirmed good: pandas 2.3.3,
  scikit-learn 1.8.0, numpy 2.4.4, xgboost 3.2.0, scipy 1.17.1, numba 0.65.1,
  hmmlearn 0.3.3. The exit suite passes 89/89 and `models/ensemble_models.pkl` loads with
  no InconsistentVersionWarning. If you change the environment at all, re-verify both
  before trusting any number. Never run `pip install -r requirements.txt` in this repo.
- **`models/ensemble_models.pkl` is NOT tracked in git.** It exists only on disk — a fresh
  clone does not contain it. Never delete, overwrite, or `git clean` it. md5 must stay
  `296e589f4da205eb1d171c2121d90f82`.
- Branch: `prototype/markov-exit-layer`. Agents NEVER commit or push. Prompt 5 prints
  literal commands for Aman. Plain `-m`, **no Co-Authored-By trailer**.
- Frozen ensemble `models/ensemble_models.pkl` md5 `296e589f4da205eb1d171c2121d90f82`
  must be unchanged at the end of every session. Verify with `md5` before and after.
- Python: `.venv/bin/python`. scipy import lag (~12 min/process on Aman's box) still
  unresolved — prefer ONE long-running process per prompt over many short ones.
- Files in play: `scripts/backtest_exit_layer.py` (472 lines), `src/exit/exit_manager.py`,
  `src/exit/zhang_optimal.py`, `src/exit/andrade_dhmm.py`. Sprint 7 harness:
  `scripts/run_holdout.py`, results in `backtests/results/sprint7_results.json`.
- Do NOT touch `src/live/scorer.py`, the feature matrix, or the frozen model.
- Prompts 2-4 may modify `scripts/backtest_exit_layer.py` (instrumentation and variants
  only — the three return paths' existing semantics must not change) and may add new
  files under `scripts/` and `backtests/`. They must NOT change any file in `src/exit/`.

---

# PROMPT 1 — Baseline reconciliation (HARD GATE)

You are working in the AI Trading Agent repo on branch `prototype/markov-exit-layer`.

**Guard first.** Confirm `basename "$PWD"` == `Ai Trading Agent` and
`git rev-parse --show-toplevel` == `$PWD`. Confirm current branch. Record
`md5 models/ensemble_models.pkl` and confirm it is `296e589f4da205eb1d171c2121d90f82`.
If any read returns `OSError [Errno 35] Resource deadlock avoided`, STOP immediately and
report it — do not retry in a loop.

**The problem.** Two runs that should describe the same thing disagree by 0.57 Sharpe:

| Source | Window | Sharpe |
|---|---|---|
| `backtests/results/sprint7_results.json` (true holdout, frozen ensemble) | 2025-01 → 2026-06 | 1.016 |
| `backtest_exit_layer.py` `baseline` path | 2025-01-01 → 2026-06-30 | +1.5871 |

Same window, same frozen model, same universe. The gap is larger than the entire
−0.5089 effect the exit layer is being judged on. Until it is explained, no number from
`backtest_exit_layer.py` can be trusted.

**Your task.** Read both harnesses — `scripts/run_holdout.py` (and whatever it calls in
`src/strategies/ensemble/`, `src/live/rebalance.py`, `src/live/scorer.py`) and
`scripts/backtest_exit_layer.py` — and produce an **itemised reconciliation bridge**: start
at 1.016, apply one adjustment at a time, arrive at 1.5871. Each step must be a number you
computed by actually re-running with that one thing changed, not an estimate.

Candidate differences already identified from a read of `backtest_exit_layer.py` — verify
each, and look for others:

1. **Risk-free subtraction.** `_annualized_sharpe` (line 136) computes `mean/sd × √12`
   with no `rf` term. Check whether Sprint 7's Sharpe subtracts a risk-free rate. At
   rf ≈ 3-4% and portfolio vol ≈ 15-18%, this alone is worth roughly 0.2-0.3 Sharpe.
2. **Regime multiplier.** `build_portfolio_weights()` applies
   `regime_gate.get_live_regime_signal()`'s multiplier. Does the exit backtest's baseline
   apply it? (Read of the code suggests it does not.)
3. **Weighting.** Exit backtest uses equal-weight 1/n (docstring line 35). What does
   Sprint 7 use?
4. **Transaction costs / slippage / dividends.** Exit backtest ignores all three
   (docstring line 33). Does Sprint 7?
5. **Book construction.** TOP_N, MIN_SCORE, universe membership, and any DO_NOT_TRADE_BAND
   (0.0025) turnover damping — same in both?
6. **Return series construction.** Month-end-to-month-end closes vs. daily compounded;
   how the first and last partial months are handled; any missing-data / delisting
   handling difference.
7. **Score source.** Sprint 7 may read `data/processed/holdout_scores.parquet` (known to
   have been clipped to 2026-06 at one point) while the exit backtest regenerates via
   `score_months()`. Confirm both cover all 17 months and that per-month `n_held` matches.

**Deliverable.** Write `backtests/exit_layer_reconciliation_2026-07-30.md` containing:

- The bridge table: `1.016 → +adj1 → +adj2 → … → 1.5871`, with residual.
- For each adjustment, the file and line number of the code that causes it.
- A one-line verdict: RECONCILED (residual ≤ 0.05) or UNRECONCILED.
- If UNRECONCILED: your best hypothesis for the remaining gap, and specifically whether
  you found any lookahead in the exit harness that is absent from the Sprint 7 harness.
  Search explicitly for: use of full-window data in per-month calibration, `.shift()`
  sign errors, any `wide[t]` slice whose upper bound exceeds `me`, and any parameter fit
  on data after the decision date.

**STOP CONDITION.** If UNRECONCILED, stop here. Report to Aman. Do not run Prompts 2-4 —
they would measure artefacts.

Print your findings; do not commit.

---

# PROMPT 2 — Exit-decision replay and hit-rate decomposition

## ⚠️ AMENDED 2026-08-19 — read this before the body of Prompt 2

Prompts 1, 1C and 1D have all run. Their results change several numbers hard-coded below.
**Where this amendment and the body disagree, the amendment wins.**

Prerequisites, all met: Prompt 1 **RECONCILED** (report:
`backtests/exit_layer_reconciliation_2026-07-30.md`). Prompt 1C froze the vintage and its
Andrade-off check **FAILED**, overturning a premise. Prompt 1D resolved the units question
to **FORK 2** (report: `backtests/exit_layer_units_and_attribution_2026-08-04.md`).

Run the four-part guard, then run everything with
`--vintage backtests/vintage_2026-08-04`. Obey the numeric-provenance rule from Prompt 1D:
every number cites the file it was read from; nothing transcribed from memory; anything
quoted rather than recomputed is labelled `[carried]`.

**Superseded facts — do NOT rely on the body's versions of these:**

- Experimental Sharpe is now **+1.2822**, not +1.0783. The as-of regime lookahead was
  fixed (`get_regime_signal_asof()`); the old figure was contaminated. Δ vs baseline is
  **−0.3049**, not −0.5089. Every mention of −0.5089 or 1.0783 below is stale.
- **"Zhang never triggers independently" is FALSE.** Zhang fires 12 times in every
  configuration. The old `zhang=0` column meant zhang-*only*; the 12 sat inside `both`.
  Zhang-only Sharpe is 1.5915 vs baseline 1.5871 — a difference to be read as noise
  (T=17, 12 sells), not as an edge.
- `Zhang SELL ⟺ Case I ∧ state 2`, verified exactly. The **price test never binds**:
  `p0 ≥ x*` in 45/45 Case I calibrations, minimum margin 33.7×. Units are correct
  (`[x*] = [K]`, dollars vs dollars) — it is economically inert, not a bug. The paper's
  own AAPL example is degenerate the same way (x* = $0.0172 against a $542.10 sell).
- **Case II is 0/225 and unreachable** (`f1` min 2.592 vs ρ = 0.03), independent of K. The
  daily hard-stop is structurally inert; `hard_stop == baseline` is a tautology.
- The only Zhang component that changes decisions is the **Φ / Case-I gate**, which blocks
  46 of 58 state-2 rows. Removing Zhang wholesale would take SELLs from 12 to 58 — so it
  is NOT a no-op, and Task D below is what decides its fate.
- Use **pivotality** semantics for attribution, not the old zhang/andrade/both labels:
  `price_test_pivotal` 8, `state_test_pivotal` 41 (62 coupled), `case_gate_pivotal` 8.
  Note the 8 price-pivotal rows are pivotal only as the last condition standing — the
  price test is True on 45/45 with a 34× minimum margin and has zero discriminating power.

**Question this answers.** The drag has never been attributed to individual decisions.
We know Andrade drives it, but not whether those overrides were *wrong* — only that the
aggregate got worse. A signal can be right 60% of the time and still lose if the misses
are large. We need the per-decision distribution.

We also need to close out a design question Aman raised: whether a "re-entry / buy-back
gate" for force-sold names could recover the drag. A read of the code says no — there is
no cross-month exclusion state, so a sold name returns automatically next month if the
ensemble still ranks it. Confirm or refute that empirically here.

**Task A — instrument.** Modify `scripts/backtest_exit_layer.py` to emit a per-decision
record for every `(month, ticker)` it evaluates, not just the `|ret_full| > 0.05` subset
currently captured in `big_moves` (line ~239). Write to
`backtests/exit_decisions_2026-07-30.parquet` with at minimum:

`month`, `ticker`, `action` (SELL/HOLD), `trigger_source` (zhang/andrade/both/none),
`andrade_action`, `andrade_confidence` (if available), `zhang_case` (I or II),
`zhang_threshold_xstar`, `zhang_xstar0` (NaN in Case I), `p0`, `ret_full`,
`regime_multiplier`, `in_book_next_month` (bool), `ret_next_month` (the ticker's return
over the *following* month, NaN if unavailable).

Constraint: the return paths must be **bit-identical** after your change. Against
`--vintage backtests/vintage_2026-08-04` the current values are baseline **+1.5871**,
experimental **+1.2822**, hard-stop **+1.5871**, Zhang-only **+1.5915**. Assert all four
and fail loudly on drift beyond 1e-6 (1C reproduced the frozen vintage at 0.0e+00, so 1e-4
is far too loose). This is instrumentation only.

Add two fields to the record beyond the list above: `price_test_pivotal`,
`state_test_pivotal`, `case_gate_pivotal` (from Prompt 1D), and `phi` — the raw value of
`Φ = (ρ+λ1−f1)(ρ+λ2−f2) − λ1λ2` — plus the four fitted parameters `f1`, `f2`, `λ1`, `λ2`.
Task D needs them.

**Task B — decompose.** From the parquet, compute and report:

1. **Hit rate.** Of all SELL decisions, what fraction had `ret_full < 0`? This is the
   core number. A layer that cuts losers should be well above 50%.
2. **Mean and median `ret_full` given SELL.** Signed. If positive, the layer is
   systematically cutting winners and the pre-committed rule says SHELVE.
3. **Forgone-return decomposition.** Split the SELL population into:
   - *correct saves* — `ret_full < 0`. Sum the loss avoided.
   - *false positives* — `ret_full > 0`. Sum the gain forgone.
   Report both sums and the net. Reconcile the net against the observed Sharpe gap
   (they should tell the same story; if they don't, say so).
4. **Tail asymmetry.** The 5 largest correct saves and the 5 largest false positives, by
   ticker and month. Is the aggregate driven by a handful of names?
5. **Concentration.** How many distinct tickers account for 50% of the total forgone
   return? Is this a systematic signal problem or three bad calls on NVDA?

**Task C — settle the re-entry question.** Report:

6. `in_book_next_month` rate among SELL-flagged names. If this is high (say >70%), the
   ensemble already re-buys and a dedicated buy-back gate has nothing to add — the layer's
   cost is one month of forgone return per decision, full stop.
7. Confirm by code inspection AND by the data that no SELL-flagged ticker was excluded for
   more than one consecutive month by exit-layer logic (as opposed to by falling out of
   top-N on its own ensemble score). State the two causes separately — do not conflate
   "the exit layer kept it out" with "the ensemble stopped ranking it."
8. **Counterfactual re-entry upper bound.** For SELL-flagged names, compute what the
   experimental path's Sharpe would have been under a *perfect-foresight* re-entry rule
   that re-buys at the month's low. This is not implementable — it is a ceiling. If even
   the ceiling does not beat baseline, no realisable re-entry gate can, and the direction
   is dead. Report the ceiling Sharpe explicitly.

## TASK D — Φ stability under estimation error (ADDED 2026-08-19; this decides the Zhang leg)

**Why.** Prompt 1D found that Φ differences two terms of order 1.65×10⁴ down to a median
`|Φ|` of 86.8 — 0.52% of the input magnitude — and that **169 of 225 calibrations sit
within 1% of the sign flip**. Φ's sign is what assigns Case I, and the Case-I gate is the
only Zhang component that changes any decision. So the question is whether that gate
survives the uncertainty in its own inputs.

Back-of-envelope motivating this, to be replaced by your measured numbers: fitted `λ1`
median 123.5/yr implies mean regime duration ≈3 days; over a 250-day window, if roughly
half the time is spent in state 1, you observe ~60 exits from it, giving a relative
standard error on `λ̂1` near `1/√60` ≈ 13%. Thirteen percent of input error against a
one percent sign-flip margin would mean the gate is noise. **Measure it; do not assume
it.**

9. **Estimate the parameter standard errors properly.** For each of the 225 calibrations,
   derive SEs for `f1`, `f2`, `λ1`, `λ2` from the estimator actually used in
   `exit_manager.calibrate_for_month` — asymptotic (observed Fisher information / inverse
   Hessian) if the estimator admits it, otherwise a nonparametric block bootstrap over the
   250-day window with block length ≥ the fitted mean regime duration. State which you
   used and why. Report the distribution of relative SE per parameter, and the realised
   transition counts per calibration (this replaces the ~60 guess above).
10. **Perturbation test.** Draw N = 1000 parameter vectors per calibration from the
    sampling distribution implied by those SEs, respecting the estimated correlation
    between parameters — a diagonal draw will understate stability if the parameters
    covary, so use the full covariance where available and say so if you cannot. For each
    draw recompute Φ, the Case assignment, x*, and the resulting SELL/HOLD decision.
11. **Report, per calibration and in aggregate:** the fraction of draws that flip the sign
    of Φ; the fraction that flip the Case assignment; the fraction that flip the final
    decision. Then the headline: **across all 225 calibrations, what fraction of decisions
    are unstable at the parameters' own estimation error?**
12. **Propagate to the portfolio.** For a sample of at least 200 perturbed worlds, re-run
    the experimental path end-to-end and report the resulting distribution of experimental
    Sharpe. Give the 5th/50th/95th percentiles. Compare that spread against the −0.3049
    point estimate. If the spread swamps the effect, the layer's measured performance is
    not identified at this sample size regardless of hit rate.
13. **Interpretation, stated plainly, no hedging.** If a majority of decisions flip under
    the parameters' own uncertainty, then the Case-I gate is noise, Zhang contributes
    nothing reliable, and no tuning of K or ρ rescues it — because the instability is in
    the sign test, not in the threshold. Say so if that is what you find. If instead the
    gate is stable, say that too, and note it becomes the one genuinely load-bearing piece
    of the layer.

Task D does not depend on Tasks A–C's conclusions and can run in the same process.

**Deliverable.** `backtests/exit_decision_analysis_2026-08-19.md` with results 1–13, the
parquet, and a plain-English reading of what they mean. State whether pre-committed
condition (a) — negative mean `ret_full` given SELL — holds, evaluating it **separately**
for the 12 Zhang-gated sells and the Andrade-override sells, since 1D showed those are
different populations. Report Task D's stability verdict as its own headline, independent
of (a).

Do not issue a KEEP/SHELVE verdict — conditions (b) and (c) are still untested and belong
to Prompts 4 and 3. Do not use the word "pass".

Print your findings; do not commit.

---

# PROMPT 3 — Mechanical variants: separate plumbing from signal

Prerequisite: Prompt 1 RECONCILED. Guard first, as before.

**Question this answers.** Part of the −0.5089 may not be signal error at all. Three
mechanical choices in the harness each impose a drag in an up-drifting tape regardless of
whether the exit signal has any information. Isolate each.

**Variant A — redistribute instead of cash.** Currently `exp_m = np.sum(exp_rets)/n`
(line ~253) keeps the exited weight in cash at 0% return. Add a fourth path,
`experimental_redist`, that renormalises the surviving book to equal weights
(`np.mean` over survivors) so the portfolio stays fully invested. Report its Sharpe.
The difference `experimental_redist − experimental` is the **pure cash drag**, attributable
to plumbing, not to Andrade. Report it as its own line item.

**Variant B — Zhang case census.** Across all 225 `(ticker, month)` calibrations, report
the Case I vs Case II split, and the distribution of `f1` vs `rho`. Case II requires
`rho > f1`. Hypothesis to test: in this tape `f1 > rho` nearly everywhere, so Case II —
and therefore `x*_0`, and therefore the entire daily hard-stop mechanism — is
**structurally undefined**, not merely untriggered. If the split is 225/225 Case I, say so
plainly: the hard-stop path cannot fire under these parameters, REV 4's
`test_daily_hard_stop_is_price_crossing` is exercising a branch that never instantiates in
practice, and `hard_stop == baseline` is a tautology rather than a result.

Also report the distribution of `x*` relative to `p0`. How far out of the money is the
Zhang threshold typically? This quantifies *how* unidentified Zhang is — a threshold 40%
below spot is a different statement from one 2% below.

**Variant C — the regime gate is not a historical signal.** The header of
`backtest_exit_layer.py` (lines ~50-56) admits `get_live_regime_signal()` has no `as_of`
parameter, so every one of the 225 calibrations caches **the same regime as of run time**.
This means Fix B was never capable of working in a backtest — the "NEUTRAL for all 17
months" result is not a reading of the 2025-26 tape, it is one reading of 2026-07-30
stamped onto all 17 months.

Per Sprint 1's commit message (3d41586), `regime_gate.py` was written with a **rule-based
VIX / yield-curve path for historical backtest** alongside the RAG blend for live mode.
Find that historical function. If it exists and takes a date, wire a fourth variant that
uses the *point-in-time* regime for each month, and report:

- the actual regime classification for each of the 17 months;
- how many months were RISK_ON (i.e. how often Fix B *would* have fired);
- `experimental` Sharpe with the point-in-time gate.

If the historical function does not exist or cannot be read (see the EDEADLK hazard note),
say so and stop this variant — do not write a new regime classifier.

**Deliverable.** `backtests/exit_layer_variants_2026-07-30.md` with an attribution table:

```
baseline                          +1.5871
  − cash drag (Variant A)          −X.XXXX
  − Andrade signal error           −X.XXXX
  − regime-gate mis-wiring (C)     ±X.XXXX
experimental (as measured)        +1.0783
```

Numbers must sum. Verify the sum programmatically with an assert; do not eyeball it.

Print your findings; do not commit.

---

# PROMPT 4 — Statistical power and risk metrics

Prerequisite: Prompt 1 RECONCILED. Guard first.

**Question this answers.** Nobody has checked whether −0.5089 is distinguishable from
noise on 17 observations, and nobody has looked at the metric an exit layer actually exists
to move.

**Task A — significance.** Using the 17 monthly return series for each path:

1. The paired difference series `experimental − baseline`. Report mean, sd, and the count
   of months where it is negative.
2. **Sign test** on that series. Exact binomial p-value. (17 months: 13/17 negative gives
   p ≈ 0.025; 9/17 gives p ≈ 1.0. This one number does more work than the Sharpe delta.)
3. **Paired t-test.** t-statistic and p-value.
4. **Stationary bootstrap** (10,000 resamples, expected block length 3) on the paired
   difference, to get a 95% CI on the Sharpe delta itself. Report the interval.
5. **Lo (2002) standard error** on each path's Sharpe individually, so the width of the
   estimate is on the record: `SE(Ŝ) ≈ sqrt((1 + Ŝ²/2)/T)` in per-period units, then
   annualise. Report each Sharpe as `point ± 1SE`.
6. State plainly whether the −0.5089 clears the 0.05 bar. If it does not, the honest
   summary is "the layer's effect is unmeasured on this window," not "the layer hurts."

**Task B — risk metrics.** Sharpe is the wrong lens for an exit layer; its job is
left-tail management. For all four paths (baseline, experimental, experimental_redist,
hard_stop) report: **max drawdown**, **Sortino ratio**, **Calmar ratio**, worst single
month, and the 5th-percentile monthly return.

If `experimental` improves max drawdown materially while losing Sharpe, that is a tradeoff
to price rather than a flat loss, and it changes the recommendation. If it degrades both,
the case closes.

**Task C — do NOT do the following**, and note in your write-up that you were told not to:
do not tune Andrade's confidence threshold, do not retune the regime detector, and do not
search over any parameter on this window. Every such search would be in-sample on the same
17 months the effect was measured on and would produce a number that is not evidence.

**Deliverable.** `backtests/exit_layer_significance_2026-07-30.md` with all Task A and
Task B results in one table, plus a two-sentence plain-English conclusion.

Print your findings; do not commit.

---

# PROMPT 5 — Consolidated verdict and commit commands

## ⚠️ AMENDED 2026-08-19 — Prompts 3 and 4 are SKIPPED. Read this instead of the body below.

**The verdict is already determined.** Prompt 2 measured pre-committed condition (a):
mean `ret_full` given SELL = **+0.1617** (positive), hit rate **9/29 = 31.0%**. The rule
states that (a) failing is *sufficient* for SHELVE, independent of (b) and (c).

Prompts 3 and 4 test (c) and (b), which only matter if (a) passes. Running them now would
be searching for grounds to reopen a settled question — the exact failure mode the
pre-commitment exists to prevent. **Do not run them.** Two cheap pieces from them are
folded into Task A below because the record deserves them, not because they can change
the outcome.

Prerequisites, all met: Prompt 1 RECONCILED; Prompt 1C froze the vintage and its
Andrade-off check failed, overturning the "Zhang never triggers" premise; Prompt 1D
resolved units to FORK 2; Prompt 2 failed condition (a).

Guard first (four checks). Run everything with `--vintage backtests/vintage_2026-08-04`.
Numeric-provenance rule applies: every figure cites its source file, nothing from memory,
`[carried]` on anything quoted rather than recomputed.

### Task A — the two folded-in measurements (run before writing)

1. **Redistribute-to-survivors variant.** Currently exited weight sits in cash at 1/n
   (`exp_m = np.sum(exp_rets)/n`). Add a path that renormalises the surviving book to
   equal weight so the portfolio stays fully invested. Report its Sharpe. The difference
   `experimental_redist − experimental` is **mechanical cash drag**, attributable to
   plumbing rather than to Andrade. The writeup must not charge plumbing to the signal.
2. **Left-tail metrics** for baseline, experimental, experimental_redist, hard_stop, and
   Zhang-only: max drawdown, Sortino, Calmar, worst single month, 5th-percentile monthly
   return. An exit layer's stated purpose is left-tail management; the record should show
   what it actually did to the left tail, even in failure. Report even if it worsens
   every one — especially then.

**Do NOT** tune Andrade's confidence threshold, retune the regime detector, or search over
any parameter on this window. Any such search is in-sample on the same 17 months the
effect was measured on and produces a number that is not evidence. Note in the writeup
that you were instructed not to.

### Task B — write `backtests/exit_layer_verdict_2026-08-19.md` (≤ 3 pages)

1. **Verdict — SHELVE.** Apply the pre-committed rule verbatim. State that (a) failed at
   +0.1617 mean / 31.0% hit rate, and that (b) and (c) were deliberately not tested
   because (a) failing is sufficient. Do not soften, do not add criteria after the fact,
   and do not use the word "pass" anywhere except when quoting the rule.
2. **The decomposition, with significance.** Zhang-gated sells: n=12, 58.3% hit, mean
   +0.0041 — break-even, and at n=12 the binomial p ≈ 0.39, i.e. no signal either way.
   Andrade-driven: n=17, **2 correct out of 17**, mean +0.2730. Under a coin-flip null
   that is p ≈ 0.0012 — Andrade's exit signal is not merely weak, it is **significantly
   anti-predictive**. Pooled 9/29 gives p ≈ 0.031. Recompute all three p-values yourself
   rather than carrying these; report your figures and flag any disagreement.
3. **Concentration caveat — state it prominently, do not bury it.** Five false positives
   (INTC, AMD, ON, ANET, AVGO) all fall in 2026-03-31 and account for roughly 63% of the
   total gain forgone. The *direction* of the failure is robust (31% over 29 decisions,
   Andrade at p ≈ 0.001). The *magnitude* rests heavily on one month of semiconductors
   rallying into a STRONG_SELL. Both facts belong in the record; a reader who sees only
   the aggregate would overstate the effect size.
4. **Attribution.** Cash drag (Task A.1) vs Andrade signal error vs the regime
   mis-wiring already corrected (−0.5089 → −0.3049, a ~0.20 Sharpe artifact of the
   lookahead). Numbers must sum; assert the sum programmatically, do not eyeball it.
5. **What is now known about Zhang.** Units are correct (`[x*] = [K]`, FORK 2). The price
   test never binds — 45/45, minimum margin 33.7×, and needing `K_fraction` ≈ 42% at the
   median to bind. Zhang's own AAPL example is degenerate the same way ($0.0172 against a
   $542.10 sell), so the rule was inert *as published* for equities; the port is faithful.
   Case II is 0/225 and unreachable (`f1` min 2.592 vs ρ = 0.03) independent of K, so the
   daily hard-stop is structurally dead and REV 4's fix A was never testable here. Record
   that the 50/50 unit tests validate the algebra and say nothing about the integration —
   future work must not re-litigate the math.
6. **Φ stability.** Φ sign flips in ~22% of bootstrap draws and 91.6% of calibrations sit
   within 1% of the flip, but decisions flip only ~6.7% because 167/225 rows are state 1
   where Zhang cannot sell regardless — the state constraint firewalls the instability.
   Note this explicitly as a correction to the earlier "the gate is a coin flip" framing.
   Then the finding that matters: across 500 perturbed worlds, experimental Sharpe spans
   1.1941–1.3666 and **every single world is below baseline** (max 1.4057 < 1.5871). The
   drag is identified above sampling noise; there is no parameter draw where this wins.
7. **The re-entry / buy-back question — CLOSED, with arithmetic.** The exit layer is
   stateless month-over-month (`exit_manager.py:266-368`); all 16 next-month absences are
   the ensemble dropping the ticker, not the layer excluding it. Perfect-foresight
   re-entry at each month's low gives 1.8108 vs baseline 1.5871. Getting from the layer's
   1.2822 back to merely *matching* baseline needs +0.3049 of the +0.5286 that perfect
   foresight buys — so any re-entry gate must capture **~58% of physically impossible
   timing just to break even with doing nothing.** Record this so the idea is not
   reopened from intuition later.
8. **Harness findings that outlive this layer.** The as-of regime lookahead
   (`get_live_regime_signal()` inside a backtest loop, against its own docstring) — check
   whether that pattern exists anywhere else in the codebase and say so. And the vintage
   problem: Sprint 7's 1.016 no longer reproduces, giving 1.0909, because the nightly
   news backfill retroactively revises historical inputs. That generalises to every
   backtest number in the project. State the two open questions plainly: is the revision
   path prices, or news → sentiment → scores; and does feature construction filter
   articles by `published_at` relative to each decision date? If it does not, there is
   lookahead in the **entry** model, which matters far more than anything here.
9. **Open and untested.** Bear-regime behaviour. Pre-register max-drawdown reduction and
   Calmar as the metrics before any such run. Note that in-sample entry contamination
   biases a 2022 test *against* the layer, making it conservative rather than invalid.
10. **Recommendation** in three sentences.

### Task C — update the repo's own memory

Update `memory/hold_sell_layer_design.md` (the repo file) with a dated section recording
the SHELVE outcome, so a future session does not restart from the design doc's optimistic
framing. Also update `memory/future_ideas.md` to move the Markov exit layer from "parked
stretch goal" to "attempted, shelved 2026-08-19, see verdict" with a pointer to the
verdict report. A parked idea and a tested-and-rejected idea are different things and the
queue should not confuse them.

Note in that entry that the queued parameter-recovery / identifiability stretch goal is
now **partly answered and no longer worth running as specified**: Prompt 2 measured 127
realised regime transitions per calibration (median) and relative SEs of 8.5–13.2%, so
identifiability is better than feared and was never the binding constraint. The binding
constraint was that Andrade's signal is anti-predictive. Say that plainly so nobody
spends a week on the simulation to rediscover it.

**Commit — Aman runs these himself. Print them; do not execute any git command.**

```
cd "$(git rev-parse --show-toplevel)"
git status
git add scripts/backtest_exit_layer.py backtests/ memory/hold_sell_layer_design.md memory/future_ideas.md MarkovExit_Diagnostics_Prompts_2026-07-30.md
git status
git commit -m "Markov exit layer: SHELVE. Condition (a) fails (mean ret_full given SELL +0.1617, hit rate 31%); Andrade anti-predictive at p~0.001; re-entry gate closed (needs 58% of perfect foresight to break even)"
git log --oneline -3
```

Check `git status` output before committing — `backtests/vintage_2026-08-04/` is 610 MB
and must NOT be staged. If it appears, stop and tell Aman; it needs a `.gitignore` entry.

Push only if Aman decides to:

```
git push origin prototype/markov-exit-layer
```

Before printing the commands, verify `md5 models/ensemble_models.pkl` is still
`296e589f4da205eb1d171c2121d90f82` and that `git status` shows nothing staged under
`src/exit/` or `models/`. If either check fails, print the failure instead of the commands.

---

# PROMPT 1C — Vintage freeze + Andrade-off sanity check
(Added 2026-08-04, after Prompt 1 RECONCILED and the as-of regime lookahead was fixed.
Run this BEFORE Prompt 2.)

Guard first, as always: `basename "$PWD"` == `Ai Trading Agent`, `git rev-parse
--show-toplevel` == `$PWD`, branch `prototype/markov-exit-layer`, `md5
models/ensemble_models.pkl` == `296e589f4da205eb1d171c2121d90f82`.

**NEW mandatory guard — assert the interpreter before anything else.** Print
`sys.executable`, `numpy.__version__`, `sklearn.__version__`, `pandas.__version__` and
assert they are the repo `.venv` with numpy 2.4.4 / sklearn 1.8.0 / pandas 2.3.3. Prompt 1
was initially run under `/opt/anaconda3/bin/python3` (numpy 1.26.3 / sklearn 1.2.2) by
mistake. Never run any rung of this batch under anaconda base. Fail loudly if the versions
do not match.

## Task A — physically freeze the data vintage

`src/data_vintage.py` fingerprints the data (sha256 over rows read) and detects revision.
That is a tripwire, not a freeze. The AV news backfill runs nightly through roughly
2026-08-11, retroactively revising historical inputs, so without a physical snapshot
Prompts 2, 3 and 4 will each read a different dataset and none of their numbers will be
comparable to each other or to the −0.3049 just measured.

1. Create `backtests/vintage_2026-08-04/` containing a byte-for-byte copy of every input
   the diagnostic path reads: `data/quant_research.db`, `data/processed/
   ensemble_feature_matrix.parquet`, and any scores parquet the harness consumes. Do not
   copy `models/ensemble_models.pkl` — reference it in place, and assert its md5.
2. Write `backtests/vintage_2026-08-04/MANIFEST.json`: for each file, absolute source
   path, size in bytes, sha256, and mtime; plus the `data_vintage.py` digest, row count
   and ticker count; plus the git HEAD sha; plus the interpreter/version block from the
   guard.
3. Add a way for `scripts/backtest_exit_layer.py` to read from the frozen snapshot — an
   `--vintage <dir>` argument or an env var, your choice, but it must be explicit and it
   must be logged in every run's output so no future run is ambiguous about what it read.
4. Re-run the three paths against the frozen snapshot and confirm they reproduce
   baseline +1.5871 and experimental +1.2822 to ≤1e-6. If they do not, STOP and report —
   that would mean the snapshot is not capturing everything the harness reads.

## Task B — Andrade-off sanity check (cheap, decisive)

Three points now suggest the layer improves monotonically as Andrade does less:

| Andrade suppressions | experimental Sharpe |
|---|---|
| 0                    | 1.0783 |
| 37 (13/17 months)    | 1.2822 |
| all (implied)        | 1.5871 = baseline |

The third row is an inference, not a measurement. Measure it.

5. Add a flag that disables the Andrade override entirely, leaving Zhang as the only exit
   trigger. Run the experimental path with it. **Expected: exactly baseline (+1.5871) to
   floating-point tolerance**, because Zhang was found to produce 0 independent triggers
   across 225 calibrations.
6. If it matches: the trigger decomposition is confirmed, Zhang contributes nothing on
   this window, and Andrade is the entire active surface. Say so plainly.
7. **If it does NOT match: STOP and report immediately.** That would mean Zhang fires
   independently somewhere, the 0-independent-triggers finding is wrong, and the whole
   attribution in the REV 3/REV 4 analysis needs redoing before any further prompt runs.
8. Either way, report the Zhang Case I vs Case II split across all 225 calibrations while
   you are in there, and the distribution of `x*` relative to `p0`.

## Task C — correct the verdict language in the Prompt 1 report

The Prompt 1 report records "MINIMUM PASS (rule 2)". That is the OLD verdict rule from
`MarkovExit_Prompts.md` (beat REV 3's 1.0783). It is NOT the rule governing this batch.

This batch's pre-committed rule requires ALL THREE of: (a) negative mean `ret_full` given
SELL, (b) paired monthly difference significant at 0.05 by BOTH sign test and paired
t-test, (c) redistribute-to-survivors variant beats baseline on Sharpe or cuts max
drawdown by ≥3pp. None of the three has been tested. Prompt 2 tests (a), Prompt 3 tests
(c), Prompt 4 tests (b).

9. Amend the report: the correct status is **NO VERDICT YET — lookahead corrected,
   experimental improved from 1.0783 to 1.2822, still −0.3049 vs baseline.** A losing
   configuration that loses less is not a pass. Do not use the word "pass" anywhere in
   the report except when quoting this batch's three-part rule.

## Deliverable

`backtests/exit_layer_vintage_and_andrade_off_2026-08-04.md` covering Tasks A, B and C,
plus the manifest and the amended Prompt 1 report. Print findings; do not commit.

---

# PROMPT 1D — Zhang units resolution + attribution redo
(Added 2026-08-04 after Prompt 1C Task B FAILED. Run BEFORE Prompt 2.)

## STANDING GUARDS (all prompts from here)

- Path/branch/md5 guard as before. `_assert_interpreter()` must pass — `.venv`,
  numpy 2.4.4 / sklearn 1.8.0 / pandas 2.3.3. Never anaconda base.
- Run everything against the frozen snapshot: `--vintage backtests/vintage_2026-08-04`.
- **NEW — numeric provenance rule.** Every number in the deliverable cites the file (and
  line, where applicable) it was read from. Never transcribe a figure from memory. If a
  value is quoted from an earlier report rather than recomputed this session, label it
  `[carried]`. Two prior prompts each shipped a numbers-provenance defect (a mislabelled
  residual, then fabricated sha256s in a first draft); both were self-caught, but the
  mitigation is now a rule, not vigilance.
- Papers live in `research_papers/quant_modeling/` (`When to Sell Markov Chain Asset.pdf`,
  `Stock Market Index Trading Algorithm.pdf`). If absent, STOP and tell Aman.

## WHAT PROMPT 1C ESTABLISHED (do not re-derive)

- Zhang-only = 1.591542820613977 vs baseline 1.587144507439707; diff +0.0044. **Treat as
  noise** — Lo SE on a Sharpe of 1.59 at T=17 is ≈±0.88, and there are only 12 sells.
  The correct reading is "Zhang-only ≈ baseline", nothing stronger.
- `Zhang SELL ⟺ Case I ∧ state 2`, verified exactly (12 = 12 = 12 in every config).
- Case I 45/225 (20%), **Case II 0/225**, never-sell 180/225 (80%).
- x*/p0 over the 45 Case I: median 0.0024, max 0.030. `p0 ≥ x*` in **45/45**.
- Therefore: the price test never binds; Zhang reduces to a Case-I-gated passthrough of
  the Andrade DHMM state; there is no Andrade-independent decision surface.

## TASK A — resolve the units of x* (HIGHEST PRIORITY; everything else waits on it)

Hypothesis to test: **the harness compares a raw dollar price against a threshold that
is not denominated in dollars.** If true, "always breached" is a dimensional artifact,
not an economic result, and the Zhang leg has never actually been tested.

Evidence motivating this, from the paper's own worked examples:

- Example 2 (p.19): `(x*, x*_0) = (0.012478, 0.033333)` with `X0 = 0.013326`. X0 sits
  *between* the two thresholds — a sensible configuration only if X is a quantity of
  order 0.01.
- Table 1 (p.22), AAPL 2H-2012: `x* = 0.017213`. AAPL traded around $600 pre-split in
  that period. `K = 0.01`. Neither figure is plausible as US dollars at that price level.
- The fitted rates in that row (`λ1 = 135.25`, `λ2 = 130.95`, per year) imply a mean
  regime duration of roughly 2.7 days, and `f1 = 4.89` implies ~489% annualised drift in
  the uptick state. Confirm the time units of f and λ while you are in there.

Your task:

1. Read the paper and state precisely **what the process X is**: raw price, normalised
   price (P/P₀), log-price, discounted price, or something else. Quote the defining
   equation and its page.
2. State the **units of K** in the paper's formulation, and whether K is absolute or
   proportional to the price scale.
3. Determine whether `p0 ≥ x*` as implemented in `src/exit/zhang_optimal.py` /
   `exit_manager.py` compares like with like. Show the dimensional argument explicitly.
4. Report the **time units** of `f1`, `f2`, `λ1`, `λ2`, `ρ` in the paper, and confirm the
   harness's `dt = 1/252` calibration produces parameters in those same units.

**Then fork, and say which fork you are in:**

- **FORK 1 — units are wrong.** Fix the comparison so both sides share a numeraire.
  Re-run all paths. Report: new Zhang trigger count, new Case I/II split, new x*/p0
  distribution, how many of the 45 Case I calibrations now have a *binding* price test,
  and the new Sharpe for baseline / experimental / Zhang-only / hard-stop. Note
  explicitly whether Case II ever appears once units are corrected — if it does, the
  daily hard-stop stops being structurally inert and REV 4's fix A becomes testable for
  the first time.
- **FORK 2 — units are right.** Then with a realistic transaction cost (K = 0.1% of
  price) x* is structurally ~0.2% of price and cannot bind for this asset class. State
  that as a finding, show the algebra for what K would have to be for x* to bind at
  realistic prices, and say whether that K is economically defensible. If it is not, the
  recommendation is to remove the Zhang leg entirely rather than tune it.

Do not proceed to Task B until Task A's fork is settled and, if FORK 1, the fix is in.

## TASK B — attribution redo with honest trigger semantics

The current taxonomy credits Zhang for sells where its price test did no work. Replace it
with a **pivotality** test: a condition counts as a trigger only if flipping it changes
the decision.

5. For every (ticker, month) decision, record: `price_test_pivotal` (would the decision
   change if the price condition were forced True? forced False?), `state_test_pivotal`
   (same for the Andrade state), and `case_gate_pivotal` (same for Case I membership).
6. Rebuild the trigger table on those three flags rather than on the old
   zhang/andrade/both labels. Report counts per month.
7. Report the distribution of the **fitted** parameters across all 225 calibrations —
   `f1`, `f2`, `λ1`, `λ2`, and implied mean regime duration `1/λ`. This is nearly free
   while you are in there and it feeds a queued follow-on: if fitted regimes are ~3 days
   (as the paper's AAPL row implies) then 250 daily closes contain ~135 transitions and
   parameter identifiability is far better than if regimes are monthly (~8 transitions).
   Do not analyse this further here — just report the distributions.

## TASK C — stale verdict block

8. `backtest_exit_layer.py` still prints the old rule-2/rule-3 verdict block. First
   **verify it is print-only** and cannot affect any computed value. If confirmed
   print-only, replace it with this batch's three-part rule and state in the report that
   run artefacts are unaffected because the block has no data path. If it is *not*
   print-only, do not touch it — report what it feeds.

## Deliverable

`backtests/exit_layer_units_and_attribution_2026-08-04.md`. Every number carries its
source per the provenance rule. Print findings; do not commit.

**No verdict on the layer in this prompt.** This batch's three-part rule is still
untested; Prompt 2 tests condition (a). Do not use the word "pass".
