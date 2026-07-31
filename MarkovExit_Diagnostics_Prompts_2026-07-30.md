# Markov Exit Layer — DIAGNOSTIC BATCH (post-REV 4 second opinion)
Date: 2026-07-30. Branch: `prototype/markov-exit-layer` (HEAD = 4a2a699 + uncommitted REV 4).

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

- **Repo path is volatile.** Do NOT hardcode it. Every session starts with:
  `basename "$PWD"` must equal `Ai Trading Agent`, and `git rev-parse --show-toplevel`
  must equal `$PWD`. If not, STOP and ask Aman for the current path. (It has moved three
  times; the path written in `MarkovExit_Prompts.md` line ~95 —
  `/Users/aman/Projects/Ai Trading Agent` — is STALE and no longer exists.)
- **ACTIVE I/O HAZARD.** As of 2026-07-30, `src/strategies/ensemble/regime_gate.py`
  returns `OSError [Errno 35] Resource deadlock avoided` on read — the iCloud
  Desktop-sync eviction problem, recurring at the current Desktop location. If any read
  fails this way: STOP, tell Aman, and have him either disable iCloud "Desktop &
  Documents Folders" sync or move the repo off Desktop. Do not retry in a loop and do not
  work around it by reconstructing the file from memory.
- Branch: `prototype/markov-exit-layer`. Agents NEVER commit or push. Prompt 5 prints
  literal commands for Aman. Plain `-m`, **no Co-Authored-By trailer**.
- Frozen ensemble `models/ensemble_models.pkl` md5 `296e589f4da205eb1d171c2121d90f82`
  must be unchanged at the end of every session. Verify with `md5` before and after.
- Python: `.venv/bin/python`. scipy import lag (~12 min/process on Aman's box) still
  unresolved — prefer ONE long-running process per prompt over many short ones.
- Files in play: `scripts/backtest_exit_layer.py` (472 lines), `src/exit/exit_manager.py`,
  `src/exit/zhang_optimal.py`, `src/exit/andrade_dhmm.py`. Sprint 7 harness:
  `scripts/run_holdout.py`, results in `sprint7_results.json`.
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
| `sprint7_results.json` (true holdout, frozen ensemble) | 2025-01 → 2026-06 | 1.016 |
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

Prerequisite: Prompt 1 returned RECONCILED. Run the same path/branch/md5 guard first.

**Question this answers.** The −0.5089 has never been attributed to individual decisions.
We know Andrade drove every override, but not whether those overrides were *wrong* — only
that the aggregate got worse. A signal can be right 60% of the time and still lose if the
misses are large. We need the per-decision distribution.

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

Constraint: the three existing return paths must produce **bit-identical** Sharpe numbers
after your change (+1.5871 / +1.0783 / +1.5871). Assert this in the script and fail loudly
if it drifts beyond 1e-4. This is instrumentation only.

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

**Deliverable.** `backtests/exit_decision_analysis_2026-07-30.md` with all eight numbered
results, the parquet, and a plain-English two-paragraph reading of what they mean. State
whether pre-committed condition (a) — negative mean `ret_full` given SELL — holds.

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

Prerequisite: Prompts 1-4 complete (or Prompt 1 failed, in which case write up the failure
and stop after step 1).

**Task.** Write `backtests/exit_layer_verdict_2026-07-30.md`, ≤ 2 pages, structured as:

1. **Verdict** — KEEP / SHELVE / INCONCLUSIVE, applying the pre-committed decision rule at
   the top of this file verbatim. State which condition (a)/(b)/(c) passed or failed and
   with what number. Do not soften and do not add new criteria after the fact.
2. **Attribution** — the Prompt 3 table: how much of −0.5089 was cash drag, how much was
   Andrade, how much was the regime mis-wiring.
3. **What is now known about Zhang** — the Case census. If Zhang is structurally Case I
   throughout, record that the Zhang integration is *unidentified at this horizon and asset
   class*, distinct from *wrong*. The 50/50 unit tests validate the formula; they say
   nothing about the integration. Future work should not re-litigate the math.
4. **The re-entry / buy-back question** — closed, with the Prompt 2C numbers, including the
   perfect-foresight ceiling. Record why it is closed so it does not get reopened from
   intuition later.
5. **Known-bad harness findings** — anything Prompt 1 turned up. If the reconciliation
   required fixing something in `backtest_exit_layer.py`, that fix matters beyond this
   layer and should be called out for the production harness too.
6. **Open, untested** — bear-regime behaviour. Note the metric that a future bear test must
   use (max drawdown reduction and Calmar, pre-registered before the run), and note that
   the in-sample-entry contamination biases such a test *against* the layer, making it
   conservative rather than invalid.
7. **Recommendation to Aman** in three sentences.

Then update `memory/hold_sell_layer_design.md` (repo file) with a dated section recording
the outcome, so the next session does not restart from the design doc's optimistic framing.

**Commit — Aman runs these himself. Print them; do not execute any git command.**

```
cd "$(git rev-parse --show-toplevel)"
git status
git add scripts/backtest_exit_layer.py backtests/ memory/hold_sell_layer_design.md
git status
git commit -m "Markov exit layer: diagnostic batch 2026-07-30 — reconciliation, decision replay, variant attribution, significance"
git log --oneline -3
```

Push only if Aman decides to:

```
git push origin prototype/markov-exit-layer
```

Before printing the commands, verify `md5 models/ensemble_models.pkl` is still
`296e589f4da205eb1d171c2121d90f82` and that `git status` shows nothing staged under
`src/exit/` or `models/`. If either check fails, print the failure instead of the commands.
