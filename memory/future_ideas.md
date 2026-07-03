# Future Ideas — AI Trading Agent

Running list of ideas surfaced during sprint work that we don't want to
lose but aren't executing right now. Each entry ends with a suggested
insertion point (which sprint, which module) so it can be pulled off
this list without re-deriving the context.

---

## Rank-based selection under rolling windows

**Status:** open. Priority: **BELOW** the live paper-trading pipeline
(Option B) below. Spun out of Sprint 6's refuted result — see the
"Sprint 6 outcome log — Rolling window (REFUTED)" entry near the bottom
of this file.

Sprint 6 showed rolling 36-month training does not fail on ranking
skill — it fails because compressed scores fall under the fixed absolute
`MIN_SCORE=0.52` gate, starving breadth (14 of ~23 test months went to
cash). The obvious fix is to decouple selection from the absolute score
level:
- **Rank-based / top-N with no absolute floor:** always hold the top-N
  by score each month regardless of level, so a compressed score
  distribution still fills the book.
- **Per-month score z-scoring:** standardise scores within each month
  before thresholding, so the gate adapts to the month's distribution
  instead of a fixed 0.52.

### ⚠️ Multiple-testing warning (read before iterating)
- **Sprint 6 counts as trial 1** on the rolling-window family. Every
  variant tested against the same 2023-24 test window inflates the
  family-wise false-positive rate (this is exactly the DSR problem the
  AlgoXpert controls guard against at the feature level).
- Any variant needs its **own pre-committed PASS rule** (fixed in
  advance, three-metric style like Sprint 6), and **do not iterate
  variants until one passes** its pre-committed rule. No p-hacking the
  threshold/selection rule against the test set.

**Suggested insertion:** a future sprint — edit
`src/strategies/ensemble/portfolio_builder.py::_select_monthly_holdings`
(selection logic only), re-enable `WINDOW_YEARS=3` in `model_trainer.py`
(mechanism is retained and dormant).

---

## Live trading pipeline — paper stage (Option B)

**Status:** open, **now the top Phase 5 candidate** — the model-refresh
question is settled (Sprint 6 refuted rolling; expanding window stays),
so there is no longer a training-window blocker ahead of this.

Sprint 5 established the current ensemble (regime gate + TimesFM +
XGBoost, expanding train) clears the Sprint 0 baseline on Sharpe (0.682
vs 0.600) and beats it on test CAGR (18.27% vs 13.81%). Risk-adjusted
profile is deployment-worthy for a paper stage.

Prerequisites before opening this (these are the real blockers):
- **FinBERT backlog** — sentiment features are stale since 2024-12-27;
  a live deployment needs current-month sentiment, so the backlog must
  be caught up (and kept fresh on a schedule).
- **Regime-gate LIMIT-7 / staleness fixes** — the live regime signal
  path needs the LIMIT-7 and staleness handling resolved before it can
  drive real rebalances.
- **Live scorer for unlabeled current months** — the current pipeline
  only scores months that have realised labels; live trading needs a
  scorer that runs on the latest *unlabeled* month.
- Decide on execution venue (broker API, paper account credentials in
  `.env`).
- Add a `src/live/` module with a scheduled rebalance harness driven by
  the same `portfolio_builder` outputs — no new alpha logic, just a
  runner that reads the latest score / weight files and posts orders.

**Suggested insertion:** Phase 5 (next) — new module
`src/live/paper_runner.py`.

---

## Sprint 5 outcome log — TimesFM (VALIDATED)

TimesFM is no longer a to-try; it is now in production as one of the
23 ensemble features.

- **Verdict:** PASS (per `backtests/results/sprint5_results.json`).
- **Test-CAGR recovery:** 163.35% of the 7.04pp Sprint-0-vs-Sprint-4
  gap. TimesFM did not just close the gap — it beat the Sprint 0 test
  baseline by +4.46pp (18.27% vs 13.81%).
- **Deflated t-stat:** 30.48 (threshold = 1.0). Signal is real, not
  multiple-testing noise.
- **XGBoost gain rank:** 17/23. Middling in tree splits, statistically
  significant.
- **Full-period Sharpe:** 0.682 — clears the Sprint 0 baseline (0.600)
  for the first time since gating was introduced.
- **2022 max DD:** improved to −29.78% (was −34.23% in Sprint 4). Gate
  protection preserved and strengthened.
- **Only regression:** test max DD widened to −21.35% (from −14.25%) as
  the model concentrated in AI winners; Calmar still improved to 0.856
  in test.

Everything that was in the "Sprint 4 diagnosis" section of this file
(stock-selection is the root cause, not gate calibration) is now
confirmed. Softer gate multipliers (0.7/1.0/1.1) are **no longer
recommended** — Sprint 5 shows the fix was in feature space, not gate
space. If future work wants to widen exposure, rank-based selection
(open idea above) is the next lever — the model-refresh lever was tried
and refuted in Sprint 6.

---

## Sprint 6 outcome log — Rolling 36-month window (REFUTED)

Rolling 3-year XGBoost retraining was tested and **rejected** as a
production configuration. `WINDOW_YEARS` is back to `None` (expanding);
the trim mechanism stays in `model_trainer.py`, dormant.

- **Verdict:** FAIL (per `backtests/results/sprint6_results.json`).
  Pre-committed rule required all three: test Sharpe > 0.9898 **(PASS,
  1.041)**, test CAGR > 18.2699% **(FAIL, 14.81%)**, 2022 max DD ≥
  −29.7815% **(PASS, −26.23%)**. Two of three is a FAIL by the rule.
- **The failure was BREADTH STARVATION, not proven ranking failure.**
  Rolling models compress scores toward 0.5, so the fixed absolute
  `MIN_SCORE=0.52` gate left **14 of ~23 test months with ZERO holdings
  (cash)**, forfeiting the 2023-24 rally → test CAGR 14.81% vs the 18.27%
  bar. Mean names/month above 0.52 fell 24.6 → 17.9 (−27.3%); cash on 357
  of 1636 days.
- **Test Sharpe 1.041 "beat" Sprint 5 only as a zero-vol cash artifact**
  — cash months carry zero volatility, inflating the ratio. Not improved
  selection; do not read it as a win.
- **The rolling-window hypothesis is therefore NOT cleanly falsified.**
  What is rejected is the *tested configuration* = rolling window +
  fixed absolute score threshold. The confound (breadth) is separable
  from the hypothesis (regime freshness).
- **What WAS confirmed (positives, held despite the FAIL):** 2022 max DD
  improved to −26.23% (from Sprint 5's −29.78%) and test max DD to
  −12.52% (from −21.35%) — the cash months also dodged drawdowns.
- **Follow-up:** see the "Rank-based selection under rolling windows"
  open idea above — decouple selection from the absolute score level,
  with a pre-committed PASS rule and the multiple-testing discipline
  (Sprint 6 = trial 1).
- **Canary for any re-run:** `last_fold_window_start = 2021-07-31`, which
  still contains the 2022 bear market. A 2026 data refresh would push the
  trailing window past 2022 and silently drop the bear market from every
  late-fold training set — re-validate before trusting a refreshed
  rolling model.
- **Production state after Sprint 6:** `models/ensemble_models.pkl`
  restored to Sprint 5 (expanding) from `ensemble_models_sprint5.pkl`;
  scores/weights regenerated; backtest spot-check reproduced test Sharpe
  0.9898 exactly.
