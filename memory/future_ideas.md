# Future Ideas — AI Trading Agent

Running list of ideas surfaced during sprint work that we don't want to
lose but aren't executing right now. Each entry ends with a suggested
insertion point (which sprint, which module) so it can be pulled off
this list without re-deriving the context.

---

## Model refresh — rolling 3-year XGBoost retrain

**Status:** open, next candidate for Phase 5 (post-Sprint 5).

Sprint 5 validated that adding forward-looking signal (TimesFM) closes
the test-CAGR gap, but the underlying XGBoost combiner is still trained
on 2018–2022 features that pre-date the 2023-24 AI-driven mega-cap
rally. Even with TimesFM's help, the model's *regime knowledge* stops
in 2022 — it hasn't seen prolonged high-VIX / inverted-curve / AI-boom
periods in-sample.

### Why this is likely the highest-leverage next step
- Sprint 5's XGBoost gain ranks show TimesFM at 17/23 — meaningful but
  middling. The bulk of predictions still come from features the pre-2022
  model interprets against pre-2022 return distributions.
- Session 4's AlgoXpert overfitting controls flagged deflated t-stats
  as healthy, but those measure statistical significance of
  contributions, not regime coverage. Regime coverage is a
  training-window problem, not a feature-selection problem.
- The test-window improvement in Sprint 5 came from letting XGBoost
  read a forward signal, but the trees themselves still don't have
  post-2022 splits. Retraining on rolling 3-year windows (drop pre-2020,
  include 2023–24) should lift the base rate further, especially in
  regimes we haven't seen before.

### Concrete Sprint 6 path (draft)
1. Modify `src/strategies/ensemble/model_trainer.py::train_walk_forward`
   to use a rolling training window (last N years) instead of expanding.
   Suggested `WINDOW_YEARS = 3`; make it a module constant, not a
   `settings.yaml` change (same pattern as `timesfm_factor.py`).
2. Rerun the full pipeline (model_trainer → score_generator →
   portfolio_builder → backtest).
3. Compare Sprint 5 (TimesFM, expanding train) vs Sprint 6 (TimesFM,
   rolling 3-year train) on the same three-way format as
   `sprint5_results.json`.
4. Decision point: if Sprint 6 test CAGR/Sharpe improves further, keep
   rolling; if it degrades (older regime data was carrying weight),
   revert and move to the live pipeline instead.

**Suggested insertion:** Sprint 6 — edit
`src/strategies/ensemble/model_trainer.py`.

---

## Live trading pipeline — paper stage

**Status:** open, secondary Phase 5 candidate.

Sprint 5 established the current ensemble (regime gate + TimesFM +
XGBoost) clears the Sprint 0 baseline on Sharpe (0.682 vs 0.600) and
beats it on test CAGR (18.27% vs 13.81%). Risk-adjusted profile is
deployment-worthy for a paper stage.

Prerequisites before opening this:
- Complete the model refresh above (avoid deploying with a known-stale
  training window).
- Decide on execution venue (broker API, paper account credentials in
  `.env`).
- Add a `src/live/` module with a scheduled rebalance harness driven by
  the same `portfolio_builder` outputs — no new alpha logic, just a
  runner that reads the latest score / weight files and posts orders.

**Suggested insertion:** Phase 5, after model refresh — new module
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
space. If future work wants to widen exposure, the model refresh above
is the higher-leverage lever.
