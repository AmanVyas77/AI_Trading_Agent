# Future Ideas — AI Trading Agent

Running list of ideas surfaced during sprint work that we don't want to
lose but aren't executing right now. Each entry ends with a suggested
insertion point (which sprint, which module) so it can be pulled off
this list without re-deriving the context.

---

## TimesFM — forward-momentum factor

**Status:** deferred to Sprint 5 (was gated CLOSED by Sprint 3's proof
run; re-evaluated after Sprint 4's regime-gate recalibration).

### Sprint 4 diagnosis — why gate tuning alone will not close the gap

Sprint 4 recalibrated the regime gate from OR to a graded gate
(`VIX>25 OR (VIX>20 AND spread<0)`), correctly recovering 18 of the 19
false-positive RISK_OFF months in 2023-24 while preserving all 10 of
2022's RISK_OFF months. In principle this should have restored most of
the test-period upside. In practice:

- Sprint 3 (OR gate) test Sharpe: **0.596**
- Sprint 4 (graded gate) test Sharpe: **0.478** — *worse* despite
  restoring 18 months of full exposure
- Test CAGR did move up (+5.41% → +6.77%, +1.36pp) but risk-adjusted
  return dropped

The signature — more exposure lifting CAGR while dropping Sharpe —
means the extra 2023-24 days the strategy is now trading are being
allocated to positions that carry more risk per unit of return than the
train-window positions. Concretely, the XGBoost ensemble model was
trained on 2018-06 → 2022-12 features, a period that includes the 2020
COVID crash, the 2021 growth-to-value rotation, and the 2022 bear
market, but *does not* include the 2023-24 AI-driven mega-cap rally.
Inspecting the top-picks logs from portfolio_builder, the model does
not overweight the actual 2023-24 outperformers (NVDA, MSFT, META, AVGO)
— they either fail to clear the 0.52 score threshold or come in behind
generic large-cap software names.

**This is a stock-selection problem, not a gate-calibration problem.**
Softer multipliers (0.7/1.0/1.1), continuous scaling, or a smarter
regime classifier will change the *magnitude* of the exposure but will
not change *which stocks* the strategy holds inside each RISK_ON window.
Without a factor that captures forward momentum in the AI-boom names,
tuning the gate further is polishing the wrong lever.

### Why TimesFM addresses the root cause

TimesFM (Google's time-series foundation model, ICML 2024) is
pre-trained on ~100B time-points and returns forward-window forecasts.
Two properties make it a natural fit here:

1. **Forward-looking.** Unlike the existing quant factors (12-1
   momentum, mean reversion, low volatility — all backward-window
   statistics on the training data), TimesFM outputs a next-N-day
   forecast per ticker. In a regime the training data doesn't cover
   (2023-24 AI rally), backward-window factors mechanically underweight
   the runaway names, while a forecast-based signal can catch them.
2. **Zero-shot.** No retraining is needed to adapt to a new regime —
   the model was pre-trained on far broader data than our
   universe-specific 2018-2022 window covers.

Loading TimesFM as an additional quant factor (alongside the existing
Phase-1 momentum, MR, and low-vol factors) is the cheapest way to test
whether the test-period gap is closable at all. If TimesFM alone
recovers ≥50% of the test-CAGR gap (i.e. lifts test CAGR from ~+6.77%
to ≥ ~10%), we have evidence the stock-selection story is right and can
justify further model work (retrain XGBoost on 2018-2024 or add a
regime-aware retraining loop). If it recovers < 20%, the diagnosis is
wrong and we come back to the gate.

### Recommended Sprint 5 path

Even though the Sprint 4 verdict formally leaves the TimesFM gate
CLOSED, proceed to TimesFM in Sprint 5 anyway. Reasons:

- The CLOSED verdict was written against the criterion "don't stack new
  factors on a Sharpe-reducing gate because it confounds evaluation."
  That risk is mitigated by holding the graded gate fixed and comparing
  TimesFM-added vs TimesFM-not-added *within* Sprint 5 (both with the
  same graded gate). The gate is a controlled variable.
- Further gate tuning without a better stock-selection signal is likely
  to yield another PARTIAL run and burn a sprint.
- The graded gate (Sprint 4) is already better than the OR gate
  (Sprint 3) on the metric that matters most for TimesFM evaluation —
  test-period exposure — so we are strictly *less* at risk of the gate
  masking a TimesFM signal than we were before Sprint 4.

Concrete Sprint 5 steps (draft):
1. Add `src/strategies/quant/timesfm_factor.py` — loads pre-trained
   TimesFM, exports a monthly per-ticker forecast score matching the
   existing quant factor interface.
2. Wire it into `ensemble/feature_matrix.py` as one more column.
3. Re-run model_trainer + score_generator + portfolio_builder + backtest.
4. Compare Sprint 4 (graded gate, no TimesFM) vs Sprint 5 (graded gate,
   + TimesFM) headline metrics — same three-way diff format as
   `sprint4_recal_results.json`.
5. Decision point after Sprint 5: if test CAGR gap closes, keep the
   graded gate as-is and move on. If it doesn't, come back and try
   softer multipliers (0.7/1.0/1.1) or a continuous version of the
   gate.

**Suggested insertion:** Sprint 5 — new module
`src/strategies/quant/timesfm_factor.py`.
