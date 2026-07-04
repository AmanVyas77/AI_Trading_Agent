# Phase Progress — AI Trading Agent

## Phase 4b: Ensemble sentiment + regime gate + EWM smoothing (Sprints 0–3)

Status: **COMPLETE** as of 2026-07-02.
Full sprint chain (Sprint 0 → Sprint 3) is done. Phase 4b closed out.

---

### Sprint 0 — Baseline rebuild (commit c0f9c65)
Wired Piotroski F, QMJ Safety/Payout, FinBERT sentiment, and Loughran-McDonald
sentiment into the ensemble feature matrix. Fixed the FinBERT text-truncation
regex bug (DOTALL + `\Z` silently ate filing bodies). Rebuilt the frozen
baseline artefacts under `backtests/results/ensemble_baseline_*.csv` and
`baseline_metrics.json` so later sprints have a stable comparison target.

### Sprint 1 — Regime gate (commit b6256e4)
Added `src/strategies/ensemble/regime_gate.py` with:
- `get_historical_regime_multipliers()` — rule-based VIX / 10y-2y yield-curve
  gate for backtest, returns monthly 0.5×/1.0×/1.2× weight multipliers.
- `get_live_regime_signal()` — RAG blend used at inference time.
Wired the historical gate into `portfolio_builder.build_portfolio_weights()`.

### Sprint 2 — EWM sentiment smoothing (commit 06c4b12)
Added `EWM_SPAN = 4` and `_ewm_smooth()` to
`src/strategies/fundamental/sentiment_pipeline.py`. Both `load_sentiment()`
and `load_lm_scores()` now apply EWM smoothing at read time (no DB schema
changes). Rationale: Arratia (2021) — ~4-quarter half-life on sentiment
noise.

### Sprint 3 — Proof run (this commit)

Re-ran the full ensemble pipeline with Sprints 1 + 2 both active, saved
the results under a `_gated` namespace so the frozen Sprint 0 baseline is
preserved for diffing. All numbers below come from
`backtests/results/sprint3_proof_results.json`.

#### Headline numbers (baseline → gated)

| Metric                         | Baseline | Gated   | Δ            |
|--------------------------------|----------|---------|--------------|
| **2022 max drawdown**          | −41.74%  | −34.23% | +7.51 pp ▲   |
| 2022 full-year return          | −38.01%  | −31.77% | +6.24 pp ▲   |
| Full-period CAGR (2018–2024)   | +14.78%  | +9.39%  | −5.39 pp ▼   |
| Full-period Sharpe             | 0.600    | 0.531   | −0.069 ▼     |
| Full-period Sortino            | 0.748    | 0.668   | −0.079 ▼     |
| Full-period Max DD             | −43.79%  | −36.61% | +7.18 pp ▲   |
| Test-period CAGR (2023–24)     | +13.81%  | +5.41%  | −8.41 pp ▼   |
| Test-period Sharpe             | 0.783    | 0.596   | −0.187 ▼     |

#### Primary objective — 2022 drawdown improvement
- Target: gated 2022 max DD better than −31.74% (≥10 pp improvement)
- Actual: −34.23% (only +7.51 pp)
- **Result: FAIL** — moved in the right direction but 2.5 pp short of the
  "meaningful" threshold.

#### Verdict (verbatim from `sprint3_proof_results.json`)

> Regime gate reduced the 2022 max drawdown from -41.74% to -34.23%
> (+7.51pp), but this falls short of the +10pp threshold set as
> 'meaningful', so the primary drawdown objective is FAIL. The tradeoff
> is worse than acceptable: full-period CAGR fell 5.39pp (14.78% -> 9.39%),
> well beyond the 3pp tolerance, and Sharpe dropped from 0.600 to 0.531 --
> lower CAGR with lower Sharpe is the signature of a gate that is reducing
> exposure rather than making smarter bets. Test-period performance
> (2023-24, which live-signal analysis flagged as largely risk-on) also
> worsened: CAGR -8.41pp and Sharpe -0.187, meaning the 1.2x multiplier
> did not deliver the promised upside amplification. Sprint 4 (TimesFM
> momentum factor) should NOT be gated open on top of the current regime
> layer -- the gate needs to be re-tuned (thresholds, multiplier
> magnitudes, or a smarter regime classifier) before stacking new factors
> on it, since building on a Sharpe-reducing gate would confound the
> TimesFM signal evaluation.

---

---

## Sprint 4 — Regime gate recalibration (this commit)

Status: **COMPLETE** as of 2026-07-03. Sprint 4 was re-scoped from
TimesFM to a targeted regime-gate recalibration after Sprint 3's proof
run showed the OR gate over-weighting the persistent 2023-24 yield-curve
inversion. Full numbers in `backtests/results/sprint4_recal_results.json`.

### Change
Single-line edit in `src/strategies/ensemble/regime_gate.py::_classify_row`.
- **Before:** `risk_off = (VIX > 25) OR (spread < 0)`
- **After (graded):** `risk_off = (VIX > 25) OR (VIX > 20 AND spread < 0)`
- Multipliers unchanged: `RISK_OFF_MULT=0.5, NEUTRAL_MULT=1.0, RISK_ON_MULT=1.2`.
- Thresholds unchanged: `VIX_RISK_OFF=25.0, VIX_RISK_ON=20.0, SPREAD_FLOOR=0.0`.

The graded gate keeps the VIX>25 arm intact (which is what actually
protects the 2022 crash months) and adds a dual-confirmation second arm
so a mild-VIX + inverted-curve regime like 2023-24 no longer trips
RISK_OFF unilaterally.

### Recon (Prompt 1, before backtesting)
- All **10** of 2022's OR-gate RISK_OFF months are preserved under the
  graded gate (4 VIX_ONLY via Arm 1, 3 SPREAD_ONLY via Arm 2, 3 BOTH).
- **18 of 19** OR-gate RISK_OFF months in the 2023-24 test window
  recover to NEUTRAL; only **Feb 2023** (VIX 20.70, spread −0.89)
  remains RISK_OFF via Arm 2.

### Three-way comparison
(Sprint 0 full Sharpe = 0.600 comes from `ensemble_baseline_stats.csv`;
the 0.721 value stored in `baseline_metrics.json` /
`sprint4_recal_results.json` is the train-period Sharpe.)

| Metric                         | Sprint 0 baseline | Sprint 3 (OR gate) | Sprint 4 (graded)  |
|--------------------------------|-------------------|--------------------|--------------------|
| **2022 max drawdown**          | −41.74%           | −34.23%            | −34.23%            |
| Full-period CAGR (2018-2024)   | +14.78%           | +9.39%             | +9.80%             |
| Full-period Sharpe             | 0.600             | 0.531              | 0.530              |
| Test-period CAGR (2023-2024)   | +13.81%           | +5.41%             | +6.77%             |

### Checks (per `sprint4_recal_results.json`)
- Primary 2022 DD (< −31.74% target): **FAIL** (−34.23%)
- Tradeoff CAGR (> 11.78%): **FAIL** (+9.80%)
- Tradeoff Sharpe (≥ 0.721 — evaluated vs the *train* Sharpe stored in
  the JSON): **FAIL** (0.530). Even against the corrected full-period
  baseline 0.600, Sharpe is a hair below (−0.07pp).
- Risk-on upside: **PARTIAL** (test CAGR up +1.36pp vs Sprint 3, still
  −7.04pp vs baseline)

### Verdict (verbatim from `sprint4_recal_results.json`)

> The graded gate (VIX>25 OR VIX>20-with-inverted-curve) fixed Sprint 3's
> over-exposure problem in direction but not magnitude: 18 of 19
> SPREAD_ONLY test months moved back to NEUTRAL, lifting test-window
> CAGR from +5.41% (Sprint 3 OR) to +6.77% (+1.36pp), yet still 7.05pp
> shy of the 13.81% baseline — the recalibration recovered only about
> 16% of the OR gate's test-CAGR gap. 2022 drawdown protection was
> preserved exactly (-34.23% max DD, identical to Sprint 3 within
> rounding), confirming Prompt 1's read that the VIX>25 arm is what
> actually protects the crash months. Against the Sprint 0 baseline the
> tradeoff is still unacceptable: full-period CAGR at 9.80% is -4.98pp
> below baseline (outside the 3pp tolerance), full Sharpe 0.530 vs 0.721
> (-0.191), and 2022 max DD -34.23% remains 2.49pp short of the primary
> -31.74% target — so the primary DD check FAILS, both tradeoff checks
> FAIL, and the risk-on upside is PARTIAL. TimesFM gate: CLOSED —
> stacking a new momentum factor on a still-Sharpe-reducing gate would
> confound its evaluation; the SPREAD arm needs further weakening
> (softer multiplier, or continuous scaling) before Sprint 4 stacks
> TimesFM.

### Artefacts (Sprint 4)
- `src/strategies/ensemble/regime_gate.py` — tracked, graded-gate edit.
- `backtests/results/sprint4_recal_results.json` — tracked; full
  three-way diff, checks, and verdict for reproducibility.
- `backtests/results/ensemble_recal_{train,test}_equity.csv`,
  `ensemble_recal_stats.csv` — local only (`*.csv` gitignored); Sprint 4
  equity curves and vectorbt stats.

---

## Sprint 5 — TimesFM 1M forward-return factor (commit 65bc3bb)

Status: **COMPLETE** as of 2026-07-03. Verdict from
`backtests/results/sprint5_results.json`: **PASS** — Phase 4b closed;
next stop is Phase 5.

### Motivation (carried from Sprint 4 diagnosis)
Sprint 4 surfaced that test-period Sharpe *worsened* when exposure was
restored (Sprint 3 0.596 → Sprint 4 0.478), pointing at the underlying
XGBoost stock selection — trained on 2018–2022, it did not overweight
the 2023–24 mega-cap AI winners. Sprint 5's hypothesis: adding a
forward-looking TimesFM 1M return forecast as a new feature lets the
model see momentum signals that don't require re-training on post-2022
data.

### Change
- **New file:** `src/strategies/quant/timesfm_factor.py`. Google's
  TimesFM 2.5 (200M params, PyTorch CPU) run per month-end over each
  ticker's daily `adj_close` history. Module-level constants (NOT in
  `settings.yaml`): `CONTEXT_LEN=252`, `HORIZON_LEN=21`, `MIN_CONTEXT=63`,
  `HF_REPO="google/timesfm-2.5-200m-pytorch"`. Batches by month-end
  (one `.forecast()` call for all eligible tickers at each date) →
  ~120 batched inferences for the full 2015–2024 window, ~4 min end-to-end
  on CPU with 2 threads. Median (quantile idx 4) at horizon step 20 is
  denormalised back to price space and expressed as a 1M return.
- **Modified:** `src/strategies/ensemble/factor_export_quant.py` — imports
  `compute_timesfm_predictions`, computes it after the resample-to-month-end
  step, and merges the long-format result into the tidy output. All
  existing 1M/3M/6M/12M momentum lookbacks, `volume_zscore`, `inv_vol`
  are **preserved** — TimesFM augments, does not replace.
- **Modified:** `src/strategies/ensemble/feature_matrix.py` — appended
  `"timesfm_pred_return_1m"` to `QUANT_COLS` (7th quant column). No
  logic changes.

### AlgoXpert DSR check (from Session 4's overfitting controls)
- **`timesfm_pred_return_1m` deflated t-stat = 30.48** (threshold = 1.0).
  30× the DSR floor — the signal survives multiple-testing correction
  comfortably.
- All 23 features (22 existing + TimesFM) survive the deflated-t cut.
- XGBoost gain rank of TimesFM: **17/23** (0.0398 mean gain vs top
  feature `vix` at 0.0587). Middling importance in tree splits but
  statistically significant contribution.

### Three-way comparison (Sprint 0 vs Sprint 4 vs Sprint 5)

| Metric                        | Sprint 0 baseline | Sprint 4 (graded) | Sprint 5 (+ TimesFM) |
|-------------------------------|-------------------|-------------------|----------------------|
| Full-period CAGR (2018-2024)  | +14.78%           | +9.80%            | **+13.60%**          |
| Full-period Sharpe            | 0.600             | 0.530             | **0.682**            |
| **2022 max drawdown**         | −41.74%           | −34.23%           | **−29.78%**          |
| Test-period CAGR (2023-2024)  | +13.81%           | +6.77%            | **+18.27%**          |
| Test-period Sharpe            | 0.783             | 0.478             | **0.990**            |

- **Test-CAGR recovery: 163.35%** of the 7.04pp Sprint-0-vs-Sprint-4
  gap. TimesFM did not just close the gap — it beat Sprint 0's test
  CAGR by +4.46pp.
- Full-period Sharpe clears the Sprint 0 baseline (0.682 > 0.600) for
  the first time since the ensemble was gated.
- 2022 max DD improved from −34.23% (Sprint 4) to −29.78% — regime-gate
  protection preserved and strengthened.
- Only regression: test max DD went from −14.25% (Sprint 4) to −21.35%
  as the model concentrated in AI winners; Calmar still improved to
  0.856 in test.

### Verdict (verbatim from `sprint5_results.json`)

> TimesFM 1-month median forecast added as 23rd feature to walk-forward
> XGBoost. Test-window CAGR rose from 6.77% (Sprint 4) to 18.27% —
> recovering 163.4% of the 7.04pp Sprint-0-vs-Sprint-4 gap and actually
> clearing the Sprint-0 test baseline (13.81%) by +4.46pp. Full-period
> Sharpe 0.682 beats Sprint 4's 0.530 AND clears the Sprint 0 baseline
> of 0.600 by ++0.082. 2022 max DD improved to -29.78% (was -34.23% in
> Sprint 4) — protection preserved and strengthened. TimesFM feature
> ranks 17/23 on avg XGBoost gain (0.0398) but its deflated t-stat is
> 30.48 — 30x the DSR threshold of 1.0 — so the signal is real, not
> multiple-testing noise. Test Sharpe 0.990 more than doubles Sprint 4's
> 0.478 and exceeds Sprint 0 baseline 0.783.

### Sprint 6 / Phase 5 recommendation (PASS branch)
Phase 4b is complete. Two candidate Phase 5 directions:
1. **Model refresh** — retrain the XGBoost combiner with a rolling
   3-year window (drop pre-2020 data, include 2023–24 AI-boom
   observations in the training set). Even with TimesFM helping, the
   base model's regime knowledge is still 2018–2022. This is the
   highest-leverage next step per Session 4's overfitting analysis.
2. **Live trading pipeline** — begin end-to-end paper-trading harness
   with the current ensemble (regime gate + TimesFM + XGBoost). Sprint 5
   established that the strategy clears the Sprint 0 baseline on Sharpe
   and beats it on test CAGR — the risk-adjusted profile is now
   deployment-worthy for a paper stage.

Suggested order: model refresh first (cheaper, faster to iterate), then
live pipeline once refreshed model is validated.

### Artefacts (Sprint 5)
- `src/strategies/quant/timesfm_factor.py` — new module (tracked).
- `src/strategies/ensemble/factor_export_quant.py`,
  `src/strategies/ensemble/feature_matrix.py` — tracked with Sprint 5 edits.
- `backtests/results/sprint5_results.json` — tracked; three-way diff,
  deltas, checks, verdict.
- `backtests/results/ensemble_timesfm_{train,test}_equity.csv`,
  `ensemble_timesfm_stats.csv` — **tracked** (force-added despite
  `*.csv` in `.gitignore`) for downstream Sprint 6+ benchmarking.

## Artefacts (all pinned in git or under `backtests/results/`)
- `backtests/results/baseline_metrics.json` — Sprint 0 baseline (tracked)
- `backtests/results/ensemble_baseline_{train,test}_equity.csv`,
  `ensemble_baseline_stats.csv` — frozen Sprint 0 equity curves (local
  only, `*.csv` is gitignored)
- `backtests/results/ensemble_gated_{train,test}_equity.csv`,
  `ensemble_gated_stats.csv` — Sprint 3 gated run (local only)
- `data/processed/ensemble_feature_matrix_gated.parquet` — smoothed
  feature matrix used by the gated run (local only, `*.parquet` gitignored)
- `backtests/results/sprint3_proof_results.json` — **tracked**; contains
  all baseline, gated, delta, and verdict fields for reproducibility.

---

## Sprint 6 — Rolling 36-month training window (REFUTED)

Status: **COMPLETE** as of 2026-07-03. Verdict from
`backtests/results/sprint6_results.json`: **FAIL** — the rolling-window
*configuration* is rejected; `WINDOW_YEARS` reverted to `None`
(expanding), production models restored to Sprint 5 state. The trim
mechanism + `window_start` bookkeeping remain in `model_trainer.py`,
dormant when `WINDOW_YEARS is None`.

### Change (tested, then reverted)
- `src/strategies/ensemble/model_trainer.py`: added `WINDOW_YEARS`
  module constant + a trailing-window trim inside the fold loop
  (`train_df = train_df[train_df["date"] > train_end - DateOffset(years=WINDOW_YEARS)]`)
  and a `window_start` key in the per-fold results dict. `target_builder.py`
  untouched. Set to `3` for the Sprint 6 run, reverted to `None` after FAIL.
- Retrained 78 folds on rolling 36-month windows: fit rows plateau at
  ~1,554 mean / 1,590 final (vs Sprint 5's expanding 5,444 final).

### Three-way comparison (Sprint 0 vs Sprint 5 vs Sprint 6)
(all Sprint 6 metrics recomputed programmatically from the saved equity CSVs)

| Metric                        | Sprint 0 baseline | Sprint 5 (+ TimesFM) | Sprint 6 (Rolling) |
|-------------------------------|-------------------|----------------------|--------------------|
| Full-period CAGR (2018-2024)  | +14.78%           | +13.60%              | **+10.59%**        |
| Full-period Sharpe            | 0.600             | 0.682                | **0.558**          |
| **2022 max drawdown**         | −41.74%           | −29.78%              | **−26.23%**        |
| Test-period CAGR (2023-2024)  | +13.81%           | +18.27%              | **+14.81%**        |
| Test-period Sharpe            | 0.783             | 0.990                | **1.041**          |
| Test-period max DD            | —                 | −21.35%              | **−12.52%**        |

### Checks (per `sprint6_results.json`, decided in advance)
- (1) test Sharpe > 0.9898: **PASS** (1.041)
- (2) test CAGR > 18.2699%: **FAIL** (14.81%)
- (3) 2022 max DD ≥ −29.7815%: **PASS** (−26.23%)
- Composite (all three required): **FAIL**

### Breadth confound (why it failed)
Smaller train sets compressed scores toward 0.5, so mean monthly names
above the fixed `MIN_SCORE=0.52` gate fell 24.6 → 17.9 (−27.3%). **14 of
~23 test months held ZERO names** → cash for 357 of 1636 days. The
strategy sat out the 2023-24 tech rally (worst relative month 2023-11:
rolling −0.1% vs Sprint 5 +10.0%), so test CAGR collapsed while test
Sharpe *rose* — a zero-vol cash artifact, not improved selection.

### Verdict (verbatim from `sprint6_results.json`)

> Rolling 36-month window FAILS: test CAGR 14.81% is far below the 18.27%
> bar (Sprint 5), though test Sharpe 1.041 edges above 0.990 and 2022 max
> DD -26.23% is comfortably inside -29.78%. The dominant cause is a
> BREADTH CONFOUND, not ranking skill: smaller train sets compress
> predicted probabilities toward 0.5, so mean monthly names above the
> 0.52 threshold fell from 24.6 (Sprint 5) to 17.9 (rolling), a -27.2%
> drop; 14 of ~23 TEST months hold zero names, leaving the strategy in
> cash for 357 of 1636 days. Month-by-month, the divergence vs Sprint 5
> is concentrated in 2023-2024: rolling sat out much of the tech rally
> (worst relative month 2023-11-30: rolling -0.1% vs Sprint 5 +10.0%),
> which is exactly why test CAGR collapsed while Sharpe stayed high (cash
> lowers volatility). The higher test Sharpe is therefore partly a
> concentration/cash artifact and must not be read as improved selection.
> Breadth fell 27.3% — under the 30% 'collapse' bar by the letter of the
> rule, but concentrated in the test window, so the Sharpe comparison is
> not apples-to-apples.

### Artefacts (Sprint 6)
- `src/strategies/ensemble/model_trainer.py` — tracked; `WINDOW_YEARS = None`
  + dormant trim mechanism + `window_start` bookkeeping.
- `backtests/results/sprint6_results.json` — tracked; three-way diff,
  deltas, fold diagnostics, breadth diagnostics, checks, verdict.
- `backtests/results/ensemble_rolling_{train,test}_equity.csv`,
  `ensemble_rolling_stats.csv` — **tracked** (force-added despite `*.csv`
  gitignore) — the rolling run's curves, preserved for the record.
- `models/ensemble_models.pkl` — restored to Sprint 5 (expanding) state
  from `ensemble_models_sprint5.pkl`; `data/processed/` scores + weights
  regenerated to match (spot-check: test Sharpe 0.9898, exact).

---

## Sprint 7 — 2025-26 true-holdout validation (PASS)

Status: **COMPLETE** as of 2026-07-04. Verdict from
`backtests/results/sprint7_results.json`: **PASS** — the frozen Sprint 5
production model (78 folds, `effective_train_end=2024-07-31`) was
scored over 2025-01 → 2026-06 with no retraining and no design changes.
This window is now **burn-once spent**: no future experiment may tune
against 2025-26.

### What ran
- Data refresh (Sprint 7 Prompt 2): prices/macro/FinBERT all advanced
  to ~2026-07-02. FinBERT backlog of 893 rows (2025 + partial 2026)
  scored on MPS in ~28 min. Authorized minimal fix to
  `src/strategies/fundamental/sentiment_pipeline.py`: parameterized
  `run_sentiment_pipeline(date_start, date_end)` + CLI flags; module
  DATE_START/DATE_END constants preserved as defaults.
- CPI backfill (Prompt 3 Part A0): `scripts/backfill_cpi.py` upserts
  CPIAUCSL Jan 2026 → today into `macro_series['cpi']`; no
  `settings.yaml` edit (CPIAUCSL is not in the FRED config). Sprint 8
  can reuse.
- Factor + feature-matrix rebuild through 2026-06-30 via the existing
  `--start/--end` CLI paths. TimesFM 135 month-ends, 0 failures.
  Feature matrix: 6981 rows × 25 cols (23 features + date + ticker);
  `fcf_yield` correctly absent (all-NaN → dropped upstream).
- Frozen scoring (Prompt 3 Part B): new `src/live/scorer.py` — loads
  the last fold from `models/ensemble_models.pkl`, reindexes X on the
  pickle's own `feature_names` (guards the fcf_yield landmine), fills
  NaN 0.0 mirroring `_prepare_xy`, asserts shape/order before
  `predict_proba[:,1]`. Emits an AUDIT log line with model fold_date,
  effective_train_end, n_features.
- Holdout run (Prompt 3 Part C): new `scripts/run_holdout.py` runs
  scoring → `build_portfolio_weights(scores_df=, prices=)` → vectorbt
  from_orders + SPY benchmark. All clamped `TIMELINE["test_end"]`
  loaders bypassed by **injection**, not by editing `portfolio_builder`
  or `backtest`. Regime gate fires automatically inside
  `build_portfolio_weights` on the injected date range.

### Pre-committed rule (fixed 2026-07-04, before any holdout metric was computed)

> PASS = holdout Sharpe (2025-01-01 → 2026-06-30, daily returns,
> mean/std·√252, computed from the strategy equity curve) > 0.600.
> Anything else = FAIL.

### Numbers (recomputed from the saved equity CSV)

| Metric                    | Sprint 0 test | Sprint 5 test | Sprint 7 holdout | SPY holdout |
|---------------------------|---------------|---------------|-------------------|-------------|
| Sharpe                    | 0.783         | 0.990         | **1.016**         | 1.007       |
| CAGR                      | +13.81%       | +18.27%       | **+32.89%**       | +17.95%     |
| Sortino                   | —             | —             | **1.473**         | 1.319       |
| Max Drawdown              | —             | −21.35%       | **−27.03%**       | −18.76%     |
| Total return              | —             | —             | **+49.33%**       | +26.21%     |
| Window                    | 2023-2024     | 2023-2024     | **2025-01-31 → 2026-06-30** | same |

Rule test: 1.016 > 0.600 → **PASS** (clears by +0.416).

### What drove it (from `sprint7_results.json`)
- **Regime skew:** 15/18 months and 290/354 days were RISK_ON (1.2×
  leverage); only 1 RISK_OFF month (2026-03-31). Leverage was on
  for 82% of the window, amplifying both return and drawdown.
- **May 2026 concentration:** +28.7% strategy vs +5.3% SPY that
  single month is roughly half the total-return excess. Verdict
  distance is highly sensitive to that month holding.
- **Under-diversification:** breadth mean 14.9 (median 14) vs 24.6
  in-sample — only 5 of 18 months hit the 20-name target.

### Verdict (verbatim from `sprint7_results.json`)

> Holdout Sharpe = 1.0160 clears the pre-committed 0.600 threshold by
> +0.416. What drove it: (1) the frozen last-fold model kept generating
> usable rank information in 2025-26 despite the 18-month training gap,
> (2) a heavily RISK_ON regime (290/354 days at 1.2× leverage)
> amplified positive selection, and (3) a single anomalous month —
> May 2026 at +28.7% strategy vs +5.3% SPY — contributed roughly half
> the excess-return gap. The strategy's Sharpe advantage over SPY
> (Δ +0.009) is thin; the +23-pt total-return excess is what would
> show up in P&L. What this proves: the Sprint 5 factor stack + graded
> regime gate did not decay to noise on a genuine out-of-sample window.
> What it does NOT prove: statistical robustness — 18 months puts a
> ~±0.30 standard error on the Sharpe estimate, and any of the caveats
> (TimesFM data-vintage leak, May 2026 concentration, regime skew) is
> individually sufficient to explain the excess. Treat this as a green
> flag to enter Sprint 8's paper-trading phase, not as evidence of a
> validated edge.

### Burn-once note
2025-01-01 → 2026-06-30 is now **spent** as a holdout. No future
experiment may tune against this window. Post-2026-06-30 months become
the next available holdout as they accrue.

### Artefacts (Sprint 7)
- `src/live/__init__.py`, `src/live/scorer.py` — new module; frozen
  last-fold scorer with X.shape[1]==23 assertion and column-order check
  against the pickle's own `feature_names`.
- `scripts/run_holdout.py` — new; runs the holdout via injection, no
  edits to `portfolio_builder` / `backtest`. Reused by Sprint 8.
- `scripts/backfill_cpi.py` — new; CPIAUCSL → `macro_series['cpi']`
  because CPIAUCSL is not in `settings.yaml`'s FRED list. Idempotent.
- `src/strategies/fundamental/sentiment_pipeline.py` — 31/2 diff:
  `run_sentiment_pipeline(date_start, date_end)` + CLI flags.
  Backward-compatible (module DATE_START/DATE_END constants preserved
  as defaults).
- `backtests/results/sprint7_results.json` — tracked; verdict + all
  metrics + informational context + seven caveats + provenance.
- `backtests/results/holdout_2025_26_equity.csv` — **tracked**
  (force-added despite `*.csv` gitignore) — 354 rows × 2 cols
  (strategy, benchmark) for reproducibility.
- `models/ensemble_models.pkl` — **untouched**; the whole sprint's
  point is that the model stayed frozen.

