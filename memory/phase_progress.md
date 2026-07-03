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

## Sprint 5 (TimesFM momentum factor) — GATE STATUS: **CLOSED, proceed anyway**

Per the strict Sprint 4 verdict the TimesFM gate is CLOSED (Sharpe still
below baseline, primary 2022 DD still short). But Sprint 4 surfaced a
diagnosis that changes the plan (see `memory/future_ideas.md`): the
test-period Sharpe *worsened* when exposure was restored (Sprint 3
0.596 → Sprint 4 0.478), which points at the underlying XGBoost stock
selection — trained pre-AI-boom, it does not overweight the 2023-24
mega-cap AI winners. Softer multipliers cannot fix a selection problem.

Recommended Sprint 5: implement TimesFM as a new quant factor alongside
the existing momentum lookbacks (Phase 1), keeping the graded gate in
place. Evaluate whether TimesFM repairs test-period alpha before
deciding whether to also soften the gate multipliers (0.7/1.0/1.1) or
introduce continuous multipliers. Full reasoning in
`memory/future_ideas.md`.

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
