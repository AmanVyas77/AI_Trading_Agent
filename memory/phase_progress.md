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

## Sprint 4 (TimesFM momentum factor) — GATE STATUS: **CLOSED**

Per the project plan, Sprint 4 was gated on Sprint 3 proving that the
regime gate improved drawdown without gutting returns. It did not clear
that bar (primary DD objective FAIL, Sharpe worsened, 1.2× risk-on did
not add value in the 2023-24 test window). Do not proceed to Sprint 4
until the regime gate is re-tuned. Suggested follow-ups before reopening
the gate:
- Widen the neutral band around the VIX / yield-curve thresholds (fewer
  aggressive risk-off months).
- Tighten the 0.5× multiplier toward 0.75× — the current setting is
  overshooting protection.
- Test a smoother regime classifier (e.g. HMM or a rolling z-score) in
  place of the rule-based bucket, so the gate rides regime transitions
  instead of stepping discretely.
- Consider evaluating the 1.2× upside multiplier separately — the fact
  that it hurt test-period returns is a stronger signal than the DD miss.

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
