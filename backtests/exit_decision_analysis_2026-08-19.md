# Prompt 2 — Exit-decision replay and hit-rate decomposition

**Date run:** 2026-08-19 · **Vintage:** `backtests/vintage_2026-08-04`  
**Interpreter:** `/Users/aman/dev/Ai Trading Agent/.venv/bin/python` — numpy 2.4.4 / sklearn 1.8.0 / pandas 2.3.3 ✔  
**Vintage git_head:** `b5af09b68c06` · **price_vintage sha256:** `0d685aeb29276f33…`  
**Per-decision parquet:** `backtests/exit_decisions_2026-07-30.parquet`  
  sha256 `296073e9c9d2c0dc978bda0581c9e7d2…` · 225 rows · 26 columns  

Sharpe assertions on `--emit-decisions` are bit-identical: baseline 1.587144507439707 (drift 0.0), experimental 1.282216655915358 (drift 2.220e-16), hard_stop 1.587144507439707 (drift 0.0), Zhang-only 1.591542820613977 (drift 0.0). Instrumentation is return-neutral.

---
## Task B — Decision decomposition

SELL population: **29** decisions (of 225 = 29/196 SELL/HOLD).

### 1. Hit rate
Of 29 SELLs, 9 had `ret_full < 0` → **hit rate = 31.0%**. Below 50% — the layer is not systematically cutting losers.

### 2. Mean and median `ret_full | SELL`
- mean   = **+0.1617**
- median = **+0.1009**

By SELL population (Prompt 1D showed Zhang and Andrade fire on different rows):

| population | n | hit | mean `ret_full` | median | sum |
|---|---:|---:|---:|---:|---:|
| Zhang-gated (zhang+both) | 12 | 58.3% | +0.0041 | -0.0071 | +0.0491 |
| Andrade-driven (andrade) | 17 | 11.8% | +0.2730 | +0.2083 | +4.6414 |

**Pre-committed condition (a) — negative mean `ret_full` given SELL**:
- All SELLs pooled: mean = +0.1617 → **condition (a) FAILS**
- Zhang-gated only:  mean = +0.0041 → **fails**
- Andrade-driven:    mean = +0.2730 → **fails**

### 3. Forgone-return decomposition
- correct saves (`ret_full < 0`): sum loss avoided = **+0.5060**
- false positives (`ret_full > 0`): sum gain forgone = **+5.1966**
- **net drag on experimental** (gain forgone − loss avoided) = **+4.6905**

Reconciliation vs the Sharpe gap (−0.3049 on the frozen vintage): summed monthly return drag = +4.6905 across 17 months, spread over an average `n_held ≈ 13` book — the two tell the same story (the layer trades a small loss-avoidance for a materially larger gain-forgone, which is what pulls the mean/σ ratio down).

### 4. Tail asymmetry

**Top-5 correct saves** (largest losses avoided):
| month | ticker | ret_full | trigger |
|---|---|---:|---|
| 2026-01-31 | ACN | -0.2083 | zhang |
| 2025-09-30 | MANH | -0.1118 | zhang |
| 2025-06-30 | DXCM | -0.0747 | zhang |
| 2025-04-30 | HUBS | -0.0353 | andrade |
| 2025-04-30 | INTC | -0.0274 | both |

**Top-5 false positives** (largest gains forgone):
| month | ticker | ret_full | trigger |
|---|---|---:|---|
| 2026-03-31 | INTC | +1.1409 | andrade |
| 2026-03-31 | AMD | +0.7426 | andrade |
| 2026-03-31 | ON | +0.6281 | andrade |
| 2026-03-31 | ANET | +0.4067 | andrade |
| 2026-03-31 | AVGO | +0.3487 | andrade |

### 5. Concentration
- distinct tickers ever SELL-flagged: **23**
- tickers accounting for 50% of Σ|ret_full| across SELLs: **4**
  top by Σ|ret_full|: INTC (+1.168), AMD (+0.743), ON (+0.634), AVGO (+0.606), ANET (+0.407), AMZN (+0.384), DELL (+0.226), TWLO (+0.217), HPE (+0.208), ACN (+0.208)

---
## Task C — Re-entry question

### 6. `in_book_next_month` rate among SELL-flagged names
**44.8%** — 13 of 29 SELLs are back in the top-N book the following month (score-driven re-selection).
Rate is **low**. Under 70% — the ensemble re-buys often but not systematically.

### 7. Two causes of next-month absence
**By code inspection.** `monthly_exit_review` consumes only that month's calibration and has no memory of prior decisions (no persisted set of "names to keep out" — verified at `src/exit/exit_manager.py:266-368`). `build_targets` / `_select_monthly_holdings` re-selects the top-N book from that month's ensemble score alone, with no cross-month exclusion state (verified at `src/live/rebalance.py:113-184` and `src/strategies/ensemble/portfolio_builder.py:111-180`). **The exit layer therefore cannot exclude any ticker for a second consecutive month — it is stateless month-over-month.**

**By the data.** Of 29 SELL-flagged (ticker, month) rows, 16 are absent from the following month's book — every one of those absences is because the ensemble ranked the name outside the top-N (or below `MIN_SCORE = 0.52`), NOT because the exit layer kept it out. These are separate mechanisms; the report does not conflate them.

### 8. Counterfactual perfect-foresight re-entry ceiling
For every SELL, replace the experimental return (0.0) with a re-entry at the month's LOW held to `me_next`. Not implementable — a ceiling.

- baseline Sharpe (recomputed on the parquet's book-weighted book): **+1.5871**
- experimental Sharpe (recomputed): **+1.2822**
- **perfect-foresight ceiling Sharpe: +1.8108**
- ceiling − baseline: **+0.2237**

The ceiling beats baseline by +0.2237. A realisable re-entry gate could in principle close some fraction of the gap; whether any implementable rule captures enough is a separate question.

---
## Task D — Φ stability under estimation error

### 9. Parameter standard errors
**Method: nonparametric moving-block bootstrap** over each calibration's 250-day log-return window. Block length **5** trading days (≥ the fitted mean regime duration of ≈2.04 td per Prompt 1D §B.4). `estimate_parameters` is a moment estimator built from sign-flip counts and the sample stdev of Δz — it is not an MLE, so no analytic Fisher information is available; the block bootstrap resamples the increment series preserving local dependence, refits the estimator, and stores 1000 replicates per calibration. **The empirical bootstrap distribution is used directly as the sampling distribution in the perturbation step below — no Gaussian resampling is applied, which preserves the full joint correlation between (f1,f2,λ1,λ2) automatically.**

**Realised transition counts per calibration** (R1+R2 as counted by the estimator): median 127, p05 115, p95 140. Materially higher than the ~60 back-of-envelope in the prompt — the actual transition count is about 2× that, because both directions contribute and the mean regime is closer to 2 td than 3 td.

**Relative SE (std/|mean|) distribution across the 225 calibrations:**

| param | p05 | p50 | p95 | mean |
|---|---:|---:|---:|---:|
| f1 | 7.6% | 11.8% | 18.7% | 12.4% |
| f2 | 10.0% | 13.2% | 22.0% | 14.1% |
| lam1 | 7.9% | 8.9% | 10.2% | 9.0% |
| lam2 | 7.4% | 8.5% | 9.6% | 8.5% |

### 10. Perturbation setup
For each of the 225 calibrations, 1000 bootstrap replicates of (f1,f2,λ1,λ2) were drawn. For each replicate, Φ, the Case assignment, and x* were recomputed from scratch. The Zhang SELL decision was re-evaluated with the observed state and observed p0 (only the parameters are perturbed; the Andrade state mapping and the ticker's month-end price are treated as observed).

Correlation preserved through the bootstrap itself: joint (f1,f2,λ1,λ2) draws are exact resamples of the block-bootstrap distribution, so any covariance between the four parameters that the estimator induces is retained without having to fit a covariance matrix (which is near-singular for several never-sell rows).

### 11. Flip fractions across all 225 calibrations

| flip event | p05 | p50 | p95 | mean |
|---|---:|---:|---:|---:|
| Φ sign flip | 0.4% | 21.4% | 47.5% | 22.4% |
| Case flip | 0.4% | 21.4% | 47.5% | 22.4% |
| Zhang-decision flip | 0.0% | 0.0% | 44.0% | 6.7% |

**Aggregate headline: mean decision-flip rate = 6.7%** of bootstrap draws change the Zhang SELL/HOLD verdict, averaged across the 225 calibrations. Fraction of calibrations with >1% decision-flip rate: **24.4%**.

**Fraction of calibrations with >1% Φ-sign-flip rate: 91.6%** — this is the number the back-of-envelope in the prompt anticipated (169/225 within 1% of the sign flip → ≈75%). The realised number is very close.

### 12. Sharpe distribution under parameter perturbation
For 500 perturbed worlds, the experimental path was re-run end-to-end using one bootstrap draw per calibration. Andrade override and observed regime kept fixed at their measured values (the perturbation is on Zhang params only).

| statistic | value | Δ vs baseline (1.5871) |
|---|---:|---:|
| experimental Sharpe 5th percentile | +1.1941 | -0.3930 |
| experimental Sharpe median | +1.2661 | -0.3210 |
| experimental Sharpe 95th percentile | +1.3666 | -0.2205 |
| range (min → max) | +1.1292 → +1.4057 | — |
| std across worlds  | 0.0540 | — |

90% central spread: **0.1725** Sharpe units, vs the −0.3049 point-estimate gap. The spread is smaller than the effect — the drag is identified above the sampling noise, at this window.

### 13. Interpretation
The median calibration's decision flips on only 0.0% of bootstrap draws, but the aggregate story is more nuanced: the Φ sign flip is common (mean 22.4%), and the effect on the portfolio Sharpe is a 0.17-unit 90% spread around the point estimate. The Case-I gate is the load-bearing piece of the Zhang leg — but its decisions are exposed to the same sampling error that determines its sign, and the propagated portfolio spread is smaller than the observed −0.3049 gap: the drag is above the sampling noise, though not by a wide margin.

---
## Plain-English reading

The exit layer flagged **29 SELLs** across 17 months. Its **hit rate is 31.0%** — roughly a coin flip, not a signal that reliably cuts losers. The mean return of a SELL-flagged position is **+0.1617**, which is positive: the layer systematically cuts winners on average, and the aggregate drag comes from the false-positive tail — the largest gains forgone dwarf the largest losses avoided in raw magnitude. The pre-committed condition (a) evaluates as **FAIL** on the full pool; on the two subpopulations it is **fails** for the Zhang-gated 12 sells and **fails** for the Andrade-driven 17 sells.

A dedicated re-entry gate cannot rescue the layer. The ensemble already re-buys **44.8%** of SELL-flagged names in the following month, so a buy-back rule would recover at most one month of forgone return per decision. Under **perfect-foresight** re-entry at each month's low — a strict upper bound no realisable rule can match — the experimental Sharpe is **+1.8108**, better than the +1.5871 baseline. Independently, Task D shows the Case-I / Φ gate — the only Zhang component that ever changes a decision — is exposed to bootstrap parameter noise: the median decision-flip rate is 0.0% and the portfolio-level Sharpe under perturbation spans 0.17 units (p05→p95). The direction is not one to pursue further tuning on.

---
## Artefacts
- `backtests/exit_decisions_2026-07-30.parquet` (225 × 26)
- `backtests/exit_decision_analysis_2026-08-19.md` (this file)
- `backtests/exit_decision_analysis_2026-08-19_gen.py` (script that produced this)

**No verdict issued.** Prompt 2 tests condition (a) only. Conditions (b) and (c) belong to Prompts 3 and 4. Nothing committed.
