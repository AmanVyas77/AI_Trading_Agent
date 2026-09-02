# Markov exit layer — SHELVE verdict (Prompt 5)

**Date:** 2026-08-19 · **Branch:** `prototype/markov-exit-layer` · **Vintage:** `backtests/vintage_2026-08-04` (git_head `b5af09b68c06`, price sha `0d685aeb2927…`)  
**Interpreter:** `/Users/aman/dev/Ai Trading Agent/.venv/bin/python` — numpy 2.4.4 / sklearn 1.8.0 / pandas 2.3.3 ✔  
**Frozen model:** `models/ensemble_models.pkl` md5 `296e589f4da205eb1d171c2121d90f82` ✔  
**Prompts 3 and 4 were deliberately NOT run** — see §1.

---

## 1. Verdict — SHELVE

Applying the standing decision rule verbatim (recorded in `scripts/backtest_exit_layer.py:687-704` and the Prompt-1C reconciliation report):

> KEEP the layer only if ALL THREE hold: (a) SELL-flagged positions have a NEGATIVE mean `ret_full`; (b) the paired monthly difference is significant at 0.05 by BOTH sign test and paired t-test in the layer's favour; (c) the redistribute-to-survivors variant beats baseline on Sharpe OR cuts max drawdown by ≥ 3 pp. SHELVE if (a) fails.

**Condition (a) fails.** Prompt 2 measured mean `ret_full` given SELL = **+0.1617** and hit rate = **9/29 = 31.0%** (source: `backtests/exit_decision_analysis_2026-08-19.md` §Task B). Conditions (b) and (c) were **not tested** because (a) failing is sufficient for SHELVE per the pre-committed rule. Running them now would be searching for grounds to reopen a settled question — the exact failure mode the pre-commitment exists to prevent.

**Ancillary measurements folded in below** (Task A: the redistribute-to-survivors path and left-tail metrics for every path) do NOT alter the verdict. No Andrade-confidence tuning, regime-detector retuning, or parameter search over this window was performed — the batch instruction forbids in-sample search.

---

## 2. Decomposition, with significance

| SELL population | n | wins | hit | mean ret_full | one-sided p (observed direction) | two-sided p |
|---|---:|---:|---:|---:|---:|---:|
| Zhang-gated (zhang+both) | 12 | 7 | 58.3% | +0.0041 | 0.3872 | 0.7744 |
| Andrade-driven | 17 | 2 | 11.8% | +0.2730 | 0.0012 | 0.0023 |
| Pooled | 29 | 9 | 31.0% | +0.1617 | 0.0307 | 0.0614 |

- **Zhang-gated (n=12).** 7 correct out of 12 → one-sided p = 0.387: hit rate is above 0.5 but n=12 is too small to distinguish from a coin flip. Break-even.
- **Andrade-driven (n=17).** 2 correct out of 17 → one-sided p = 0.0012: **significantly anti-predictive**. This is the substantive finding — Andrade's exit signal on this window does not merely fail to help, it opposes the outcome.
- **Pooled (n=29).** 9 correct out of 29 → one-sided p = 0.031: below 0.05, driven by the Andrade sub-population.

**Cross-check vs the amendment.** Amendment quoted 0.39 (Zhang), 0.0012 (Andrade), 0.031 (pooled). This script gets **0.387**, **0.0012**, **0.031** — one-sided in the observed direction, exact binomial. The amendment's Andrade figure (0.0012) is P(X ≤ 2 | Bin(17, 0.5)); we get 0.0012, exact agreement to the reported precision.

---

## 3. Concentration caveat — read prominently, not buried

Five false positives — **INTC, AMD, ON, ANET, AVGO, all on 2026-03-31** — account for 3.267 of 5.197 = **62.9%** of total gain forgone. Every one is Andrade-driven (the semis rallied hard into a STRONG_SELL cluster on that month-end).

**Direction of the failure is robust.** Hit rate 9/29 at p = 0.031; Andrade sub-population 2/17 at p = 0.0012; the Zhang-gated leg is flat (mean +0.0041). Even if the March semis cluster is set aside, the residual population is still anti-predictive at the aggregate level.

**Magnitude of the failure is concentrated.** Absent that single month, gain-forgone shrinks by ~63%. Aman reading only the aggregate drag (net +4.6905, Sharpe Δ −0.3049) without this note would overstate the effect size the layer would see out-of-sample. Both facts belong on the record.

---

## 4. Attribution — numbers sum

Historic gap (pre-lookahead-fix): baseline **1.5871** − REV3 experimental **1.0783** = **+0.5088**. Decomposed:

| component | Sharpe delta | source |
|---|---:|---|
| Lookahead artifact removed (as-of regime fix, 2026-08-03) | +0.2039 | REV3 experimental → current experimental |
| Cash drag (plumbing — SELLs → cash at 1/n vs redistribute) | +0.2730 | experimental → experimental_redist |
| Andrade signal error (residual: signal-attributable) | +0.0320 | experimental_redist → baseline |
| **Sum** | **+0.5088** | — |
| **Historic gap (baseline − REV3)** | **+0.5088** | — |
| **|sum − historic|** | **0.000000** | asserted ≤ 1e-9 |

**Reading.** Roughly two-fifths of the original −0.5088 drag was plumbing (lookahead + cash drag) and about three-fifths was Andrade signal error. The lookahead artifact is already fixed. The cash drag is a book-construction choice, not attributable to the signal. The remainder — the actual Andrade-attributable drag — is **+0.0320 Sharpe units**.

---

## 5. What is now known about the Zhang leg

Findings from Prompt 1D (units + attribution) and the FORK 2 verdict, reproduced here so future work does not re-litigate:

- **Units are correct.** `[x*] = [K]` = price. `p0 ≥ x*` is dimensionally sound. Verified against the paper's own Example 2 and Table 1 AAPL row [carried: exit_layer_units_and_attribution_2026-08-04.md §A].
- **Price test never binds.** 45/45 Case I calibrations have `p0 ≥ x*` with minimum margin **33.7×**, median **423×**. To make it bind on this asset class would need `K_fraction ≈ 42%` at the median — three orders of magnitude beyond real US-equity round-trip costs. The paper's own AAPL market test is degenerate the same way: $0.0172 threshold against a $542.10 sell price [carried §F.1-F.3].
- **Case II is 0/225 and structurally unreachable.** `f1` min 2.592 vs `ρ = 0.03`, so `ρ > f1` never holds on this asset class. The daily hard-stop is not merely untriggered, it is *inert by construction*, independent of any choice of `K`. REV 4's fix A was never testable here [carried §F.4].
- **Unit tests validate the algebra only.** The 60+ tests in `tests/unit/test_zhang_optimal.py` reproduce paper Example 2 and Table 1 exactly — they say nothing about the integration with the Andrade/portfolio harness. Future work must not confuse "algebra passes" with "layer works."

---

## 6. Φ stability — corrected framing

Prompt 1D's flag "the gate is a 0.5%-of-magnitude sign test on two ~1.6×10⁴ terms — likely a coin flip" was tested in Prompt 2 Task D and is only partly right:

- Φ sign flips in a **mean 22.4%** of bootstrap draws (block bootstrap, block length 5 td, 1000 replicates per calibration). **91.6%** of calibrations have >1% Φ-flip rate, confirming the back-of-envelope.
- **But decisions flip only in a mean 6.7%** of draws (median 0.0%). **The state constraint firewalls the Φ instability**: 167 of 225 rows are state 1 (Andrade uptick) where Zhang cannot sell regardless of Φ's sign, so ~74% of the parameter noise never reaches the decision surface.
- **Portfolio-level identification is above sampling noise but unambiguously below baseline.** Across 500 perturbed worlds, experimental Sharpe spans **+1.1941 to +1.3666** (90% CI); the maximum across all draws was **+1.4057, still below baseline +1.5871**. *No parameter draw in the ensemble produces a winner.*

**Correction to the record.** The earlier "the gate is a coin flip" framing (Prompt 1D §F.5) was too strong on decision impact. Sign is coin-flip-like; decisions are not, because of the state firewall. The updated summary: the Zhang leg contributes 12 SELLs that are collectively break-even; parameter uncertainty widens the propagated Sharpe by ~0.17 units; **every world in the ensemble loses to baseline**.

---

## 7. Re-entry / buy-back — CLOSED

**Statelessness.** The exit layer has no cross-month memory. `monthly_exit_review` consumes only that month's calibration (`src/exit/exit_manager.py:266-368`); `_select_monthly_holdings` re-selects the book from that month's ensemble score alone (`src/strategies/ensemble/portfolio_builder.py:111-180`). All **16** of the 16 next-month absences among the 29 SELLs are the ensemble dropping the ticker on its own signal, not the exit layer excluding it. A dedicated buy-back gate would only affect the 13 names the ensemble re-selects anyway.

**Arithmetic ceiling.** Perfect-foresight re-entry at each month's low gives experimental Sharpe **+1.8108** vs baseline **+1.5871** (source: `exit_decision_analysis_2026-08-19.md` §8).

- What perfect foresight buys: **+0.5286** Sharpe over the layer's current +1.2822.
- What we need just to match baseline: **+0.3049** Sharpe.
- Fraction of physically-impossible timing that any realisable re-entry rule would need to capture merely to break even with doing nothing: **57.7%**.

The idea is closed on this window. Any future re-opening from intuition alone should re-read this section first.

---

## 8. Harness findings that outlive the layer

### 8.1 As-of regime lookahead pattern

The demonstrated fault was `get_live_regime_signal()` being called inside a backtest loop against its own docstring (fixed in `scripts/backtest_exit_layer.py`, 2026-08-03). **Check across the codebase (`grep -rn get_live_regime_signal src/`):**

- `src/live/rebalance.py:140` — LIVE build_targets path. Correct usage.
- `src/exit/exit_manager.py:249` — fallback inside `calibrate_for_month` when `regime_signal=None`. Correct for live; the docstring already flags that backtests MUST pass a point-in-time reading. The backtest harness supplies one (`backtest_exit_layer.py:343`).
- No other backtest-loop caller found. **The pattern is contained.**

### 8.2 Vintage / revision path — much bigger open question

Sprint 7's headline Sharpe of 1.016 no longer reproduces — re-running the same code today gives 1.0909 [carried: Prompt 1C]. The revision path was investigated:

- **Prices.** The nightly AlphaVantage backfill retroactively revises historical `prices` rows (documented, and the reason for `scripts/freeze_vintage.py`). This IS a revision path in the ensemble feature construction.
- **News → sentiment → scores.** Looked at: `news_articles` is populated by `src/data/news_pipeline.py` but **no file outside `news_pipeline.py` references it** (`grep -rn news_articles src/`). The `sentiment_scores` table feeding the feature matrix (`feature_matrix.py:24, 70`) is sourced entirely from SEC 8-K filings (SQL check on the vintage DB: `sentiment_scores.source` distribution is `sec_8k|4578` rows, zero `news` rows). **News does not feed the current entry model** — the revision path is prices, not news.
- **Does feature construction filter articles by `published_at` relative to each decision date?** N/A for the ensemble entry model (news isn't in it). Sentiment features enter via `filing_date` from `sentiment_scores`, which IS the point-in-time reference (SEC filing date). The `load_sentiment_daily` function forward-fills from `filing_date` to daily — correct as long as scored filings don't have their scores retroactively rewritten. FinBERT is deterministic on the same input, so the score for a given `(ticker, filing_date)` is stable across re-runs; the only silent-drift risk is a re-scoring with a different model.

**Bottom line.** The vintage-revision problem is real for prices (hence the frozen vintage) and does NOT apply to the current entry features via a news path. If news is ever added as an ensemble input, the temporal filter must be `published_at ≤ decision_date` with no future information leaking through re-scoring.

---

## 9. Open and untested — bear-regime behaviour

This window (2025-01 → 2026-06, 17 months) is a strong bull tape. The layer's stated purpose is left-tail management, which was not exercised. Bear-regime behaviour is genuinely untested.

**Pre-registered metrics** for any future bear-regime run (fixed here so the goalposts cannot be moved after the fact):
- **max-drawdown reduction ≥ 3 pp** vs baseline
- **Calmar improvement > 0**
- Sample: a 2022-style drawdown window of similar length (≥ 12 months, drawdown ≥ 20%).

**Caveat on interpreting a 2022 run.** The ensemble entry model was trained through 2024-07-31 on data that includes 2022. Any 2022 backtest has in-sample entry contamination biasing the baseline high, which makes the exit layer look **conservatively** bad relative to its true out-of-sample value. If it wins under that bias, that is meaningful evidence; if it loses, the bias makes the finding weaker than it would otherwise be.

---

## A. Folded-in measurements (Prompt 5 Task A)

### A.1 Redistribute-to-survivors variant

| path | Sharpe | Sortino | Calmar | max DD | worst month | 5th %ile month |
|---|---:|---:|---:|---:|---:|---:|
| baseline | +1.5871 | +4.3039 | +5.694 | -0.1183 | -0.1183 | -0.1064 |
| experimental | +1.2822 | +2.9893 | +3.625 | -0.1183 | -0.1183 | -0.1064 |
| experimental_redist | +1.5552 | +3.9034 | +5.073 | -0.1183 | -0.1183 | -0.1064 |
| hard_stop | +1.5871 | +4.3039 | +5.694 | -0.1183 | -0.1183 | -0.1064 |
| zhang_only | +1.5915 | +4.2678 | +5.644 | -0.1183 | -0.1183 | -0.1064 |

**Cash drag = experimental_redist − experimental = +0.2730** Sharpe. That is the plumbing cost of holding exited weight in cash at 1/n; it is not attributable to the Andrade or Zhang signal and the record should not charge it against the layer.

**Condition (c) — for completeness only.** The redistribute-to-survivors variant Sharpe is +1.5552 vs baseline +1.5871 (does not beat baseline). Max drawdown change vs baseline: **+0.00 pp** (worse; threshold was ≥ 3 pp improvement). **Condition (c) would fail** — but (c) does not matter once (a) has failed.

### A.2 Left-tail metrics

The table above reports max drawdown, Sortino, Calmar, worst single month, and 5th-percentile monthly return for every path. Reading them honestly, even though every one worsens:

- **Max drawdown** — baseline -0.1183, experimental -0.1183 (worse), redist -0.1183, Zhang-only -0.1183. The layer did NOT reduce the worst drawdown on this window.
- **Sortino** — baseline +4.3039, experimental +2.9893 (worse). Even the downside-only Sharpe worsens.
- **Calmar** — baseline +5.694, experimental +3.625 (worse).
- **Worst month** — baseline -0.1183, experimental -0.1183 (worse).
- **5th %ile month** — baseline -0.1064, experimental -0.1064.

Every left-tail metric on this window is either equal to baseline or worse. The layer's stated purpose was not delivered here. That does not settle its bear-regime behaviour, which remains untested.

---

## 10. Recommendation (three sentences)

**SHELVE the Markov exit layer.** Do not tune it further on this window — in-sample search over 17 months would produce numbers that are not evidence, and even the perfect-foresight ceiling for a re-entry gate captures only half the gap needed to break even with doing nothing. If the layer is ever revisited, it should be on a bear-regime window with the pre-registered max-drawdown and Calmar metrics in §9, and against Andrade's exit signal specifically (Zhang's leg is either inert or break-even on equities at this price scale).

