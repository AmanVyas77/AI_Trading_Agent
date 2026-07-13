# Future Ideas — AI Trading Agent

Running list of ideas surfaced during sprint work that we don't want to
lose but aren't executing right now. Each entry ends with a suggested
insertion point (which sprint, which module) so it can be pulled off
this list without re-deriving the context.

---

## Runbook patch — quarterly fundamentals ingestion + xbrl staleness guard

**Status:** open, logged 2026-07-12 (Aman: "add it to memory for future
things to include"). Small single-session job, NOT a full sprint.
Ideally before the 2026-08-03 rebalance (Q2-2026 10-Qs land mid-July →
August).

**Gap:** src/live/refresh.py (Sprint 8) refreshes prices, FRED macro,
CPI, FinBERT 8-Ks, TimesFM, and the feature matrix — but does NOT call:
- the EDGAR/XBRL pipeline (src/data/edgar_pipeline.py → xbrl_facts) —
  so new quarterly statements (10-Qs) never flow in automatically;
- the 10-K collection + LM scoring increment;
- SimFin estimates (eps_revisions stale since 2026-04; also carries a
  TIMELINE clamp found in Sprint 7 recon: simfin_pipeline.py
  fetch_simfin_eps dict literal "end": TIMELINE["test_end"] — needs a
  param/module-override, NOT a settings.yaml edit).
And the staleness assertions don't watch xbrl_facts at all — quarterly
fundamentals can drift from normal reporting lag to genuinely stale
without failing loud.

**Fix (2-prompt patch: implement + verify):**
1. Add steps to run_refresh(): EDGAR/XBRL re-ingestion (incremental),
   10-K/LM increment, SimFin estimates (after de-clamping).
2. Add staleness assertion: max(xbrl_facts.end_date) ≤ ~120 days old
   (calibrated to SEC 10-Q deadlines ~40d after quarter end, same
   philosophy as the CPI 75-day calibration).
3. Same fail-loud test pattern as the Sprint 8 freshness gates.
Note: quarterly ffill lag is in-sample-consistent (feature matrix
ffills quarterlies), so weeks of lag are fine — the guard exists to
catch QUARTERS of lag.

**Suggested insertion:** standalone Runbook_Patch_Prompts.md session,
before the first August rebalance; independent of Sprint 9 (different
files).

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

## END-GOAL — Unified quant→analyst funnel (the project's target architecture)

**Status:** open — this is the stated end-vision (Aman, 2026-07-04). It is the
unfinished half of the original Phase 4 two-table plan: Table 1 papers became
code features (done, Sessions 1-5 + Sprints 0-5); Table 2 papers were curated
into ChromaDB specifically as a knowledge base for an LLM stock evaluator
(built, never wired into the decision loop — see research_wiki/_index.md
"LLMs for Financial Analysis" section: Kim 2024, Cao 2024, Kim 2023a/b,
Sharma 2024, etc.).

**Vision:** quant side runs its strategies AND flags top momentum candidates →
Claude API receives a per-ticker dossier (XBRL fundamentals, FinBERT/LM
sentiment, macro snapshot, RAG retrieval over the Table 2 corpus) → renders a
structured verdict: is the momentum research-supported (CONFIRMED) or a likely
false positive (REJECT)? Systems run in PARALLEL during a testing stage to
measure each side's standalone performance, then in UNITY (analyst verdicts
influence the book) once the parallel stage proves value.

**Staged implementation:**
1. Live paper harness first (Option B below) — the funnel needs a monthly live loop.
2. `src/analyst/` — candidate feed (top-N ensemble/momentum), dossier builder
   (reuse rag/query/context_builder.py), Claude API call via LLMClient
   anthropic backend (NOT ollama — kernel-panics this machine), append-only
   verdict log in SQLite (timestamp, model version, prompt hash, verdict,
   confidence, falsifiable checkpoint).
3. Shadow/parallel stage: analyst scores ALL 54 names monthly (not just top-20 —
   3× the sample), verdicts logged but never trading. Pre-committed PASS rule
   written BEFORE the window opens (Sprint 3-6 discipline).
4. Unity stage (only on PASS): veto-only integration first (capped — max ~5
   vetoes/month, replaced by next-ranked, never left as cash: Sprint 6 breadth
   lesson), sizing tilts / score blending later.

**Known hazards (agreed in advance):**
- NO historical backtest of the analyst — Claude knows 2018-24 outcomes;
  hindsight contamination makes any backtest spuriously good. Forward
  paper-trading only.
- Blind the analyst to the quant flag direction (sycophancy: Sharma 2024;
  anchoring: Tversky & Kahneman 1974 — both in the corpus).
- Thin statistical power: ~6 months × 54 names before the comparison means much.
- Pin model version + temperature; store prompts; log is the experiment.

**Suggested insertion:** Phase 6, immediately after Option B is live.

---

## Multi-sector basket expansion (post-analyst-funnel)

**Status:** open, logged 2026-07-04 at Aman's request. Slot: after the analyst
funnel is validated.

Replicate the full strategy across S&P 500 sectors as independent baskets —
same pipeline per basket (own universe list, own sector-SPDR benchmark for the
outperformance label, own fine-tuned XGBoost) — then combine the sleeves into
one portfolio via an allocation layer (equal-weight sleeves first; regime-driven
sector tilts later = the full RegimeFolio vision already in the RAG corpus).

**Sectors to include (Aman, 2026-07-04):** Healthcare, Communications,
Consumer Discretionary, and similar GAAP-friendly sectors (Industrials,
Consumer Staples, Energy, Materials, Utilities as candidates).
**Explicitly EXCLUDE: Financial Services** (Aman's call) — and note REITs share
the same accounting problem: Piotroski / FCF-yield / gross-profitability break
structurally for banks/insurers/REITs (no meaningful gross profit or capex;
xbrl_features.py would produce silent garbage through the ffill/fillna(0) path).

**Honest cost assessment (from the 2026-07-04 sizing discussion):**
- ~80% of ensemble machinery transfers as-is (trainer, purge/DSR, scorer,
  portfolio builder, backtest, TimesFM, EWM, regime gate). BENCHMARK swap per
  basket is trivial.
- Real work #1 — sector-shaped fundamentals: deferred_revenue_yoy and
  rd_intensity are tech-centric; every sector needs a feature audit (per-sector
  feature engineering, not just model fine-tuning).
- Real work #2 — data backfill: ~450 new tickers × 10y EDGAR XBRL + 8-K FinBERT
  on an 8 GB Mac = days-to-weeks of CPU; wants overnight batches or a cloud box.
- New component: sector-sleeve allocator.
- Multiple-testing explosion: 10 sectors × experiments — pre-committed rules
  mandatory everywhere.
- Estimate: 12-18 sprint-equivalents (~2-3 months at observed cadence).

**Execution plan: PILOT FIRST** — one sector (Healthcare or Industrials),
3-4 sprints, to surface every generalization problem at ~10% of the cost.
Bonus: each new sector's 2025-26 window is still a usable holdout (no design
decision has touched it). Strategic payoff: the -29.8% 2022 drawdown is
substantially tech beta; multi-sector sleeves are the most direct
diversification of it.

---

## STRETCH GOAL — Markov-chain hold/sell portfolio optimization

**Status:** parked by design (Aman, 2026-07-04: "do not take this into
consideration now"). Nothing concrete yet — recorded so it isn't lost.

Idea: use advanced quant methods — Markov chains / regime-switching state
models — for the hold-vs-sell decision on individual positions, i.e. proper
portfolio optimization on top of the monthly top-N book. Natural connections
already in the codebase/corpus: Hamilton regime-switching + Ang & Bekaert are
in the RAG corpus (regime_detection/), and the regime gate is already a crude
2-state version of this at the portfolio level. A per-position Markov layer
would model transition probabilities between hold/trim/exit states conditioned
on regime + position P&L state.

**Suggested insertion:** end of project, after the unified funnel is validated.
Needs its own design pass — do not bolt onto Option B or the analyst layer.

---

## Sprint 7 — 2025-26 true-holdout validation — **VALIDATED** (burn-once, spent)

**Status:** closed 2026-07-04. Verdict: **PASS** — holdout Sharpe
**1.016** > pre-committed 0.600 bar (Δ +0.416). See
`backtests/results/sprint7_results.json` and `memory/phase_progress.md`
Sprint 7 section for the numbers and the seven caveats.

Key numbers (2025-01-31 → 2026-06-30, 354 daily obs):

| | Strategy | SPY |
|---|---|---|
| Sharpe | **1.016** | 1.007 |
| CAGR | +32.89% | +17.95% |
| Total return | +49.33% | +26.21% |
| Max DD | −27.03% | −18.76% |

What the result does and does not prove:
- **Proves:** the Sprint 5 factor stack + graded regime gate did not
  decay to noise on a genuine out-of-sample window.
- **Does not prove:** statistical robustness. 18 months puts the SE on
  Sharpe at ~±0.30; May 2026's +23-pt single-month excess is roughly
  half the total-return gap; 82% of days were RISK_ON (1.2× leverage);
  TimesFM's pretrained-checkpoint data vintage is a possible subtle
  leak channel for 2025. Any one of those is individually sufficient
  to explain the excess.

**Burn-once — the window is now spent.** No future experiment may tune
against 2025-01-01 → 2026-06-30. Post-2026-06-30 months become the
next available holdout as they accrue (~monthly cadence). If a future
sprint's changes need OOS validation, it must either wait for enough
new months to build a fresh window, or rely on in-sample CV alone
until the next natural holdout is available.

**Inheritance for Sprint 8 (Option B, below):**
- `src/live/scorer.py` — the frozen-model scorer is reusable as-is
  for the live pipeline's unlabeled-current-month scoring path (it
  already handles X.reindex/fillna and the last-fold selection).
- `scripts/run_holdout.py` — the injection pattern
  (`build_portfolio_weights(scores_df=, prices=)`) is the template
  for the paper-trading harness; internal loaders' `TIMELINE`
  clamps are bypassable without editing tracked modules.
- `scripts/backfill_cpi.py` — Sprint 8's live schedule can call it
  monthly to keep `macro_series['cpi']` current until CPIAUCSL is
  added to `settings.yaml`'s FRED list.
- Data feeds current as of 2026-07-02: prices, VIX/T10Y2Y/DFF/etc.,
  FinBERT (2026-07-02), LM (2026-06-05), CPI (2026-05-01).
- Sentiment pipeline now takes `--date-start/--date-end` — the daily
  live loop can pass yesterday-to-today windows for incremental
  updates without touching the module constants.

---

## Sprint 9 candidate — News sentiment as 24th feature (FMP source verified)

**Status:** open (2026-07-04). Must come AFTER the 2025-26 holdout (see above).

Verified via the FMP connector: per-ticker timestamped stock news (Benzinga,
Seeking Alpha, Fool, press releases) with usable coverage from ~2018-19 (NVDA
2019-03 rich; 2016 empty) — aligns with the first walk-forward test month
(2018-06); earlier months take neutral NaN→0 like other features.
Pipeline: ingest to new SQLite table (point-in-time publishedDate = no
lookahead) → score with the EXISTING FinBERT → aggregate per ticker-month →
Arratia EWM smoothing → `news_sentiment` as 24th feature → full sprint with
pre-committed rule (counts as another multiple-testing trial). Dedupe sources;
consider separating press-release tone from commentary tone. Corpus support:
Lopez-Lira 2023 (headlines→returns). Same feed later serves the live pipeline
daily and the analyst-funnel dossiers.

---

## Sprint 8 outcome log — Live paper-trading pipeline (Option B) — **LIVE**

**Status:** closed 2026-07-12. Verdict: **PASS** — dress rehearsal on
the Alpaca paper account cleared all six pre-committed operational
criteria. See `backtests/results/sprint8_results.json` and
`memory/phase_progress.md` Sprint 8 section for the criteria table
and evidence.

**first_live_month:** 2026-06. The runbook is a three-command monthly
cycle:

```
.venv/bin/python -m src.live.refresh
.venv/bin/python -m src.live.paper_runner --skip-refresh              # dry-run
.venv/bin/python -m src.live.paper_runner --skip-refresh --execute    # submit
```

Prerequisites that were open pre-Sprint 8 have all been resolved:

- **FinBERT backlog** — cleared in Sprint 7; live sentiment is
  ≤ 14d fresh, checked by `refresh.py`'s freshness assertions.
- **Regime-gate LIMIT-7 / staleness fixes** — done in Prompt 2:
  `_load_macro_snapshot` is per-series `MAX(date)`; `MAX_STALENESS_DAYS`
  and a critical-series (vix/spread) NEUTRAL fallback are in place.
- **Live scorer for unlabeled current months** — Sprint 7 delivered
  `src/live/scorer.py`; Sprint 8's `build_targets()` calls it directly.
- **Execution venue** — Alpaca paper; `ALPACA_API_KEY`/`ALPACA_SECRET_KEY`
  in `.env`; `broker_alpaca.py` hard-locks to the paper endpoint URL.
- **`src/live/` module with a scheduled rebalance harness** — shipped:
  `refresh.py`, `rebalance.py`, `broker_alpaca.py`, `paper_runner.py`.

Known limitations (recorded pre-Sprint 9):

- LLM arm on the regime gate is disabled by default and hard-blocks
  ollama by name (kernel-panic history). Rules-only signal drives
  every live rebalance.
- Whole-share qty fallback is wired but was not exercised in the
  rehearsal (all 6 names accepted fractional notional).
- CPI staleness limit is 75d (BLS cadence calibration); no metric
  attached, just calendar reality.
- The dress-rehearsal orders were queued (Sunday); fill verification
  against ±0.5% happens at the next market open by design.
- RISK_ON gross is 1.2× — relies on Alpaca paper's 2-4× buying_power;
  `paper_runner` asserts `buying_power ≥ gross` before submission.

**Next open item is now the END-GOAL analyst funnel** (entry above).
Option B being live is the funnel's prerequisite (item 1 of its staged
implementation is now met). The shadow-stage prerequisite list under
that entry is **unchanged** — it still requires an `src/analyst/`
module, blinding, model-version pinning, and a pre-committed PASS rule
written before the shadow window opens.

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
