# Future Ideas — AI Trading Agent

Running list of ideas surfaced during sprint work that we don't want to
lose but aren't executing right now. Each entry ends with a suggested
insertion point (which sprint, which module) so it can be pulled off
this list without re-deriving the context.

---

## Runbook patch — quarterly fundamentals ingestion + xbrl staleness guard

**Status: DONE 2026-07-14** — implemented in Runbook_Patch_Prompts (2
prompts). refresh.py now runs edgar_xbrl, lm_10k_increment (LEFT-JOIN
gated no-op when nothing new), simfin_estimates (de-clamped fetch), and
factor_export_fundamental between sentiment and quant_factors; new
120-day guards on xbrl_facts + eps_revisions. First-time run pulled
xbrl_facts max 2026-06-11 → 2026-06-17, eps_revisions 2026-04-03 →
2026-05-31, and rebuilt the fundamental parquet through Q2-2026 (46/54
tickers gained a 2026-Q2 period end). Idempotent on re-run (row counts
==). See [[runbook-patch-2026-07]] for full outcome.

**Follow-up (small, not urgent): LM collection PK-skip.** The 10-K
collection loop re-downloads all ~430 filings from SEC each refresh
(23 min) because it doesn't skip already-stored (ticker, filing_date)
PKs before hitting the network. Adding a pre-check against
edgar_10k_filings would cut lm_10k_increment from ~23 min to seconds
when there's nothing new. Not a correctness issue — INSERT OR REPLACE
keeps state consistent — just wasted bandwidth.

**Follow-up (Aman decision): SimFin API key.** Prompt 2 provenance
check confirmed `analyst_estimates.source = 'seasonal_random_walk'`
for every row from 2006-2026 → historical training data was ALWAYS
synthetic. `.env` has `SIMFIN_API_KEY=` with an empty value. Filling
it in would INTRODUCE drift going forward (real API rows for 2026+
vs synthetic history), so either leave empty (status quo — matches
training) or fill and backfill history in one shot.

---

## Original ask (kept for provenance)

Logged 2026-07-12 (Aman: "add it to memory for future
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

---

## RAG layer — FinDER benchmark DECLINED, three items kept

**Status:** decided 2026-08-08 (Aman). Logged as a full entry rather than
deleted so a future session does not re-derive the same recommendation from
the same paper. Supersedes the "FinDER-driven RAG evaluation program"
proposal written earlier the same day.

### The decision

**Do not run FinDER as a benchmark.** The dataset
(`huggingface.co/datasets/Linq-AI-Research/FinDER`, 5,703 expert-annotated
query-evidence-answer triplets over 490 S&P 500 companies' 10-Ks) is public
and directly on-topic, but running it is not decision-relevant to this
project.

**The deciding argument.** Ask what a FinDER result would change. If it says
LLM reranking gives +8pp, we implement reranking. If it says +0, we would not
believe it — we would assume it failed to transfer to our corpus and our
section-aware `SECFilingChunker` (FinDER's baselines strip HTML to flat
paragraphs). An experiment whose outcome does not change the action is not
worth the cost.

**Supporting reasons:**

- The useful content is the paper's *findings*, not the dataset. Those are
  extracted and recorded below and in [[papers/choi_2025_finder]] — available
  for free, already paid for by reading it.
- FinDER ships its own document corpus and its gold evidence is specific to
  those 10-Ks. Running it means ingesting a foreign multi-GB corpus through
  our pipeline on an 8 GB Mac, then standing up a RAGAS LLM-as-judge harness,
  to obtain a score on someone else's documents.
- The two things it would validate — that reranking helps, that expanding
  terse queries helps — are close to settled findings. Benchmarking to confirm
  them is ceremony.

**The one thing that would reopen this:** FinDER is the only way to get an
*absolute* calibration ("is our stack embarrassing or merely mediocre"),
because an in-house recall@k has no external reference point. Judged not worth
a sprint. If the analyst funnel later underperforms and retrieval is the prime
suspect, this is the diagnostic to reach for.

### Kept item 1 — LLM reranking (implement, no benchmark needed)

Retrieve wide (top-20), have Claude rerank down to 6, instead of taking raw
cosine rank. Small diff in `src/rag/query/context_builder.py`
(`MAX_CHUNKS = 6` currently consumes the retriever's top-n directly).
Paper's supporting result: Claude-3.7-Sonnet best reranker at F1 63.05, and
*"retrieval sets need not be perfectly precise — a diverse pool is beneficial,
since reasoning-focused models discern relevant information despite noise."*

### Kept item 2 — query expansion (implement, no benchmark needed)

Expand the query before it hits Chroma: resolve ticker to company name, expand
acronyms, add metric synonyms. The funnel emits terse template queries
(ticker + metric), which is precisely the failure mode measured — expert
rewriting lifted precision 25.7 → 33.9 in the paper.

### Kept item 3 — in-house retrieval eval, AFTER the funnel exists

The real gap is that the RAG layer has never been measured on *our* data.
FinDER does not fix that either. But the in-house version is small, because
the funnel's queries come from a dossier template — roughly 10-20 question
types, known in advance. We do not need a 5,703-query distribution. We need:
for those specific questions, does retrieval surface the right chunks?
Metric = recall@k against hand-checked chunk IDs. An afternoon of annotation,
not a sprint.

**Sequencing:** build the funnel first, with reranking and query expansion on
by default; measure once real dossiers exist. Measuring a component before the
system that consumes it optimises for the wrong thing.

**Burn-once discipline still applies.** If the in-house set is used to tune
retrieval configs, it cannot also serve as the funnel's acceptance gate —
same hazard as the 2025-26 holdout. Split dev / held-out at creation time.

### DROPPED — embedding model swap

`all-MiniLM-L6-v2` (384-dim, general-purpose) is weaker than everything in the
paper's comparison set; `gte-large-en-v1.5` (434M) scores 17.83 context recall
vs BM25's 11.68, and E5-Mistral tops the table at 25.95 but is 7B (~14 GB
fp16, not viable on the 8 GB Mac). **Not doing this speculatively** — it costs
a full corpus re-embed for a gain nothing currently measures. Revisit only if
item 3 shows retrieval is the funnel's binding constraint.

### Findings retained regardless (these are the paper's actual value)

- **Retrieval is the binding constraint, not the generator.** Best-in-class
  context recall on 10-Ks is **25.95%**. Claude's answer correctness runs
  **9.4 (no context) → 33.9 (top-10 retrieved) → 66.5 (perfect context)**.
  Roughly half the achievable quality is lost before the LLM sees anything —
  prompt engineering and model choice are second-order until retrieval
  improves.
- **RULING: RAG never supplies numbers to the analyst.** Numbers come from
  `xbrl_facts` and the factor parquets — exact, point-in-time. RAG supplies
  narrative context only. `context_builder.py` already injects structured
  scores; this is a reason not to expand RAG's role, not to widen it. Useful
  corollary: because RAG is narrative-only, bad retrieval yields a *vague*
  verdict rather than a *wrong number* — the blast radius is lower than it
  first appears, which is part of why the benchmark was declined.
- **Faithfulness ≫ correctness** (~85 vs ~29 on partial context). Models are
  consistent with whatever context they are handed; the context is what is
  wrong. Therefore the analyst verdict log must record **retrieved chunk
  IDs**, not just the verdict — otherwise a bad verdict is un-diagnosable
  after the fact.
- **News is a different retrieval problem** and FinDER says nothing about it
  (10-K only). Our 254k news rows are already chunk-sized, recency-dominated,
  and need cross-source dedup. Open design question, decide before building:
  news probably should not enter the dossier via semantic retrieval at all,
  but as a date-filtered SQL pull ("last 30 days of headlines for this
  ticker"). Semantic search over news mainly buys "what has been said about
  X", which may not be what the analyst needs.

### Two gates — do not conflate

- **Gate 1 (RAG quality)** — retrieval good enough to be worth wiring in?
  A prerequisite filter only.
- **Gate 2 (analyst value)** — shadow-stage forward paper trading with a
  pre-committed rule, per the END-GOAL entry above. The only thing that
  constitutes evidence of value.

Passing Gate 1 buys the right to run Gate 2, nothing more.

**Suggested insertion:** items 1 and 2 fold into the first `src/analyst/`
session as defaults. Item 3 is a follow-up once dossiers exist. No standalone
sprint.

---

## Sprint 9 flag — FinBERT is the weakest scorer on the published ladder

**Status:** open, logged 2026-08-08. Affects Sprint 9 in flight — not a
blocker, a second arm worth running.

Lopez-Lira & Tang (JFE 2026, the published version of the
`lopez_lira_2023_chatgpt_returns` draft already in the corpus) rank LLMs by
drift-strategy Sharpe: **GPT-4 2.97 > GPT-3.5 1.66 > DistilBART-MNLI 1.26 >
basic models negative**. FinBERT-class domain models sit at the bottom of
that ladder. In their regressions the GPT-4 score **subsumes** RavenPack
commercial sentiment for drift — RavenPack's coefficient goes insignificant
once GPT-4 is included.

Sprint 9 scores all 254k ingested articles with FinBERT, i.e. the weakest
arm they tested. **This is not a reason to kill Sprint 9** — monthly
aggregation into a cross-sectional feature is a genuinely different task
from daily drift trading, and FinBERT may well be adequate for it. But it
is a strong reason not to assume FinBERT is the ceiling.

**Proposed second arm:** score the same corpus with Claude and carry
`news_sentiment_llm` alongside FinBERT's `news_sentiment`. The articles are
already ingested (163,921 FNSPID 2015-2023 + 90,491 AV 2022-01→2026-08);
only the scoring pass differs. Cost control: the LLM arm only needs the
post-2022 AV window, or a subsample — the monthly aggregate is a mean over
many articles and is robust to sampling.

**Multiple-testing note:** a second feature arm is a second trial against
the same 2023-24 test window (which has now judged 5+ experiments). Each arm
needs its own pre-committed rule, and they must be declared **before**
either is evaluated — not "run both, keep the winner."

### Do NOT implement their trading strategy

Recorded so it is not revisited. Their signal is daily-rebalanced
long-short, ~190% daily turnover, 1-2 trading day drift horizon,
concentrated in **small caps** and the **short leg** (short Sharpe 2.01 vs
long 0.78). Unprofitable at 20 bps round-trip. Our system is monthly,
long-only, 54 large-cap tech names. Every dimension that makes their signal
work is one we have excluded by design.

**Two findings worth keeping anyway:**

- **Alpha decay is documented.** Their Sharpe falls 6.54 (2021Q4) → 3.68
  (2022) → 2.33 (2023) → 1.22 (Jan-May 2024) as LLM adoption rose. Any
  news-sentiment feature should be expected to decay; argues for periodic
  re-validation rather than a one-time PASS.
- **Topic-conditional underreaction.** Markets process earnings,
  partnerships and clinical trials efficiently (strong initial alignment,
  no significant drift), but underreact to insider transactions, dividend
  announcements and healthcare conference presentations (significant
  Drift×GPT of 26.3 / 22.3 / 34.2 bps). This is a gate on *when* an analyst
  verdict should carry weight — relevant to the funnel, not to Sprint 9.

---

## Analyst funnel — prompt design (buy side near-final, hold/sell OPEN)

**Status:** open, logged 2026-08-08 (Aman). Buy-side prompt believed close to
final design; selection still to be tested. Hold/sell prompt is an unanswered
design question. Belongs to the END-GOAL entry above (staged item 2).

### Provenance

The candidate prompt is Kim, Muhn & Nikolaev (2024) — already in the corpus as
[[papers/kim_2024_financial_statement_llm]], the 60.35% (GPT-4 + CoT) vs 52.71%
(human analysts, 1-month horizon) result on next-period earnings direction.
Their setup: **anonymised** standardised statements, no company name, no dates,
CoT scaffold of trend → ratio → synthesis → prediction. Anonymisation is load-
bearing in their design — it is what kills the memorisation channel.

### BUG TO FIX BEFORE BUILDING — ContextBuilder leaks the quant verdict

`src/rag/query/context_builder.py::_scores_context()` injects
**`ensemble_score`** — the XGBoost model's own output — into the prompt, plus
every fundamental factor score, whenever a ticker is detected. Verified by
reading the function, not inferred.

If the funnel reuses `ContextBuilder` to build dossiers, the analyst sees the
quant verdict before forming its own. That is precisely the channel the
END-GOAL entry rules out ("blind the analyst to the quant flag direction —
sycophancy: Sharma 2024; anchoring: Tversky & Kahneman 1974"). It is not a
hypothetical risk; it is the *default behaviour* of the function one would
naturally reuse.

**Fix:** `src/analyst/` gets its own dossier builder, or `ContextBuilder` grows
a `blind=True` path that suppresses the ensemble block. Do not rely on
remembering to pass `ticker=None`.

### RULING — the LLM does not compute numbers

Extension of the RAG ruling recorded in the FinDER entry above. Numbers come
from `xbrl_facts` and the factor parquets. RAG supplies narrative. **The LLM
supplies interpretation only.**

The draft prompt asked for eight ratios "showing calculations explicitly."
Dropped. Rationale:

- We already compute all of it. `piotroski_f` alone subsumes ROA, leverage,
  current ratio and asset turnover as components; plus `gross_profitability`,
  `qmj_safety`, `qmj_payout`, `fcf_yield`, `revenue_acceleration`,
  `deferred_revenue_yoy`, `rd_intensity`, `sue_score`.
- [[papers/bubeck_2023_sparks_agi]] is in the corpus specifically documenting
  GPT-4 numerical-reasoning failure modes. Hand-computed ratios are the single
  most likely source of a wrong verdict.
- The explicit-calculation block is the longest and most expensive output
  section, and it produces nothing we do not already hold exactly.

Corollary: do not paste raw two-year statements either. Emit a compact
structured table from `xbrl_facts` with derived ratios pre-attached
(~3-5× input token reduction).

### Cache the static half

Persona + tag schema + analytical instructions are identical across all 54
tickers every month; only the dossier varies. Prompt-cache the prefix. At
54 names × monthly × two arms (below), this is a material cost line, not a
micro-optimisation.

### Output design — emit BOTH a decision and a calibration claim

Three-way target mismatch, resolved deliberately:

| | Target |
|---|---|
| Kim et al. | next-period **earnings direction** |
| Our XGBoost label | **outperformance vs XLK** |
| Funnel's stated job | **CONFIRM / REJECT** a momentum flag |

Keep earnings direction *alongside* the verdict — not because it is the
decision, but because it is **falsifiable on a known date**, which the END-GOAL
entry explicitly asks for. CONFIRM/REJECT is the actionable output but is mushy
to grade; earnings direction is a hard dated claim gradeable without waiting on
price action or arguing attribution.

`<confidence>` (HIGH/MEDIUM/LOW self-report) is poorly calibrated in LLMs. Log
it, but do **not** let it gate the veto cap until it is shown to predict
anything. Better candidates: agreement across N samples at temperature > 0, or
agreement between the two arms below.

### Candidate prompt (buy side)

```
[CACHED PREFIX — static across all tickers]

You are a senior financial analyst. You will receive a structured
financial summary. All figures are pre-computed and exact — do NOT
recompute them, and do not perform arithmetic. Interpret what they mean.

If a figure needed for a judgment is absent, say so explicitly rather
than estimating it.

Respond using exactly the XML tags below, in order.

<trend_analysis>
The 3-5 most important trends across revenue, cost structure,
profitability, and balance sheet composition. For each: direction,
magnitude, and implication. Cite the specific line item or ratio.
</trend_analysis>

<ratio_interpretation>
Interpret the provided ratios and their year-over-year changes. Focus
on what drove each change and whether it is likely to persist. Flag
any ratio that contradicts the others.
</ratio_interpretation>

<risk_factors>
2-3 concerns evident from the data. Cite the relevant line items.
</risk_factors>

<reasoning>
Synthesize the above into a judgment about the company's earnings
trajectory and the durability of its recent price momentum. State
which single factor most influenced your conclusion, and what would
have to be true for you to be wrong.
</reasoning>

<earnings_direction>INCREASE or DECREASE</earnings_direction>

<falsifiable_checkpoint>
One specific, dated, checkable claim implied by your reasoning
(e.g. "FY2026 Q3 revenue YoY growth below 12%").
</falsifiable_checkpoint>

<verdict>CONFIRMED or REJECTED</verdict>

<confidence>HIGH, MEDIUM, or LOW — one sentence of justification.</confidence>

[VARIABLE BLOCK — per ticker]

<financial_summary>
  [compact xbrl_facts table, 2 fiscal years]
  [pre-computed ratios: gross_profitability, piotroski_f components,
   qmj_safety, fcf_yield, revenue_acceleration, ...]
</financial_summary>

<narrative_context>
  [RAG chunks — reranked top-6, NARRATIVE ONLY, with chunk IDs]
</narrative_context>

<recent_news>
  [date-filtered headlines, last 30d]
</recent_news>

<macro_snapshot>
  [vix, spread, regime — WITHOUT the multiplier decision]
</macro_snapshot>
```

**Deliberately absent from the variable block:** `ensemble_score`, momentum
rank, regime multiplier, and any indication of *why* this ticker was selected.
`<verdict>` asks CONFIRMED/REJECTED without ever stating what is being
confirmed — the model evaluates the business; the funnel maps that onto its own
flag afterwards.

### Two arms in the shadow stage — this replaces the FinDER measurement

- **Arm A** — anonymised, statements only, no RAG, no ticker. Kim's exact
  setup, with a published 60.35% external reference point.
- **Arm B** — named, full RAG dossier + news + macro.

**The A→B delta IS the RAG pipeline's contribution**, measured on our data, on
the decision we actually care about. This is the number the FinDER benchmark
could not have given us (see the DECLINED entry above), and it arrives free
with a stage already planned. Arm A doubles as an upstream sanity check: if we
cannot reproduce ~60% on the anonymised task, something is broken before RAG
and no amount of retrieval tuning will save it.

### Prompt selection can happen OFFLINE and cheaply — anonymisation is the key

Aman (2026-08-08): "we can run some tests to fully identify which prompt to use
in buy side." Important methodological point that makes this cheap:

**Anonymisation is exactly what makes historical evaluation valid.** The
END-GOAL entry's ban on backtesting the analyst exists because Claude knows
2018-24 outcomes for *named* companies. Strip the name, dates and identifying
detail — Kim's own design — and the memorisation channel closes. So prompt
variants can be baked off **offline against hundreds of historical
ticker-quarters**, graded immediately against realised earnings direction, with
no shadow months consumed and no holdout burned.

Only the winning prompt then enters the forward shadow stage.

**Limits of the offline bake-off (do not overclaim it):**
- It selects only for the *statements-reasoning scaffold*. It is silent on how
  well a prompt uses RAG context, news, or macro, because all three are
  inherently identifying and cannot be anonymised.
- So: offline bake-off picks the Arm A scaffold; the dossier-integration parts
  of Arm B still need forward evaluation.

**Multiple-testing discipline applies to prompts too.** Testing n prompt
variants against the same shadow window and keeping the winner overfits that
window exactly the way a feature sweep would. Either pre-commit the selection
rule, or split the shadow months dev/holdout at the start. Offline selection
is the cheap way to avoid spending shadow months on this at all.

### OPEN — hold/sell layer prompt

Aman flagged (2026-08-08) that the hold/sell layer needs its own prompt
decision. Not yet designed. Recorded observations:

**It is a structurally different question from the buy side.** Buy is
cross-sectional and comparative ("is this a top-20 name this month?").
Hold/sell is absolute, position-specific and path-dependent — conditioned on
entry date, entry price, holding period, and *what has changed since entry*.

**The prior verdict becomes an input — this is what `falsifiable_checkpoint`
is for.** Month N's checkpoint is graded at month N+1 and fed back: "you
predicted X; here is what happened; does the thesis still hold?" That turns the
append-only verdict log into a self-grading loop and gives the hold/sell layer
an input the buy layer structurally cannot have. Strong argument for keeping
the checkpoint tag even though it is not the decision.

**Behavioural hazard — decide whether the model sees P&L at all.**
Showing unrealised P&L invites disposition-effect and anchoring behaviour
([[papers/tversky_kahneman_1974_anchoring]], already in corpus). Argument for
blinding: the correct question is "would I buy this today?", not "am I
underwater?" Argument against: a genuine exit layer legitimately needs position
state. Unresolved.

**Cheapest candidate answer — there may be no second prompt.** The classic
reduction is *"if I did not own this, would I buy it today?"* If yes, hold; if
no, sell. That collapses hold/sell into the buy-side prompt plus a different
downstream rule, and avoids validating a second prompt entirely. Counter: it
ignores sell-specific signals (thesis broken, checkpoint failed, a named risk
factor materialised) that a fresh cross-sectional buy evaluation would not
weight properly. **Test the reduction first** — it is nearly free, and if it
holds, a whole workstream disappears.

**Interaction with the stalled Markov exit layer — resolve before building.**
`src/exit/` (Zhang + Andrade DHMM) currently has **no verdict** and a known
degeneracy: Zhang has no Andrade-independent decision surface, its price
threshold is always breached (x\* ≈ 0.15-3% of p0), and there are zero Case II
calibrations on the tested window, so the hard-stop path is structurally inert.
See `backtests/exit_layer_vintage_and_andrade_off_2026-08-04.md`.

An LLM hold/sell layer is a *different approach to the same problem*. Do not
build it as a silent replacement — decide first whether the Markov layer is
dead, parked, or pending re-derivation. **Two unvalidated exit layers is worse
than one**, and stacking them would confound both.

**Suggested insertion:** buy-side prompt + offline bake-off is the first
`src/analyst/` session. Hold/sell prompt is a later session, gated on (a) the
Markov exit layer's disposition and (b) the buy-side reduction test above.
