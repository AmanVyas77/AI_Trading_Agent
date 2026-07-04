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

## Sprint 7 candidate — 2025-26 TRUE HOLDOUT validation (burn-once, run FIRST)

**Status:** open, proposed next sprint (2026-07-04). Prices in DB to 2026-05-08,
macro to 2026-05, LM to 2026-06; FinBERT stale since 2024-12 (must backfill first).

Run the FROZEN Sprint 5 production model over 2025-01 → current day. Every
design decision (Sprints 0-6) used data ≤ 2024-11 only, so 2025-26 is a genuine
out-of-sample holdout — the strongest validation evidence this project can
produce. RULES: (1) pre-commit the expectation before running (one rule, one
run, one verdict — no iteration against this window, it is burn-once);
(2) run BEFORE any new feature (news sentiment below) is added — adding
features first destroys the frozen-design claim; (3) needs end-date override
params in the loaders that clamp to TIMELINE["test_end"] (module-level, NOT
settings.yaml); ~18 new TimesFM month-end batches; XLK/labels through 2026.
Caveat to note in the writeup: TimesFM pretraining data vintage is a possible
subtle leak channel for 2025 — acknowledge, don't overclaim.
Synergy: the data-freshness work IS Option B's prerequisite list — this sprint
doubles as the live pipeline's plumbing dry-run.

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
