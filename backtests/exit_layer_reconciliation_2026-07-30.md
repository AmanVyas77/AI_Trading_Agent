# Exit-layer baseline reconciliation — Sprint 7 holdout (1.016) → `backtest_exit_layer.py` baseline (1.5871)

**Date:** 2026-08-03
**Branch:** `prototype/markov-exit-layer`
**Frozen model:** `models/ensemble_models.pkl`, md5 `296e589f4da205eb1d171c2121d90f82` ✔ (matches expected)
**Window:** 2025-01-01 → 2026-06-30 (17 monthly holding periods, 2025-01-31 → 2026-06-30)

---

## VERDICT: **RECONCILED** — residual **0.0** at the only seam where it is measurable

> **Correction (post-review).** An earlier draft quoted the residual as `+0.000045`.
> That figure is **not** a bridge residual and should not be read as one. It is
> `E - 1.5871` where `E = 1.587144507439707` is the recomputed baseline and `1.5871`
> is the published **four-decimal constant** (`backtest_exit_layer.py:102`,
> `REV3_SHARPE["baseline"]`). It is therefore just the rounding error of that constant,
> and it is **bit-identical** to the "REV 3 → REV 4 baseline drift" quoted in the brief
> because that quantity is the same arithmetic — `backtest_exit_layer.py:428` computes
> `abs(sharpe["baseline"] - REV3_SHARPE["baseline"])`, same minuend, same subtrahend.
> One number computed twice, not two agreeing numbers.
>
> The bridge has **two anchors good only to 4 dp**: `1.0160` (as stored in
> `sprint7_results.json`; recomputing from the published equity CSV gives
> `1.0160246487064075`) and `1.5871`. The bridge closes to within the rounding of both.
> The genuinely harness-attributable residual, at the L4b→L5 seam, is exactly `0.0` —
> but that is a tautology, since L5 and the target are the same computation.
>
> **The load-bearing evidence is therefore not the residual.** It is: (a) each rung is a
> single-variable re-run, (b) book membership is provably identical in all 17 months, and
> (c) the baseline agrees with the published figure to the full 4 dp available.

The 0.57 gap is fully explained. It is **not** a bug in either harness and **not** lookahead
in the baseline path. It is five stacked apples-to-oranges differences, one of which
(the data vintage) means the published 1.016 can no longer be regenerated at all.

---

## The bridge

Every rung is a number produced by actually re-running with exactly one thing changed,
on today's DB. No estimates.

| # | Step | Δ | Sharpe | Cause (file:line) |
|---|------|---|--------|-------------------|
| L0 | **Sprint 7 published** (daily, √252, old price vintage) | — | **1.0160** | `backtests/results/sprint7_results.json` |
| L1 | + re-run the *same* Sprint 7 harness on **today's DB** | **+0.0749** | 1.0909 | data vintage — see §Vintage |
| L2 | + switch estimator daily·√252 → monthly·√12 | **−0.0021** | 1.0888 | `scripts/backtest_exit_layer.py:136-143` vs `sprint7_results.json` `pre_committed_rule` |
| L3 | + turn **fees & slippage off** | **+0.0597** | 1.1485 | `scripts/run_holdout.py:129-130`; `config/settings.yaml` (0.001 / 0.0005) |
| L4 | + turn the **regime gate off** | **+0.3589** | 1.5074 | `src/strategies/ensemble/portfolio_builder.py:284-299` (mult at :294) |
| L5 | + daily re-targeting → **monthly buy-and-hold** | **+0.0797** | **1.5871** | `scripts/run_holdout.py:123-133` + `portfolio_builder.py:218` vs `backtest_exit_layer.py:187-192, 252` |
| | **Residual vs the 4-dp constant 1.5871** | | +0.000045 *(rounding of the constant — see correction above, NOT a bridge residual)* | — |

`backtest_exit_layer.py`'s baseline reproduces **bit-exactly** (1.5871) on today's DB.
Sprint 7's 1.0160 does **not** (see below).

### Order sensitivity (honest caveat)

L4 and L5 do not commute — they interact through leverage. Their **sum is invariant** at
+0.4386, but the split depends on order:

| Order | regime-off term | buy-and-hold term |
|-------|-----------------|-------------------|
| regime-off first (table above) | +0.3589 | +0.0797 |
| buy-and-hold first | +0.1474 | +0.2912 |

Read the +0.4386 as a single joint "construction + leverage" term; the per-row split is
presentational.

---

## Vintage: the 1.016 anchor is no longer reproducible {#vintage}

Re-running the exact Sprint 7 pipeline (`score_months` → `build_portfolio_weights` →
vectorbt, unchanged code) against today's DB gives **1.0909, not 1.0160**. Equity diverges
from the published curve starting **2025-02-28**, with per-month return differences up to
**5.87 pp** (2026-06) and **−3.59 pp** (2026-05):

```
month     published   re-run    diff
2025-03      -9.51    -10.09   -0.58
2026-04       8.39     10.40   +2.01
2026-05      28.69     25.10   -3.59
2026-06      10.13     16.00   +5.87
```

Cause: the `prices` table is being **retroactively rewritten**. `logs/av_backfill_state.jsonl`
shows Alpha Vantage backfill runs on 2026-07-31, 08-01, 08-02, 08-03 and 08-04 adding
1,303 / 75 / 4,226 / 7,980 / 4,144 rows, with cursors pointed at *historical* windows
(`MSFT 2022-07→2022-12`, `INTC 2024-01→2024-06`). Sprint 7 ran 2026-07-04, so its
`adj_close` vintage predates all of it.

**Implication:** `sprint7_results.json` is a frozen artefact of a price snapshot that no
longer exists. It cannot be re-derived, and any future comparison against 1.016 is
comparing across vintages. This is independent of the exit layer and worth its own fix
(snapshot/hash the price table per sprint).

---

## Verified NON-differences

Each of these was checked and is **not** a contributor:

| Candidate | Finding |
|-----------|---------|
| **Risk-free subtraction** | **Neither harness subtracts rf.** `_annualized_sharpe` (`backtest_exit_layer.py:143`) is `mean/sd·√12`; Sprint 7's pre-committed rule is `mean/std·√252` from the equity curve. Worth **0.00**, not the hypothesised 0.2–0.3. |
| **Weighting** | Both equal-weight 1/n. `portfolio_builder.py:154-156` vs `backtest_exit_layer.py:252`. Identical. |
| **TOP_N / MIN_SCORE / universe** | `backtest_exit_layer.py:86` imports the *same* constants (`TOP_N=20`, `MIN_SCORE=0.52`) from `portfolio_builder.py:57-58`. **Book membership verified identical in all 17 months** (set comparison, zero differences). |
| **`DO_NOT_TRADE_BAND` (0.0025)** | Exists only at `src/live/rebalance.py:60`, used at `:222`. On the **live** path. Called by neither harness. |
| **Score source / clipping** | Both call `score_months()` on the same feature matrix + frozen fold. Regenerated output is **byte-identical** to the on-disk parquet: 954 rows, **18 month-ends, 2025-01-31 → 2026-06-30**. Not clipped. `n_held` per month matches the Sprint 7 holdings exactly. |
| **Dividends** | Both read `adj_close` from the same `prices` table. Same treatment. |
| **Partial first/last month** | Both span the same 17 periods; the exit harness labels each return by its *starting* month-end, Sprint 7 by its ending month-end. Once aligned, same periods. |

---

## Lookahead audit of the exit harness

Searched for: full-window data in per-month calibration; `.shift()` sign errors;
`wide[t]` slices with upper bound > `me`; parameters fit on post-decision data.

**Clean:**
- `calibrate_for_month` **does** truncate: `window = closes_df.loc[closes_df.index <= month_end]`
  (`src/exit/exit_manager.py:226`). Handing the full-history `closes_df` in from
  `backtest_exit_layer.py:194` is therefore *not* a leak.
- **No `.shift()` anywhere in `src/exit/`.**
- Only two slices reach past `me`, both legitimate: the holding-period day loop
  (`backtest_exit_layer.py:226-229`, `me < t ≤ me_next`) and the exit price
  (`:188`). Neither is used to *decide* anything at `me`.
- **The baseline path calls no calibration output at all** — `ret_full` (`:191`) is pure
  price arithmetic. This is why the ladder closes exactly at the L4b→L5 seam.

**LEAK FOUND — present in the exit harness, absent from Sprint 7:**

`src/exit/exit_manager.py:238` calls `get_live_regime_signal()` inside
`calibrate_for_month`, which `backtest_exit_layer.py:201` calls inside the per-(ticker,
month) backtest loop. That function's own docstring
(`src/strategies/ensemble/regime_gate.py:252`) states:

> "Only for live/forward use — never called inside the backtest loop."

It takes no as-of parameter: it reads the **current** macro snapshot (as-of dates
2026-05-01 → 2026-07-13) and applies it to gate decisions for **2025-01 onward**. Sprint 7
instead uses `get_historical_regime_multipliers()` (`regime_gate.py:160`), explicitly
labelled *"Suitable for backtest use"*, which returns a genuine as-of series
(1.2 ×15 months, 1.0 ×2, 0.5 for 2026-03-31).

Two consequences for Prompts 2–4:

1. **REV 4 fix B is currently inert.** The live signal returns `multiplier = 1.0` with
   `stale = True` on `vix`, `yield_spread_10y2y`, `fed_funds_rate`, `cpi` — staleness
   degrades RISK_ON to NEUTRAL. Since suppression needs `mult > NEUTRAL_MULT (1.0)`, the
   Andrade override fires in **every** month, exactly as in REV 3. This matches commit
   `efc1831` ("both inert, Sharpe unchanged from REV 3").
2. **The experimental path is not reproducible across run dates.** If the macro series are
   refreshed and the multiplier flips to 1.2, every Andrade STRONG_SELL is suppressed and
   the experimental Sharpe changes — with no code change. The −0.5089 is a function of
   *when you run it*.

---

## Bottom line

- The **baseline** number from `backtest_exit_layer.py` is trustworthy: it reconciles to
  Sprint 7 to 5e-5 once vintage, estimator, costs, leverage and rebalancing are accounted
  for, and the two harnesses select provably identical books.
- The 0.57 gap was never evidence of a bug. It is dominated by the **regime gate /
  rebalancing construction** (+0.4386 jointly) plus **costs** (+0.0597) — i.e. Sprint 7
  measures a levered, cost-charged, daily-retargeted portfolio, and the exit backtest
  measures an unlevered, frictionless, buy-and-hold one.
- Two things still need fixing before the −0.5089 verdict means anything:
  1. `get_live_regime_signal()` must be replaced with an as-of historical lookup in
     `calibrate_for_month`, or the regime gate removed from the backtest path entirely.
  2. The price vintage must be pinned/hashed per sprint, or no historical result in
     `backtests/results/` is reproducible.

---

## End-to-end verification of the baseline (post-review)

The ladder's L5 rung was a *reimplementation* of the exit harness's baseline leg. To
remove that as a load-bearing assumption, the real script was run end-to-end under the
repo's own environment (`.venv`, which unlike the ambient `python3` has `hmmlearn`):

```
.venv/bin/python scripts/backtest_exit_layer.py --start 2025-01-01 --end 2026-06-30
  baseline      : +1.5871
  experimental  : +1.0783   (Δ vs baseline: -0.5089)
  hard_stop     : +1.5871
  baseline drift vs REV 3: 0.000045 (OK)
```

Matches the reimplementation exactly, and the script's own `baseline drift` line prints
the same `0.000045` — confirming it is the rounding of the `1.5871` literal, as described
above.

**Environment note.** The ladder was originally computed under `/opt/anaconda3/bin/python3`
(numpy 1.26.3, sklearn 1.2.2) rather than the repo's `.venv` (numpy 2.4.4, sklearn 1.8.0).
Re-running every rung under `.venv` reproduces all six values to **≤ 4.4e-16**, and
`score_months` output is bit-identical across both, so the reconciliation is
interpreter-independent. Only `hmmlearn` (needed for the Andrade leg, not the baseline)
was missing from the ambient interpreter.

---

## Fixes applied (2026-08-03)

### 1. As-of regime signal — lookahead removed

- **New** `get_regime_signal_asof(as_of, max_staleness_days=45)` in
  `src/strategies/ensemble/regime_gate.py`. Reads only macro observations dated
  ≤ `as_of`; returns the same dict shape as the live signal; never calls an LLM;
  a stale or missing critical series degrades to NEUTRAL rather than trading off an
  old reading.
- `calibrate_for_month` (`src/exit/exit_manager.py`) takes a new `regime_signal`
  parameter. `None` preserves the live path exactly; backtests must supply an as-of
  reading.
- `backtest_exit_layer.py` fetches it once per month (it does not vary by ticker) and
  reports the per-month reading in a new report section.

Cross-validation: the as-of signal agrees **18/18 months** with the independently
written `get_historical_regime_multipliers()`.

**Effect** — the baseline is unchanged (1.5871, drift 0.000045), confirming
the fix touches only the experimental path:

| | before (run-time regime) | after (as-of regime) |
|---|---|---|
| baseline | 1.5871 | 1.5871 |
| experimental | 1.0783 | **1.2822** |
| Δ vs baseline | −0.5089 | **−0.3049** |
| Andrade suppressions | 0 (0/17 months) | 37 (13/17 months) |
| status | no verdict | **NO VERDICT YET** (see below) |

> **Status correction (Prompt 1C Task C, 2026-08-04).** An earlier draft of this
> section recorded "MINIMUM PASS (rule 2 only)". That label is emitted by
> `backtest_exit_layer.py`'s built-in verdict block, which encodes the **old** rule
> from `MarkovExit_Prompts.md` (beat REV 3's 1.0783). **That is not the rule
> governing this batch, and the label should not have been carried into this
> report.**
>
> This batch's pre-committed rule requires **all three** of:
> (a) negative mean `ret_full` given SELL; (b) the paired monthly difference
> significant at 0.05 by **both** sign test and paired t-test; (c) a
> redistribute-to-survivors variant that beats baseline on Sharpe or cuts max
> drawdown by ≥3 pp. Prompt 2 tests (a), Prompt 3 tests (c), Prompt 4 tests (b).
>
> **None of the three has been tested.** The correct status is therefore
> **NO VERDICT YET** — the lookahead was corrected and the experimental path
> improved from 1.0783 to 1.2822, but it remains **−0.3049 below baseline**.
> A losing configuration that loses less is not a passing one.

The old code applied a flat, staleness-degraded `multiplier = 1.0` to all 17 months, so
REV 4 fix B **never fired**. Point-in-time, 14/17 months are RISK_ON (1.2) and the gate
engages. **The −0.5089 that Prompts 2-4 were to be judged against was inflated by the
lookahead by ~0.20 Sharpe.** The exit layer still hurts, but by −0.3049, not −0.5089.

### 2. Price-vintage pin

- **New** `src/utils/data_vintage.py` — `price_vintage(db, start, end)` returns a sha256
  over the exact `(date, ticker, adj_close)` rows a backtest reads, plus row/ticker counts
  and span. A revised price changes the digest even at constant row count.
- Recorded by both `backtest_exit_layer.py` and `run_holdout.py` (log line + report
  header).

Current holdout vintage:
`sha256=0d685aeb29276f339d5c36a06a9dbe9db55d2da0a0b9b05e9fa9e147ca018b69`,
rows=20142, tickers=54, span 2025-01-02→2026-06-30.

### Tests

`tests/unit/test_regime_asof_and_vintage.py` — 10 tests, all green. Covers as-of/
historical agreement, live-dict shape compatibility, the no-data-after-`as_of` invariant,
a **regression guard that the signal is not constant across the window** (the signature of
the original bug), staleness→NEUTRAL, and vintage determinism / revised-price detection.

Full suite: **114 green, 3 failing** — all 3 failures verified pre-existing (identical set
on stashed pristine code): two `test_factors` assertions and one live-macro freshness
assertion (`vix` 24d stale in the DB).

---

### Reproduction

Guard: `basename $PWD` = `Ai Trading Agent`, `git rev-parse --show-toplevel` = same,
branch `prototype/markov-exit-layer`, model md5 `296e589f4da205eb1d171c2121d90f82`.
No `OSError [Errno 35]` encountered on any read.

Ladder script: `scratchpad/recon.py` (loaders mirror both harnesses verbatim;
`exit_baseline_monthly` reimplements `backtest_exit_layer.run_backtest` lines 168-261,
baseline leg only — verified to reproduce 1.5871 exactly).
