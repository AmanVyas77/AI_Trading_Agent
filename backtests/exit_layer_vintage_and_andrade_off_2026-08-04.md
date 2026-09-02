# Prompt 1C — Vintage freeze + Andrade-off sanity check

**Date:** 2026-08-04
**Branch:** `prototype/markov-exit-layer` · **git HEAD** `b5af09b68c060e4aa7fa3666ef52f769c1065973`
**Frozen model:** `models/ensemble_models.pkl` md5 `296e589f4da205eb1d171c2121d90f82` ✔
**Interpreter:** `/Users/aman/dev/Ai Trading Agent/.venv/bin/python` — numpy 2.4.4 / sklearn 1.8.0 / pandas 2.3.3 ✔

All guards green, including the new interpreter assertion (now also enforced in-process by
`_assert_interpreter()` at `scripts/backtest_exit_layer.py`, so no future run can silently
use anaconda base).

---

## HEADLINE: **STOP — Task B failed. Do not run Prompts 2-4.**

Task A succeeded: the vintage is frozen and reproduces bit-for-bit.
**Task B did not.** Zhang-only ≠ baseline, so the "0 independent Zhang triggers" premise is
wrong and the REV 3 / REV 4 trigger attribution needs redoing before any further prompt runs.

| Task | Result |
|------|--------|
| A — vintage freeze | ✅ frozen snapshot reproduces live to **0.0e+00** on all three paths |
| B — Andrade-off sanity check | ❌ **1.5915 ≠ 1.5871 baseline** (diff **+0.004398**, tolerance 1e-6) |
| C — verdict language | ✅ amended; status is **NO VERDICT YET** |

---

## Task A — Physical vintage freeze ✅

`backtests/vintage_2026-08-04/` — 639.5 MB, three files copied byte-for-byte:

| file | size | sha256 (first 16) |
|------|------|-------------------|
| `quant_research.db` | 638,754,816 | `3c6a7e0c62afb69b…` |
| `ensemble_feature_matrix.parquet` | 702,666 | `534169a30e4b6caf…` |
| `holdout_scores.parquet` | 4,942 | `cacc0e515899fc06…` |

*(Authoritative values are in `MANIFEST.json`; the table is a convenience copy.)*

`models/ensemble_models.pkl` is **referenced in place, not copied**, md5 asserted at freeze
time and re-asserted on every `--vintage` run. `config/settings.yaml` is likewise recorded
but not copied (git-tracked config, not data).

`MANIFEST.json` records for every file: absolute source path, size, sha256, mtime (UTC);
plus the `data_vintage` digest (`0d685aeb2927…`, 20,142 rows, 54 tickers,
2025-01-02→2026-06-30), git HEAD/branch/dirty, and the interpreter block.

**Wiring.** `scripts/backtest_exit_layer.py --vintage <dir>` repoints all inputs. It
sha256-verifies every frozen file against the manifest before use and aborts on mismatch.
It rebinds **both** `backtest_exit_layer.DB_PATH` *and* `regime_gate.DB_PATH` — the as-of
regime signal reads `macro_series` through its own module global, so repointing only the
first would have left the regime leg reading live data. The data source is logged on every
run and printed in the report header; without `--vintage` the run logs a **warning** that
it is reading revisable live data.

**Reproduction (Task A step 4)** — full precision, not the 4-dp report values:

| path | live | frozen | \|diff\| |
|------|------|--------|--------|
| baseline | 1.587144507439707 | 1.587144507439707 | **0.0e+00** |
| experimental | 1.282216655915358 | 1.282216655915358 | **0.0e+00** |
| hard_stop | 1.587144507439707 | 1.587144507439707 | **0.0e+00** |

All 17 monthly returns are bit-identical across the two runs, and the regenerated scores
match the frozen `holdout_scores.parquet` bit-for-bit (asserted in-run — this is what
proves the snapshot captures every input the scorer reads, not just the prices).

*Note:* the brief refers to `src/data_vintage.py`; the actual path is
`src/utils/data_vintage.py`.

---

## Task B — Andrade-off sanity check ❌ **STOP**

Zhang left as the only exit trigger (`--no-andrade`), run against the frozen vintage:

```
baseline                = 1.587144507439707
experimental (Zhang-only) = 1.591542820613977
diff                    = +0.004398313174270      (tolerance 1e-6 → FAIL)
```

**Zhang fires 12 independent triggers**, not 0. Seven months are affected:

| month | baseline | Zhang-only exp | diff |
|-------|----------|----------------|------|
| 2025-03-31 | +0.012421 | +0.010543 | −0.001878 |
| 2025-04-30 | +0.099825 | +0.088939 | −0.010887 |
| 2025-05-31 | +0.084663 | +0.076268 | −0.008396 |
| 2025-06-30 | +0.003855 | +0.007590 | +0.003735 |
| 2025-08-31 | +0.158111 | +0.149158 | −0.008953 |
| 2025-09-30 | +0.088824 | +0.095743 | +0.006919 |
| 2026-01-31 | −0.003971 | +0.008283 | +0.012254 |

### Where the "0 independent triggers" claim came from — and why it was misread

The **pre-fix** (REV 3/REV 4, run-time regime lookahead) attribution table reads:

```
TOTAL   zhang=0   andrade=46   both=12   none=167
```

`zhang = 0` is the *zhang-only* column — Zhang SELLs where Andrade did **not** also fire.
Zhang actually fired **12** times in that run too, all of them in the `both` column. With
the leaked flat regime (`multiplier = 1.0` everywhere → Andrade override always active),
Andrade co-fired on every name Zhang sold, so Zhang never appeared alone.

Post-fix, Andrade is suppressed in 13/17 months and the same 12 Zhang firings become
visible as `zhang=8, both=4`. **Zhang fires exactly 12 times in every configuration** —
`8 + 4 = 12` with Andrade on, `12` with Andrade off. Its decision is untouched by the
regime gate, which is the correct scoping of the as-of fix.

So the premise was a misreading of a column header, not a measurement. The third row of
the brief's monotonicity table ("all suppressed → 1.5871 = baseline") is now **measured
and false**: Zhang-only is **above** baseline, at 1.5915.

---

## Zhang Case I / Case II split — 225 calibrations

| regime | n | share |
|--------|---|-------|
| Case I (ρ ≤ f1) | 45 | 20.0 % |
| **Case II (ρ > f1)** | **0** | **0.0 %** |
| never-sell (Φ ≤ 0) | 180 | 80.0 % |

Andrade-derived state: 167 state-1 (uptick), 58 state-2 (downtick). Zhang can only SELL in
state 2.

**There is not a single Case II calibration on this window.** The daily hard-stop is
therefore *structurally* inert, not merely untriggered — `daily_hard_stop` returns
`no_action` for Case I and never-sell regimes before it ever evaluates a price. The
reported `hard_stop = baseline, 0 triggers` is a tautology on this window, and the whole
REV 4 "fix A" rationale (making x\*_0 a genuine crossing level) cannot be exercised here at
all.

### Distribution of x\* relative to p0

Over the 45 Case I calibrations (the only ones with a finite threshold):

```
n=45  min=0.001485  p05=0.001619  median=0.002367  p95=0.008608  max=0.02966
```

x\* sits at roughly **0.15 % – 3 % of p0** — about 24 cents on a $100 stock. The SELL test
is `price ≥ x*`, so:

- **45 / 45** Case I calibrations have `p0 ≥ x*`. The price test is satisfied *every time*.
- Of those, **12** are also in state 2 — and 12 is exactly the observed Zhang trigger count.

### The consequence — Zhang is not an independent signal

`Zhang SELL ⟺ (Case I) ∧ (state 2)`, verified exactly: `case1 ∧ state2 = 12` = Zhang
triggers with Andrade off = 12.

The price threshold contributes **nothing** — it is always breached, because
`x* ∝ K_absolute = 0.001 × entry_price` makes it inherently ~0.2 % of price. And `state` is
mapped directly from the Andrade DHMM forecast (`_ANDRADE_TO_STATE`, `exit_manager.py`).

**So `--no-andrade` does not produce an Andrade-free path.** It disables the STRONG_SELL
*override* while Zhang continues to consume the Andrade state mapping. The exit layer has
**no Andrade-independent decision surface at all** — both triggers are functions of the same
DHMM output, and Zhang's optimal-stopping mathematics is degenerate to a state passthrough
on this window. This is the same class of degeneracy the REV 4 docstring flagged for the
hard-stop under `K = K_fraction × current_price`; referencing K to `entry_price` fixed the
*algebraic* collapse but left the threshold numerically irrelevant.

---

## Task C — verdict language corrected ✅

`backtests/exit_layer_reconciliation_2026-07-30.md` recorded **"MINIMUM PASS (rule 2
only)"**. That label is emitted by `backtest_exit_layer.py`'s built-in verdict block, which
encodes the **old** rule from `MarkovExit_Prompts.md` (beat REV 3's 1.0783). It is not the
rule governing this batch and should not have been carried into the report.

This batch's pre-committed rule requires **all three** of: (a) negative mean `ret_full`
given SELL; (b) the paired monthly difference significant at 0.05 by **both** sign test and
paired t-test; (c) a redistribute-to-survivors variant that beats baseline on Sharpe or cuts
max drawdown by ≥3 pp. Prompt 2 tests (a), Prompt 3 tests (c), Prompt 4 tests (b).

**None of the three has been tested.** The report now records:

> **NO VERDICT YET** — lookahead corrected, experimental improved from 1.0783 to 1.2822,
> still −0.3049 vs baseline. A losing configuration that loses less is not a passing one.

The word "pass" no longer appears in that report except where it quotes the old label it is
correcting and this batch's three-part rule. (Test-suite wording changed to "green" /
"failing" to keep the sweep unambiguous.)

Note that `backtest_exit_layer.py` still *prints* the stale rule-2/rule-3 verdict block at
the end of every run. It is now actively misleading and should be either deleted or
repointed at the three-part rule — flagged, not changed, since it is out of scope here and
changing it mid-diagnostic would alter run artefacts.

---

## What this means for Prompts 2-4

Per the stop condition, they should not run. Beyond the literal trigger:

1. **The attribution is wrong.** Every REV 3 / REV 4 statement of the form "Zhang
   contributes nothing, Andrade is the whole effect" rests on the `zhang-only = 0` column.
   Zhang fires 12 times and, alone, is mildly *positive* (+0.0044 vs baseline).
2. **The monotonicity story collapses.** The suggested progression 1.0783 → 1.2822 → 1.5871
   is not "less Andrade is better" converging on baseline. The measured Zhang-only endpoint
   is 1.5915, *above* baseline — so the relationship is not monotone toward the baseline.
3. **Prompt 2 measures the wrong thing as specified.** It tests mean `ret_full` given SELL
   across all SELLs. With Zhang and Andrade both being functions of one DHMM state, that
   pools two non-independent triggers; the decomposition needs re-deriving first.
4. **The hard-stop path cannot be evaluated on this window at all** (0 Case II
   calibrations), so any Prompt 3/4 conclusion about it would be vacuous.

Suggested order before resuming: re-derive the trigger attribution against the frozen
vintage with the corrected columns; decide whether a threshold that is always breached
should count as a Zhang trigger at all; and re-scope the hard-stop question to a window (or
a K) where Case II actually occurs.

---

## Artefacts

- `backtests/vintage_2026-08-04/` + `MANIFEST.json` — frozen snapshot
- `scripts/freeze_vintage.py` — snapshot builder (interpreter + model-md5 guards, verifies
  each copy against its source sha256 to catch a concurrent backfill write)
- `scripts/backtest_exit_layer.py` — `--vintage DIR`, `--no-andrade`,
  `_assert_interpreter()`, Zhang case/threshold report section, data-source provenance in
  every report header
- `src/exit/exit_manager.py` — `monthly_exit_review(..., allow_andrade=True)`
- `backtests/exit_layer_reconciliation_2026-07-30.md` — amended per Task C

Nothing committed.
