# Exit layer — Zhang units resolution + attribution redo (Prompt 1D)

Date run: 2026-08-08 · Branch: `prototype/markov-exit-layer` · Vintage: `backtests/vintage_2026-08-04`

**Fork: FORK 2 — the units are right.** The comparison `p0 ≥ x*` is dimensionally sound.
The price test is inert for an economic reason, not a dimensional one, and the same
degeneracy is present in the paper's own published market test.

---

## 0. Provenance and guards

Every number below carries a source tag. Nothing is transcribed from memory.

| Tag | Source |
|---|---|
| `[paper pN]` | `research_papers/quant_modeling/When to Sell Markov Chain Asset.pdf`, page N |
| `[rows]` | `backtests/exit_layer_units_and_attribution_2026-08-04_rows.csv` — 225 data rows, sha256 `2c1abaa8c93a52098012c082af7f9c074ec6f398940201372876e32fdc3dfe38` |
| `[gen]` | `backtests/exit_layer_units_and_attribution_2026-08-04_gen.py` — script that produced `[rows]`, sha256 `41ee682fa061bbbc808f6371a243f70645a622906a21c64def6c2d386a207abd` |
| `[logA:L]` | `logs/exit_backtest_20260808_112506.log` line L (Andrade ON) |
| `[logZ:L]` | `logs/exit_backtest_20260808_112602.log` line L (`--no-andrade`, Zhang-only) |
| `[src f:L]` | repo source file f, line L |
| `[con:L]` | `backtests/exit_layer_units_and_attribution_2026-08-04_console.txt` line L — captured run console. The `AUDIT:`/`VINTAGE:` guard lines are emitted by `logger`, **not** written into the `.log` report, so they are persisted here to be citable. |
| `[carried]` | quoted from an earlier report, not recomputed this session |

Guards, all verified before any number was produced:

- Interpreter: `/Users/aman/dev/Ai Trading Agent/.venv/bin/python`, numpy 2.4.4 / sklearn 1.8.0 / pandas 2.3.3 `[con:1]`, asserted by `_assert_interpreter()` `[src backtest_exit_layer.py:133]`. Not anaconda. Independently probed directly at the interpreter, same three versions.
- Vintage: `backtests/vintage_2026-08-04`, all 3 frozen files sha256-verified against `MANIFEST.json` `[con:2]`; `models/ensemble_models.pkl` md5 `296e589f4da205eb1d171c2121d90f82` matches `manifest.referenced.ensemble_models.pkl.md5` — asserted in `use_vintage` `[src backtest_exit_layer.py:196-204]` and confirmed by a direct `md5` of the file.
- `price_vintage` sha256 `0d685aeb29276f339d5c36a06a9dbe9db55d2da0a0b9b05e9fa9e147ca018b69`, git_head `b5af09b6` `[con:3]`; 20142 rows, 54 tickers, 2025-01-02→2026-06-30 `[logA:5-6]`.
- Regenerated holdout scores match the frozen copy **bit-for-bit** `[con:10]`, in all runs this session.
- Papers present at `research_papers/quant_modeling/`.

---

## TASK A — units of x*

### A.1 What the process X is

The paper's state variable is the **raw stock price**, not a normalised, log, or discounted
price. Defining equation, §2 Problem Formulation `[paper p4]`:

> `dS_t / S_t = f(α_t) dt,  S_0 = x ≥ 0,  t ≥ 0`

with `f(1) = f1 > 0` the uptick return rate and `f(2) = f2 < 0` the downtick rate `[paper p4]`.
The `x` that appears throughout as the threshold argument is exactly this `S_0 = x` — the
price level itself. There is no normalisation anywhere in the derivation: the HJB system
`[paper p6, eq. 1]` is written in `x` directly, and the value functions `v(x,1), v(x,2)`
are functions of the raw price.

### A.2 Units of K — absolute, not proportional

The objective `[paper p5]` is

> `J(x,i,τ) = E[ e^{-ρτ} (S_τ − K) I_{τ<∞} ]`

`K` is **subtracted from the price**. It therefore carries the same units as `S`, i.e.
currency per share, and it is an **absolute** quantity in the paper's formulation — it is
introduced as "the fixed transaction cost" `[paper p5]` and is a constant of the problem,
never scaled to the price level. The paper reinforces this on `[paper p6]`: "V(x,i) ≥ 0
implies x* ≥ K", a statement that only typechecks if `x*` and `K` share a numeraire.

### A.3 Does `p0 ≥ x*` compare like with like? — Yes.

Dimensional argument, explicit. Write `[P]` for the price dimension.

The Case I threshold `[paper eq. 9]`, as implemented at `[src src/exit/zhang_optimal.py:128]`:

```
x* = ((ρ + λ1 − f1) / (ρ + λ1)) · (K · β2 / (β2 − 1))
```

- `ρ, λ1, λ2, f1, f2` are all **rates**, dimension `[1/time]`. The factor
  `(ρ+λ1−f1)/(ρ+λ1)` is a ratio of rates → **dimensionless**.
- `β2` solves `f1·f2·β² − D1·β + D2 = 0` `[paper p7, eq. 4-5]` with `D1, D2` built only
  from rates `[src zhang_optimal.py:74-82]`. `β2` is a pure exponent → **dimensionless**,
  hence `β2/(β2−1)` is dimensionless.
- Therefore `[x*] = [K]`.

On the harness side, `K_absolute = K_fraction × entry_price` `[src src/exit/exit_manager.py:246]`
with `K_fraction = 0.001` `[src exit_manager.py:134]` and `entry_price` a dollar close.
So `[K_absolute] = [P]`, hence `[x*] = [P]`, and `p0` is a dollar close `[src scripts/backtest_exit_layer.py:349]`.
Both sides of `price >= x_star` `[src zhang_optimal.py:247]` are `[P]`.

**The comparison is dimensionally correct. The hypothesis that motivated Task A is refuted.**

Independent confirmation that the implementation is faithful to the paper: `zhang_optimal`
reproduces the paper's worked examples exactly.

| Quantity | Paper | Harness recomputation |
|---|---|---|
| Example 2, `(x*, x*_0)` | `(0.012478, 0.033333)` `[paper p19]` | `(0.012478, 0.033333)` |
| Table 1 AAPL 2H-2012, `x*` | `0.017213` `[paper p22]` | `0.017213` |

(Φ(0.03) for that row recomputes to `36.40` against the paper's tabulated `35.79` `[paper p22]`;
the table prints `f1, f2, λ1, λ2` rounded to 2 dp and Φ is a difference of two ~1.6×10⁴
quantities, so this is rounding in the table's inputs, not a discrepancy in the formula. `x*`,
which is far less sensitive, matches to all six published digits.)

### A.4 Time units of f, λ, ρ — all per annum; the harness agrees

The paper's rates are **annualised**. Evidence internal to the paper: `ρ = 0.03` is stated
as "the risk free rate" `[paper p21]`, which is an annual rate; Example 2 uses `ρ = 0.10`
with `λ1 = λ2 = 1` `[paper p19]`; and Table 1 `[paper p22]` reports `λ1 ∈ [97.98, 135.25]`,
`λ2 ∈ [107.95, 141.44]` fitted from **daily** AAPL closes over half-year windows — values
only interpretable per year (mean regime duration `1/λ ≈ 1/135 yr ≈ 1.9 trading days).
`f1 = 4.89` on that row is likewise ~489 % annualised drift, not a daily figure.

The harness calibration is in the same units. `estimate_parameters` `[src zhang_optimal.py:287]`
takes `dt = 1/252` and forms `T = n·dt` in **years** `[src zhang_optimal.py:330]`,
`mu = mean(Δz)/dt` `[src zhang_optimal.py:315]`, `σ_annual = σ0/√dt` `[src zhang_optimal.py:336]`,
so `λ1 = (R1 + R2/R)/T` and `f1 = μ + σ1` are per annum. `RHO = 0.03` `[src backtest_exit_layer.py:115]`
is annual, matching the paper. **Confirmed: `dt = 1/252` produces parameters in the paper's units.**

The fitted distributions land on top of the paper's AAPL row, which is the strongest
available check that units agree end to end (§B.3 below): our `λ1` median 123.5 vs the
paper's 97.98–135.25; our `f1` median 5.434 vs the paper's 1.80–10.45 `[rows]`, `[paper p22]`.

---

## FORK 2 — units are right; x* cannot bind for this asset class

### F.1 The structural result

`x*` is **exactly linear in K** — `β2` and the rate ratio do not depend on `K`, so
`x*(cK) = c·x*(K)`. Verified numerically to 12 dp across `c ∈ {1, 10, 10³, 10⁶}`.
Define the K-multiplier

```
M ≡ x* / K = ((ρ + λ1 − f1)/(ρ + λ1)) · β2/(β2 − 1)
```

`M` is a function of the fitted rates alone. Across the 45 Case I calibrations `[rows]`:

| | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| `M = x*/K` | 1.089 | 1.321 | **1.820** | 2.651 | 27.461 |

`M ≥ 1` in all 45 — the theorem `x* ≥ K` `[paper p6]` holds row by row.

With `K = 0.001 × entry_price`, `x*` is therefore pinned at roughly `M × 0.1 %` of the
price, i.e. **~0.2 % of price at the median**. Measured `[rows]`:

| `x*/p0` | min | p05 | median | p95 | max |
|---|---|---|---|---|---|
| | 0.001485 | 0.001619 | **0.002367** | 0.008608 | 0.02966 |

`p0 ≥ x*` in **45/45** `[rows]`, `[logA:167]`. This reproduces Prompt 1C exactly
(median 0.0024, max 0.030, 45/45) `[carried]`.

The margin is not marginal. Expressed the other way round, `p0/x*` `[rows]`:

| `p0/x*` | min | p05 | median | p95 | max |
|---|---|---|---|---|---|
| | **33.7** | 116.4 | 422.5 | 618.0 | 673.5 |

The *tightest* case on the entire book has the price exceeding the sell threshold by a
factor of 34. The price test is not close to binding anywhere.

### F.2 What K would have to be for x* to bind

Binding requires `x* > p0` at least sometimes:

```
x* > p0
⇔ M · K_fraction · entry_price > p0
⇔ K_fraction > (p0 / entry_price) / M
```

Evaluated per Case I calibration `[rows]`:

| required `K_fraction` | min | p05 | median | p95 | max |
|---|---|---|---|---|---|
| | **0.0337 (3.4 %)** | 0.1164 (11.6 %) | **0.4225 (42.2 %)** | 0.6180 (61.8 %) | 0.6735 (67.4 %) |

Current `K_fraction = 0.001` (0.1 %) `[src exit_manager.py:134]`. The median required value
is **422× larger**. Even the single most favourable calibration on the book needs a 3.4 %
round-trip transaction cost before the price test does anything at all, and to make the
test bind *generally* you need transaction costs of roughly **42 % of the share price**.

**Is that economically defensible? No.** US equity round-trip costs — commission plus
spread plus slippage — are basis points, not tens of percent. A 42 % cost assumption is
three orders of magnitude outside anything this strategy could face, and at that level the
strategy has no positive expectancy to optimise in the first place. There is no admissible
`K` that makes the Zhang price test informative for exchange-traded equities at these
price levels.

### F.3 The paper's own market test is degenerate the same way

This is the decisive evidence that the defect is not in the harness. In the paper's only
empirical test `[paper p21-22]`: AAPL daily closes, `ρ = 0.03`, `K = 0.01`, and for the
selected 2H-2012 row `x* = 0.017213` `[paper p22]`. The paper then states the resulting
decision `[paper p22]`:

> one should sell as soon as α_t turns to 2 after the new year

and executes it at a close of **$542.10/share** `[paper p22]`. That is a price exceeding
the threshold by a factor of **31,494**. To make `x*` reach $542.10 on that row you would
need `K = 542.10/M = $314.94`, i.e. **58.1 % of the share price**.

So in the paper's own worked market test the price condition is vacuous and the author's
stated rule collapses to *"sell when the chain enters state 2"* — precisely the reduction
Prompt 1C measured in the harness (`Zhang SELL ⟺ Case I ∧ state 2`, 12 = 12 = 12) `[carried]`.
The harness is a faithful implementation of a rule that is degenerate as published, for
any asset whose price is large relative to a fixed per-share transaction cost.

### F.4 Case II never appears, and cannot

Case II requires `ρ > f1`, i.e. `f1 < 0.03` per annum. Observed `f1` minimum across all
225 calibrations is **2.592** `[rows]` — 86× the threshold. Case II count: **0/225**
`[rows]`, `[logA:156-159]`. Because `f1 = μ + σ1` and `σ1` is an annualised volatility
term scaled by `√(λ1(λ1+λ2)/(2λ2))` `[src zhang_optimal.py:337-339]` with `λ ~ O(100)`,
`f1` is dominated by the volatility term and is structurally of order units-to-tens per
annum. `f1 < 0.03` would require a near-zero-volatility asset.

**Consequence: the daily hard-stop remains structurally inert.** `daily_hard_stop` returns
`no_action` whenever `ρ ≤ f1` `[src exit_manager.py:408]`, which is every row.
Hard-stop triggers: **0/17 months** `[logA:129-130]`. Correcting units would not have
changed this — there were no units to correct. **REV 4's fix A is still not testable**, and
it cannot be made testable by any choice of `K`, because the Case I / Case II branch does
not depend on `K` at all.

### F.5 Recommendation

Per the FORK 2 rule: the required `K` is not economically defensible, so the recommendation
is **removal rather than tuning** — but the removal must be surgical, because "the Zhang leg"
is not one thing. Decomposed:

1. **The price test (`p0 ≥ x*`) is inert and should be removed.** It is True on 45/45
   Case I rows with a minimum margin of 34× `[rows]`. It has zero discriminating power: it
   never separates one (ticker, month) from another on this window, and §F.2 shows no
   admissible `K` changes that.
2. **The state test is not Zhang's** — it is the Andrade DHMM action mapped through
   `_ANDRADE_TO_STATE` `[src exit_manager.py:137-142]`. Removing Zhang does not remove it.
3. **The Case I / Φ gate is *not* inert and cannot be silently dropped.** It blocks 46 of
   the 58 state-2 rows from selling `[rows]`. Deleting the Zhang leg wholesale would convert
   the exit rule into "sell on Andrade state 2", which fires 58 times instead of 12 — a
   materially different strategy, not a simplification.

That third point deserves a flag, because the gate that survives is doing its work on a
knife edge. `Φ(ρ) = (ρ+λ1−f1)(ρ+λ2−f2) − λ1λ2` is a difference of two quantities of median
magnitude ~1.65×10⁴, and the result has median `|Φ| = 86.8` — **0.52 % of the terms being
differenced** `[rows]`. Φ ranges over `[−521.6, +195.8]`, and **169 of 225** calibrations sit
within 1 % of the sign flip `[rows]`. The one component of the Zhang leg that changes
decisions is a catastrophic-cancellation sign test on two fitted rates. Whether it carries
signal or is a noise-driven coin flip is not settled by this prompt — but it is the thing
to test, not the price threshold.

---

## TASK B — attribution on pivotality

### B.1 Method

A condition counts as a trigger only if **flipping it changes the decision**. For each of the
225 (ticker, month) decisions `[rows]`, the final monthly SELL was decomposed into primitives
and each was forced True and forced False with all others held fixed `[gen]`:

```
SELL = (Φ>0 ∧ Case I ∧ state==2 ∧ price≥x*) ∨ (Andrade STRONG_SELL ∧ regime permits)
```

`price_test_pivotal`, `state_test_pivotal`, `case_gate_pivotal` are each
`decide(cond=True) ≠ decide(cond=False)`.

Two guards on the method:

- **Self-check.** The composer was re-evaluated at the *observed* conditions and asserted
  equal to the real `monthly_exit_review` output on **all 225 rows** `[gen]`; a mismatch
  aborts. The counterfactual table cannot drift from the production decision path silently.
- **State flip is applied to the Zhang leg only** in the headline figure, holding the Andrade
  override fixed, so the two paths through which the Andrade signal enters are not confounded.
  A coupled variant (uptick also forces `STRONG_SELL` off) is reported alongside.

Case-gate note: `Φ>0 ⟺ Case I` exactly on this window (45 = 45, Case II = 0) `[rows]`, so the
Φ gate and the Case I gate are the same partition here and are reported as one flag.

### B.2 Trigger table on pivotality flags

| month | n | SELLs | price pivotal | state pivotal (Zhang leg) | state pivotal (coupled) | case gate pivotal |
|---|---:|---:|---:|---:|---:|---:|
| 2025-01-31 | 4 | 0 | 0 | 0 | 0 | 0 |
| 2025-02-28 | 3 | 0 | 0 | 1 | 1 | 0 |
| 2025-03-31 | 7 | 3 | 0 | 2 | 5 | 0 |
| 2025-04-30 | 15 | 9 | 0 | 2 | 11 | 0 |
| 2025-05-31 | 20 | 1 | 1 | 4 | 4 | 1 |
| 2025-06-30 | 20 | 1 | 1 | 8 | 8 | 1 |
| 2025-07-31 | 5 | 0 | 0 | 1 | 1 | 0 |
| 2025-08-31 | 11 | 2 | 2 | 3 | 3 | 2 |
| 2025-09-30 | 16 | 3 | 3 | 3 | 3 | 3 |
| 2025-10-31 | 12 | 0 | 0 | 0 | 0 | 0 |
| 2025-11-30 | 10 | 0 | 0 | 1 | 1 | 0 |
| 2025-12-31 | 17 | 0 | 0 | 3 | 3 | 0 |
| 2026-01-31 | 17 | 1 | 1 | 3 | 3 | 1 |
| 2026-02-28 | 14 | 0 | 0 | 1 | 1 | 0 |
| 2026-03-31 | 20 | 9 | 0 | 8 | 17 | 0 |
| 2026-04-30 | 20 | 0 | 0 | 1 | 1 | 0 |
| 2026-05-31 | 14 | 0 | 0 | 0 | 0 | 0 |
| **TOTAL** | **225** | **29** | **8** | **41** | **62** | **8** |

Source: `[rows]`. Only three joint patterns occur across all 225 rows `[rows]`:

| price / state / case pivotal | count | meaning |
|---|---:|---|
| F / F / F | 184 | no single condition is decisive (196 total non-SELL minus the 33 below, plus the 21 Andrade-driven SELLs) |
| F / T / F | 33 | state-2 would create a SELL that does not exist; price and case gate not decisive |
| T / T / T | 8 | the Zhang-only SELLs — all three conditions decisive together |

### B.3 Reading it honestly — pivotality ≠ discriminating power

`price_test_pivotal = 8` must not be read as "the price test did work in 8 cases." Those 8
are exactly the `trigger_source == "zhang"` rows of the old taxonomy `[logA:45-61]`; in each,
the price condition is the last condition standing, so flipping it flips the outcome. But the
condition is **True on 45/45 Case I rows** with a minimum margin of 34× (§F.1). A condition
that never varies is pivotal wherever it is unopposed and yet carries **zero information** —
it cannot distinguish a good exit from a bad one. This is the precise sense in which the old
taxonomy over-credited Zhang, and the pivotality test alone does not fully repair it: the
margin distribution has to be read next to it. Both are reported above.

The other two counts are informative:

- **`case_gate_pivotal = 8`** is identical to the price count, and for a different reason: the
  gate is decisive only on the 8 rows where the Zhang path is the sole SELL source. On the
  other 37 Case I rows it is either blocked by state 1 (33 rows) or made redundant by a
  concurrent Andrade STRONG_SELL (4 rows, `trigger_source == "both"` `[logA:45-61]`).
- **`state_test_pivotal = 41`** = the 8 Zhang-only SELLs plus 33 rows where forcing state 2
  would manufacture a SELL that did not occur. Under the coupled flip this rises to **62**,
  the difference being the 21 Andrade-driven SELLs (17 `andrade` + 4 `both` `[logA:45-61]`)
  where the Andrade signal is decisive through the override path rather than the Zhang path.

Old taxonomy for comparison `[logA:44-62]`: zhang 8, andrade 17, both 4, none 196. Total
Zhang firings 8 + 4 = 12, matching Prompt 1C `[carried]`.

**Net:** on 225 decisions the Zhang leg contributes 8 pivotal SELLs, every one of which turns
on a knife-edge Φ sign test (§F.5) and none of which turns on a price threshold that ever
varied.

### B.4 Fitted parameter distributions (all 225 calibrations, per annum)

| param | min | p05 | p25 | median | p75 | p95 | max |
|---|---|---|---|---|---|---|---|
| `f1` | 2.592 | 3.119 | 4.142 | **5.434** | 6.716 | 8.041 | 10.68 |
| `f2` | −11.81 | −7.851 | −6.446 | **−5.537** | −4.415 | −3.276 | −2.531 |
| `λ1` | 92.47 | 104.3 | 114.9 | **123.5** | 130.2 | 141.6 | 149.9 |
| `λ2` | 113.6 | 118.6 | 129.7 | **135.6** | 142.9 | 153.2 | 163.3 |

Source: `[rows]`.

Implied mean regime duration `[rows]`:

| | median | min | max |
|---|---|---|---|
| `1/λ1` | 0.0080956 yr = **2.04 trading days** (2.95 calendar days) | 1.68 td | 2.73 td |
| `1/λ2` | 0.0073729 yr = **1.86 trading days** (2.69 calendar days) | 1.54 td | 2.22 td |

At the median `1/λ1`, a 250-day calibration window `[src backtest_exit_layer.py:120]`
contains ≈ **122 regime transitions** `[rows]`.

These sit directly on top of the paper's AAPL Table 1 range (`λ1` 97.98–135.25, `λ2`
107.95–141.44, `f1` 1.80–10.45, `f2` −1.85 to −10.61) `[paper p22]` — which is both the
unit-agreement check promised in A.4 and confirmation that the ~3-day regime the paper's
row implies is what this calibration produces too.

Per the prompt, reported only — not analysed here.

---

## TASK C — stale verdict block

### C.1 Print-only? Verified yes, two independent ways.

**Static.** An AST walk of `write_report` (lines 489–700 of the pre-edit file) found
**no** subscript assignments, **no** attribute assignments, and **no** `global` statements
anywhere in the function; the only calls with side effects are `lines.append`,
`LOG_DIR.mkdir`, `path.write_text`, and `print`. Every other call is a read-only pandas or
builtin accessor. The verdict locals `rule2`, `rule3`, `verdict`, `base_drift`, `exp_r4`
appear **only** inside `lines.append(...)` f-strings; `REV3_SHARPE` `[src backtest_exit_layer.py:123]`
is read nowhere outside `write_report`. `write_report` is called once, at
`[src backtest_exit_layer.py:734]`, *after* `run_backtest` has fully returned, and its
return value is used only for a log message.

**Empirical.** Diffing the full report produced before the edit against the one produced
after, on identical inputs: the only differences are the generation timestamp (line 3) and
the replaced block itself (lines 182+). **All 181 preceding lines of computed output are
byte-identical.**

### C.2 Replacement

Confirmed print-only, so the block was replaced. It previously emitted
`FULL PASS / MINIMUM PASS / FAIL` by comparing this run's experimental Sharpe against the
REV 3 numbers — a verdict on the layer that the batch's actual KEEP/SHELVE rule does not
authorise this script to reach. It now restates this batch's three-part rule and explicitly
declines to adjudicate:

```
  STANDING DECISION RULE — diagnostic batch 2026-07-30
  KEEP the layer only if ALL THREE hold:
    (a) SELL-flagged positions have a NEGATIVE mean ret_full
    (b) the paired monthly difference is significant at 0.05 by
        BOTH sign test and paired t-test, in the layer's favour
    (c) the redistribute-to-survivors variant still beats baseline
        on Sharpe OR cuts max drawdown by ≥ 3 percentage points
  SHELVE if (a) fails.  INCONCLUSIVE (→ shelve, noted) if (a) holds
  but (b) fails.

  NOT EVALUATED HERE — this script measures none of (a), (b), (c).
  Sharpe alone does not decide the layer. Prompt 2 tests (a).
```

None of (a)/(b)/(c) is computable from what `run_backtest` returns — (a) needs per-SELL
`ret_full`, (b) needs the paired monthly series under a sign test and a paired t-test,
(c) needs the redistribute-to-survivors variant and a drawdown series — so the block
restates the rule and reports only the integrity check it can actually run (baseline drift
vs REV 3, `0.000045`, OK `[logA:182]`).

**Run artefacts are unaffected: the block has no data path.** Confirmed statically and by
the byte-identical diff above.

---

## Reference Sharpe figures (recomputed this session, not carried)

| path | Andrade ON | Zhang-only (`--no-andrade`) |
|---|---|---|
| baseline | +1.5871 `[logA:14]` | +1.5871 `[logZ:14]` |
| experimental | +1.2822 `[logA:15]` | +1.5915 `[logZ:15]` |
| hard_stop | +1.5871 `[logA:16]` | +1.5871 `[logZ:16]` |

Zhang-only vs baseline: **+0.0044** `[logZ:15]`, reproducing Prompt 1C `[carried]`. Per 1C's
standing reading this is noise — Lo SE on a Sharpe of 1.59 at T=17 is ≈ ±0.88, on 12 sells —
and the correct statement remains "Zhang-only ≈ baseline", nothing stronger `[carried]`.
These are reported for continuity; **no verdict on the layer is drawn in this prompt.**

---

## Changes made

- `scripts/backtest_exit_layer.py` — verdict block in `write_report` replaced (Task C).
  This is the only source edit. Not committed.
- Written (untracked, not committed): this report, `[rows]`, `[gen]`, the console capture
  `[con]`, and four run logs under `logs/`.

Ordering note: `[logA]` and `[logZ]` were produced **before** the Task C edit and are the
source for every Sharpe and attribution figure quoted here. `[con]` was captured **after**
it, purely to persist the guard lines. Per the C.1 diff, the two report bodies are
byte-identical apart from the timestamp and the verdict block itself, so the citations do
not straddle a change in any computed value.

Test status: `tests/unit/test_zhang_optimal.py` + `tests/unit/test_exit_manager.py` —
**70 passed**. The full suite has 6 pre-existing failures in unrelated modules
(`test_factors.py`, `test_pipeline.py` derived-features, and `test_regime_gate_live.py`
asserting VIX freshness ≤ 14d against a DB that is 29d stale under the frozen vintage).
No test references `write_report`; the edit is not implicated in any of them.

## Open items handed forward

1. The Φ sign test is the only Zhang component that changes decisions, and it is a
   0.5 %-of-magnitude difference of two ~1.6×10⁴ terms with 169/225 rows within 1 % of the
   flip. Its stability under resampling is untested.
2. Removing the Zhang leg is not a no-op: it would take the Φ/Case-I gate with it and
   triple the SELL count from 12 to 58. Any removal needs to specify what replaces the gate.
3. Regimes fit at ~2 trading days (~122 transitions per 250-day window) — the identifiability
   question queued for the follow-on. Distributions reported in B.4; not analysed here.
