# BREADTH CAP — implement the pre-committed n=1 per-name weight cap

Implements the rule recorded in `backtests/results/sprint8_results.json` under
`breadth_rules`, decided 2026-09-02 blind to any affected ticker. **Read that
entry first** — it carries the evidence, the rejected options and the honest
statement of what the cap costs. This prompt implements it; it does not
re-litigate it.

## Deadline — this is not a backlog item

**Must land before the October rebalance (2026-10-01.)**

Breadth above `MIN_SCORE = 0.52` has been collapsing: Dec 18 → Apr 24 → Jun 3 →
Jul 9 → **Aug 2**. The live book for 2026-08 was `n_selected = 2`, and AMAT/HPE
cleared the gate by 0.0012. **n=1 is plausibly the next month, not a distant
tail.** Until this lands, `rebalance.py` is unbounded and an n=1 month puts
100% (NEUTRAL) or 120% (RISK_ON) of equity in a single name.

## Repo guard

```bash
cd ~/dev/"Ai Trading Agent"
git rev-parse --show-toplevel      # must end: Ai Trading Agent
git branch --show-current          # main
git status --short | grep -v '^??' || echo "(clean)"
```

## HARD RULES

- **Read-only git only** — `status`, `diff`, `log`, `show`, `rev-parse`,
  `branch --show-current`. No `add`, `commit`, `push`, `checkout`, `reset`,
  `stash`. Print every write for the operator to paste.
- **No `Co-Authored-By` trailer, ever.** Five commits were `filter-branch`ed in
  July to strip them.
- `.venv` only. Never `/opt/anaconda3`. Never install from `requirements.txt` —
  only `requirements.lock.txt`. Assert `sys.executable` before running python.
- **No model changes.** Do not retrain, do not touch `FEATURE_COLS`,
  `MIN_SCORE` or `TOP_N`. `md5 -q models/ensemble_models.pkl` must still be
  `296e589f4da205eb1d171c2121d90f82` when you are done.
- This is not a strategy change. Option D (rank-based selection) is **not**
  adopted and must not enter the live path.

## The rule

| n_selected | per-name weight |
|---|---|
| 0 | `0.0` — all cash. Unchanged. |
| 1 | `min(multiplier/1, CAP)`, remainder to cash |
| ≥2 | `min(multiplier/n, CAP)` — inactive in every observed case |

`CAP = 0.60`. The cap is **post-multiplier**. It binds at NEUTRAL (1.0 → 0.60)
and RISK_ON (1.2 → 0.60), and deliberately does **not** bind at RISK_OFF
(0.5 → 0.5, unclipped).

## THE TRAP — read this before writing any code

The two code paths apply the regime multiplier at **different stages**:

- `src/live/rebalance.py:146` — `scaled_weight = regime["multiplier"] / n_selected`.
  The multiplier is already applied here. Capping in place is correct.
- `src/strategies/ensemble/portfolio_builder.py:155` — `weight = 1.0 / n_selected`.
  The multiplier is **not** applied until ~line 296,
  `daily_weights.multiply(daily_mults, axis=0)`.

So `min(1.0/n, CAP)` at line 155 caps the **pre-multiplier** weight and
implements a *different rule*. It is wrong in exactly the two n=1 cases the cap
exists for:

| regime | n | correct (post-mult) | naive cap at line 155 | |
|---|---|---|---|---|
| RISK_ON | 1 | **0.60** | 0.72 | fails to bound the case the rule exists for |
| RISK_OFF | 1 | **0.50** | 0.30 | wrongly clips the case meant to pass through |
| RISK_ON | 2 | 0.60 | 0.60 | ok |
| NEUTRAL | 1 | 0.60 | 0.60 | ok |

In `portfolio_builder.py` the cap must therefore be applied **after** the
multiplier — clip `daily_weights` once the regime scaling is done, not at the
`1.0/n_selected` step.

## Implementation

1. **One shared constant, not two literals.** Define it once — the natural home
   is `portfolio_builder.py` alongside `MIN_SCORE` / `TOP_N`, which
   `rebalance.py` already imports from there (`rebalance.py:45`):

   ```python
   MAX_PER_NAME_WEIGHT = 0.60   # post-multiplier cap; see sprint8 breadth_rules
   ```

   Import it into `rebalance.py` on the existing import line. Do not restate
   `0.60` anywhere else, tests included — reference the constant.

2. **Live path** — `rebalance.py:146`, cap in place (multiplier already applied).
3. **Backtest path** — `portfolio_builder.py`, cap after the regime scaling.
4. Surface the cap in the persisted targets payload so a capped month is
   auditable after the fact: record whether the cap bound, and the pre-cap
   weight, next to `per_name_weight`.

## Required tests

Add to `tests/unit/`. Reference `MAX_PER_NAME_WEIGHT`, never the literal.

1. **n=2 × RISK_ON is bit-identical before and after.** Assert against the
   **uncapped computation**, not the literal `0.60`:

   ```python
   assert min(RISK_ON_MULT/2, MAX_PER_NAME_WEIGHT) == RISK_ON_MULT/2
   ```

   Rationale: on CPython `1.2/2` and `0.60` happen to round to the same double
   (`0x1.3333333333333p-1`), so a literal assertion passes today — but by
   coincidence, and it would break silently if `RISK_ON_MULT` or the cap were
   retuned. The uncapped-computation form tests the invariant that matters:
   **the cap must not perturb the n=2 book.** This is what proves the live
   2026-08 AMAT/HPE 60/60 book and its successor months are untouched.
2. **n=1 × RISK_ON → 0.60.**
3. **n=1 × NEUTRAL → 0.60.**
4. **n=1 × RISK_OFF → 0.50, unclipped.** Assert it equals `RISK_OFF_MULT/1`, so
   the test states the intent (pass-through) rather than a magic number.
5. **n=0 → 0.0**, unchanged.
6. **Both paths agree.** Parametrise over
   `{RISK_OFF, NEUTRAL, RISK_ON} × n ∈ {1,2,3,20}` and assert the live and
   backtest paths produce the same per-name weight. This is the test that would
   have caught the pre/post-multiplier trap above.

## Regression check

A holdout re-run on the frozen vintage must be **unchanged**:

```bash
source .venv/bin/activate
python -c "import sys; assert '/dev/Ai Trading Agent/.venv/' in sys.executable"
md5 -q models/ensemble_models.pkl   # 296e589f4da205eb1d171c2121d90f82
```

Score the holdout from `backtests/vintage_2026-08-04/ensemble_feature_matrix.parquet`
with `folds[-1]` and confirm it still reproduces
`backtests/vintage_2026-08-04/holdout_scores.parquet` bit-identically
(`max|diff| == 0.0`, n=954). Then rebuild the holdout equity curve and confirm
Sharpe / max-DD / total return are **identical, not merely close** — the frozen
vintage's holdout minimum breadth is **3**, so the cap must be a provable no-op
there. Any movement at all means the cap is binding where it must not.

Also run the suite and report new failures separately from the known ones:

```bash
python -m pytest tests/unit -q
```

Known pre-existing failures as of 2026-09-02: `test_factors.py`
(`test_gross_profitability_formula`, `test_known_values`) and
`test_regime_gate_live.py::test_snapshot_covers_every_series`. The last one
fails on a hardcoded CPI-month window (`assert cpi_date.month in (4,5,6)`, now
7) — it is stale *test*, not stale data, and it will keep failing every month
until rewritten as a relative-age check. Fixing it is out of scope here; flag it.

## Finally

- Update `breadth_rules.implemented` to `true` in
  `backtests/results/sprint8_results.json`, and drop the `deadline` note.
- Print the `git add` / `git commit` / `git push` block. **No trailer.**
