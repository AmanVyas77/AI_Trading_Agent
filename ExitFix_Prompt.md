# EXIT FIX — make `paper_runner` able to fully exit a fractional position

Sibling to `BreadthCap_Prompt.md`. Fixes the defect recorded in
`backtests/results/sprint8_results.json` under `live_gaps[0].exit_defect` and
listed in `known_limitations`. **Read the `exit_defect` entry first** — it
carries the measured evidence. This prompt specifies the fix; it does not
re-derive the diagnosis.

## Deadline — this is not a backlog item

**Must land before the October rebalance (2026-10-01).**

Every live book since 2026-07 has held fractional quantities, because notional
buys always produce them. Any month that rotates a name out hits this defect.
On 2026-09-02 it left **4.66pp of equity** ($3,931.74) in five names the
strategy had marked for complete exit, and recovery required manual
`close_position` calls outside `paper_runner`.

## Repo guard

```bash
cd ~/dev/"Ai Trading Agent"
git --no-optional-locks rev-parse --show-toplevel   # ends: Ai Trading Agent
git --no-optional-locks branch --show-current       # main
git --no-optional-locks status --short | grep -v '^??' || echo "(clean)"
```

## HARD RULES

- **Read-only git only** — `status`, `diff`, `log`, `show`, `rev-parse`,
  `branch --show-current`. No `add`, `commit`, `push`, `checkout`, `reset`,
  `stash`. Print every write for the operator to paste.
- **No `Co-Authored-By` trailer, ever.**
- `.venv` only. Never `/opt/anaconda3`. Never install from `requirements.txt` —
  only `requirements.lock.txt`. Assert `sys.executable` before running python.
- **Do not run `--execute`.** The operator places all orders. Build it, test it
  offline against a mocked broker, hand over the command.
- **No model changes.** `md5 -q models/ensemble_models.pkl` must still be
  `296e589f4da205eb1d171c2121d90f82` when you are done. This is an order-layer
  fix; it must not touch scoring, selection or weighting.

## The defect, precisely

Two independent bugs in `src/live/broker_alpaca.py::_submit_one` (line 151),
both of which **under-sell**:

1. **Notional overshoot → rejection.** The primary path submits a notional sell
   equal to the position's market value as read at query time. Any adverse move
   between the read and the submission makes that notional exceed the live
   position value and Alpaca rejects it with code `40310000`.
2. **The retry truncates and uses a stale price.** Line 184:
   `qty = max(1, int(notional / price))`, where `price` comes from
   `_latest_price` (line 202) — the most recent `adj_close` in the `prices`
   table. On 2026-09-02 that was the **2026-08-31** close: one session stale
   and, that day, higher than the live price for all five names. `int()` throws
   away the fraction; the stale-high price buys fewer shares per dollar.

Verified, not inferred — `int(target_notional / 2026-08-31 adj_close)`
reproduces all five realised fills exactly:

| ticker | notional | stale close | live avg | int() | filled |
|---|---:|---:|---:|---:|---:|
| ACLS | 15,326.40 | 115.33 | 108.38 | 132 | 132 |
| FORM | 16,740.11 | 100.61 | 93.62 | 166 | 166 |
| MRVL | 18,088.83 | 211.66 | 204.03 | 85 | 85 |
| ON | 15,510.17 | 74.09 | 71.90 | 209 | 209 |
| QCOM | 17,787.30 | 170.48 | 165.41 | 104 | 104 |

**Re-running `--execute` does not self-heal it.** Simulated: the stubs
($466.91–$1,126.20) clear the $210.92 do-not-trade band on pass 2, so orders are
generated — and `int()` truncates again. On pass 3 every name falls *below* the
band and becomes a permanent orphan, stranding **$495.18** across the five.
Repeated runs converge to an un-exitable remainder, not to flat.

## Required fix

### 1. Full exits must be quantity-based, not notional-based

This is the primary fix and it removes both bugs for the common case. When the
target weight is `0.0` — or more generally when the sell would close the entire
position — do not compute a notional at all. Submit the **exact fractional
quantity held**, read from the broker (`position.qty`), or call
`close_position`. Alpaca accepts fractional sell quantities against an existing
position, which is exactly why the manual recovery worked.

`diff_orders` in `src/live/rebalance.py` currently emits only
`{ticker, side, notional}`. Extend the order dict so a full exit is
distinguishable from a partial one — e.g. carry `qty` and/or an explicit
`close_position: True` flag — rather than having the broker layer infer intent
by comparing floats.

### 2. Partial sells must size off a live price, never the DB

If a notional sell is still used for partial reductions, size the retry from a
**live** price — the broker's own `position.current_price` or a quote — not
`_latest_price`'s DB `adj_close`. Keep `_latest_price` only as a last-resort
fallback, and **log loudly** when it is used, including the age of the row.

### 3. Never `int()`

Use the fractional quantity where the asset is fractionable. Fall back to whole
shares only for non-fractionable symbols, and when that happens record the
residual explicitly in the run record rather than discarding it silently.

### 4. Post-execute verification — the control that was missing

After fills, `paper_runner` must re-read positions and assert every name is
within the do-not-trade band of its target weight. If any is not, **raise
loudly** and write the discrepancy into the run record.

This is the point. The defect was not caught by the order layer; it was caught
by a human reading fill quantities afterwards. Nothing errored — five orders
came back `status=filled` and the run reported `n_submitted: 7`. **A silent
partial exit must become a loud failure.** This is the same failure mode as the
scaffold bug: a step reporting success while doing the wrong thing.

## Required tests

Add to `tests/unit/`. Mock the broker; do not hit the network.

1. **Full exit of a fractional position submits the exact held quantity.**
   Position `141.41288361` → order carries `qty == 141.41288361`, not a notional
   and not `132`.
2. **`int()` truncation is gone.** Given the five 2026-09-02 cases above, assert
   the sized quantity equals the full held quantity, not the historical
   truncated value. Use those real numbers as a regression fixture.
3. **Stale-price fallback is not used when a live price is available**, and when
   it is used it logs a warning carrying the row's age.
4. **Post-execute verification raises** when a simulated partial fill leaves a
   position outside the band, and passes when all names are within it.
5. **Non-fractionable fallback** records the residual in the run record instead
   of dropping it.
6. **No regression on the buy path.** Notional buys still fill to within $0.01:
   AMAT 50,621.08 → 50,621.07 and HPE 29,705.07 → 29,705.06 are the reference
   cases. The buy path was correct on 2026-09-02 and must not change.

## Regression check

```bash
source .venv/bin/activate
python -c "import sys; assert '/dev/Ai Trading Agent/.venv/' in sys.executable"
md5 -q models/ensemble_models.pkl   # 296e589f4da205eb1d171c2121d90f82
python -m pytest tests/unit -q
```

Known pre-existing failures as of 2026-09-02, to be reported separately from any
new ones: `test_factors.py` (`test_gross_profitability_formula`,
`test_known_values`) and `test_regime_gate_live.py::test_snapshot_covers_every_series`.
The last fails on a hardcoded CPI-month window (`assert cpi_date.month in (4,5,6)`,
now 7) — a stale *test*, not stale data. Out of scope; flag it.

A dry run must still produce the same targets for 2026-08: `n_selected=2`,
AMAT + HPE, `per_name_weight=0.6`. This fix touches order placement only.

## Secondary item — the run-record overwrite

Smaller, same code area, recorded under `live_gaps[0].record_keeping_hazard`.
Every `paper_runner` invocation writes the unsuffixed
`logs/live_runs/<month>_run.json`, **including dry runs**. The post-execute
idempotency dry run overwrote August's, so the canonical filename now reads
`mode=dry_run, n_orders=0` for a month that traded seven orders.

Either stop dry runs from writing the unsuffixed record, or have them write to a
distinct `_dryrun` name. An execute record must never be overwritten by a
subsequent dry run. The authoritative record for 2026-08 is
`logs/live_runs/2026-08_run_20260902T095341.json`.

## Finally

- Update `live_gaps[0].exit_defect.status` in
  `backtests/results/sprint8_results.json`, and update the `known_limitations`
  entry to say the defect is fixed and in which commit.
- Print the `git add` / `git commit` / `git push` block. **No trailer.**
