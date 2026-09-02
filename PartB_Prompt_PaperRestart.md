# PART B — Restart the paper-trading book on the 2026-08-31 month-end

You are working in the AI Trading Agent repo. This task **touches a live
(paper) brokerage account**. Read the whole file before running anything.

## Repo guard — run FIRST, stop if it fails

```bash
cd ~/dev/"Ai Trading Agent"
git rev-parse --show-toplevel      # must end in: Ai Trading Agent
```

## Context

Sprint 8 put a 6-name book live on 2026-07-13: **ACLS, FORM, HPE, MRVL, ON,
QCOM**, 20% each, 1.2× gross, $100k Alpaca paper. The runbook is monthly:
month-end scores → refresh → dry-run → execute.

**It has not run since.** Two rebalances were missed (2026-08-03 and
2026-09-01). `data/live/` holds only `targets_2026-06.json`;
`logs/live_refresh_log.jsonl` ends at `as_of: 2026-07-14`; `prices.max(date)`
is 2026-07-10. Nothing errored, because nothing was invoked.

Over that gap the frozen book ran **−11.01% vs SPY +1.94%** (36 sessions,
DD −22.97%). It is an *unmanaged* book, so part of that is the missing
rebalance rather than the strategy — which is the reason to restart rather
than the reason not to.

**Decision already made: rebalance now on the 2026-08-31 book**, and document
July/August as a gap. Do not re-litigate that.

## Standing constraints

- **Interpreter.** `.venv` only. Never `/opt/anaconda3`. Never install from
  `requirements.txt` (its `>=` pins pull pandas 3.x / sklearn 1.9 and break the
  frozen pickle) — only `requirements.lock.txt`.
- **No model changes.** Production is the Sprint 5 expanding-window ensemble on
  the frozen pickle. Do not retrain, do not touch `FEATURE_COLS`, do not adjust
  `MIN_SCORE` or `TOP_N`. A book that is down is not evidence for changing the
  model — 36 sessions is far below the power needed to separate skill from noise.
- **No Markov code.** That layer was shelved 2026-08-19. Entry routing only.
- **You do not run `--execute`.** You run everything up to and including the
  dry-run, then hand me the execute command and wait. I place the orders.
- **You do not run git commit.** Print commands; I run them.

## Step 0 — environment assertion

```bash
source .venv/bin/activate
python -c "import sys,numpy,sklearn,pandas,xgboost; print(sys.executable); print('numpy',numpy.__version__,'sklearn',sklearn.__version__,'pandas',pandas.__version__,'xgb',xgboost.__version__)"
md5 -q models/ensemble_models.pkl
```

**STOP unless all four hold:** path contains `/dev/Ai Trading Agent/.venv/`;
versions are numpy 2.4.4 / sklearn 1.8.0 / pandas 2.3.3 / xgboost 3.2.0; md5 is
`296e589f4da205eb1d171c2121d90f82`; and the pickle loads without an
`InconsistentVersionWarning`.

## Step 1 — reconcile the broker BEFORE sizing anything

The June book has been held unmanaged for ~7 weeks.

```bash
python - <<'PY'
from src.live.broker_alpaca import AlpacaPaperBroker
b = AlpacaPaperBroker()
acct, pos = b.get_account(), b.get_positions()
print("base_url:", b.base_url)
print("equity:", acct.get("equity"), " cash:", acct.get("cash"))
print(f"{'ticker':<8}{'qty':>10}{'mkt_value':>14}{'unreal_pl':>14}")
for p in sorted(pos, key=lambda x: x["symbol"]):
    print(f"{p['symbol']:<8}{float(p.get('qty',0)):>10.4f}"
          f"{float(p.get('market_value',0)):>14.2f}"
          f"{float(p.get('unrealized_pl',0)):>14.2f}")
print("open orders:", len(b.get_open_orders()))
PY
```

Compare against `logs/live_runs/2026-06_run_20260712T191958.json` (6 buys, each
exactly $20,000.00 notional; equity was $96,216 on 2026-07-13).

**Report to me, and STOP if any of these is true:**

1. `base_url` is not the paper endpoint. (It should be impossible —
   `broker_alpaca.py` asserts `BaseURL.TRADING_PAPER` — but check the printed
   value anyway.)
2. Held symbols are not exactly those six, or a split / symbol change happened.
3. **There are open orders.** `diff_orders` sizes against *positions*, not
   pending orders, so a stranded July order would double the position. These
   must be cancelled before Step 3.
4. Equity disagrees with the reconstruction by much more than ~1%. (The
   `spy_relative_scorecard` reconstruction marks day-1 at $97,156 vs the
   broker's $96,216 — 0.97% fill slippage, which is the known baseline.)

## Step 2 — refresh, standalone, once

**Run this on its own. Do not let `paper_runner` do it.** `paper_runner` calls
`run_refresh()` internally, so if you skip this every dry-run and every retry
re-pays the full ~45 minutes. Running it here lets Steps 3 and 4 use
`--skip-refresh`.

```bash
python -m src.live.refresh 2>&1 | tee "logs/live_runs/refresh_$(date +%Y%m%dT%H%M%S).log"
```

`--as-of` is omitted deliberately: it defaults to today, and
`_last_completed_month_end` resolves that to **2026-08-31**, which is the book
we want.

Budget ~45 min. Set a long timeout; do not kill it. What differs from July:

- **`quant_pipeline` actually runs this time** — it was `skipped` on both July
  runs, and prices are ~7.5 weeks stale. Real download.
- **`sentiment` has a ~7-week 8-K backlog.** Runs on Apple MPS.
- **`quant_factors` is the long pole**, ~26 min (1,585s measured 2026-07-13).
- **CPI needs the backfill step.** It sits at 2026-05-01, i.e. ~123 days against
  a 75-day limit. This resolves itself: `_assert_freshness()` runs *after* all
  steps, so `backfill_cpi` gets its chance first. If the assertion still fires
  on cpi, run `python scripts/backfill_cpi.py` directly and check FRED actually
  has the July/August CPIAUCSL prints before doing anything else.

`refresh` raises loudly on staleness by design. **Do not pass `--skip` to work
around a failure** — read which series failed and fix that step.

Then confirm:

```bash
tail -1 logs/live_refresh_log.jsonl | python -m json.tool | head -40
```

Every source `"ok": true`, and `month_end` reads `2026-08-31`.

## Step 3 — dry run

```bash
python -m src.live.paper_runner --skip-refresh
```

No `--execute`, so this submits nothing.

**Read out and interpret these four for me:**

- **`month` must be `2026-08`.** If it says 2026-06 or 2026-07, the feature
  matrix did not rebuild — go back to Step 2.
- **`n_selected`.** Sprint 7 found breadth structurally thinner out-of-sample
  (mean 14.9 names vs 24.6 in-sample; June-2026 gave 6). A small book is normal
  here. **Zero is not** — that means the MIN_SCORE=0.52 gate left us in cash,
  which is a decision for me, not a trade.
- **`Regime: mult / rule_signal / stale`.** If `stale=True`, the multiplier was
  forced to NEUTRAL because a critical series (vix or spread) is stale — meaning
  Step 2 did not fully succeed. Go back.
- **Turnover.** We are rolling a 7-week-old book to a new month, so expect more
  churn than a normal monthly step. `DO_NOT_TRADE_BAND` is 0.25% of equity; it
  filters noise, not genuine rotation. Tell me the names entering and leaving.

## Step 4 — hand me the execute command, then stop

Do not run it. Print it, along with the follow-up idempotency check:

```bash
python -m src.live.paper_runner --skip-refresh --execute
```

Note for me in your handoff: it needs **US market hours** (Alpaca fractional is
market + DAY only, so an out-of-session submission queues to the next open), and
the second invocation must report `n_submitted: 0` with everything in
`skipped_already_open`. If it submits a second set, that is a bug — stop.

Sprint 8's fill criterion was 6/6 filled within ±0.5% of target notional per
name. After I run it, help me verify against
`logs/live_runs/2026-08_run_*.json`.

## Step 5 — the gap record

Draft (do not commit) a `live_gaps` entry for `sprint8_results.json`:

- Months missed: 2026-07 and 2026-08. The scheduled 2026-08-03 rebalance never
  ran; 2026-09-01 was late.
- Cause: attention went to the Markov diagnostics (2026-07-30 → 08-19) and the
  news-backfill fix (2026-08-11). No code failure — nothing was invoked.
- Book held unmanaged throughout: ACLS, FORM, HPE, MRVL, ON, QCOM.
- Measured over the gap: −11.01% vs SPY +1.94%, DD −22.97%, 36 sessions
  (reconstruction from Stooq EOD, not a broker read).
- Broker state at restart: the Step 1 output.
- Decision: resumed on the 2026-08-31 book rather than waiting for 2026-09-30,
  accepting a documented two-month hole.

Add a line stating that any future performance claim on this book must cite the
gap — it is not a continuous track record and must not be presented as one.

Then print the `git add` / `git commit` commands for me to run.
