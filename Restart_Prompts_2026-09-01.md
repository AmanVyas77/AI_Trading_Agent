# Restart batch — 2026-09-01

Two independent jobs. **Part A** closes the Markov branch (~10 min, pure git).
**Part B** restarts the paper book on the 2026-08-31 month-end (~1 hour, mostly
waiting on `refresh`).

Run everything from the repo root:

```bash
cd ~/dev/"Ai Trading Agent"
```

Verified against the repo on 2026-09-01. Facts this batch relies on:

- `models/ensemble_models.pkl` md5 = `296e589f4da205eb1d171c2121d90f82` ✔ unchanged
- `_last_completed_month_end(2026-09-01)` → **2026-08-31**
- `_resolve_month_end(None)` → **2026-08-31** — so `build_targets()` needs no `--as-of`
- `prices.max(date)` = 2026-07-10 (53 days stale), `cpi.max(date)` = 2026-05-01 (123 days, limit 75)
- Branch `prototype/markov-exit-layer`, HEAD `510916c`

---

## PART A — close out the Markov exit layer

The SHELVE verdict is written but untracked. Commit it so the branch stops
being an open question. **Copy-paste these; do not let an agent run them.**

```bash
git status --short
```

Expect: modified `.gitignore`, `MarkovExit_Diagnostics_Prompts_2026-07-30.md`,
`Sprint7_Prompts.md`, `backtests/exit_layer_reconciliation_2026-07-30.md`,
`memory/future_ideas.md`, `research_wiki/_index.md`,
`scripts/backtest_exit_layer.py`, `src/exit/exit_manager.py` — plus the
untracked reports.

```bash
git add scripts/backtest_exit_layer.py src/exit/exit_manager.py scripts/freeze_vintage.py
git add backtests/exit_layer_verdict_2026-08-19.md \
        backtests/exit_layer_verdict_2026-08-19_gen.py \
        backtests/exit_decision_analysis_2026-08-19.md \
        backtests/exit_decision_analysis_2026-08-19_gen.py \
        backtests/exit_layer_units_and_attribution_2026-08-04.md \
        backtests/exit_layer_units_and_attribution_2026-08-04_gen.py \
        backtests/exit_layer_units_and_attribution_2026-08-04_console.txt \
        backtests/exit_layer_vintage_and_andrade_off_2026-08-04.md \
        backtests/exit_layer_reconciliation_2026-07-30.md
git add MarkovExit_Diagnostics_Prompts_2026-07-30.md \
        News_Backfill_DataQuality_Fix_Prompt_2026-08-11.md \
        Sprint7_Prompts.md memory/future_ideas.md research_wiki/ .gitignore
```

**Do NOT add** `backtests/vintage_2026-08-04/` (639 MB), `data/`, `logs/`.
Confirm before committing:

```bash
git status --short | grep '^A' | wc -l          # count what is staged
git diff --cached --stat | tail -1              # sanity: no 600MB blob
```

```bash
git commit -m "Markov exit layer: SHELVE verdict — condition (a) fails, Andrade anti-predictive

Prompt 2 (2026-08-19): mean ret_full given SELL = +0.1617, hit rate 9/29.
Andrade-driven sells 2/17 correct, one-sided p=0.0012 — significantly
anti-predictive. Zhang leg break-even (12 sells, mean +0.0041) and
structurally inert: 45/45 Case I have p0>=x* at min margin 33.7x; Case II
0/225 and unreachable. Attribution of the original -0.5088 drag sums exact:
lookahead +0.2039, cash drag +0.2730, Andrade signal error +0.0320.
Across 500 perturbed worlds max experimental Sharpe +1.4057 < baseline
+1.5871. Re-entry gate closed (perfect foresight captures only 57.7% of
what break-even needs). Prompts 3-4 deliberately not run per the
pre-committed rule. Layer is not in production and never was."
```

```bash
git push origin prototype/markov-exit-layer
```

Optional, if you want the Sprint 9 news-pipeline fix on `main` too — memory
flags this as still pending:

```bash
git log --oneline main..prototype/markov-exit-layer | tail -5
# then, from main:  git cherry-pick 510916c
```

---

## PART B — restart the paper book on the 2026-08-31 book

### Why now, briefly

The book has not been touched since 2026-07-13. Two rebalances were missed
(2026-08-03, 2026-09-01). Over those 36 sessions the frozen book is
**−11.01% vs SPY +1.94%** — but that is an *unmanaged* book, so part of the
gap is the absence of the rebalance, not the strategy. Restarting is how you
find out which. Every skipped month is out-of-sample track record that cannot
be recreated later.

### B0 — interpreter assert (non-negotiable)

The anaconda-vs-venv trap has bitten this project before.

```bash
source .venv/bin/activate
python -c "import sys,numpy,sklearn,pandas,xgboost; \
print(sys.executable); \
print('numpy',numpy.__version__,'sklearn',sklearn.__version__, \
'pandas',pandas.__version__,'xgb',xgboost.__version__)"
md5 -q models/ensemble_models.pkl
```

**STOP unless** the path contains `/dev/Ai Trading Agent/.venv/`, versions are
numpy 2.4.4 / sklearn 1.8.0 / pandas 2.3.3 / xgboost 3.2.0, and the md5 is
`296e589f4da205eb1d171c2121d90f82`. If any differ, reinstall from
`requirements.lock.txt` — never `requirements.txt`.

### B1 — reconcile the broker BEFORE anything else

The June book has been held unmanaged for ~7 weeks. Read what Alpaca actually
holds and compare it to the run record before any order is sized.

```bash
python - <<'PY'
import json
from src.live.broker_alpaca import AlpacaPaperBroker
b = AlpacaPaperBroker()
acct = b.get_account()
pos  = b.get_positions()
print("base_url:", b.base_url)          # must be the paper endpoint
print("equity:  ", acct.get("equity"))
print("cash:    ", acct.get("cash"))
print(f"{'ticker':<8}{'qty':>10}{'mkt_value':>14}{'unreal_pl':>14}")
for p in sorted(pos, key=lambda x: x["symbol"]):
    print(f"{p['symbol']:<8}{float(p.get('qty',0)):>10.4f}"
          f"{float(p.get('market_value',0)):>14.2f}"
          f"{float(p.get('unrealized_pl',0)):>14.2f}")
print("open orders:", len(b.get_open_orders()))
PY
```

Expected book from `logs/live_runs/2026-06_run_20260712T191958.json`:
**ACLS, FORM, HPE, MRVL, ON, QCOM**, six buys at $20,000.00 notional each,
1.2× gross on $100k.

**Checklist — write the answers down, they go in the gap record at B5:**

1. Are exactly those six symbols still held? Any corporate action, split, or
   symbol change over 7 weeks?
2. Any stranded open orders from July? If yes, cancel them before B4 —
   `diff_orders` sizes against positions, not against pending orders.
3. Current equity vs the $96,216 recorded on 2026-07-13. The
   `spy_relative_scorecard` reconstruction (Stooq EOD) marks day-1 at $97,156,
   a 0.97% fill-slippage difference. Treat the broker read as truth and the
   reconstruction as an estimate — if they now disagree by much more than 1%,
   find out why before trading.

### B2 — refresh, standalone

**Run `refresh` on its own first.** `paper_runner` calls `run_refresh()`
internally, so if you skip this step every dry-run and every retry re-pays the
full ~45 minutes. Doing it once here lets B3/B4 use `--skip-refresh`.

```bash
python -m src.live.refresh --as-of 2026-09-01 2>&1 | tee logs/live_runs/refresh_$(date +%Y%m%dT%H%M%S).log
```

What to expect, and what is different from July:

- **`quant_pipeline` will actually run this time.** It was `skipped` on both
  July runs. Prices are 53 days stale, so this is a real download.
- **`sentiment` has ~7 weeks of 8-K backlog.** On Apple MPS this ran ~28 min
  for an 18-month backlog, so expect single-digit minutes here — but it is the
  second-longest pole.
- **`quant_factors` is the long pole**, ~26 min (1,585s measured 2026-07-13).
- **`backfill_cpi` matters this time.** CPI is at 2026-05-01, i.e. 123 days
  against a 75-day limit. The good news: `_assert_freshness()` runs *after*
  all steps, so the `backfill_cpi` step gets its chance to fix this before the
  assertion fires. If it still fails, run `python scripts/backfill_cpi.py`
  directly and check FRED actually has the July/August CPIAUCSL prints.
- Total ≈ 45 min.

`refresh` fails **loud** on staleness — that is the design, not a bug. If it
raises, read which series and re-run that step; do not pass `--skip`.

Confirm before moving on:

```bash
tail -1 logs/live_refresh_log.jsonl | python -m json.tool | head -40
```

Every source should be `"ok": true` and `month_end` should read `2026-08-31`.

### B3 — dry run

```bash
python -m src.live.paper_runner --as-of 2026-09-01 --skip-refresh
```

`--execute` is absent, so this is the safe default — it prints the book and
the orders and submits nothing.

**Read these before proceeding:**

- `month` must be **2026-08**. If it says 2026-06 or 2026-07, the feature
  matrix did not rebuild — go back to B2.
- `n_selected` — Sprint 7 found breadth structurally thinner out-of-sample
  (mean 14.9 names vs 24.6 in-sample; June-2026 gave 6). A small book is
  normal here, not a bug. **Zero names is not** — that is the MIN_SCORE=0.52
  gate leaving you in cash, and it needs a decision rather than a trade.
- `Regime: mult / rule_signal / stale`. **If `stale=True` the multiplier was
  forced to NEUTRAL** — that means a critical series (vix or spread) is stale
  and B2 did not fully succeed.
- Turnover. You are rolling a 7-week-old book straight to a new month, so
  expect more churn than a normal monthly step. The `DO_NOT_TRADE_BAND` is
  0.25% of equity, which filters noise, not genuine rotation.

### B4 — execute

Only after B3 reads clean. **US market hours** — Alpaca fractional orders are
market + DAY only, so a submission outside the session sits queued until the
next open.

```bash
python -m src.live.paper_runner --as-of 2026-09-01 --skip-refresh --execute
```

Then verify — the run is idempotent by design, so a second invocation should
submit **zero** new orders:

```bash
python -m src.live.paper_runner --as-of 2026-09-01 --skip-refresh --execute
```

Expect `n_submitted: 0` with everything in `skipped_already_open`. If it
submits a second set, stop and do not run it again.

Fills: Sprint 8's C4 criterion was 6/6 filled within ±0.5% of target notional
per name. Check the run record:

```bash
ls -t logs/live_runs/2026-08_run_*.json | head -1
python -m json.tool "$(ls -t logs/live_runs/2026-08_run_*.json | head -1)" | head -60
```

### B5 — record the gap (do not skip this)

A live track record with an *undocumented* hole is worse than one with a
documented hole. Add to `sprint8_results.json` a `live_gaps` entry:

- Months missed: **2026-07 and 2026-08** (scheduled 2026-08-03 rebalance never
  ran; 2026-09-01 was late by this restart).
- Cause: attention went to the Markov diagnostics (2026-07-30 → 08-19) and the
  news-backfill fix (2026-08-11). No code failure — nothing was invoked, so
  nothing errored.
- Book held unmanaged throughout: ACLS, FORM, HPE, MRVL, ON, QCOM.
- Measured over the gap: **−11.01% vs SPY +1.94%**, DD −22.97% (36 sessions,
  reconstruction from Stooq EOD — see `spy_relative_scorecard`).
- Broker state at restart: paste the B1 reconcile output.
- Decision taken: resumed on the 2026-08-31 book rather than waiting for
  2026-09-30, accepting a documented two-month hole.

Any future performance claim on the live book must cite this gap. It is not
a continuous track record and must not be presented as one.

---

## What this batch deliberately does NOT do

- **No model retrain, no feature change, no threshold tuning.** Production
  stays the Sprint 5 expanding-window ensemble on the frozen pickle. The book
  being down is not, by itself, evidence for changing the model — 36 sessions
  is far below the power needed to distinguish skill from noise.
- **No Markov code in the live path.** It is shelved. `src/live/scorer.py`
  routes entry only.
- **No Sprint 9 news feature.** The corpus is complete (439,720 articles) and
  wired to nothing; that is the next sprint, and it needs an SPY leg added to
  its verdict rule first.
