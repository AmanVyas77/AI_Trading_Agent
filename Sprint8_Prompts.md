# Sprint 8 — Option B: Live Paper-Trading Pipeline (Alpaca paper account)
Copy-paste each prompt into a fresh Cowork/coding-agent session, one at a time, in order.
Each prompt is self-contained.

Sprint 8 turns the validated Sprint 5 ensemble into a running monthly paper-trading
loop: data refresh → frozen-model scoring → weights (regime-gated) → order diff →
paper orders on Alpaca → logs. NO new alpha logic anywhere — this sprint is pure
infrastructure. The frozen model (Sprint 7 holdout PASS, Sharpe 1.016 on spent
2025-26 window) is used as-is.

VERDICT RULE — OPERATIONAL, pre-committed (this sprint ships plumbing, not returns;
the "performance" of paper trading is judged over months, not at commit time):
Sprint 8 = PASS only if ALL SIX hold in one end-to-end dress rehearsal (Prompt 4):
  (1) Refresh completes with every feed within staleness limits (prices ≤ 3 trading
      days; VIX/spread ≤ 3 days; FinBERT ≤ 14 days; cpi ≤ 75 days — amended from 45
      on 2026-07-10 pre-rehearsal: CPI's reference-month dating means 40-75d age is
      normal BLS cadence; see Prompt 4 PRE-FLIGHT FIX 2).
  (2) Scorer emits current-month scores with the 23-feature assert passing and a
      logged audit line (model effective_train_end 2024-07-31).
  (3) Weights respect MIN_SCORE=0.52 / TOP_N=20 / regime multiplier (log proves it).
  (4) Orders submitted to the Alpaca PAPER account and filled; resulting positions
      match target weights within ±0.5% of portfolio value per name.
  (5) Idempotency: immediately re-running the rebalance submits ZERO new orders.
  (6) A complete run record persists (SQLite log + JSON snapshot) sufficient to
      reconstruct the run without the session transcript.
Any failure → FAIL: fix-forward in a follow-up session, but the verdict is recorded.

Verified context (treat as fact; verified 2026-07-04/05):
- ⚠️ HASHES REWRITTEN: git history was rewritten on 2026-07-05 (filter-branch, run
  by Aman natively) to strip Co-Authored-By trailers — ALL previously quoted commit
  hashes (af3d449, f858e10, 13a2b6c, 5c367c6, 65bc3bb) are STALE. Identify commits
  by MESSAGE, never by old hash. Expect HEAD's message to be the Sprint 7 PASS
  commit (or later).
- ⚠️ SANDBOX GIT HAZARD: bulk git operations on this repo from inside a Cowork
  sandbox mount can SIGBUS/deadlock. Read-only git (log/status/ls-files/diff) is
  fine; anything heavier (rebase, filter-branch, gc) must be handed to Aman as
  commands. Normal add/commit is NOT done by agents anyway (see Prompt 5 policy).
- TWO NESTED REPOS: authoritative = INNER repo rooted at "Ai Trading Agent/";
  verify git rev-parse --show-toplevel ends "Ai Trading Agent" before ANY git
  command; NEVER touch the outer Stock_Project/ repo. No force-push, ever.
- COMMIT POLICY (standing, from Aman): agents NEVER run git commit/push in this
  project. Prompt 5 verifies the tree and prints copy-paste commands for Aman.
  Commit messages are plain -m with NO Co-Authored-By / Generated-with trailer.
- CRITICAL RULE: never edit config/settings.yaml or .env. Credentials: the agent
  NEVER types, pastes, or handles API keys/secrets. If Alpaca keys are missing
  from .env, STOP and ask Aman to create the paper account and add
  ALPACA_API_KEY / ALPACA_SECRET_KEY to .env himself, then resume.
- Sprint 7 inheritance (all committed and working):
  * src/live/scorer.py — frozen-model scorer: score_months(start, end); reindexes
    to the pickle's 23 feature_names, fillna(0.0), asserts shape+order.
  * scripts/run_holdout.py — the INJECTION pattern: build_portfolio_weights(
    scores_df=…, prices=…) bypasses every TIMELINE-clamped loader. Reuse this
    pattern; do not patch portfolio_builder/backtest.
  * scripts/backfill_cpi.py — CPIAUCSL upsert (cpi is NOT in settings.yaml's FRED
    list; this script is the ongoing fix; ~1-month publication lag is normal).
  * sentiment_pipeline now takes --date-start/--date-end; module default DATE_END
    is still "2024-12-31" — fixing that default IS authorized in this sprint.
  * Refresh runtimes (measured): prices+FRED ~40s; FinBERT weekly increment ~2min
    on Apple MPS (auto-detected); full TimesFM factor rebuild ~5min.
- Live regime gate KNOWN DEFECTS to fix this sprint (from the Phase 5 code audit):
  * regime_gate._load_macro_snapshot (~line 78) does ORDER BY date DESC LIMIT 7 on
    long-format macro_series → monthly CPI is structurally never in the snapshot.
  * No staleness guard — a months-old VIX silently produces a "current" signal.
  * get_live_regime_signal's LLM arm defaults to Ollama — FORBIDDEN on this
    machine (kernel panics). Paper stage runs RULES-ONLY by default; the LLM arm
    may remain as optional code behind an env check but must never be required.
- Universe: 54 tech tickers; production book: top-20 above 0.52, equal weight,
  monthly rebalance at month-end scores (trade the first session after month-end).
- Machine: 8 GB Mac, .venv/bin/python for everything. Broker: Alpaca paper
  (alpaca-py SDK) — paper endpoint https://paper-api.alpaca.markets. Fractional
  orders where needed; whole-share rounding acceptable if fractional unsupported
  for a name (tolerance in rule (4) allows it).

---

## PROMPT 1 of 5 — Recon (read-only)

```
You are picking up the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. This is Sprint 8
(live paper-trading pipeline), Prompt 1 of 5. READ-ONLY — no edits, no
installs, no orders.

REPO GUARD: git rev-parse --show-toplevel must end "Ai Trading Agent".
NOTE: history was rewritten 2026-07-05 (trailer strip) — identify
commits by MESSAGE. Confirm HEAD's message chain contains "Sprint 7:
2025-26 true holdout PASS…" and record the NEW hashes for Sprint 7,
"Dashboard v2…", "Sprint 6…", "Catch-up…", "Sprint 5…". List working-
tree state (Sprint*_Prompts.md modifications expected; data/ untracked
expected).

Use .venv/bin/python throughout.

1. Inheritance check: read src/live/scorer.py and scripts/run_holdout.py
   in full. Record scorer's public API + audit line, and exactly how
   run_holdout injects scores/prices into
   portfolio_builder.build_portfolio_weights (the pattern Prompt 3
   reuses for the live path).

2. Credentials + SDK (do NOT read secret values):
   - .env: report whether keys named ALPACA_API_KEY / ALPACA_SECRET_KEY
     (or similar) EXIST — check names only, e.g. grep -o '^[A-Z_]*' .env.
     If absent: flag prominently — Prompt 4 is blocked until Aman adds
     them himself. Do not ask for the values; do not echo .env contents.
   - .venv/bin/pip show alpaca-py — installed? Version? (Install happens
     in Prompt 4, not now.)

3. Regime gate live-path audit: read regime_gate.py lines ~54-95 and
   ~158-220. Quote: the LIMIT 7 snapshot query, the blanket except that
   returns {}, the absence of any date/staleness check, the LLM default
   backend resolution in llm_client.py (~line 70), and the
   disagree→NEUTRAL blend. These are Prompt 2's fix targets — record
   exact line numbers.

4. Data freshness deltas since Sprint 7 (2026-07-04 baseline): max
   dates for prices, vix, yield_spread_10y2y, cpi, sentiment_scores.
   Estimate what a monthly refresh must touch (expected: prices+FRED
   ~40s, FinBERT increment ~2min MPS, backfill_cpi, TimesFM factor
   rebuild ~5min, feature_matrix rebuild).

5. Rebalance timing logic: confirm from portfolio_builder how month-end
   scores map to tradable dates (weights ffill from rebalance date;
   live equivalent = trade first session after month-end). Note the
   current month-end the dress rehearsal will use.

6. Alpaca capability check (docs knowledge only, no API calls):
   fractional shares on paper accounts, market vs limit for a monthly
   rebalance of ~20 liquid tech names, and the positions/orders
   endpoints the diff logic needs.

Output: new-hash table, inheritance summary, credentials/SDK status,
regime-gate defect quotes with line numbers, freshness table, rebalance
timing note. State whether Prompt 4 is blocked on credentials. State
you're ready for Prompt 2.
```

---

## PROMPT 2 of 5 — Ops hardening (regime gate live path + sentiment default)

```
You are continuing Sprint 8 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. Prompt 2 of 5.
REPO GUARD first (inner repo; commits identified by message — hashes
were rewritten 2026-07-05). Use .venv/bin/python.
CRITICAL RULES: never edit config/settings.yaml or .env; never handle
credential values; Ollama must never run on this machine.

Three authorized, tightly-scoped edits:

EDIT 1 — regime_gate._load_macro_snapshot (fix the LIMIT-7 bug):
Replace the ORDER BY date DESC LIMIT 7 query with a per-series latest
read (e.g. GROUP BY series_name with MAX(date), or a window query) so
EVERY series — including monthly cpi and industrial_production —
contributes its own most recent value. Return, alongside each value,
its as-of date: change the snapshot to
{series: {"value": float, "date": "YYYY-MM-DD"}} internally, keeping a
values-only view for backward compatibility with the prompt builder.

EDIT 2 — staleness guard in get_live_regime_signal:
Module-level constants (NOT yaml):
  MAX_STALENESS_DAYS = {"vix": 5, "yield_spread_10y2y": 5,
                        "fed_funds_rate": 10, "cpi": 60}
If a series is missing or older than its limit: log a WARNING naming
the series and its age, and set a "stale": true flag + "stale_series"
list in the returned dict. A stale vix or yield_spread forces the
returned multiplier to NEUTRAL_MULT with reasoning "stale macro —
defaulting to neutral" (never silently RISK_ON/OFF on old data).

EDIT 3 — rules-only default for the live signal + sentiment default:
- get_live_regime_signal(use_llm: bool = False): the LLM arm runs ONLY
  when use_llm=True AND the resolved backend is NOT ollama (hard-block
  ollama with a clear error naming the kernel-panic history). Default
  call path is rules-only: llm_signal="SKIPPED", blend = rule signal.
  Keep the LLM code intact for the future analyst work.
- sentiment_pipeline: change the module DATE_END default from
  "2024-12-31" to today's date at import time (datetime.now strftime),
  so a live run without --date-end can no longer silently clamp to 2024.
  DATE_START default stays "2015-01-01" (callers pass --date-start for
  increments).

TESTS (offline-safe, no network beyond the local DB):
1. Snapshot test: call the new _load_macro_snapshot; assert cpi present
   with date ≈ 2026-05, vix/spread dates within days of the DB max.
2. Staleness test: monkeypatch the snapshot to return a 90-day-old vix;
   assert multiplier == 1.0 and stale flag set.
3. Rules-only test: get_live_regime_signal() with no LLM configured
   returns a dict with llm_signal == "SKIPPED" and a valid multiplier;
   assert no ollama import is attempted (e.g. sys.modules check).
4. Sentiment default test: import module, assert DATE_END == today.
Run all four; paste results. git diff --stat must show exactly:
regime_gate.py, sentiment_pipeline.py (+ a tests/ file if you add one —
allowed). settings.yaml must show NO diff.

Output: the diffs, the four test results, and the exact live-signal
dict shape Prompt 3 will consume. State you're ready for Prompt 3.
```

---

## PROMPT 3 of 5 — src/live/refresh.py + src/live/rebalance.py (dry-run, no broker)

```
You are continuing Sprint 8 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. Prompt 3 of 5.
Prompt 2 hardened the live regime gate (per-series snapshot, staleness
guard, rules-only default) and fixed the sentiment DATE_END default.
REPO GUARD first. Use .venv/bin/python. No broker calls in this prompt.
CRITICAL RULES: no settings.yaml/.env edits; injection over patching
(the run_holdout.py pattern); no edits to portfolio_builder/backtest/
model_trainer/score_generator.

PART A — src/live/refresh.py (the monthly data runbook, one command):
def run_refresh(as_of: str | None = None) -> dict   (+ CLI)
Steps, in order, each timed and logged:
  1. quant_pipeline (prices + FRED)            (~40s)
  2. scripts/backfill_cpi.py logic (import or subprocess)
  3. sentiment pipeline increment: --date-start = (max(filing_date) in
     sentiment_scores minus 7 days, for overlap safety; idempotent via
     INSERT OR REPLACE) --date-end = today     (~2min MPS)
  4. factor_export_quant --start 2015-01-01 --end <last completed
     month-end>                                 (~5min, TimesFM inside)
  5. feature_matrix --start 2015-01-01 --end <same>
  6. Staleness assertions (the rule-(1) limits from the sprint header):
     prices ≤ 3 trading days old; vix/spread ≤ 3 days; FinBERT ≤ 14
     days; cpi ≤ 45 days. Any violation → raise with a named-series
     error (fail LOUD — this is the guard against silent-stale trading).
Returns a freshness report dict; also appended as JSON to
logs/live_refresh_log.jsonl.

PART B — src/live/rebalance.py (target book construction, dry-run):
def build_targets(as_of_month_end: str | None = None) -> dict:
  1. scorer.score_months(month_end, month_end) → current scores
     (23-feature assert inherited from scorer).
  2. Selection: MIN_SCORE=0.52, TOP_N=20, equal weight — import the
     constants from portfolio_builder (single source of truth), apply
     the same nlargest logic (mirror _select_monthly_holdings; cite the
     lines you mirrored in a comment).
  3. Live regime multiplier via get_live_regime_signal() (rules-only
     default from Prompt 2) — scale weights; log signal dict verbatim.
  4. Output dict: {as_of, regime: {...}, n_selected, targets:
     [{ticker, score, weight}...], cash_weight} where cash_weight =
     1 - sum(weights) (regime < 1.0 or thin breadth leaves cash).
  5. Persist to data/live/targets_<YYYY-MM>.json AND a new SQLite table
     live_targets (month PK, generated_at, json payload) — append-only;
     regeneration for the same month INSERT OR REPLACEs and logs that
     it did so.
def diff_orders(targets: dict, current_positions: list[dict],
                equity: float) -> list[dict]:
  Pure function (broker-agnostic): positions [{ticker, qty, market_value}]
  + equity → orders [{ticker, side, notional_or_qty}] that move the book
  to target weights, with a DO-NOT-TRADE band: skip any order under
  0.25% of equity (keeps idempotency and avoids dust churn).

PART C — Dry-run (no broker):
1. run_refresh() end-to-end; paste the freshness report.
2. build_targets() for the current completed month-end; paste the
   targets summary (n_selected, top 5 by weight, regime line, cash%).
3. diff_orders() against a MOCK current book: (a) empty book at
   $100,000 equity → expect ~n_selected buy orders; (b) the book that
   (a) would produce → expect ZERO orders (idempotency at the pure-
   function level).
4. Unit-check: sum(target weights) + cash_weight == 1.0 ± 1e-9;
   no weight > 1/TOP_N × RISK_ON_MULT + 1e-9.

git diff --stat: new files only (src/live/refresh.py, rebalance.py,
tests if added) + logs/data dirs. Output: freshness report, targets
summary, both mock diffs, unit-check results. State you're ready for
Prompt 4 (broker + dress rehearsal) and whether credentials were
confirmed present in Prompt 1.
```

---

## PROMPT 4 of 5 — Broker adapter + dress rehearsal + operational verdict

```
You are continuing Sprint 8 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. Prompt 4 of 5.
Prompts 2-3 delivered the hardened gate, refresh runbook, target
builder, and a pure order-diff function (mock-tested idempotent).
REPO GUARD first. Use .venv/bin/python.

CREDENTIALS GATE (before anything else): confirm ALPACA_API_KEY and
ALPACA_SECRET_KEY exist in .env BY NAME ONLY (never print values). If
missing: STOP — output instructions for Aman (create Alpaca account →
Paper Trading section → generate keys → add the two lines to .env) and
end the session. NEVER ask him to paste keys into chat.

PRE-FLIGHT FIXES (from the Prompt 3 review — do BOTH before Part A;
they correct spec errors in the original prompts, not implementation
errors):

FIX 1 — rebalance.py weighting must match the VALIDATED strategy.
Prompt 3's invariant ("weight ≤ 1/TOP_N × RISK_ON_MULT") wrongly forced
slot-based 1/TOP_N sizing. The backtested + holdout-validated strategy
(portfolio_builder.py:154-156) weights 1/n_selected among chosen names,
then scales by the regime multiplier — it CONCENTRATES when breadth is
thin (validated behavior; e.g. 2 names → 60% each under RISK_ON).
Change build_targets to: weight = multiplier / n_selected per name.
New invariants: |Σweights − multiplier| ≤ 1e-9; max_weight ==
multiplier/n_selected; cash_weight = 1 − Σweights MAY BE NEGATIVE under
RISK_ON (1.2 gross — Alpaca accounts have 2× margin; log gross exposure
and confirm buying_power covers it before submitting). Regenerate
data/live/targets_2026-06.json + the live_targets row, and re-run the
two mock diffs from Prompt 3 Part C.3 (empty book → n_selected buys at
multiplier/n_selected × equity notional; round-trip → 0 orders).

FIX 2 — CPI staleness limits were mis-calibrated vs the BLS calendar.
CPI rows are dated by reference-month start; the latest print's age
oscillates ~40-75 days in NORMAL operation (May data, dated 05-01,
releases mid-June). Set the cpi limit to 75 days in BOTH
refresh.py's assertions AND regime_gate.MAX_STALENESS_DAYS (was 45/60).
This amends operational criterion (1) in the sprint header to
"cpi ≤ 75 days" — a calendar calibration recorded here before the
dress rehearsal; no performance metric is involved. Re-run run_refresh()
and confirm it now completes with all feeds ✓.

PART A — SDK + adapter:
1. .venv/bin/pip install alpaca-py  (project venv, not system).
2. src/live/broker_alpaca.py — thin adapter, paper endpoint HARDCODED
   (https://paper-api.alpaca.markets — a module constant; there must be
   no code path to a live-money endpoint):
     get_account() -> {equity, cash, buying_power}
     get_positions() -> [{ticker, qty, market_value}]
     submit_orders(orders) -> submits MARKET DAY orders serially
       (fractional/notional where supported; whole-share fallback);
       returns order ids; logs each to a new SQLite table live_orders
       (order_id PK, month, ticker, side, qty/notional, status,
       submitted_at).
     poll_fills(order_ids, timeout) -> final statuses.
   Adapter refuses to run if the resolved base URL is not the paper
   endpooint constant (assert at init).
3. src/live/paper_runner.py — the one-command loop:
     run_refresh() → build_targets() → get_positions()/get_account() →
     diff_orders() → [--dry-run: print orders and stop] → submit →
     poll fills → write run record: logs/live_runs/<YYYY-MM>_run.json
     (freshness report, targets, signal dict, orders, fills, final
     positions vs targets deltas) + append to a live_runs SQLite table.
   Flags: --dry-run (default true!), --execute for the real submission.

PART B — Dress rehearsal (paper account, real orders; market hours
matter — if closed, orders queue as DAY: run during a session or use
notional market orders at next open; note which happened):
1. paper_runner --dry-run → paste the order list.
2. paper_runner --execute → submit, poll fills, paste final
   positions-vs-targets table with per-name weight deltas.
3. Immediately re-run paper_runner --execute → assert ZERO new orders
   submitted (live idempotency, rule (5)).
4. Kill-switch sanity: assert the adapter's endpoint constant is the
   paper URL and account.equity is paper money (~$100k default).

PART C — Verdict against the six pre-committed operational criteria
(sprint header). Evaluate each with evidence (log lines / table refs),
then write backtests/results/sprint8_results.json:
  {generated_at, sprint: "Sprint 8 — live paper-trading pipeline",
   criteria: {c1_freshness: PASS/FAIL + evidence, … c6_run_record: …},
   verdict, verdict_notes (3-6 sentences), first_live_month,
   monthly_runbook: ["…exact commands in order…"],
   known_limitations: [DATE_END default now dynamic; LLM arm disabled;
   whole-share rounding; anything found]}

NO code/model changes in response to the verdict. State the verdict and
that you're ready for Prompt 5 (commit — commands go to Aman).
```

---

## PROMPT 5 of 5 — Prepare commit (COMMANDS FOR AMAN — agent does NOT commit)

```
You are finishing Sprint 8 of the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. Prompt 5 of 5.
Prompt 4 wrote backtests/results/sprint8_results.json with an
operational verdict.

STANDING POLICY (Aman, 2026-07-05): you do NOT run git commit or git
push in this project — ever. You verify the tree, then print exact
copy-paste commands for Aman to run in his own Terminal. Commit
messages are plain -m with NO Co-Authored-By, Generated-with, or any
other trailer.

REPO GUARD: inner repo (toplevel ends "Ai Trading Agent"); identify
commits by message (hashes were rewritten 2026-07-05); read-only git
only (status/diff/log — bulk git operations from the sandbox can
SIGBUS on this mount).

1. Update memory/phase_progress.md: append "## Sprint 8 — live paper
   trading pipeline" in the established format (what was built, the six
   criteria table with evidence, verdict verbatim, first_live_month,
   the monthly runbook commands).
2. Update memory/future_ideas.md: convert the Option B entry to an
   outcome log (LIVE or FAILED-DRESS-REHEARSAL); the analyst-funnel
   END-GOAL entry becomes the next open item; note the shadow-stage
   prerequisite list is unchanged.
3. Verify with git status --short + git diff --stat that the change set
   is exactly: src/live/{refresh,rebalance,broker_alpaca,paper_runner}.py,
   regime_gate.py + sentiment_pipeline.py (Prompt 2), any tests/ files,
   backtests/results/sprint8_results.json, both memory files, and
   (untracked, DO NOT STAGE) logs/, data/live/. settings.yaml and .env
   must show NO diff — if either does, STOP and report.
4. Print for Aman — exactly this structure, with the real file list:

   cd "/Users/aman/Desktop/Stock_Project/Ai Trading Agent"
   git status
   git add <each Sprint 8 file, explicitly — no 'git add .'>
   git commit -m "Sprint 8: live paper-trading pipeline — <PASS/FAIL> dress rehearsal (sprint8_results.json)"
   git push origin main

   Plus the reminder lines: run from the inner folder; expect a
   fast-forward; if the push is rejected, stop and bring the error to
   the knowledge-base chat; do not force-push.

Output: the memory diffs, the verification result, and the command
block. Do not execute the commands.
```

---

## Usage notes

- Prompt 1 tells you immediately whether Prompt 4 is blocked on Alpaca keys —
  if so, create the paper account and add the two .env lines yourself between
  Prompts 3 and 4 (never paste keys into a chat).
- Prompt 4's dress rehearsal is market-hours sensitive: fills confirm fastest
  during a US session; off-hours the orders queue as DAY orders — either is
  acceptable, the prompt records which.
- After Sprint 8 passes, the monthly cadence is: run paper_runner --dry-run,
  eyeball the order list, then --execute — first trading day after each
  month-end. The run records under logs/live_runs/ are the raw material for
  the analyst-funnel shadow comparison later.
- Feed results back to the knowledge-base chat after Prompts 1, 3, and 4.
