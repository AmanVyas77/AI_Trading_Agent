# Runbook Patch — Quarterly Fundamentals Ingestion + xbrl Staleness Guard
Two prompts, one session each. Small ops patch — NOT a sprint: no model changes, no
new alpha, no verdict rule. Goal: the monthly refresh must automatically ingest new
quarterly statements (Q2-2026 10-Qs are landing mid-July → August) BEFORE the
2026-08-03 rebalance, and fail loud if fundamentals go genuinely stale.

Context (treat as fact):
- Repo: INNER repo only ("Ai Trading Agent/"); rev-parse guard; commits by MESSAGE
  (hashes rewritten 2026-07-05); read-only git from sandboxes; expect HEAD at/after
  the Sprint 8 commit. COMMIT POLICY: agent never commits/pushes — Prompt 2 ends by
  printing copy-paste commands for Aman, plain -m, NO Co-Authored-By trailer.
- CRITICAL RULE: never edit config/settings.yaml or .env. Module-level overrides only.
- PAPER TRADING IS LIVE (book: ACLS/FORM/HPE/MRVL/ON/QCOM @20%, 1.2 gross). This
  patch touches src/live/refresh.py + src/data pipelines only. It must NOT touch
  models/ensemble_models.pkl, src/live/{scorer,rebalance,broker_alpaca,paper_runner}.py,
  or any portfolio logic.
- THE GAP (from the 2026-07-12 review): run_refresh() steps are prices/FRED →
  backfill_cpi → FinBERT increment → factor_export_quant → feature_matrix. Missing:
  (a) EDGAR/XBRL ingestion (src/data/edgar_pipeline.py → xbrl_facts) — new 10-Qs
      never flow in;
  (b) **factor_export_fundamental re-export** — even if xbrl_facts had new rows,
      feature_matrix reads fundamental_factor_scores.parquet, which nothing in the
      runbook regenerates. Without this step the fundamentals in the live features
      are frozen at whatever the last manual export saw. This is the sneakiest part
      of the gap — do not skip it;
  (c) 10-K collection + LM scoring increment (annual cadence, cheap check);
  (d) SimFin estimates (eps_revisions stale since 2026-04) — fetch_simfin_eps has a
      dict-literal clamp "end": TIMELINE["test_end"] (simfin_pipeline.py ~143-144,
      found in Sprint 7 recon) needing a param/default fix;
  (e) no staleness assertion on xbrl_facts or eps_revisions.
- Staleness calibration (same philosophy as the CPI 75-day fix): 10-Qs are due ~40
  days after quarter end → max(xbrl_facts.end_date) age across the universe should
  never exceed ~120 days in healthy operation (by early August, most Q2/06-30 rows
  exist → age ~35d; if NOTHING has ingested by then, age hits ~125d and the guard
  fires — which is exactly the failure this patch exists to catch).
  eps_revisions: limit 120 days (SimFin refresh cadence is loose; calibrate, don't
  strangle).
- Machine: 8 GB Mac, .venv/bin/python. SEC rate limit: ≤10 req/s with the
  SEC_USER_AGENT already in .env (read by name, never echo values).

---

## PROMPT 1 of 2 — Recon + implement

```
You are working on the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent — the Runbook Patch,
Prompt 1 of 2. REPO GUARD first (inner repo; commits by message).
Use .venv/bin/python. Never edit config/settings.yaml or .env. Do not
touch models/, src/live/{scorer,rebalance,broker_alpaca,paper_runner}.py,
or portfolio logic.

RECON (before editing):
1. src/data/edgar_pipeline.py — identify the ingestion entry point and
   whether it is incremental or full re-pull per ticker; how it handles
   already-present (ticker, end_date) rows (upsert? skip?); estimated
   runtime for 54 tickers under the SEC rate limit. Quote signatures.
2. src/strategies/ensemble/factor_export_fundamental.py — confirm its
   CLI (--tickers/--start/--end, verified in Sprint 7 recon) and what
   it writes (fundamental_factor_scores.parquet).
3. src/data/simfin_pipeline.py ~lines 143-144 — quote the dict-literal
   TIMELINE clamp in fetch_simfin_eps; identify the minimal
   param/default fix (mirror the sentiment_pipeline date-args pattern
   from Sprint 8). Check SIMFIN_API_KEY exists in .env BY NAME.
4. src/live/refresh.py — quote the current step list and the staleness-
   assertion block the new checks extend.
5. Current max(end_date) in xbrl_facts and max(date) in eps_revisions —
   report ages as of today.

IMPLEMENT (keep each edit minimal; mirror existing patterns):
A. simfin_pipeline.fetch_simfin_eps: replace the dict-literal clamp
   with start/end params defaulting to module constants
   (DATE_START="2015-01-01", DATE_END=today at import — same pattern
   sentiment_pipeline now uses). CLI passthrough if it has a main().
B. refresh.py — insert new steps in this order (between the FinBERT
   increment and factor_export_quant):
     3b. EDGAR/XBRL ingestion for the universe (incremental if the
         pipeline supports it; if it only does full per-ticker pulls,
         run it as-is — idempotent upserts make that safe, just slower;
         log per-ticker timing).
     3c. 10-K/LM increment: run collection+scoring only if the
         unscored-10-K count > 0 (the LM pipeline's LEFT JOIN makes
         this a cheap no-op check).
     3d. SimFin estimates refresh via the de-clamped fetch.
     3e. factor_export_fundamental --start 2015-01-01 --end <last
         completed month-end>  ← regenerates the parquet feature_matrix
         reads; without this, steps 3b-3d change nothing downstream.
   feature_matrix (existing last step) now picks up fresh fundamentals.
C. Staleness assertions (extend the existing fail-loud block):
     max(xbrl_facts.end_date) age ≤ 120 days
     max(eps_revisions.date)  age ≤ 120 days
   Same named-series error style as the existing guards.
D. Module constants for the new limits next to the existing ones; a
   comment citing the 10-Q ~40-day deadline calibration.

Run the full patched run_refresh() once, end-to-end. Expect step 3b to
pull Q2-2026 rows for whichever names have filed by today; report how
many tickers gained a 2026-06-30 (or fiscal-equivalent) quarter, the
new max(end_date) ages, per-step timings, and the final all-✓ freshness
table. If the SEC pull is projected >30 min, chunk it and report
progress. git diff --stat at the end — expected: refresh.py,
simfin_pipeline.py only (+ tests if you add them now).

Output: recon quotes, the diffs, the refresh run report. State you're
ready for Prompt 2 (verify + commit prep).
```

---

## PROMPT 2 of 2 — Verify + commit prep (commands for Aman)

```
You are finishing the Runbook Patch for the "AI Trading Agent" project
at /Users/aman/Desktop/Stock_Project/Ai Trading Agent — Prompt 2 of 2.
Prompt 1 added EDGAR/XBRL + LM + SimFin + fundamental-export steps to
run_refresh() and 120-day staleness guards, and ran it once. REPO GUARD
first. Use .venv/bin/python. Read-only git.

ADDENDUM (from the Prompt 1 review, 2026-07-14):
0a. EPS-DATA PROVENANCE CHECK: Prompt 1 found SIMFIN_API_KEY present but
    EMPTY → the new eps_revisions rows came from the synthetic fallback
    (seasonal random walk on XBRL EPS), not the SimFin API. Determine
    what produced the HISTORICAL rows the model trained on: query
    analyst_estimates.source (and any equivalent marker on
    eps_revisions) grouped by year — if training-era rows are
    API-sourced and 2026 rows are fallback-sourced, that is SOURCE
    DRIFT between backtest and live: report it prominently and record
    it in the memory note (Aman decides: fill the key or accept the
    fallback; live must match training either way).
0b. Note for memory/future_ideas: LM 10-K collection re-downloads all
    historical filings every run (~23 min) because the collection gate
    doesn't skip already-stored PKs before hitting SEC — log as a small
    follow-up patch candidate ("LM collection PK-skip"), do NOT fix it
    in this session.

VERIFY (fail-loud test pattern, offline-safe where possible):
1. Data flow proof: pick 2 tickers that filed Q2 before today. Trace
   one all the way: new xbrl_facts row (end_date 2026-06-30 or fiscal
   equivalent) → fundamental_factor_scores.parquet row for the new
   quarter → feature_matrix rebuild shows updated fundamental values
   for the 2026-07 month-end rows (vs the ffilled Q1 values they had
   before). Paste the before/after values for piotroski_f and
   gross_profitability for those 2 tickers.
2. Staleness guard test: monkeypatch/temp-view the xbrl max(end_date)
   to 130 days old → assert run_refresh's check RAISES with the named
   series; restore.
3. Idempotency: run the patched run_refresh() a second time — EDGAR
   upserts must not duplicate rows (count xbrl_facts before/after ==),
   LM step must no-op (0 unscored), and the run must end all-✓.
4. LIVE-PATH SAFETY: prove the live scorer still works — one
   score_months() smoke call; md5(models/ensemble_models.pkl) unchanged
   vs models/ensemble_models_sprint5.pkl; confirm no diff in any
   src/live/ file except refresh.py.
5. Full unit suite: report pass/fail counts (the 2 pre-existing
   test_factors.py failures are known tech debt — unchanged is fine;
   NEW failures are not).

MEMORY: append a short "## Runbook patch (2026-07)" note to
memory/phase_progress.md (what was added, the two new limits + their
calibration, the factor_export_fundamental catch) and mark the
future_ideas.md entry DONE with a one-line outcome.

COMMIT PREP (standing policy — you do NOT run git commit/push):
Verify with git status --short + git diff --stat that the change set is
exactly: src/live/refresh.py, src/data/simfin_pipeline.py, tests (if
added), memory/phase_progress.md, memory/future_ideas.md — and that
settings.yaml/.env show NO diff. Then print for Aman:

  cd "/Users/aman/Desktop/Stock_Project/Ai Trading Agent"
  git status
  git add src/live/refresh.py src/data/simfin_pipeline.py \
          memory/phase_progress.md memory/future_ideas.md <tests-if-any>
  git commit -m "Runbook patch: EDGAR/LM/SimFin ingestion + fundamental re-export in monthly refresh; 120d staleness guards"
  git push origin main

Plus the reminders: inner repo only; expect fast-forward; rejected push
→ stop, no force-push, bring it to the knowledge-base chat.

Output: the five verification results, memory diffs, and the command
block. Do not execute the commands.
```

---

## Usage notes
- Prompt 1's EDGAR pull is the only slow part (SEC rate-limited); everything else
  is minutes. Run it whenever — it doesn't touch the live book.
- The single most important line in this patch is step 3e
  (factor_export_fundamental): ingestion without re-export changes nothing the
  model sees. Prompt 2's trace test exists to prove that chain end-to-end.
- After this lands, the 2026-08-03 rebalance scores on Q2 fundamentals for every
  name that has filed — automatically, forever.
