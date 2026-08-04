# Sprint 9 — News Sentiment 24th Feature (FNSPID + Alpha Vantage stitch) — REV 2
Copy-paste each prompt into a fresh Cowork/coding-agent session, one at a time, in order.
Each prompt is self-contained. REV 2 (2026-07-15) replaces the FMP design — FMP's free
tier has NO news endpoints (all 402, verified in the Rev-1 Prompt 1 recon; that recon's
FinBERT-reuse and feature-wiring findings are carried forward below and remain valid).

DATA ARCHITECTURE (the Rev-2 core — read carefully):
Two free sources, stitched with the seam INSIDE the training window:
  * FNSPID (HuggingFace: Zihan1004/FNSPID; CC BY-NC 4.0 — personal research OK,
    revisit license if this project ever commercializes): 15.7M ticker-associated news
    records 1999-2023 from Nasdaq/Bloomberg/Reuters/Benzinga. USED FOR: 2015-01 →
    2021-12 feature months only.
  * Alpha Vantage NEWS_SENTIMENT (free key, ~25 requests/day cap): archive from
    ~2022-03. USED FOR: 2022-01 → live, forever. (AV's own sentiment values are
    IGNORED — we score everything with our FinBERT for consistency.)
  * SEAM at 2022-01: inside training, safely before the 2023-24 test window. The
    2022-2023 OVERLAP (both sources have it) is used ONLY for calibration — verify
    the two sources produce consistent monthly FinBERT score distributions; z-align
    or document if they differ. Test window + live months are PURE single-source AV
    → no drift where verdicts and trading happen.
  * FNSPID alone is disqualified (ends 2023-12 = feature switches off mid-test-window).
    AV alone loses 2015-2021 depth. The stitch is the point.

PRE-COMMITTED VERDICT RULE (unchanged from Rev 1, fixed before any ingestion):
  PASS requires ALL FOUR:
    (1) full-period Sharpe  > 0.6818   (Sprint 5)
    (2) test Sharpe (2023-24) > 0.9898 (Sprint 5)
    (3) 2022 max DD ≥ −29.7815%        (no worse than Sprint 5)
    (4) news_sentiment deflated t-stat ≥ 1.0
  Anything else = FAIL → restore production, record verdict. The 2023-24 window has
  judged 5 prior experiments — note this in verdict_notes either way. The 2025-26
  holdout is SPENT — never evaluate on it.

DEPLOYMENT POLICY (unchanged): PASS does NOT deploy. Live paper stays on the frozen
Sprint 5 model; model swap is Aman's separate explicit decision. Every session restores
models/ensemble_models.pkl to Sprint 5 state before ending (md5 vs
ensemble_models_sprint5.pkl = 296e589f4da205eb1d171c2121d90f82, byte size 4,120,378).

Verified context (carried from the Rev-1 recon of 2026-07-15 — do not re-derive):
- Repo: INNER repo only; rev-parse guard; commits by MESSAGE (Sprint 8 = "Sprint 8:
  live paper-trading pipeline — PASS dress rehearsal", runbook patch = "Runbook
  patch: EDGAR/LM/SimFin ingestion…"). Agents NEVER commit/push — Prompt 5 prints
  commands for Aman, plain -m, NO trailers. Never edit settings.yaml/.env; never
  handle credential values. 8 GB Mac, .venv/bin/python, MPS available.
- FinBERT reuse (verified line numbers): FinBERTScorer class at
  sentiment_pipeline.py:376; score_text() at :448 returns P(pos)−P(neg) in [-1,1];
  pipeline built with truncation=True, max_length=512 → headline+snippet needs NO
  chunking: FinBERTScorer().score_text(title + ". " + snippet) directly.
- Feature wiring (verified): feature_matrix.py:59-79 — add NEWS_COLS =
  ["news_sentiment"] after :73; FACTOR_COLS += NEWS_COLS (:76); OUTPUT_COLS ordering
  (:79); new _load_news_scores() parallel to _load_quant_scores (:100); merge in
  _merge_all (:231) after quant+fund, before macro. model_trainer.py:46/95 META_COLS
  proves any new column auto-flows into training — zero trainer edits.
- Labels: target_builder rebuild only needs the training window; XLK cache
  (to 2025-01-14) covers the 2024-11 label horizon — no XLK fetch.
- Production state: ensemble_models.pkl md5 296e589f… == sprint5 backup ✓. Prompt 2
  of this sprint snapshots data/processed/{ensemble_labeled,ensemble_scores,
  ensemble_weights}.parquet to data/processed/_pre_sprint9_backup/ before anything.
- EWM convention: monthly per-ticker mean of article scores → EWM(span=4) at read
  time (mirror load_sentiment/load_lm_scores); zero-article months = NaN (not 0).
- 8 GB CONSTRAINT for FNSPID: the raw news CSVs are multi-GB (potentially >20 GB).
  NEVER load them whole. Stream-filter with chunked reads (or DuckDB) down to the
  54-ticker universe; check free disk BEFORE downloading; delete raw files after
  extraction. The filtered subset for 54 tickers should be a few hundred MB at most.

---

## PROMPT 1 of 5 — Recon (FNSPID sizing + AV probe)

```
You are picking up the "AI Trading Agent" project at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. Sprint 9 REV 2
(news sentiment via FNSPID + Alpha Vantage stitch), Prompt 1 of 5.
Read-only on the repo; network limited to the small probes below.
REPO GUARD first (inner repo, commits by message).

1. CREDENTIALS: .env must contain ALPHAVANTAGE_API_KEY (check BY NAME
   only). If absent: STOP and tell Aman to get a free key at
   alphavantage.co/support/#api-key and add the line himself.

2. Alpha Vantage probe (≤ 6 requests — the free cap is ~25/day and
   Prompt 2 needs the rest):
   (a) NEWS_SENTIMENT for NVDA, time_from=20220301T0000 limit=50 —
       confirm archive reaches back to ~2022-03; record the actual
       earliest item returned.
   (b) Same for AAPL 2022-H2 — confirm breadth beyond mega-tickers.
   (c) One call with limit=1000 — learn max page size, response fields
       (title, url, time_published, summary, ticker relevance), and
       whether a sort=EARLIEST param behaves as documented.
   Record: fields, effective page size, items-per-call, daily-cap
   evidence (header or documented), and a request-count estimate for
   54 tickers × 2022-01→today (with limit=1000 + time slicing, expect
   roughly 2-6 calls/ticker/year — compute the multi-day plan at
   25 req/day; ~2 weeks is acceptable, state the schedule).

3. FNSPID sizing (metadata only — do NOT download data files yet):
   Query the HuggingFace API for Zihan1004/FNSPID file listing + sizes
   (e.g. GET huggingface.co/api/datasets/Zihan1004/FNSPID/tree/main,
   recurse into Stock_news/). Identify which file(s) contain the
   ticker-keyed news (expect a large nasdaq_exteral_data.csv or
   similar, possibly >20 GB) and any smaller per-exchange or subset
   files that would serve our 54 tickers cheaper. Check local free
   disk (df -h). Produce the download+filter plan: streamed download →
   chunked filter to universe tickers → SQLite/parquet subset → delete
   raw. If total disk needed exceeds free space minus 20 GB headroom,
   plan a curl-piped streaming filter (never materializing the full
   file) and say so explicitly.

4. Universe ticker list: dump the 54 tickers from the prices table —
   Prompt 2's filter needs the exact symbols; note any that FNSPID's
   S&P500-oriented coverage might miss (small caps like ACLS, CAMT,
   FORM — flag for coverage checking, not exclusion).

5. Confirm the carried-forward wiring facts still hold (spot-check the
   quoted line numbers in feature_matrix.py and sentiment_pipeline.py
   — files may have drifted since the Rev-1 recon).

Output: AV probe table + request schedule, FNSPID file inventory +
disk plan, universe list with coverage flags, wiring spot-check. State
you're ready for Prompt 2 and whether its AV phase starts today or
must wait for the daily cap to reset.
```

---

## PROMPT 2 of 5 — Ingestion (two phases, both resumable)

```
You are continuing Sprint 9 REV 2 at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. Prompt 2 of 5.
Prompt 1 sized FNSPID, probed Alpha Vantage, and produced the request
schedule — honor both plans. REPO GUARD first. No settings.yaml/.env
edits. FIRST ACTION: snapshot data/processed/{ensemble_labeled,
ensemble_scores,ensemble_weights}.parquet to
data/processed/_pre_sprint9_backup/ (the Prompt 4 restore anchor).

RULINGS FROM THE PROMPT 1 REVIEW (2026-07-17 — binding for this prompt
and Prompt 3):
R1. SYMMETRIC FEATURE CONSTRUCTION: AV items carry a per-ticker
    relevance_score; FNSPID has none. STORE relevance in the AV
    staging rows, but the feature aggregation uses plain per-ticker
    article means for BOTH sources — no relevance filtering or
    weighting anywhere in the feature path (asymmetric construction
    would fabricate a seam-calibration gap). Relevance-weighted
    variant = future idea, not this sprint.
R2. GOOG/GOOGL: fetch AV news for GOOG only (saves budget); keep raw
    tables unmirrored; mirror monthly aggregates to GOOGL at
    feature-assembly time in Prompt 3 (same company, same news). Apply
    the same mirroring to FNSPID aggregates if its coverage is
    GOOG-only or GOOGL-only.
R3. AV pacing: ≥ 1.5s sleep between calls (burst limit is 1/sec on top
    of the ~25/day cap); state cursor in news_ingest_log as designed;
    FNSPID stream runs today, AV day-1 starts tomorrow.

Build src/data/news_pipeline.py with a unified store:
- Table news_articles(ticker, published_at, title, snippet, site,
  source TEXT,  -- 'fnspid' | 'alphavantage'
  article_hash TEXT PRIMARY KEY)  -- sha256(ticker|published_at|title)
- news_ingest_log(source, ticker, period, completed_at) for resume.

PHASE A — FNSPID (one-time, local-heavy):
Execute Prompt 1's disk plan: streamed download + chunked filter
(chunksize ≤ 100k rows or DuckDB) to the 54 tickers; keep ONLY
2015-01-01 → 2023-12-31 rows (we use ≤2021-12 for features and
2022-23 for seam calibration); normalize to the table schema with
source='fnspid'; delete raw files after verification. Report:
rows kept per year, per-ticker coverage (which of the 54 have <100
articles total — expected for small caps; report, don't exclude),
disk used/freed.

PHASE B — Alpha Vantage (multi-day trickle, ~25 req/day, AUTOMATED):
1. Build the fetch loop per Prompt 1's schedule: 2022-01-01 → today,
   sort=EARLIEST, limit=1000, GOOG-only per ruling R2, ≥1.5s sleep per
   ruling R3, source='alphavantage', resumable via news_ingest_log.
   Behavior per invocation: consume up to ~20 requests (leave daily
   headroom), then exit 0 with a one-line summary appended to
   logs/av_backfill_state.jsonl (date, requests_used, tickers_done/54,
   current cursor). If the API returns the daily-cap message early,
   record it and exit cleanly — never busy-retry against the cap.
   A --status flag prints coverage per ticker + % complete from the
   log, human-readable.
2. AUTOMATE THE DAILY RUN — set up a launchd agent (macOS-native;
   preferred over cron here) firing daily at 00:05 local:
   - Write ~/Library/LaunchAgents/com.aitrading.av-backfill.plist
     running the resume command with absolute paths (.venv/bin/python,
     project cwd), StandardOut/ErrPath appended to
     logs/av_backfill_launchd.log.
   - Load it (launchctl load / bootstrap) and VERIFY: launchctl list
     shows the label, then trigger one manual kickstart to prove the
     plist executes end-to-end (this doubles as AV day-1 if the cap
     allows; if today's cap is spent, the run exits cleanly and
     that clean exit IS the verification).
   - The job is self-terminating: once news_ingest_log shows all 54
     tickers covered 2022-01→today, the script logs "BACKFILL
     COMPLETE" and exits without making requests; instructions for
     Aman to unload the agent afterward go in the final report
     (launchctl unload <plist>) — do NOT auto-unload.
   - Machine-asleep-at-00:05 caveat: note in the report that launchd
     skips missed firings unless the Mac is awake; the job simply
     catches up the next time it fires — no data risk, just calendar
     stretch. (Do NOT change power settings.)
3. Report cadence: Aman checks in OCCASIONALLY (every few days) by
   running the --status command and pasting it to the knowledge-base
   chat — no daily agent sessions needed. This prompt's session is
   COMPLETE once the launchd agent is verified; PHASE B as a whole is
   complete when --status shows 54/54, which gates Prompt 3.

Verification (on whatever is complete):
- Year histogram per source; the 2022-23 overlap must show BOTH
  sources present (that's the calibration corpus).
- Dedup stats; no published_at outside [2015, now]; 5 random rows
  spot-read per source.

Output: Phase A report, Phase B automation proof (plist path, launchctl
list line, kickstart result), the --status command for Aman's periodic
check-ins, and the unified coverage table so far. Prompt 3 is GATED on
--status reaching 54/54 — state the projected completion date.
```

---

## PROMPT 3 of 5 — FinBERT scoring, seam calibration, feature wiring

```
You are continuing Sprint 9 REV 2 at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. Prompt 3 of 5.
Prompt 2 filled news_articles from both sources (fnspid ≤2023-12,
alphavantage 2022-01→today). REPO GUARD first. Do NOT touch the
production pickle.

PART A0 — COVERAGE PRE-FLIGHT (added after Prompt 2, 2026-07-17):
1. Confirm the FNSPID_NASDAQ stream completed (logs/fnspid_stream.log
   + a news_ingest_log row); if not, re-run it (idempotent) before
   anything else. Re-verify the per-ticker coverage table — the
   Prompt 2 zero-row flags (APPF DELL HPE ICHR MSFT ON SWKS TWLO)
   should mostly be resolved by the NASDAQ file; report any ticker
   still at zero FNSPID rows (AV covers it from 2022; pre-2022 it
   will be all-NaN — acceptable, but must be listed in the results
   caveats).
2. THE 2021 CLIFF: FNSPID rows collapse 18.5k (2019) → 10.6k (2020) →
   2.6k (2021, pre-NASDAQ numbers). Report per-year totals after
   NASDAQ, and per-ticker MONTHLY article density for 2020-2021
   specifically (median articles/ticker/month). If 2021 density is
   materially starved vs 2019 (>70% drop), say so prominently — the
   seam calibration (Part B) must then note it is comparing a thin
   FNSPID year against healthy AV years, and the results JSON caveats
   must record 2021 feature sparsity.

PART A — Score everything: news_sentiment_scores(article_hash PK,
ticker, published_at, source, finbert_score) via the existing
FinBERTScorer.score_text(title + ". " + snippet) — import, don't
duplicate; MPS batches; incremental via LEFT JOIN; resumable. Report
throughput after 500 and a per-source score distribution (mean/std/
%positive); a degenerate distribution (>90% one sign) = STOP finding.

PART B — SEAM CALIBRATION (the stitch's integrity check):
On the 2022-01→2023-12 overlap, compute per-month, per-source monthly
mean FinBERT scores for tickers covered by both. Compare distributions
(means, stds, rank-correlation of ticker-months across sources).
  * If aligned (|Δmean| small vs cross-sectional std, rank-corr
    strong): no adjustment; document the numbers.
  * If shifted but correlated: z-score normalize EACH SOURCE's monthly
    aggregates within its own history before stitching; document.
  * If uncorrelated: STOP — the stitch premise fails; bring the
    numbers back to the knowledge-base chat before proceeding.
Feature assembly rule after calibration: months ≤ 2021-12 from
fnspid; months ≥ 2022-01 from alphavantage ONLY (overlap fnspid rows
are calibration-only, never features).

PART C — Monthly feature + wiring (as Rev-1 recon mapped):
- load_news_sentiment(start, end): per (ticker, month) mean over
  articles published ≤ month-end (point-in-time), source per the
  assembly rule, then per-ticker EWM(span=4) mirroring
  load_sentiment/load_lm_scores; zero-article months = NaN. Write
  data/processed/news_sentiment_scores.parquet (monthly) for the
  feature_matrix loader + audit.
- feature_matrix.py edits exactly as the header's wiring map (NEWS_COLS
  after :73, FACTOR_COLS :76, OUTPUT_COLS :79, _load_news_scores,
  merge in _merge_all :231). Quote the full diff.
- Rebuild TRAINING matrix (--start 2015-01-01 --end 2024-12-31) +
  target_builder. Verify: 24 feature cols; news_sentiment NaN% by year
  (high pre-2018 fnspid-thin years is expected — report the actual
  curve); label months/rows unchanged (117 / 5,974).
- Live-path check: regenerate the LIVE feature matrix last (--end
  <last completed month-end>) and prove src/live/scorer.py still runs
  (it reindexes to the frozen 23 — the extra column is dropped
  harmlessly; smoke-call proof).

Output: scoring stats per source, THE CALIBRATION VERDICT with
numbers, aggregation spot-checks (3 tickers × 3 months hand-traced),
feature_matrix diff, labeled-parquet verification, live smoke proof.
State you're ready for Prompt 4.
```

---

## PROMPT 4 of 5 — Retrain, backtest, verdict — then RESTORE PRODUCTION

```
You are continuing Sprint 9 REV 2 at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. Prompt 4 of 5.
Prompt 3 delivered the 24-feature labeled dataset with the calibrated
FNSPID+AV stitch. REPO GUARD first.

RULE (pre-committed, apply exactly): PASS = full Sharpe > 0.6818 AND
test Sharpe > 0.9898 AND 2022 max DD ≥ −29.7815% AND news_sentiment
deflated-t ≥ 1.0. Else FAIL. 2025-26 holdout is SPENT — touch nothing
there. RESTORE DISCIPLINE: live trades read ensemble_models.pkl —
verify md5 == 296e589f4da205eb1d171c2121d90f82 before starting; all
sprint outputs go to versioned names; the finally-block below runs in
EVERY branch including crashes (if a session dies, run it FIRST on
recovery).

1. model_trainer (WINDOW_YEARS must be None) on the 24-feature
   labeled parquet → copy result to
   models/ensemble_models_sprint9_news.pkl.
2. score_generator → portfolio_builder → backtest; copy fixed-name
   outputs to backtests/results/ensemble_news_{train,test}_equity.csv
   + ensemble_news_stats.csv.
3. Recompute ALL verdict metrics programmatically from the equity CSVs.
   Extract news_sentiment deflated-t + gain rank /24 from the new
   pickle; note where the 5 lowest-t features now sit.
4. Breadth diagnostic: monthly names >0.52 vs Sprint 5's 24.6 mean
   (a breadth-driven Sharpe move is a confound — report either way).
   Also report the SEAM diagnostic: model performance contribution of
   news_sentiment in 2015-2021 (fnspid months) vs 2022-24 (AV months)
   folds if separable from fold importances — best-effort, one
   paragraph.
5. Verdict per the rule → backtests/results/sprint9_results.json
   (sprint5 schema + news_feature diagnostics + seam_calibration
   summary from Prompt 3 + multiple_testing_note [6th experiment on
   this window] + deployment_policy line + caveats [CC BY-NC license
   note; AV-era coverage starts 2022-03 not -01 if the probe showed
   that]).
6. FINALLY: cp models/ensemble_models_sprint5.pkl
   models/ensemble_models.pkl; restore
   data/processed/_pre_sprint9_backup/ parquets or regenerate via
   score_generator+portfolio_builder; PROVE: pickle md5, live scorer
   smoke call, and a rerun backtest on restored state reproducing test
   Sharpe 0.9898 ± 0.001.

Output: three-way table (Sprint 0 / 5 / 9), four checks, verdict +
notes (breadth + seam + multiple-testing caveats), news diagnostics,
restoration proof. NO deployment regardless of verdict. Ready for
Prompt 5.
```

---

## PROMPT 5 of 5 — Prepare commit (COMMANDS FOR AMAN — agent does NOT commit)

```
You are finishing Sprint 9 REV 2 at
/Users/aman/Desktop/Stock_Project/Ai Trading Agent. Prompt 5 of 5.
Prompt 4 wrote sprint9_results.json and PROVED production restoration.
STANDING POLICY: no agent commits/pushes; print commands for Aman;
plain -m, no trailers; read-only git.

1. memory/phase_progress.md: append "## Sprint 9 — news sentiment
   (FNSPID+AV stitch)" — rule verbatim, three-way table, four checks,
   verdict verbatim, seam-calibration numbers, news diagnostics,
   deployment-policy line, restoration proof.
2. memory/future_ideas.md: outcome-log the Sprint 9 entry
   (VALIDATED-NOT-DEPLOYED or REFUTED). If PASS: open item "Deploy
   sprint9 model?" (Aman's call, needs a dress-rehearsal-style
   dry-run; note live news feed = AV incremental, ~54 req/month —
   fits the free cap trivially). If FAIL: record whether diagnostics
   say dead feature vs drowned feature; no immediate variants
   (window is tired).
3. Verify change set: src/data/news_pipeline.py, feature_matrix.py
   diff, sprint9_results.json, ensemble_news_*.csv (git add -f), both
   memory files, tests if added. NOT staged: models/*.pkl, data/**,
   raw FNSPID remnants (should be deleted), _pre_sprint9_backup/.
   settings.yaml/.env: NO diff. Production pickle md5: re-verify,
   STOP if wrong.
4. Print the command block (cd inner repo; git status; explicit git
   add list; git add -f for the three CSVs; commit -m "Sprint 9: news
   sentiment 24th feature via FNSPID+AlphaVantage stitch —
   <PASS-not-deployed/REFUTED> (sprint9_results.json)"; push origin
   main; reminders: fast-forward expected, no force-push, rejects come
   to the knowledge-base chat).

Output: memory diffs, verification incl. pickle md5 line, command
block. Do not execute.
```

---

## Usage notes
- Aman's prerequisite: ALPHAVANTAGE_API_KEY in .env (free, instant). FMP key stays
  for the future analyst-funnel work — not used this sprint.
- Prompt 2 Phase B trickles ~2 weeks at 25 req/day by design; the paper account and
  monthly runbook are completely unaffected throughout. Report between days.
- The two moments most likely to need a knowledge-base-chat decision: Prompt 1's
  FNSPID disk plan (if the file is >free-disk) and Prompt 3's seam calibration (if
  sources disagree). Bring numbers, not vibes.
- If a Prompt 4 session dies mid-run: recovery session runs the FINALLY block first.
