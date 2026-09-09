# Sprint 9B — News Sentiment as the 24th Feature (modelling half) — REV 2

## REV 2 CHANGELOG (2026-09-09) — the stitch is dead, read this first

Prompt 2 ran and **landed on the table's STOP**. The design changed materially as
a result. Read this before anything else in the file; several sections below now
describe a plan that is no longer being executed and are marked as such.

**What Prompt 2 measured** (`backtests/results/sprint9b_calibration.json`, panel
of 451 ticker-months, 27 tickers × 24 months, 2022-01 → 2023-12):

```
dmean  = +0.350531        (AV +0.327130 vs FNSPID -0.023401)
sratio =  1.037942        (dispersions near-identical)
rho    = +0.207759        (p = 8.6e-06, n = 451)
xsd    =  0.271471   ->   |dmean| = 1.29 x xsd
bootstrap 95% CI on rho = [+0.1129, +0.2965]   P(rho < 0.20) = 43.3%
```

**Three findings, in order of importance.**

1. **STEP 4 is unexecutable by construction.** FNSPID stopped emitting
   `title_only` rows on **2020-06-11**; Alpha Vantage began on **2022-01-01**.
   Zero paired ticker-months exist at any article threshold, so `rho_title`
   cannot be computed. Inside the overlap window the FNSPID side is 100%
   `title_snippet`, which is why `rho_snippet` equals `rho` exactly. The
   diagnostic that was meant to separate a *source* seam from an *instrument*
   seam does not exist on this corpus and cannot be repaired by moving a
   threshold.
2. **AMENDMENT 2's premise was wrong, and it was mine.** It argued that
   `fnspid/title_only` (+0.085) "sits BETWEEN the two snippet cells, so
   title_only is NOT the outlier". That ordering came from full-corpus cell
   means — but `fnspid/title_only` is entirely 2015–2020 and never coexists with
   AV. I compared cell means as though they were contemporaneous when they are
   not. The comparison was confounded and the conclusion drawn from it does not
   hold.
3. **The seam is a pure level shift larger than the signal itself.**
   `sratio ≈ 1.04` means the dispersions match almost exactly; `|dmean|` is
   **1.29× the entire cross-sectional spread**. Combined with weak rank
   agreement (rho 0.21, CI straddling the STOP line at P = 43.3%), the two
   sources are not interchangeable instruments.

**The ruling (Aman, 2026-09-09): abandon the stitch. Go AV-only, retrain both
arms.** The STOP is honoured — for the structural reason in finding 1, not for
the knife-edge in the CI. Three consequences, each written into the relevant
ruling below:

- **R4 is SUPERSEDED.** There is no assembly rule to select, because there is
  nothing to assemble: one source, `alphavantage`, 2022-01 → present, 56 months.
  It matches the pre-registered forward window exactly, which is 100% AV.
- **R3 is AMENDED** to add the cross-sectional standardisation that every other
  ticker-varying feature in this repo already gets, and that R3 omitted. This is
  a correction of an inconsistency, verifiable by reading
  `factor_export_quant._cs_zscore` and `xbrl_features` — it does not depend on
  any calibration number.
- **R6/L1 is AMENDED.** Both arms must train on the identical 2022+ window, or
  the comparison confounds "news" with "less training data". This makes it a
  clean test of whether news adds anything, and **no longer** a test of whether
  news24 should replace production.

**What AV-only does NOT fix: density.** AV-only ticker-month medians run
2022: 13 → 2023: 8 → 2024: 9 → 2025: 20 → 2026: 199, a 22× swing *inside* the
single-instrument era. R3's shrinkage remains the remedy for that, unchanged.
The stitch decision fixed the **instrument** problem only.

---

**Supersedes the modelling half of `Sprint9_Prompts.md` REV 2 (Prompts 3–5).**
The ingestion half of REV 2 is DONE and is not repeated here: 439,727 articles,
2015-01-01 → 2026-09-01, 54 tickers, `news_ingest_log` 257 `ok` / 8 `empty` /
0 `error`, nightly launchd job in steady state.

Written 2026-09-07 after a recon pass over the actual corpus. The recon changed
the design in four places; those changes are recorded as rulings R3–R6 below and
were signed off by Aman before this file was written. **Do not "restore" the REV 2
design — it was drafted before the corpus existed and its assumptions are now
measurably wrong.**

---

## SEQUENCING — this sprint is THIRD

Two live-path fixes are due before the **2026-10-01 rebalance** and are NOT part
of this sprint. Do them first; Sprint 9B has no deadline and must not crowd them out.

| order | work | spec | why first |
|---|---|---|---|
| 1 | Fractional-exit defect | `ExitFix_Prompt.md` | LIVE and UNFIXED. Every live book since 2026-07 has been fractional, so the next rotation hits the broken exit path **with certainty**. Re-running `--execute` does not self-heal — it converges to a permanently un-exitable $495.18 orphan. |
| 2 | Breadth cap | `BreadthCap_Prompt.md` | `breadth_rules.implemented` is `false`. n=1 would put 100–120% of equity in one name. Mind the pre/post-multiplier trap. |
| 3 | **This file** | — | Research. Nothing here touches the live book. |

Prompt 0 of this file (SPY ingestion) is the one exception: it is small, it gates
everything downstream, and it should be done early because the forward window
needs SPY prices from its first day.

---

## WHAT THE RECON FOUND (do not re-derive — these are measured)

All figures below were computed on 2026-09-02 against
`data/quant_research.db` (832 MB, 439,727 rows in `news_articles`).

### 1. Four density regimes, not one seam

Median articles per ticker-month, GOOGL excluded (ruling R2 mirrors it from GOOG):

| era | window | ticker-month coverage | median n |
|---|---|---|---|
| Train (fit) | 2018-07 → 2022-12 | 78.4% | 20 |
| **Test fold** | 2023-01 → 2024-12 | 86.4% | **13** |
| Holdout (SPENT) | 2025-01 → 2026-06 | 94.5% | 46 |
| Live | 2026-07 → 2026-08 | 100% | **229** |

By calendar year: 2019: 26 → 2020: 19 → 2021: 11 → 2022: 22 → **2023: 21 →
2024: 9** → 2025: 20 → **2026: 230**.

Source split by year (`source` column):

```
fnspid       2015: 7,980   2016: 10,563  2017: 14,794  2018: 19,876
             2019: 23,495  2020: 16,370  2021: 9,189   2022: 21,161  2023: 40,493
alphavantage 2022: 10,677  2023: 7,372   2024: 9,045   2025: 61,748  2026: 186,964
```

FNSPID stops at 2023-12. **2024 is AV-only and is the sparsest year in the
corpus.** The 12× step-up at 2025-10 is the cursor-pagination backfill
(`510916c`) landing — AV rows for 2025 were fetched in 2026-07 and 2026-08, not
in 2025. Confirmed from `news_ingest_log`: the AV windows
`2022-01..2022-12` through `2025-01..2025-12` all have `completed_at` in
2026-07/08; only `2026-01..2026-12` is still being touched.

**Therefore: article density is a function of ingestion effort, not newsflow.**
It is an artifact of how many requests each ticker-window received before the
budget ran out, decaying backwards through history. This is the single most
important fact in the file.

### 2. No assembly rule can make the 2023-24 fold homogeneous

FNSPID dies exactly halfway through it. Under the REV 2 rule (AV-only from
2022-01), 2023 drops from median 21 → **8** and coverage 91% → 75%; 2024 stays
at 9 either way. This is one of the two reasons the judging window moved (R5).

### 3. `NaN not 0` was unrepresentable

`model_trainer._prepare_xy` does `df[feature_cols].fillna(0.0)` (line 105) and
`live/scorer.py` does `.fillna(0.0)` (line 84). A NaN news feature would silently
become "perfectly neutral news". Worse, the NaN rate is itself a clean time trend
(21.6% train → 13.6% test → 5.5% holdout → 0% live), so the missingness pattern
encodes the calendar. R3 removes the problem rather than patching it.

### 4. Text shape is a second seam, confounded with era

**58,872 of 163,921 FNSPID rows (35.9%) have no snippet** — title only, mean 71
characters. AV snippets average 475 characters (only 20 AV rows lack one). FinBERT
therefore sees systematically different inputs by source, and source is
collinear with time. The calibration in Prompt 2 must separate this from the
source effect proper.

### 5. Article volume is a size/attention proxy

NVDA 49,484 articles vs TWLO 222 — a 223× range. Top 5: NVDA 49,484, MSFT
37,656, AMZN 27,399, AAPL 25,235, AMD 24,772. Bottom 5: COHU 986, CAMT 960,
APPF 480, ICHR 430, TWLO 222. Any cross-sectional aggregate partly encodes "is
this a mega-cap", which the strategy already trades. This is why R3 forbids an
article-count feature and why the shrinkage target is an **equal-weighted mean
of per-ticker means**, never a pooled mean of articles.

### 6. Syndication is mild but present

439,727 articles vs 427,857 distinct (ticker, day, title) triples — 2.7%
same-day exact-title duplication. Largest same-timestamp cluster sampled for
AAPL in 2026 was 5 articles. Not large enough to justify a dedup pass; record it
as a known limitation.

### 7. What is already correct and needs no work

- `live/scorer.py` does `df.reindex(columns=feat_names)` then asserts
  `X.shape[1] == 23`. A 24th column in the matrix is **dropped harmlessly** on
  the live path. Verified by reading, not assumed. Prompt 3 still smoke-proves it.
- `scripts/freeze_vintage.py` copies `data/quant_research.db` **byte-for-byte**
  (`FROZEN_INPUTS`), so a vintage freeze already covers `news_articles` and will
  cover `news_sentiment_scores`. 832 MB per snapshot — budget the disk.
- `data/processed/_pre_sprint9_backup/` exists with `SHA256SUMS.txt` and the
  three restore parquets. The restore anchor is in place.
- `_get_feature_cols` derives features as "every column except
  `{date, ticker, fwd_return, xlk_return, label}`", so a new matrix column joins
  the model automatically. No trainer edit needed to admit the feature.

### 8. A weak leg inherited from REV 2

`deflated-t` is computed in `_deflated_importance_tstats` from XGBoost
`feature_importances_` across folds — it measures *"the model kept splitting on
this column"*, not *"this column made money"*. It is retained in R6 as a
necessary-not-sufficient gate and is explicitly labelled as an importance-stability
statistic. Note also that going 23 → 24 features changes the deflator
`sqrt(2·ln(n_features))` from 2.5043 to 2.5219 for **every** feature, so prior
sprints' deflated-t values are not directly comparable to this sprint's.

### 9. SPY is not in the database

`SELECT COUNT(*) FROM prices WHERE ticker='SPY'` returns **0**. The `prices`
table holds the 54 universe tickers only. `backtest.py:84` downloads SPY from
**yfinance at run time**. So today an SPY-relative verdict would compare a frozen
strategy against an unfrozen, network-dependent benchmark that `freeze_vintage.py`
does not snapshot. Prompt 0 fixes this, and it must be fixed before the forward
window opens.

---

## STANDING RULINGS

**R1 (carried, unchanged).** Alpha Vantage `relevance_score` is stored but
**never used in features**. Using it would fabricate a seam gap, because FNSPID
has no equivalent field and the asymmetry would be perfectly collinear with era.

**R2 (carried, unchanged).** Fetch **GOOG only**; mirror its aggregates to GOOGL
at assembly time. GOOGL's `MAX(published_at)` of 2020-06-10 is by design, not a
gap.

**R3 (new — aggregation). AMENDED 2026-09-09 — see R3-A below.** Shrinkage
toward the contemporaneous cross-sectional mean, weight `n/(n+k)`, **k = 10 fixed
a priori**. No article-count floor. No NaN ever emitted. At n=0 the formula returns the cross-sectional mean directly, so
`_prepare_xy`'s and `live/scorer.py`'s `fillna(0.0)` never fires on this column
and **`_prepare_xy` is not modified** — the frozen 23 features keep their exact
current treatment and comparability with prior sprints is preserved.
*Accepted cost, recorded deliberately:* "no news" and "exactly average news"
become indistinguishable. *Rejected:* a `has_news` indicator — the missingness
rate runs 21.6% → 5.5% → 0% across eras, so such an indicator would hand the
model a clean calendar variable. Report realized shrinkage weight and effective
n per era.

**R3-A (amendment to R3, 2026-09-09) — cross-sectional standardisation.**
After shrinkage and after the EWM, **z-score `news_sentiment` across tickers
within each month**, clipping to the same cap `factor_export_quant._cs_zscore`
uses (read the constant, do not guess it). Order is **shrink → EWM →
cross-sectional z**, so the z-score is the final step, matching how the quant
factors are built (`factors[label] = _cs_zscore(ret)`).

*Why.* Every other ticker-varying feature in this repo is already
cross-sectionally standardised — the quant factors by `_cs_zscore` per date, the
fundamental factors by `xbrl_features.build_feature_matrix`, which states it
"handles cross-sectional standardisation for every feature in FEATURE_COLS".
Macro is time-series z-scored, correctly, since it does not vary across tickers.
R3 as originally written would have made `news_sentiment` **the only
ticker-varying feature entering the matrix un-standardised**, on a raw
[−1, +1] FinBERT scale. That was an oversight in my design, findable by reading
the code and independent of any calibration number.

*Two consequences worth knowing.* It makes `dmean` and `sratio` irrelevant by
construction — a level shift common to all tickers in a month cancels under
per-month demeaning, and a dispersion ratio divides out — which is why two of
Prompt 2's three statistics stop mattering once the stitch is gone. And it means
a ticker-month with zero articles, which R3 maps to `xsec(m)`, lands at
approximately **0** after standardisation: the cross-sectional centre. The
"never emits NaN" design and the standardisation converge on the same place.

**R4 (assembly rule) — SUPERSEDED 2026-09-09. Do not execute Prompt 2.**
Prompt 2 ran, landed on the table's STOP, and the stitch was abandoned. There is
no assembly rule left to select: the corpus is **`alphavantage` only, 2022-01 →
present**, one instrument, no seam, no blending, no calibration. The full record
of what R4 was, how it was amended on 2026-09-08, what Prompt 2 measured and why
the STOP was honoured is in the REV 2 changelog at the head of this file and in
`backtests/results/sprint9b_calibration.json`. **FNSPID rows stay in the
database and stay scored — they simply do not feed the feature.**

**R5 (new — judging window).** Sprint 9B does **not** take a verdict on the
2023-01 → 2024-12 test fold. That fold has judged ~6 experiments and it straddles
the corpus's worst density break. Sprint 9B **builds and freezes** the 24-feature
model and **pre-registers a forward window** whose density is uniform. The
2023-24 fold is run once as a smoke diagnostic, recorded with
`verdict_eligible: false`, and may never be cited to accept the feature.

**R6 (new — verdict rule with an SPY leg). AMENDED 2026-09-09.** The rule is
stated in full in Prompt 5. Its SPY leg is a **paired** comparison of SPY-excess
Sharpe **over the same forward window**, not a comparison against Sprint 5's
−0.876 (which was measured on a different window and does not transfer).

> **AMENDMENT (2026-09-09).** The paired baseline is no longer the production
> model. Going AV-only means news24 can only train from 2022-01, while
> production trains on the full history — so comparing them would confound
> "news" with "less training data". **Both arms now train on the identical
> 2022-01 → 2026-08 window, identical folds, identical hyperparameters, the only
> difference being the news column.** The shadow harness therefore carries
> **three** arms:
>
> - `base23` — 23 features, retrained on 2022+. **This is L1's baseline.**
> - `news24` — 24 features, retrained on 2022+. Identical to `base23` but for
>   the one column.
> - `prod23` — the untouched production model. **Reported for context only, never
>   a verdict leg**, since it trains on a different window.
>
> This is a cleaner test of *"does news add anything"* than the original design.
> It is explicitly **not** a test of *"should news24 replace production"* — that
> question is not asked by this sprint and a PASS must not be read as answering
> it. Say so in the pre-registration.

**House conventions (unchanged).** The agent designs and executes but **never
runs `git commit` or `git push`** — Prompt 6 prints literal copy-paste commands
for Aman. **No `Co-Authored-By` trailer in this repo, ever.** Use
`git --no-optional-locks` for read-only git inspection through the Cowork device
bridge (the mount grants read/write but not delete, so a normal git command that
takes `.git/index.lock` leaves an unremovable 0-byte lock corpse).

---

## PROMPT 0 of 6 — Prerequisites: SPY into the DB, and freeze a vintage

```
You are starting Sprint 9B at /Users/aman/dev/Ai Trading Agent (NOT the
Desktop or Projects paths — those are dead). Prompt 0 of 6. This prompt
is small and gates everything after it.

REPO GUARD FIRST: confirm you are on branch main; confirm
`.venv/bin/python -c "import sys; print(sys.executable)"` reports the
repo's .venv and NOT /opt/anaconda3; confirm
md5 models/ensemble_models.pkl == 296e589f4da205eb1d171c2121d90f82.
If any check fails, STOP and report. Install only from
requirements.lock.txt — `pip install -r requirements.txt` is a trap in
this repo (the >= pins pull pandas 3.x / sklearn 1.9 and break the
frozen pickle).

TASK A — Put SPY in the prices table.
`SELECT COUNT(*) FROM prices WHERE ticker='SPY'` currently returns 0.
backtest.py:84 downloads SPY from yfinance at run time, so the benchmark
is unfrozen, network-dependent, and NOT covered by
scripts/freeze_vintage.py. A pre-registered SPY-relative verdict twelve
months out cannot rest on that.

1. Ingest SPY daily OHLCV + adj_close into `prices` for 2015-01-01 →
   today, through the SAME code path the 54 universe tickers use
   (src/data/quant_pipeline.py). Do NOT add SPY to universe.csv — it is a
   benchmark, not a candidate. Verify that nothing downstream treats a
   55th row in `prices` as a 55th universe member: grep for places that
   derive the ticker list from `prices` rather than from universe.csv and
   report every one you find BEFORE writing any rows.
2. Add a benchmark loader that reads SPY from `prices` and falls back to
   yfinance ONLY with a loud logged warning naming the fallback.
3. Reconcile: recompute SPY total return over 2025-01-31 → 2026-06-30
   from the DB rows and confirm it reproduces the +26.21% recorded in
   the SPY-relative scorecard to within 0.1pp. If it does not, report the
   discrepancy and STOP — do not "fix" the scorecard.

TASK B — Freeze a vintage, before anything else reads the corpus.
The nightly AV backfill revises history; Sprint 7's 1.016 now reproduces
as 1.0909. Run:
    .venv/bin/python scripts/freeze_vintage.py --out backtests/vintage_2026-09-XX
using today's date. Confirm the MANIFEST.json records the DB sha256, the
git HEAD sha, and the interpreter block, and that the copy-verify step
passed (it aborts if the backfill wrote mid-copy — if it aborts, wait
until the launchd job is idle and re-run).

TASK C — Add a news vintage fingerprint.
src/utils/data_vintage.py has price_vintage() but nothing for news.
Design (do not yet wire into the verdict path) a news_vintage() that
digests: row count, MIN/MAX published_at, and per-source per-year counts
from news_articles. Report the digest for the frozen vintage. This is how
a later session detects that the corpus moved underneath a comparison.

Output: the grep results from A1 BEFORE the write, SPY row count and
date range after, the reconciliation figure, the vintage path +
MANIFEST digest, and the news_vintage digest. State you are ready for
Prompt 1.
```

---

## PROMPT 1 of 6 — FinBERT-score the corpus

```
You are continuing Sprint 9B at /Users/aman/dev/Ai Trading Agent.
Prompt 1 of 6. Prompt 0 put SPY in `prices` and froze a vintage.
REPO GUARD first (branch, .venv interpreter, pickle md5). Do NOT touch
models/ensemble_models.pkl.

Score every row of news_articles with the EXISTING FinBERTScorer in
src/strategies/fundamental/sentiment_pipeline.py. Import it — do not
duplicate it. Its score_text() returns P(positive) − P(negative) in
[-1, +1], which is the convention this project already uses for
finbert_score, and keeping it means the news column is on the same
scale as the existing 8-K column.

NEW TABLE (do not reuse `sentiment_scores` — its PRIMARY KEY is
(ticker, filing_date), which cannot hold multiple articles per ticker
per day, and it is 100% sec_8k today; mixing sources in it would
silently corrupt the existing finbert_score feature):

    CREATE TABLE IF NOT EXISTS news_sentiment_scores (
        article_hash   TEXT PRIMARY KEY,
        ticker         TEXT NOT NULL,
        published_at   TEXT NOT NULL,
        source         TEXT NOT NULL,
        finbert_score  REAL NOT NULL,
        text_shape     TEXT NOT NULL,   -- 'title_only' | 'title_snippet'
        model_name     TEXT NOT NULL,
        scored_at      TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_nss_ticker_pub
        ON news_sentiment_scores(ticker, published_at);

IMMUTABILITY RULE — this is load-bearing, not hygiene. Insert with
`INSERT OR IGNORE`, never `INSERT OR REPLACE`. Once an article_hash has
a score, that score is frozen forever. A re-score pass must not
retroactively rewrite a historical value, because the moment news feeds
the ensemble, a rewritten history is a silent lookahead. If a model
change ever makes re-scoring necessary, that is a new `model_name` and a
new table, not an overwrite.

SCORED TEXT: `title + ". " + snippet` when a snippet exists, `title`
alone when it does not, and set text_shape accordingly. 58,872 of
163,921 FNSPID rows (35.9%) are title-only; AV is almost entirely
title+snippet. text_shape is recorded because Prompt 2 must test whether
it, rather than `source`, drives the seam.

MECHANICS: MPS if available (--cpu-only fallback; MPS is flaky for some
transformer ops on this machine), batched, incremental via
`LEFT JOIN news_sentiment_scores ... WHERE nss.article_hash IS NULL`,
fully resumable — assume the session may die and be restarted. Commit in
batches, not per-row. Report throughput after the first 500 rows and
extrapolate the full-corpus estimate BEFORE continuing; if the estimate
exceeds 8 hours, stop and report rather than running blind. Watch RAM —
this machine has 8 GB.

REPORT, per source AND per text_shape (four cells):
  count, mean, std, %positive (score > 0), %negative, %|score| < 0.05
A degenerate distribution — any cell >90% one sign — is a STOP finding:
report it and do not proceed to Prompt 2.

Also hand-verify 5 articles: print the raw title/snippet and the score,
and confirm by eye that the sign is defensible. A model that scores
"Company X beats earnings, raises guidance" negative is broken and no
downstream statistic will reveal it.

Output: table schema confirmation, throughput + total runtime, the four
distribution cells, the 5 hand-verified examples, and total rows scored
vs total rows in news_articles (they should match exactly; explain any
gap). State you are ready for Prompt 2.
```

---

## PROMPT 2 of 6 — Seam calibration — ✅ EXECUTED 2026-09-09, OUTCOME: STOP

> **DO NOT RUN THIS PROMPT AGAIN.** It was executed once, at HEAD `53b71bf`, and
> it reached the table's STOP. The text below is preserved verbatim as the record
> of what was actually asked — including AMENDMENT 2, whose premise turned out to
> be wrong. Do not "fix" it retroactively; an amended record of a pre-registered
> rule is worthless. What it produced, and the ruling that followed, are in the
> REV 2 changelog at the head of this file and in
> `backtests/results/sprint9b_calibration.json`.
>
> **Outcome in one line:** rho = 0.2078 (CI [0.113, 0.297], P(rho<0.20) = 43.3%),
> |dmean| = 1.29 × xsd, sratio = 1.04, and STEP 4 unexecutable because FNSPID's
> title-only cell ends 2020-06 while AV starts 2022-01 — zero paired months at
> any threshold. STEP 3 pointed at RULE D on a knife-edge; STEP 4 landed on the
> uncovered combination the table answers with STOP. **The stitch was abandoned.**

```
You are continuing Sprint 9B at /Users/aman/dev/Ai Trading Agent.
Prompt 2 of 6. Prompt 1 scored the corpus into news_sentiment_scores.
REPO GUARD first.

This prompt DECIDES the assembly rule (ruling R4). Apply the table
mechanically. Do not improvise an outcome the table does not name, and
do not adjust a threshold because a statistic landed just outside it —
if that happens, report it and STOP.

── AMENDMENTS TO R4, RECORDED 2026-09-08, BEFORE THIS PROMPT RAN ──────

The table was first fixed on 2026-09-07, before the corpus had been
scored. Prompt 1 then measured two facts about the INPUTS that the
original table did not anticipate. Both amendments were approved by
Aman on 2026-09-08 and are recorded here, in the file, before Prompt 2
executes.

  AMENDMENT 1 — U-z standardises per (source x text_shape) cell, four
  streams, not per source, two streams. Reason: the measured means run
  +0.284 / +0.147 / +0.085 / -0.008 across the four cells, so pooling
  by source alone would leave a ~0.09 gap inside the FNSPID stream and
  a ~0.29 gap between the two title_snippet cells unaddressed.

  AMENDMENT 2 — RULE T's trigger and remedy both change. It now fires
  on the fnspid/title_snippet cell rather than on title_only, and its
  remedy is to re-score those rows on title alone rather than to
  discard them. Reason: 86.4% of that cell is hard-truncated at 2,000
  characters, and fnspid/title_only sits BETWEEN the two snippet cells,
  so the original remedy would have dropped 58,872 articles that are
  not the problem while leaving 105,049 that are.

  WHY THIS IS NOT PEEKING, stated so a later reader can check the
  claim rather than take it on trust: both amendments respond to
  properties of the INPUT TEXT — cell means, string lengths, truncation
  rates. At the time they were made, no forward return, no label, no
  rank IC and no Sharpe had been computed against the news feature; the
  dependent variable was entirely unseen and remains so until Prompt 4.
  R4 exists to stop the assembly rule being selected by OUTCOME, and
  that constraint is intact. The amendments are recorded with their
  date and reasoning precisely so the ordering is auditable.

  What did NOT change: the rho thresholds (0.50 / 0.20), the |dmean|
  threshold (0.25 * xsd), the >= 5 articles pairing requirement, the
  four named branches of STEP 3, and the rho < 0.20 STOP. Those are
  untouched from the 2026-09-07 fixing.

── END AMENDMENTS ─────────────────────────────────────────────────────

STEP 1 — Build the paired panel.
On the overlap window 2022-01 → 2023-12, for every (ticker, month) where
BOTH sources have >= 5 articles, compute the per-source RAW monthly mean
FinBERT score (no shrinkage yet — shrinkage is Prompt 3). Report how many
paired ticker-months this yields; if fewer than 200, say so prominently,
because every statistic below is then thin.

STEP 2 — Three statistics.
  (a) dmean = mean over pairs of [ AV_value - FNSPID_value ]
  (b) sratio = std(AV values) / std(FNSPID values)
  (c) rho = Spearman rank correlation of the paired values
Also compute xsd = the pooled cross-sectional std of monthly means over
the same window, which is the yardstick dmean is judged against.

STEP 3 — The pre-committed decision table.

  rho >= 0.50 AND |dmean| <= 0.25 * xsd
      -> RULE U-raw. Union both sources; pool articles directly, so a
         ticker-month's mean is over all its articles regardless of
         source. Simplest rule; justified only when the two sources are
         measurably interchangeable.

  rho >= 0.50 AND |dmean| >  0.25 * xsd
      -> RULE U-z. Union, but first z-score each of the FOUR
         (source x text_shape) CELLS within its OWN history (expanding,
         backward-only — never a full-sample z-score, which would be
         lookahead), then pool the standardised values weighted by
         article count. See AMENDMENT 1 at the head of this prompt: the
         cells are per-cell, NOT per-source, because the measured
         heterogeneity runs along text_shape at least as strongly as
         along source.

  0.20 <= rho < 0.50
      -> RULE D. Denser source per era: FNSPID through 2023-12,
         alphavantage from 2024-01. Do NOT blend. Record that the seam
         and the density cliff then coincide at 2024-01 and are no
         longer separable.

  rho < 0.20
      -> STOP. The two sources are not measuring the same quantity. The
         stitch premise fails. Report the numbers and end the session —
         do not pick a rule.

STEP 4 — Isolate the text-shape confound (REVISED, see AMENDMENT 2).
Repeat STEP 2 twice more, splitting the FNSPID side by text_shape:
  rho_snippet  = rho on pairs where the FNSPID side is title_snippet
  rho_title    = rho on pairs where the FNSPID side is title_only

Prompt 1 measured, on the full scored corpus:

  source        text_shape       count     mean    %|s|<.05  capped@2000
  alphavantage  title_snippet  278,799   +0.2843     7.7%       0.0%
  alphavantage  title_only          20   +0.1466    30.0%          —
  fnspid        title_only      58,872   +0.0851    24.3%          —
  fnspid        title_snippet  105,049   -0.0078    16.0%      86.4%

So the suspect cell is fnspid/title_snippet — 86.4% of it is hard-
truncated at exactly 2,000 characters (mean length 1,909, cut mid-word),
i.e. ~490 tokens of raw article body, against AV's coherent ~475-char
summaries. fnspid/title_only (+0.085) sits BETWEEN the two snippet
cells, so title_only is NOT the outlier and must not be what gets
dropped.

  If rho_title >= 0.50 AND rho_snippet < 0.20
      -> RULE T fires. The seam is a TEXT-SHAPE seam, not a source seam,
         and the truncated bodies are the cause. Remedy: RE-SCORE those
         105,049 rows on TITLE ALONE, discarding the truncated body.
         Do NOT discard the articles. Re-scoring writes NEW rows under a
         distinct model_name/text_shape marker — never an overwrite; the
         immutability rule (INSERT OR IGNORE, a score frozen once
         written) is not suspended for this. Decide and state the schema
         change needed, since article_hash is the PRIMARY KEY and the
         same article now carries two scores under two text treatments.
         Then re-run STEP 2 and STEP 3 on the re-levelled panel and
         confirm the selected rule is stable.
  If both rho_title and rho_snippet clear 0.50
      -> no text-shape remedy; the per-cell z-scoring in U-z already
         absorbs the level difference. Say so explicitly.
  Any other combination
      -> report it and STOP. The table does not cover it, and improvising
         a fifth outcome after seeing the numbers is exactly what the
         pre-registration exists to prevent.

Report the article count affected by whichever branch fires, by year.

STEP 5 — Report the density context alongside the verdict, always.
Whatever rule is selected, print the resulting median articles per
ticker-month per year under that rule, so the reader sees what the rule
costs. For reference, measured on 2026-09-02: all-sources medians are
2019: 26, 2021: 11, 2022: 22, 2023: 21, 2024: 9, 2025: 20, 2026: 230;
the suspended REV 2 rule would give 2022: 13, 2023: 8.

Output: the paired-panel size, dmean / sratio / rho / xsd with the
arithmetic shown, rho_snippet and rho_title, THE SELECTED RULE named
explicitly with the branch of the table that selected it, the density
table under that rule, and whether RULE T applies. Write all of it to
backtests/results/sprint9b_calibration.json. State you are ready for
Prompt 3.
```

---

## PROMPT 3 of 6 — The feature: shrinkage, EWM, and wiring

```
You are continuing Sprint 9B at /Users/aman/dev/Ai Trading Agent.
Prompt 3 of 6. Prompt 2 selected the assembly rule. REPO GUARD first.

⚠ REV 2 — READ THE CHANGELOG AT THE HEAD OF THIS FILE FIRST. Prompt 2
reached the table's STOP and the stitch was abandoned. Two things this
prompt now does differently from how it was first drafted:

  (a) THE CORPUS IS alphavantage ONLY, published_at >= 2022-01-01.
      There is no assembly rule, no calibration, no per-source or
      per-cell z-scoring, no RULE T, and no FNSPID row feeds the
      feature. FNSPID rows stay in the DB and stay scored; they are
      simply not read. Filter on source explicitly and assert the
      filter caught what you expect: 278,819 AV rows, 0 fnspid, in the
      frame you aggregate from.
  (b) R3-A ADDS CROSS-SECTIONAL STANDARDISATION as the final step.
      See the formula section below.
  (c) R2 STILL APPLIES — MIRROR GOOG -> GOOGL AT ASSEMBLY. This was
      dropped from REV 2 by mistake when the file was rewritten for
      AV-only; R2 was never rescinded. universe.csv contains BOTH
      tickers, GOOG has 7,163 AV articles since 2022 and GOOGL has
      ZERO, so without the mirror one ticker in the matrix carries a
      structurally dead news column for all 56 months. After computing
      GOOG's monthly aggregate, copy it to GOOGL BEFORE the
      cross-sectional z-score, so both share classes contribute to and
      are measured against the same monthly cross-section. Assert
      afterwards that GOOGL's n, raw and shrunk equal GOOG's exactly.

Note what AV-only does NOT fix. Density inside the AV era still runs
2022: 13 -> 2023: 8 -> 2024: 9 -> 2025: 20 -> 2026: 199 median articles
per ticker-month, a 22x swing. R3's shrinkage is still the remedy and is
unchanged. Report the per-year realised shrinkage weights as specified.

Build `load_news_sentiment(start, end, engine=None)` returning
[ticker, date (month-end), news_sentiment], and write
data/processed/news_sentiment_scores.parquet for audit.

── THE FORMULA, STATED IN WORDS BEFORE SYMBOLS ────────────────────────

For one ticker i in one calendar month m:

  n(i,m) is the number of that ticker's articles published inside month
  m, counted from news_sentiment_scores after the Prompt 2 assembly rule
  (and after RULE T's title_only exclusion, if it applied).

  raw(i,m) is the plain arithmetic mean of those articles' FinBERT
  scores. It is undefined when n(i,m) = 0.

  xsec(m) is the cross-sectional mean for that month: take every ticker
  that has at least one article in month m, take that ticker's raw
  value, and average those per-ticker values giving each ticker equal
  weight. It is NOT the mean of all articles pooled — NVDA has 223x
  TWLO's article count, so a pooled mean would make xsec(m) mostly a
  reading of NVDA's sentiment.

  k = 10, fixed in advance, never tuned. It is the number of articles at
  which the ticker's own evidence and the cross-sectional prior get
  equal weight.

  shrunk(i,m) is a weighted average of the ticker's own mean and the
  cross-sectional mean, where the weight on the ticker's own mean is
  n/(n+k) — that is, the more articles the ticker had, the more we
  trust its own number, and the fewer it had, the more we fall back on
  what the cross-section did that month.

  In symbols:  shrunk(i,m) = [n/(n+k)] * raw(i,m) + [k/(n+k)] * xsec(m)

  At n = 0 the first weight is exactly 0 and the second is exactly 1, so
  shrunk = xsec(m). IMPLEMENT THIS AS AN EXPLICIT BRANCH:
      if n == 0: shrunk = xsec(m)
  Do NOT rely on the formula, because raw is NaN there and in numpy
  0 * NaN is NaN, not 0. This one line is the difference between "never
  emits NaN" and "silently emits NaN in 21.6% of training rows".

  smoothed(i,m) is the per-ticker exponentially weighted moving average
  of the shrunk series over months, span = 4 (the Arratia convention
  this repo already uses in _ewm_smooth for the 8-K and LM columns).
  SHRINK FIRST, SMOOTH SECOND. Smoothing first and shrinking second
  would apply a single month's article count to a value that already
  mixes four months.

  news_sentiment(i,m) — R3-A, ADDED 2026-09-09 — is then the CROSS-
  SECTIONAL z-score of smoothed(i,m) across all tickers within month m,
  clipped to the same cap factor_export_quant._cs_zscore uses. READ that
  constant from the code; do not guess it. In words: within each month,
  subtract the mean of smoothed across the 53 tickers and divide by its
  standard deviation, then clip.

  ORDER IS: shrink -> EWM -> cross-sectional z. The z-score is LAST,
  matching how the quant factors are built (factors[label] =
  _cs_zscore(ret)), so the column arrives at the matrix on the same
  scale as its 23 neighbours instead of on FinBERT's raw [-1, +1].

  Two things this changes, state both in your report:
    - A ticker-month with zero articles maps to xsec(m) by the shrinkage
      branch, and xsec(m) is the cross-sectional mean, so after
      standardisation it lands at approximately 0 — the cross-sectional
      centre. The "never emits NaN" rule and the standardisation agree.
    - Any level shift common to all tickers in a month cancels. This is
      what makes the abandoned stitch's dmean and sratio irrelevant, and
      it is worth confirming numerically rather than asserting.

── WORKED EXAMPLE, INTERMEDIATE STEPS SHOWN ───────────────────────────

Month 2024-03. Suppose across the 44 covered tickers that month the
equal-weighted mean of their raw values is xsec = +0.11. Take three:

  NVDA: n = 180, raw = +0.42
        weight on own mean = 180 / (180 + 10) = 180/190 = 0.94737
        weight on xsec     = 10 / 190          = 0.05263
        shrunk = 0.94737 * 0.42 + 0.05263 * 0.11
               = 0.397895 + 0.005789
               = +0.403684
        (barely moved — 180 articles is plenty of evidence)

  AMAT: n = 6, raw = -0.30
        weight on own mean = 6 / (6 + 10) = 6/16 = 0.37500
        weight on xsec     = 10/16        = 0.62500
        shrunk = 0.37500 * (-0.30) + 0.62500 * 0.11
               = -0.112500 + 0.068750
               = -0.043750
        (pulled hard toward the cross-section — 6 articles is thin)

  TWLO: n = 0
        branch fires: shrunk = xsec = +0.110000

Now the EWM, span = 4, on AMAT's shrunk series. pandas
ewm(span=4, min_periods=1).mean() with the default adjust=True computes
alpha = 2/(span+1) = 2/5 = 0.4, so the decay factor (1-alpha) = 0.6, and
the value at time t is the sum of (0.6^i * x_{t-i}) divided by the sum of
(0.6^i), for i = 0 up to t.

Say AMAT's last three shrunk values are x0 = +0.05 (2024-01),
x1 = -0.02 (2024-02), x2 = -0.04375 (2024-03). Then at 2024-03:

  numerator   = 1.00 * (-0.04375) + 0.60 * (-0.02) + 0.36 * (+0.05)
              = -0.043750 - 0.012000 + 0.018000
              = -0.037750
  denominator = 1.00 + 0.60 + 0.36 = 1.96
  news_sentiment(AMAT, 2024-03) = -0.037750 / 1.96 = -0.019260

Note that because shrinkage removed every gap, each ticker now has an
unbroken monthly series, so "span = 4" means the same four calendar
months for every ticker. Under the old NaN convention it would have
meant "the last 4 months that happened to have articles", which differs
per ticker and per era. State this in the results JSON as a side benefit
of R3.

── POINT-IN-TIME DISCIPLINE (load-bearing from now on) ────────────────

Until now news fed nothing, so lookahead was N/A. From this prompt
onward it is real. Three rules:

1. Every aggregation filters `published_at <= <month-end 23:59:59 UTC>`
   for the month being computed. Do not rely on the month bucket alone —
   filter explicitly, so the intent is visible in the code and a future
   reader cannot mistake it.
2. A PARTIAL current month must never be emitted as if complete. When
   `end` falls mid-month, drop that month entirely and log that you
   dropped it. Today (2026-09-07) a naive build would emit a 2026-09 row
   from 7 days of articles and it would look like a normal row.
3. xsec(m) uses only month m's own articles. This is safe — it is
   contemporaneous, not forward-looking — but write a comment saying so,
   because it looks like cross-sectional leakage at a glance and someone
   will flag it later.

── WIRING ─────────────────────────────────────────────────────────────

Edit src/strategies/ensemble/feature_matrix.py:
  - add NEWS_COLS = ["news_sentiment"] after the MACRO_COLS block
  - FACTOR_COLS = QUANT_COLS + FUND_COLS + NEWS_COLS
  - OUTPUT_COLS = ["date","ticker"] + QUANT_COLS + FUND_COLS + NEWS_COLS
    + MACRO_COLS
  - add _load_news_scores() alongside _load_quant_scores()
  - merge it in _merge_all() on ["date","ticker"]
Quote the FULL diff.

⚠ WATCH THE NaN-DROP DENOMINATOR. build_feature_matrix drops rows where
more than NAN_DROP_THRESHOLD (50%) of FACTOR_COLS are NaN. Adding a 24th
entry to FACTOR_COLS changes that denominator from 20 to 21 and could
therefore change WHICH HISTORICAL ROWS SURVIVE, silently altering the
matrix in months that have nothing to do with news. Because R3
guarantees news_sentiment is never NaN, adding it can only ever LOWER
each row's NaN fraction, so no row that survived before can drop now —
but PROVE it rather than reasoning about it: rebuild with and without
NEWS_COLS in FACTOR_COLS and assert the surviving (date, ticker) index
is identical.

── VERIFY ─────────────────────────────────────────────────────────────

1. Rebuild the FULL matrix first: --start 2015-01-01 --end 2026-08-31.
   Assert 24 feature columns, and assert the (date,ticker) index matches
   the pre-change matrix EXACTLY — adding a never-NaN column must not
   change which historical rows survive the NaN-drop (see the warning
   above). news_sentiment will be exactly 0.0 for every month before
   2022-01, because with no AV articles every ticker falls to xsec(m)
   and standardising a constant vector gives zeros. That is correct and
   expected; confirm it rather than treating it as a bug, and confirm it
   is EXACTLY zero, not merely small.
   Then note for Prompt 4: the SPRINT WINDOW is 2022-01-01 -> 2026-08-31
   (56 months of AV). Months before 2022 carry a structurally
   uninformative news column and both arms will be trained on 2022+
   only, per the R6 amendment.
2. Report, per year: mean realized shrinkage weight n/(n+k), median
   effective n, and the fraction of ticker-months that fell back
   entirely to xsec (n=0). This is the R3 honesty report — it shows
   exactly how much of the early history is prior rather than evidence.
3. Hand-trace 3 tickers x 3 months end to end (article count -> raw ->
   xsec -> shrunk -> EWM) and show every intermediate number, as in the
   worked example above.
4. target_builder: the invariant to test is that adding a never-NaN
   feature column does not move the label set AT ALL. Test it that way
   — build labels from the pre-change matrix and the post-change matrix
   and compare — rather than against a remembered row count.
   ⚠ DO NOT expect 117 / 5,974. That figure was carried forward from
   REV 1 and is STALE: the Sep-1 quant/fundamental rebuild grew the
   matrix 5,938 -> 7,086 rows and dropped (2015-03-31, ACLS), so the
   current full-span answer is 117 / 5,973 and was 5,973 before this
   sprint touched anything. Confirmed against the frozen vintage. If you
   find yourself investigating a one-row delta, check the baseline
   before you check the code.
   Then report how many label months and rows fall inside the
   2022-01 -> 2026-08 sprint window — that is what both arms train on.
   ⚠ AS OF 2026-09-09 THIS IS THE KNOWN BLOCKER, resolved by Prompt 3b
   below: only 35 label months exist (2022-01 .. 2024-11) against the
   39 required by MIN_TRAIN_MONTHS=36 + PURGE_MONTHS=3, so zero folds
   can be generated. Do not shrink a guard to make it fit. Run Prompt 3b
   first, then re-run this check.
5. LIVE-PATH SMOKE PROOF: rebuild the live matrix to the last COMPLETED
   month-end (2026-08-31) and run src/live/scorer.py against the
   untouched production pickle. It reindexes to the frozen 23 names and
   asserts shape[1] == 23, so the 24th column must be dropped
   harmlessly. Show the assertion passing and the scores reproducing.
   If this fails, the live book is at risk and you STOP immediately.

Output: the full feature_matrix diff, the identical-index proof, the
per-year shrinkage honesty report, the 3x3 hand-trace, label
verification, and the live smoke proof. State you are ready for Prompt 4.
```

---

## PROMPT 3b — XLK into the DB, then extend the label window

**Added 2026-09-09 after Prompt 3 hit the fold STOP.** Run this between
Prompt 3 and Prompt 4. It is the SPY fix again, for a benchmark that matters
more.

```
You are continuing Sprint 9B at /Users/aman/dev/Ai Trading Agent.
Prompt 3b. REPO GUARD first (branch, .venv interpreter, pickle md5).

WHY THIS EXISTS. target_builder.py:55 sets BENCHMARK = "XLK", line 132
downloads it from yfinance with end="2025-01-15" HARD-CODED, and caches
to data/raw/xlk_monthly.csv — which is gitignored and in no frozen
vintage. label = 1 if a ticker's forward return beats XLK's. So THE
DEPENDENT VARIABLE OF THIS PROJECT rests on an unfrozen network fetch,
and every verdict in the project's history was computed against a
benchmark nobody snapshotted. XLK is not in `prices` (verified: 55
tickers, SPY present, XLK absent). This is the SPY defect from Prompt 0,
still live, and worse — SPY was a comparison, XLK makes the labels.

TASK A — XLK into `prices`.
1. Ingest XLK daily OHLCV + adj_close for 2015-01-01 -> today through
   the SAME path SPY took (src/data/quant_pipeline.py). Do NOT add XLK
   to universe.csv — it is a benchmark.
2. Add "XLK" to benchmarks.tickers in settings.yaml, so it is now
   ["SPY", "XLK"]. The guard built in Prompt 0 was designed for exactly
   this: re-run the four-site check and CONFIRM each site now excludes
   two benchmarks rather than one. Re-run the equality proof and expect
   (prices - benchmarks) == universe with both diff sets empty. If it
   does not hold, STOP — the guard is not generalising and a second
   benchmark row is loose in the z-scoring paths.
3. Point _load_xlk_monthly() at `prices` via the Prompt 0 benchmark
   loader. Keep a yfinance fallback ONLY behind the same loud warning
   and allow_download=False. Remove the hard-coded end="2025-01-15".
4. Reconcile before regenerating anything: compare the DB XLK monthly
   series against the existing data/raw/xlk_monthly.csv cache over their
   overlap. Report max |delta| per month. They will not match exactly —
   the cache came from a different vintage — so REPORT the differences,
   do not silently adopt either. If any month differs by more than 0.5%,
   show it and say which months' labels it could flip.

TASK B — Extend the label window to the pre-registered span.
REV 2 already committed both arms to 2022-01-01 -> 2026-08-31, so this
is an implementation fix to reach a window already on the record, NOT a
new pre-registration decision. The blocker is target_builder.py:89
clamping prices to TIMELINE["test_end"] (2024-12-31) while prices run to
2026-09-04.
1. Thread explicit start/end parameters through target_builder. DO NOT
   mutate TIMELINE["test_end"] in settings.yaml — it is read elsewhere,
   including the live matrix build, and changing it would ripple into
   the production path. Default behaviour with no arguments must stay
   bit-identical to today.
2. Regenerate labels to 2026-08-31. Report: label months and rows over
   the full span, and inside the sprint window. Re-run the fold
   feasibility check and show months > MIN_TRAIN_MONTHS + PURGE_MONTHS
   arithmetic explicitly.
3. Report how many labels CHANGED versus the pre-existing labeled
   parquet over the overlapping period, and attribute the change: XLK
   source (DB vs yfinance cache) versus window extension. If labels move
   in months before 2024-11, that is the XLK swap and it must be
   quantified, not waved through.

ON TRAINING OVER THE SPENT HOLDOUT — state this in the output.
Extending labels means both arms TRAIN on 2025-01 -> 2026-06, which is
the Sprint 7 holdout. That holdout is SPENT, which bars re-EVALUATING
there; it does not bar training. The forward window (2026-09-30 ->
2028-09-30) sits strictly after it, so there is no leakage into what
Sprint 9B will actually be judged on, and nothing is lost that was not
already spent. Ruled acceptable by Aman on 2026-09-09. Write it into
sprint9b_preregistration.json so a later reader does not mistake it for
a broken rule.

TASK C — Re-freeze.
Before freezing, CONFIRM `git status --porcelain --untracked-files=all`
is empty for src/ and scripts/. Three artifacts in this sprint were
frozen or committed while the code that produced them was still loose;
do not make it four. Then freeze a vintage capturing prices-with-XLK,
the regenerated labels, and the news feature parquet.

Output: the guard re-run with two benchmarks, the XLK reconciliation
with per-month deltas, the label regeneration report with attribution,
the fold arithmetic, and the freeze manifest. State whether Prompt 4 is
now unblocked.
```

---

## PROMPT 4 of 6 — Retrain, freeze, and the reference-only diagnostic

```
You are continuing Sprint 9B at /Users/aman/dev/Ai Trading Agent.
Prompt 4 of 6. Prompt 3 delivered the 24-feature matrix. REPO GUARD
first, and re-assert the production pickle md5 before you start.

READ THIS BEFORE ANYTHING ELSE: per ruling R5, this prompt does NOT
take a verdict. Nothing you compute here can accept or reject the news
feature. You are building the artifact that a future session will judge,
and running one diagnostic whose results are marked ineligible. If you
find yourself writing the words "PASS" or "FAIL" about the 2023-24 fold,
you have misread the sprint.

RESTORE DISCIPLINE: live trades read models/ensemble_models.pkl. All
sprint outputs go to versioned names. The FINALLY block at the end runs
in EVERY branch including crashes — if a session dies mid-run, the
recovery session runs it FIRST, before anything else.

1. Train TWO arms, identically (R6 amendment, 2026-09-09). Same window
   2022-01-01 -> 2026-08-31, same folds, same hyperparameters,
   WINDOW_YEARS None (Sprint 6 REFUTED the rolling config). The ONLY
   difference between them is the presence of the news column.
     base23 -> models/ensemble_models_sprint9b_base.pkl   (23 features)
     news24 -> models/ensemble_models_sprint9b_news.pkl   (24 features)
   Build base23 by dropping news_sentiment from the labeled parquet, not
   by reusing any earlier pickle — an arm trained on a different window
   is not a control. Record both md5s. Confirm feature_names is 23 and
   24 respectively and that news_sentiment appears in exactly one of
   them. Confirm both report the same fold count and the same
   effective_train_end per fold; if they differ, the arms are not paired
   and you must find out why before continuing.

2. NO 2023-24 REFERENCE BACKTEST. It was in REV 1 and is deliberately
   removed. Under AV-only training the 2023 test fold would train on
   about twelve months of 2022, so the run would measure the thinness of
   the training window rather than anything about news. Running it
   anyway would spend a 7th experiment on that fold to learn nothing.
   Instead, three cheap sanity checks that do NOT touch an equity curve:
     - both arms train to completion and produce the expected fold count
     - news_sentiment has non-zero mean importance in news24
     - in-sample monthly cross-sectional rank IC of news_sentiment over
       2022-01 -> 2026-08, reported with mean, sd, n and t
   If news_sentiment has literally zero importance across every fold,
   say so prominently — that is a STOP finding, not a footnote.

3. News diagnostics from the new pickle:
   - news_sentiment deflated-t, and its gain rank out of 24
   - where the 5 lowest-t features now sit
   - note explicitly that the deflator sqrt(2*ln(n_features)) moved from
     2.5043 (23 features) to 2.5219 (24), so every feature's deflated-t
     shifted slightly and prior sprints' values are not directly
     comparable
   - monthly cross-sectional Spearman rank IC of news_sentiment against
     next-month forward return, across the whole 2015-2024 span, split by
     era (pre-2022 / 2022-23 / 2024). Report mean IC, std, and t. This
     is the statistic the forward window will lean on, so establish its
     in-sample scale now.

4. Breadth diagnostic: monthly count of names scoring above MIN_SCORE
   0.52 under the 24-feature model vs the production model, same months.
   A breadth-driven Sharpe move is a confound, not a result — report it
   either way. Recall breadth above 0.52 has been collapsing: Dec 18 ->
   Apr 24 -> Jun 3 -> Jul 9 -> Aug 2.

5. BUILD THE SHADOW HARNESS. This is the real deliverable of this
   prompt, and it long outlives this sprint — the analyst-funnel shadow
   stage (the project's actual end goal) will reuse it. Design it for
   N ARMS, not two.

   An "arm" is a named scoring strategy: {name, description, score_fn}
   where score_fn(feature_matrix_slice, month_end) -> Series indexed by
   ticker. Sprint 9B registers THREE (R6 amendment, 2026-09-09):
     "base23"  — models/ensemble_models_sprint9b_base.pkl, 2022+ window.
                 THIS IS L1's BASELINE.
     "news24"  — models/ensemble_models_sprint9b_news.pkl, 2022+ window.
                 Identical to base23 but for the one column.
     "prod23"  — the untouched production pickle. CONTEXT ONLY, never a
                 verdict leg, because it trains on a different window.
                 Mark it in the JSON with "verdict_eligible": false so a
                 later reader cannot mistake it for the control.
   A future sprint will register a fourth whose score_fn calls the Claude
   analyst instead of a pickle. Do NOT hard-code the arm list, do not
   name the JSON keys after pickles, and do not assume an arm's score
   comes from a model file at all.

   For each month-end the harness scores every registered arm against
   the SAME feature matrix and writes
   backtests/results/shadow/<YYYY-MM>.json with an arm-keyed structure:
     {"month_end": ..., "feature_matrix_sha256": ...,
      "arms": {"<name>": {"scores": {...}, "weights": {...},
                          "selected": [...], "n_selected": int,
                          "hypothetical_return": float|null}}}
   The forward return is filled in on the FOLLOWING month's run, never
   estimated at write time.

   Constraints: runnable unattended each month; NEVER submits an order;
   never writes to models/ensemble_models.pkl; records the feature
   matrix hash so a later reader can prove all arms saw identical
   inputs. Dry-run on 2026-08-31 and show both arms' score vectors side
   by side.

6. FINALLY (runs in every branch): cp
   models/ensemble_models_sprint5.pkl models/ensemble_models.pkl;
   restore data/processed/_pre_sprint9_backup/ parquets (verify against
   its SHA256SUMS.txt); then PROVE restoration three ways — pickle md5
   == 296e589f4da205eb1d171c2121d90f82, a live scorer smoke call, and a
   rerun that reproduces test Sharpe 0.9898 +/- 0.001 on the frozen
   vintage from Prompt 0.

NO DEPLOYMENT. The production book keeps trading the 23-feature model
regardless of anything in this prompt.

Output: new pickle md5 + feature count, the reference-only table clearly
labelled as such, news diagnostics including the era-split rank IC,
breadth comparison, shadow harness dry-run, and the three-way
restoration proof. State you are ready for Prompt 5.
```

---

## PROMPT 5 of 6 — Pre-register the forward window and the rule

```
You are continuing Sprint 9B at /Users/aman/dev/Ai Trading Agent.
Prompt 5 of 6. Prompt 4 froze models/ensemble_models_sprint9b_news.pkl
and built the shadow harness. REPO GUARD first.

This prompt writes a pre-registration document. Its whole value is that
it is written BEFORE the data exists. Write it, commit it (via Aman, per
Prompt 6), and do not edit it afterwards — an amended pre-registration
is not a pre-registration.

Write backtests/results/sprint9b_preregistration.json AND a human-
readable Sprint9B_Preregistration.md containing:

── THE WINDOW ─────────────────────────────────────────────────────────
Start: the first month-end AFTER the frozen pickle's md5 is recorded —
2026-09-30, scored 2026-09-30, first shadow trade date 2026-10-01.
End: 2028-09-30 (24 monthly observations).
Interim read: 2027-09-30 (12 observations), FAIL-FAST ONLY — see below.

WHY 24 MONTHS, STATED HONESTLY.

  Sharpe precision. For an annualised Sharpe S estimated from n monthly
  returns, the standard error is approximately
        SE(S) = sqrt( (12 + S^2 / 2) / n )
  Word for word: take 12 (the periods-per-year scaling), add half the
  square of the Sharpe you are estimating, divide by the number of
  monthly observations, take the square root. At S = 1.0:
        n = 12:  sqrt((12 + 0.5) / 12) = sqrt(1.04167) = 1.021
        n = 24:  sqrt((12 + 0.5) / 24) = sqrt(0.52083) = 0.722
  A standard error of 0.72 around a Sharpe estimate is enormous. Sharpe
  alone cannot settle this question at any window length available here.

  Rank IC precision. The design therefore leans on the monthly
  cross-sectional rank IC, which has 53 names behind each of n monthly
  observations. Its t-statistic is
        t = mean(IC) * sqrt(n) / sd(IC)
  With a typical monthly IC standard deviation near 0.20 and a mean IC
  of 0.04 (a realistic figure for news sentiment in the published
  literature):
        n = 12:  t = 0.04 * 3.464 / 0.20 = 0.693
        n = 24:  t = 0.04 * 4.899 / 0.20 = 0.980
  So 24 months roughly doubles the Sharpe precision and lifts a
  realistic IC to the L2 bar of 1.0 — but only just.
SAY THIS PLAINLY IN THE DOCUMENT: even at 24 months this design can only
detect a LARGE news effect. It cannot distinguish "small real effect"
from "no effect". That is a limitation of having 53 names and a monthly
rebalance, not something a longer window inside this project fixes.
The bars below are set as SCREENING bars accordingly, and a PASS means
"worth a deployment prompt", never "demonstrated alpha".

── THE RULE (ruling R6), FIXED NOW ────────────────────────────────────
Over the pre-registered window, PASS requires ALL FOUR:

  L1  SPY LEG, PAIRED. ExcessSharpe(news24) > ExcessSharpe(base23),
      where ExcessSharpe = strategy annualised Sharpe minus SPY
      annualised Sharpe over the IDENTICAL window, SPY read from the
      `prices` table. Both arms are simulated from the same shadow
      records and were trained on the same 2022+ window with the same
      folds, so this is a paired comparison differing in exactly one
      feature column — which is the only reason 24 monthly observations
      can say anything at all.
      THE BASELINE IS base23, NOT prod23 (R6 amendment 2026-09-09).
      prod23 trains on the full history; comparing against it would
      confound "news" with "more training data". Report prod23's numbers
      alongside for context, clearly marked as not a verdict leg.
      NOTE: this leg deliberately does NOT require beating SPY. Sprint
      5's -0.876 excess Sharpe was measured on 2023-24 and does not
      transfer. A PASS on L1 means "adding the news column improved the
      SPY-excess Sharpe of an otherwise identical model", NOT "the
      strategy beats the index" and NOT "news24 should replace
      production". Report the absolute gap to SPY alongside, always.

  L2  NEWS IC LEG. Mean monthly cross-sectional Spearman rank IC of
      news_sentiment vs next-month forward return > 0, with
      t = mean(IC) * sqrt(n) / sd(IC) >= 1.0.

  L3  IMPORTANCE-STABILITY GATE. news_sentiment deflated-t >= 1.0 in
      the frozen 24-feature pickle. Necessary, not sufficient. Label it
      in the document as what it is: a measure of whether the model kept
      splitting on the column across folds, NOT a measure of whether the
      column made money.

  L4  NO DRAWDOWN REGRESSION. MaxDD(news24) >= MaxDD(base23) - 5.0pp
      over the window. Baseline is base23, same amendment as L1.

WHAT THIS SPRINT DOES NOT ASK. Because both arms train on 2022+ only,
neither resembles production, and a PASS says nothing about whether
news24 should be deployed. The question answered is narrow and worth
stating in one sentence in the document: does adding a news-sentiment
column improve an otherwise identical model over the forward window?
"Should we deploy it" is a separate question needing its own sprint,
its own window and its own pre-registration.

FAIL if any leg misses. There is no partial credit and no "directionally
encouraging" outcome.

── FAIL-FAST AT 12 MONTHS ─────────────────────────────────────────────
At 2027-09-30, evaluate all four legs on the 12 observations available.
If ALL FOUR miss, stop the experiment early and record FAIL. If any leg
is met, continue to 24 months and do not report an interim verdict. This
is asymmetric on purpose: early stopping for futility costs nothing,
early stopping for success is how a spent window becomes six spent
windows.

── WHAT IS FROZEN ─────────────────────────────────────────────────────
Record, so a future session can prove nothing moved:
  - md5 of models/ensemble_models_sprint9b_news.pkl  (news24)
  - md5 of models/ensemble_models_sprint9b_base.pkl  (base23, the L1
    baseline — without this the comparison cannot be reproduced)
  - md5 of models/ensemble_models.pkl (production, 296e589f...)
  - the latest vintage path + MANIFEST sha256 + news_vintage digest.
    Use a vintage frozen AFTER the feature parquet exists, and confirm
    `git status --porcelain --untracked-files=all` is empty for src/ and
    scripts/ BEFORE freezing — three artifacts in this sprint were
    frozen or committed while the code that produced them was still
    loose.
  - THAT THE STITCH WAS ABANDONED: source = alphavantage only, from
    2022-01-01, and the Prompt 2 numbers that ruled it out (rho 0.2078,
    CI [0.113, 0.297], |dmean| 1.29 x xsd, STEP 4 unexecutable)
  - k = 10, EWM span = 4, shrink -> EWM -> cross-sectional z (R3-A)
  - the training window 2022-01-01 -> 2026-08-31, identical for both
    arms, and the fold count each produced
  - git HEAD sha at pre-registration

⚠ A NOTE ON THE MANIFEST'S git_dirty FIELD. It was fixed in 0380363 to
ignore untracked files, so a manifest frozen after that commit means
what it says. Any OLDER manifest cited here reads dirty=true purely
because this repo carries ~13 permanently-untracked paths, and needs
that caveat attached. The manifest's hard-coded `purpose` string may
still describe the shelved Markov diagnostic — fix that before it is
recorded for two years.

── MULTIPLE-TESTING LEDGER ────────────────────────────────────────────
State: the 2023-24 fold has been touched ~6 times and is retired from
verdict duty for this project — Sprint 9B deliberately did NOT spend a
7th on it (the REV 1 reference backtest was removed, see Prompt 4 step
2). The 2025-26 holdout is SPENT. This forward window is experiment #1
on genuinely unseen data, and it gets ONE evaluation at 24 months plus
one futility check at 12. No variant sweeps, no re-runs with a different
k, no re-scoring with a different sentiment model, no re-opening the
stitch. If any of those become desirable, they are a NEW pre-registration
on a NEW window, not an amendment to this one.

Record honestly that the pre-registration itself was amended three times
before the window opened — R4 on 2026-09-08 (per-cell z-scoring,
retargeted RULE T), then on 2026-09-09 R4 superseded outright, R3
amended to add cross-sectional standardisation, and R6/L1 repointed at
base23. Every one of those responded to a property of the INPUTS with
the dependent variable unseen, and each is dated in this file with its
reasoning. That is the standard being claimed, and a reader should be
able to check it against the git history rather than take it on trust.

── OPERATIONS ─────────────────────────────────────────────────────────
Specify a scheduled task that runs the shadow harness monthly, one day
after the live rebalance (the live task fires the 2nd of each month at
10:00 ET), writes backtests/results/shadow/<YYYY-MM>.json, and emails a
one-line summary. It must not be able to submit orders. Do NOT create
the scheduled task in this prompt — specify it, and let Aman create it
after he has read the specification.

The task must run whatever arms are registered, not a fixed two, so
that adding the analyst arm later needs no change to the schedule. Note
in the specification that a new arm joining mid-window is NOT part of
this pre-registration and gets its own — the L1-L4 rule above is a
statement about news24 vs prod23 and nothing else.

Output: both files, and a plain-English summary of what will be true in
September 2028 under each outcome. State you are ready for Prompt 6.
```

---

## PROMPT 6 of 6 — Prepare the commit (COMMANDS FOR AMAN — agent does NOT commit)

```
You are finishing Sprint 9B at /Users/aman/dev/Ai Trading Agent.
Prompt 6 of 6. The agent does NOT run git. Print literal copy-paste
commands for Aman and stop.

1. Inventory what changed. Use `git --no-optional-locks status
   --porcelain` and `git --no-optional-locks diff --stat` (the
   --no-optional-locks flag matters: the Cowork mount grants read/write
   but not delete, so a git command that takes .git/index.lock leaves a
   0-byte lock corpse that cannot be removed and the next local `git
   add` fails with "Another git process seems to be running").
2. Separate what SHOULD be committed from what should not:
   COMMIT: src/ and feature_matrix changes, the news scoring module, the
   shadow harness, Sprint9B_Prompts.md, Sprint9B_Preregistration.md,
   backtests/results/sprint9b_*.json.
   DO NOT COMMIT: data/quant_research.db (832 MB), the vintage snapshot
   directory, any .pkl, any parquet under data/processed/.
   Verify .gitignore already covers these; if it does not, print the
   .gitignore lines to add as part of the commit.
3. Print the exact commands. One commit. Subject line under 72
   characters, body explaining the four rulings R3-R6 and why the
   verdict is deferred.
4. NO Co-Authored-By trailer. Five commits were filter-branched in July
   to strip them; do not reintroduce one.
5. Remind Aman that `main` is the working branch, that
   prototype/markov-exit-layer has permanently diverged (no fast-forward
   in either direction), and that ExitFix_Prompt.md and
   BreadthCap_Prompt.md are still outstanding ahead of the 2026-10-01
   rebalance.

Output: the inventory, the include/exclude split with reasons, and the
literal command block.
```

---

## Usage notes

- Prompts 0, 2 and 5 are the ones most likely to need a decision from Aman
  mid-flight: Prompt 0 if the SPY ingest touches something that derives the
  universe from `prices`, Prompt 2 if the calibration lands on `rho < 0.20`
  (STOP), and Prompt 5 because it fixes a rule for two years.
- Prompt 1 is the only long-running compute job. If its 500-row extrapolation
  says more than 8 hours, stop and reconsider batching before committing a night
  to it.
- Every prompt begins with the repo guard for a reason: the project has three
  dead paths (`~/Desktop/Stock_Project/Ai Trading Agent`,
  `~/Projects/Ai Trading Agent`, `~/Desktop/Ai Trading Agent`), and Cowork
  sessions may still have two of them attached as folders.
- Standing lesson from 2026-09-02, demonstrated three times in one day:
  **`filled` is not `correct`, `ok` is not `worked`, no-error is not success.**
  Verify resulting state, never the absence of a complaint. Prompt 3's
  identical-index proof and Prompt 4's three-way restoration proof exist because
  of that lesson.
