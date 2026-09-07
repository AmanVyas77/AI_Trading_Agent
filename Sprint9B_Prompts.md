# Sprint 9B — News Sentiment as the 24th Feature (modelling half) — REV 1

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

**R3 (new — aggregation).** Shrinkage toward the contemporaneous cross-sectional
mean, weight `n/(n+k)`, **k = 10 fixed a priori**. No article-count floor. No NaN
ever emitted. At n=0 the formula returns the cross-sectional mean directly, so
`_prepare_xy`'s and `live/scorer.py`'s `fillna(0.0)` never fires on this column
and **`_prepare_xy` is not modified** — the frozen 23 features keep their exact
current treatment and comparability with prior sprints is preserved.
*Accepted cost, recorded deliberately:* "no news" and "exactly average news"
become indistinguishable. *Rejected:* a `has_news` indicator — the missingness
rate runs 21.6% → 5.5% → 0% across eras, so such an indicator would hand the
model a clean calendar variable. Report realized shrinkage weight and effective
n per era.

**R4 (new — assembly rule is decided by measurement, against a table fixed in
advance).** The REV 2 rule (`fnspid ≤ 2021-12`, `alphavantage ≥ 2022-01`) is
**suspended**, not adopted. Prompt 2 runs the calibration and selects the rule
from the pre-committed decision table in that prompt. The table is fixed *now*,
before any number is seen, so the choice is evidence-driven rather than
outcome-driven.

**R5 (new — judging window).** Sprint 9B does **not** take a verdict on the
2023-01 → 2024-12 test fold. That fold has judged ~6 experiments and it straddles
the corpus's worst density break. Sprint 9B **builds and freezes** the 24-feature
model and **pre-registers a forward window** whose density is uniform. The
2023-24 fold is run once as a smoke diagnostic, recorded with
`verdict_eligible: false`, and may never be cited to accept the feature.

**R6 (new — verdict rule with an SPY leg).** The rule is stated in full in
Prompt 5. Its SPY leg is a **paired** comparison of SPY-excess Sharpe between the
24-feature model and the 23-feature production model **over the same forward
window**, not a comparison against Sprint 5's −0.876 (which was measured on a
different window and does not transfer).

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

## PROMPT 2 of 6 — Seam calibration, and the rule it selects

```
You are continuing Sprint 9B at /Users/aman/dev/Ai Trading Agent.
Prompt 2 of 6. Prompt 1 scored the corpus into news_sentiment_scores.
REPO GUARD first.

This prompt DECIDES the assembly rule (ruling R4). The decision table
below was fixed before any number was seen. Apply it mechanically. Do
not improvise a fifth outcome, and do not adjust a threshold because a
statistic landed just outside it — if that happens, report it and STOP.

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
      -> RULE U-z. Union, but first z-score EACH source's monthly
         aggregates within its OWN history (expanding, backward-only —
         never a full-sample z-score, which would be lookahead), then
         pool the standardised values weighted by article count.

  0.20 <= rho < 0.50
      -> RULE D. Denser source per era: FNSPID through 2023-12,
         alphavantage from 2024-01. Do NOT blend. Record that the seam
         and the density cliff then coincide at 2024-01 and are no
         longer separable.

  rho < 0.20
      -> STOP. The two sources are not measuring the same quantity. The
         stitch premise fails. Report the numbers and end the session —
         do not pick a rule.

STEP 4 — Isolate the text-shape confound.
Repeat STEP 2 twice more, splitting the FNSPID side by text_shape:
  rho_snippet  = rho computed on pairs where the FNSPID side is title_snippet
  rho_title    = rho computed on pairs where the FNSPID side is title_only
If rho_snippet >= 0.50 while rho_title < 0.20, the seam is a TEXT-SHAPE
seam, not a source seam. In that case RULE T applies on top of whichever
rule STEP 3 selected: exclude title_only articles from feature
aggregation entirely (they stay in the DB, they just do not feed the
mean), and re-run STEP 2 and STEP 3 on the filtered panel to confirm the
selected rule is stable. Report the article loss this causes, by year.

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

  news_sentiment(i,m) is then the per-ticker exponentially weighted
  moving average of the shrunk series over months, span = 4 (the Arratia
  convention this repo already uses in _ewm_smooth for the 8-K and LM
  columns). SHRINK FIRST, SMOOTH SECOND. Smoothing first and shrinking
  second would apply a single month's article count to a value that
  already mixes four months.

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

1. Rebuild the training matrix: --start 2015-01-01 --end 2024-12-31.
   Assert 24 feature columns. Assert the (date,ticker) index matches the
   pre-change matrix exactly. Assert news_sentiment has ZERO NaN.
2. Report, per year: mean realized shrinkage weight n/(n+k), median
   effective n, and the fraction of ticker-months that fell back
   entirely to xsec (n=0). This is the R3 honesty report — it shows
   exactly how much of the early history is prior rather than evidence.
3. Hand-trace 3 tickers x 3 months end to end (article count -> raw ->
   xsec -> shrunk -> EWM) and show every intermediate number, as in the
   worked example above.
4. target_builder: confirm label months/rows are UNCHANGED at 117 /
   5,974. If they moved, something upstream changed and you must find
   out what before continuing.
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

1. Train. model_trainer on the 24-feature labeled parquet, WINDOW_YEARS
   must be None (Sprint 6 REFUTED the rolling config). Save to
   models/ensemble_models_sprint9b_news.pkl. Record the md5.
   Confirm feature_names has 24 entries and news_sentiment is among them
   — _get_feature_cols picks it up automatically, so if it is missing,
   the matrix did not save what you think it saved.

2. Reference-only diagnostic on 2023-01 -> 2024-12. Run
   score_generator -> portfolio_builder -> backtest. Copy outputs to
   backtests/results/ensemble_news_ref_{train,test}_equity.csv. Compute
   the usual metrics AND the SPY-relative ones (SPY now comes from the
   `prices` table per Prompt 0, not yfinance).
   Every one of these numbers is written to the results JSON under a
   block literally named "reference_only_not_verdict_eligible": true.
   Write one sentence in the JSON explaining why: the fold has judged ~6
   prior experiments, and the corpus's median density inside it runs 21
   in 2023 and 9 in 2024, so the feature does not mean the same thing
   across the fold.

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
   ticker. Sprint 9B registers exactly two:
     "prod23"  — production 23-feature pickle
     "news24"  — models/ensemble_models_sprint9b_news.pkl
   A future sprint will register a third whose score_fn calls the Claude
   analyst instead of a pickle. Do NOT hard-code two pickles, do not
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

  L1  SPY LEG, PAIRED. ExcessSharpe(news24) > ExcessSharpe(prod23),
      where ExcessSharpe = strategy annualised Sharpe minus SPY
      annualised Sharpe over the IDENTICAL window, SPY read from the
      `prices` table. Both books are simulated from the same shadow
      records, so this is a paired comparison on shared market
      exposure — which is the only reason 24 monthly observations can
      say anything at all.
      NOTE: this leg deliberately does NOT require beating SPY. Sprint
      5's -0.876 excess Sharpe was measured on 2023-24 and does not
      transfer to a different window; the baseline here is the
      production model measured on THIS window. A PASS on L1 therefore
      means "news narrowed the gap to the index", not "the strategy
      beats the index". Report the absolute gap alongside it, always.

  L2  NEWS IC LEG. Mean monthly cross-sectional Spearman rank IC of
      news_sentiment vs next-month forward return > 0, with
      t = mean(IC) * sqrt(n) / sd(IC) >= 1.0.

  L3  IMPORTANCE-STABILITY GATE. news_sentiment deflated-t >= 1.0 in
      the frozen 24-feature pickle. Necessary, not sufficient. Label it
      in the document as what it is: a measure of whether the model kept
      splitting on the column across folds, NOT a measure of whether the
      column made money.

  L4  NO DRAWDOWN REGRESSION. MaxDD(news24) >= MaxDD(prod23) - 5.0pp
      over the window.

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
  - md5 of models/ensemble_models_sprint9b_news.pkl
  - md5 of models/ensemble_models.pkl (production, 296e589f...)
  - the Prompt 0 vintage path + MANIFEST sha256 + news_vintage digest
  - the selected assembly rule from Prompt 2 and the rho/dmean that
    selected it
  - k = 10, EWM span = 4, and the shrink-then-smooth order
  - git HEAD sha at pre-registration

── MULTIPLE-TESTING LEDGER ────────────────────────────────────────────
State: the 2023-24 fold has now been touched ~7 times and is retired
from verdict duty for this project. The 2025-26 holdout is SPENT. This
forward window is experiment #1 on genuinely unseen data, and it gets
ONE evaluation at 24 months plus one futility check at 12. No variant
sweeps, no re-runs with a different k, no re-scoring with a different
sentiment model. If any of those become desirable, they are a NEW
pre-registration on a NEW window, not an amendment to this one.

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
