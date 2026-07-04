# Dashboard Redesign — Claude Design Prompts
Target tool: **Claude Design** (claude.ai/design or the desktop sidebar), NOT a coding agent.
Unlike the Sprint prompts, these run sequentially **inside ONE Claude Design project** —
Prompt 0 sets up the design system, Prompt 1 generates the full prototype, Prompts 2–4 are
refinement passes on specific views, Prompt 5 is the export. Refine freely between prompts
with inline comments and sliders; the numbered prompts are the structural checkpoints.

Division of labor (important): Claude Design owns LOOK and LAYOUT only. Its generated JS
uses hardcoded mock data — do not ask it to wire real data loaders, and do not trust its
chart math. The exported HTML comes back to the Cowork session for integration
(export_data.py extensions + real JSON wiring + verification against the tracked CSVs).

Files to have ready for upload:
- `src/dashboard/index.html`  (design system source — Prompt 0)
- `backtests/results/sprint5_results.json` and `sprint6_results.json`  (Prompt 1 & 3)
- Optional: `src/dashboard/data/universe.json` (real tickers/sectors for the holdings view)
- Do NOT upload prices.json (2 MB, unnecessary — the prompts contain real numbers)

---

## PROMPT 0 — Register the design system

Attach: `src/dashboard/index.html`

```
Create a design system from the attached index.html. It is an existing
quant research dashboard with a GitHub-dark aesthetic. Extract and
register exactly these tokens — do not invent replacements:

Colors: background #0d1117, surface #161b22, surface-alt #1c2333,
border #30363d, text #c9d1d9, text-muted #8b949e, accent #58a6ff,
accent-alt #7ee787, red #f85149, orange #d29922, purple #bc8cff,
cyan #39d2c0, green #3fb950.
Typography: Inter (300-700) for UI, JetBrains Mono (400/500) for numbers,
tickers, and code-like values. All financial figures render in the mono
font.
Shape: 8px radius (12px for large cards), subtle shadows
(0 2px 12px rgba(0,0,0,0.4)), 1px borders in #30363d, 0.2s ease
transitions.
Components to extract: the sticky header with gradient logo tile, the
tab-button row, stat cards, and chart panel containers.

Name the design system "Quant Research Platform". Every subsequent
screen in this project must use it.
```

---

## PROMPT 1 — Full dashboard prototype (all views)

Attach: `sprint5_results.json`, `sprint6_results.json`
Template: Prototype. Design system: Quant Research Platform.

```
Design a single-page quant research dashboard using the Quant Research
Platform design system. Desktop-first (1440px), dark only. It extends
an existing dashboard, so keep the same skeleton: sticky header with
logo + title "Quant Research Platform" and a data-freshness timestamp
on the right; below it a horizontal tab row. Five tabs:

TAB 1 — "Performance" (default):
- Large line chart: cumulative return (%) 2018-2024 with five series —
  Ensemble (accent blue #58a6ff, 3px, visually dominant), Quant,
  Fundamental (thinner, muted), XLK benchmark (orange), SPY (purple).
  Realistic shapes: sharp COVID dip Mar 2020, grinding 2022 bear
  (ensemble bottoms near -30%), steep 2023-24 AI-boom recovery.
- Vertical dashed marker at Jan 2023 labeled "TRAIN | TEST".
- Shaded vertical bands behind the series: red-tinted
  (rgba(248,81,73,0.08)) for RISK_OFF months (cluster them across 2022
  and scattered in 2020), green-tinted (rgba(63,185,80,0.08)) for
  RISK_ON stretches in 2023-24. A small legend chip row explains the
  bands: "Regime gate: risk-off 0.5x / neutral 1.0x / risk-on 1.2x".
- Stat card row above the chart: CAGR 13.60%, Sharpe 0.682, Test Sharpe
  0.990, 2022 Max DD -29.78%, each with a small delta-vs-baseline
  sub-label (e.g. "+0.082 vs baseline").

TAB 2 — "Sprint History":
- A comparison table across experiment sprints, columns = Sprint 0
  (Baseline), Sprint 4 (Graded gate), Sprint 5 (TimesFM), Sprint 6
  (Rolling window); rows = Full CAGR, Full Sharpe, Test CAGR, Test
  Sharpe, 2022 Max DD. Use the real values from the two attached JSON
  files. Best value per row subtly highlighted in green.
- Verdict badges per sprint column header: Sprint 4 "FAIL" (red pill),
  Sprint 5 "PASS" (green pill), Sprint 6 "REFUTED" (red pill).
- Below the table, one expandable "verdict notes" card per sprint
  showing the verdict_notes text from the JSONs (collapsed by default,
  JetBrains Mono, muted).

TAB 3 — "Holdings":
- Month selector (dropdown or horizontal scrubber, default 2024-11).
- Grid of the top-20 holdings for the selected month: ticker (mono,
  large), company name (muted, truncated), model score as both number
  (0.5-0.7 range) and a thin horizontal score bar, equal weight badge
  "5.0%". Use real tech tickers: NVDA, MSFT, AVGO, META, AMD, CRM,
  NOW, PANW, ANET, KLAC, CDNS, SNPS, ADBE, AMAT, LRCX, MU, ORCL,
  FTNT, CSCO, TXN.
- A compact "regime this month" chip: NEUTRAL 1.0x.

TAB 4 — "Breadth":
- Bar chart: one bar per month 2018-2024, height = number of names
  scoring above the 0.52 model threshold (range ~0-45, average ~25,
  visibly thinner in stress months). Horizontal reference lines at 20
  ("max portfolio size") and 10 ("starvation warning" — label it).
  Months with zero names get a small red dot marker at the baseline.
- Caption card explaining why breadth matters: "Months where no name
  clears the 0.52 threshold leave the strategy in cash. Breadth
  starvation, not ranking skill, sank the Sprint 6 rolling-window
  experiment."

TAB 5 — "Macro & Regime":
- Two stacked charts sharing an x-axis: VIX (with a dashed threshold
  line at 25 and a dotted one at 20) and the 10y-2y yield spread (with
  a zero line; shade the sub-zero inversion stretch of 2022-24).
- The same red/green regime bands as Tab 1 so the rule's behavior is
  visually traceable to its inputs.

Global: numbers in JetBrains Mono; generous 24px card padding; charts
get panel containers with 1px borders; no build-step-dependent
components — this must be exportable as a single standalone HTML file.
```

---

## PROMPT 2 — Refinement: Performance tab

Run after inspecting Prompt 1's output. Edit the bracketed judgment calls first.

```
Refine the Performance tab:
1. The regime bands must sit BEHIND the line series at low opacity —
   they currently [compete with / sit above] the lines.
2. Add a drawdown subplot (25% height) under the main chart, sharing
   its x-axis: underwater curve for the Ensemble only, filled red at
   8% opacity, with the -29.78% floor of 2022 annotated.
3. The TRAIN | TEST marker label should sit at the top of the plot
   area, small caps, muted — not overlapping any series.
4. Stat cards: add hover tooltips spelling out definitions ("Sharpe =
   daily mean/std, annualized sqrt(252)").
5. Add a range selector row: 1Y / 3Y / Train / Test / All chips,
   Test selected by default.
```

---

## PROMPT 3 — Refinement: Sprint History tab

```
Refine the Sprint History tab:
1. Add a small sparkline column to the table: each sprint's test-period
   equity shape in miniature (Sprint 4 flat-weak, Sprint 5 steep,
   Sprint 6 stair-stepped with flat cash stretches).
2. Add a "What changed" row above the metrics: one short phrase per
   column — "Frozen baseline", "Graded regime gate", "+TimesFM 23rd
   feature", "Rolling 36-month window".
3. The Sprint 6 column needs a footnote glyph on its Test Sharpe cell
   (1.041, which BEAT Sprint 5) linking to a note: "Cash artifact —
   14 of 23 test months held zero positions; see Breadth tab."
4. Verdict pills: add the deciding number under each ("Test CAGR
   6.77% < bar", "163% gap recovery", "Test CAGR 14.81% < 18.27%").
```

---

## PROMPT 4 — Refinement: Holdings + Breadth tabs

```
Refine Holdings and Breadth:
1. Holdings: add a sector-color left border on each holding card
   (semis cyan, software blue, hardware purple, other muted) and a
   sector mini-legend.
2. Holdings: month scrubber should show a tiny breadth indicator under
   each month tick (echo of the Breadth chart) so starved months are
   visible while scrubbing; months with zero holdings render an empty
   state: "Strategy in cash — no names above threshold" with a muted
   cash icon.
3. Breadth: color bars by regime of that month (red/green/neutral
   tint), keep the 20 and 10 reference lines crisp.
4. Add cross-navigation: clicking a starved month in Breadth jumps to
   that month in Holdings.
```

---

## PROMPT 5 — Export

```
Export this prototype as a single standalone HTML file with all CSS
and JS inline. Requirements for the handoff: keep all mock datasets in
clearly named const arrays at the top of the script block (PERF_DATA,
SPRINT_DATA, HOLDINGS_DATA, BREADTH_DATA, MACRO_DATA) so a coding
agent can swap them for fetch() calls against real JSON files without
touching the rendering code. No external dependencies except Plotly
from cdn.plot.ly and Google Fonts.
```

---

## HANDOFF — back to the Cowork session (not Claude Design)

Bring the exported HTML file into the Cowork knowledge-base chat. The integration work
there, in order:
1. Extend `src/dashboard/export_data.py` with the missing exports: ensemble equity
   (from `backtests/results/ensemble_timesfm_*.csv`), sprint comparison (from the tracked
   `sprint*_results.json`), monthly holdings + scores (from `ensemble_scores.json` logic),
   breadth per month (names above `MIN_SCORE=0.52`), regime multiplier series (from
   `regime_gate.get_historical_regime_multipliers`).
2. Replace the prototype's mock const arrays with fetch() loaders against
   `src/dashboard/data/*.json`, preserving the existing no-build-step constraint.
3. Verify programmatically: chart endpoints must match the tracked equity CSVs; the
   sprint table must match the results JSONs; breadth counts must match the Sprint 6
   diagnostic (rolling mean 17.9 vs expanding 24.6 was the reference computation).
4. Keep the old dashboard reachable until the new one passes verification; commit as a
   single "Dashboard v2" change with before/after screenshots.
