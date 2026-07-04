/* ═══════════════════════════════════════════════════════════════════════
 *  Quant Research Platform — Dashboard v2
 *  ─────────────────────────────────────────
 *  Pure JavaScript, no build step. Plotly.js from CDN.
 *  Layout/design ported from the Claude Design prototype (2026-07-03);
 *  all data loaded from src/dashboard/data/*.json (export_data.py).
 * ═══════════════════════════════════════════════════════════════════════ */

'use strict';

/* ── Palette ─────────────────────────────────────────────────────────── */
const C = {
  bg: '#0d1117', surface: '#161b22', surfaceAlt: '#1c2333', border: '#30363d',
  text: '#c9d1d9', muted: '#8b949e', accent: '#58a6ff', accentAlt: '#7ee787',
  red: '#f85149', orange: '#d29922', purple: '#bc8cff', cyan: '#39d2c0',
  green: '#3fb950',
  redBand: 'rgba(248,81,73,0.08)', greenBand: 'rgba(63,185,80,0.08)',
};

const AXIS = {
  gridcolor: '#21262d', zerolinecolor: '#30363d', linecolor: '#30363d',
  tickcolor: '#30363d', tickfont: { color: C.muted, size: 11 },
};

const PLOT_BASE = {
  paper_bgcolor: C.bg, plot_bgcolor: C.bg,
  font: { color: C.text, family: "'Inter', sans-serif", size: 12 },
  legend: { bgcolor: 'rgba(0,0,0,0)', font: { size: 11 }, x: 0.01, y: 0.99,
            xanchor: 'left', yanchor: 'top' },
  hoverlabel: { bgcolor: C.surface, bordercolor: C.border,
                font: { color: C.text, size: 12 } },
  margin: { t: 36, r: 24, b: 48, l: 64 },
};

const PLOT_CONFIG = { responsive: true, displaylogo: false,
  modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'] };

/* Industry string → sector color (Holdings cards) */
function sectorColor(industry) {
  const s = (industry || '').toLowerCase();
  if (s.includes('semiconductor')) return C.cyan;
  if (s.includes('software') || s.includes('application') ||
      s.includes('infrastructure') || s.includes('electronic gaming')) return C.accent;
  if (s.includes('network') || s.includes('security') ||
      s.includes('communication equipment')) return C.purple;
  return C.orange;
}

/* ── Data store ──────────────────────────────────────────────────────── */
const D = {};   // filled by loadAll()

async function fetchJSON(name) {
  const r = await fetch(`data/${name}`);
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  return r.json();
}

async function loadAll() {
  const names = ['ensemble_performance.json', 'performance.json', 'sprints.json',
                 'holdings.json', 'breadth.json', 'regime.json', 'macro.json'];
  const results = await Promise.allSettled(names.map(fetchJSON));
  const keys = ['ensPerf', 'perf', 'sprints', 'holdings', 'breadth', 'regime', 'macro'];
  const failed = [];
  results.forEach((res, i) => {
    if (res.status === 'fulfilled') D[keys[i]] = res.value;
    else { failed.push(names[i]); console.warn('Load failed:', names[i], res.reason); }
  });
  if (failed.length) showLoadBanner(failed);
}

/* Visible diagnostics when data files fail to load — never fail silently. */
function showLoadBanner(failed) {
  const div = document.createElement('div');
  div.style.cssText = 'background:rgba(248,81,73,0.12);border:1px solid rgba(248,81,73,0.4);' +
    'border-radius:8px;margin:16px 32px 0;padding:14px 18px;font-size:13px;' +
    'color:#f85149;line-height:1.6';
  const isFile = location.protocol === 'file:';
  div.innerHTML = isFile
    ? '<strong>Data cannot load over file:// (browser security).</strong> ' +
      'Serve the dashboard over HTTP instead:<br>' +
      '<code style="font-family:\'JetBrains Mono\',monospace;color:#c9d1d9;background:#161b22;' +
      'padding:2px 8px;border-radius:4px;display:inline-block;margin-top:6px">' +
      'cd "&lt;project&gt;/src/dashboard" && python3 -m http.server 8080</code><br>' +
      'then open <code style="font-family:\'JetBrains Mono\',monospace;color:#58a6ff">' +
      'http://localhost:8080/index_v2.html</code>'
    : `<strong>Failed to load ${failed.length} data file(s):</strong> ${failed.join(', ')}. ` +
      'Run <code style="font-family:\'JetBrains Mono\',monospace;color:#c9d1d9">' +
      'python3 -m src.dashboard.export_data</code> from the project root, and make sure ' +
      'the HTTP server is started from the src/dashboard directory.';
  document.body.insertBefore(div, document.body.children[2] || null);
}

/* ── Regime helpers ──────────────────────────────────────────────────── */

/* Contiguous month ranges per label → Plotly rect shapes */
function regimeShapes(fromDate) {
  if (!D.regime) return [];
  const rows = D.regime.filter(r => r.month >= fromDate.slice(0, 7));
  const shapes = [];
  let run = null;
  const flush = () => {
    if (!run || run.label === 'NEUTRAL') { run = null; return; }
    shapes.push({
      type: 'rect', xref: 'x', yref: 'paper',
      x0: run.start + '-01', x1: nextMonth(run.end), y0: 0, y1: 1,
      fillcolor: run.label === 'RISK_OFF' ? C.redBand : C.greenBand,
      line: { width: 0 }, layer: 'below',
    });
    run = null;
  };
  for (const r of rows) {
    if (run && r.label === run.label && r.month === nextMonth(run.end).slice(0, 7)) {
      run.end = r.month;
    } else { flush(); run = { label: r.label, start: r.month, end: r.month }; }
  }
  flush();
  return shapes;
}

function nextMonth(ym) {
  const [y, m] = ym.split('-').map(Number);
  const d = new Date(Date.UTC(y, m, 1)); // month is 0-based → this is ym+1
  return `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, '0')}-01`;
}

function regimeOfMonth(ym) {
  if (!D.regime) return null;
  return D.regime.find(r => r.month === ym) || null;
}

/* ── Tabs ────────────────────────────────────────────────────────────── */
const CHART_IDS = ['perf-chart', 'breadth-chart', 'macro-chart'];
const rendered = new Set();

function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const panel = document.getElementById(btn.dataset.tab);
      if (panel) panel.classList.add('active');
      if (btn.dataset.tab === 'breadth' && !rendered.has('breadth')) {
        rendered.add('breadth'); renderBreadthChart();
      }
      if (btn.dataset.tab === 'macro' && !rendered.has('macro')) {
        rendered.add('macro'); renderMacroChart();
      }
      setTimeout(() => CHART_IDS.forEach(id => {
        const el = document.getElementById(id);
        if (el && el.data) Plotly.Plots.resize(el);
      }), 80);
    });
  });
}

/* ── Stat cards (Performance tab) ────────────────────────────────────── */
function renderStatCards() {
  const el = document.getElementById('stat-cards');
  if (!el || !D.sprints) return;
  const by = Object.fromEntries(D.sprints.sprints.map(s => [s.id, s.metrics]));
  const s0 = by.sprint0 || {}, s5 = by.sprint5 || {};
  const pp = v => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2);
  const cards = [
    { label: 'Full-Period CAGR', value: `${s5.cagr_full?.toFixed(2)}<span class="unit">%</span>`,
      top: C.accent, delta: s5.cagr_full - s0.cagr_full,
      deltaTxt: `${pp(s5.cagr_full - s0.cagr_full)}pp vs baseline` },
    { label: 'Full-Period Sharpe', value: s5.sharpe_full?.toFixed(3),
      top: C.green, delta: s5.sharpe_full - s0.sharpe_full,
      deltaTxt: `${pp(s5.sharpe_full - s0.sharpe_full)} vs baseline` },
    { label: 'Test-Period Sharpe', value: s5.sharpe_test?.toFixed(3),
      top: C.accent, delta: s5.sharpe_test - s0.sharpe_test,
      deltaTxt: `${pp(s5.sharpe_test - s0.sharpe_test)} vs baseline` },
    { label: '2022 Max Drawdown',
      value: `−${Math.abs(s5.max_dd_2022).toFixed(2)}<span class="unit">%</span>`,
      top: C.green, valueColor: C.red, delta: s5.max_dd_2022 - s0.max_dd_2022,
      deltaTxt: `${pp(s5.max_dd_2022 - s0.max_dd_2022)}pp vs baseline` },
  ];
  el.innerHTML = cards.map(c => `
    <div class="stat-card" style="border-top-color:${c.top}">
      <div class="stat-label">${c.label}</div>
      <div class="stat-value" style="color:${c.valueColor || C.text}">${c.value}</div>
      <div class="stat-delta" style="color:${c.delta >= 0 ? C.green : C.red}">${c.deltaTxt}</div>
    </div>`).join('');
}

/* ── Performance chart ───────────────────────────────────────────────── */
const ENSEMBLE_START = '2018-07-02';

function renderPerfChart() {
  const el = document.getElementById('perf-chart');
  if (!el) return;
  const traces = [];

  // Phase 1/2 + benchmarks from performance.json (fractions → %), rebased
  // to the ensemble inception date so all series share a zero point.
  if (D.perf && D.perf.data) {
    const rows = D.perf.data.filter(d => d.date >= ENSEMBLE_START);
    const base = rows[0] || {};
    const mk = (key, name, color, width) => {
      if (base[key] == null) return null;
      const pts = rows.filter(d => d[key] != null);
      return {
        x: pts.map(d => d.date),
        y: pts.map(d => ((1 + d[key]) / (1 + base[key]) - 1) * 100),
        name, type: 'scatter', mode: 'lines', line: { width, color },
        connectgaps: true,
        hovertemplate: `<b>%{x|%b %Y}</b><br>${name}: %{y:.1f}%<extra></extra>`,
      };
    };
    [['xlk', 'XLK Benchmark', C.orange, 2.0],
     ['spy', 'SPY', C.purple, 2.0],
     ['fundamental', 'Fundamental', C.cyan, 1.5],
     ['quant', 'Quant', C.accentAlt, 1.8]]
      .forEach(([k, n, c, w]) => { const t = mk(k, n, c, w); if (t) traces.push(t); });
  }

  // Ensemble (already cumulative %) — drawn last, dominant
  if (D.ensPerf && D.ensPerf.data) {
    traces.push({
      x: D.ensPerf.data.map(d => d.date),
      y: D.ensPerf.data.map(d => d.strategy),
      name: 'Ensemble', type: 'scatter', mode: 'lines',
      line: { width: 3.0, color: C.accent }, connectgaps: true,
      hovertemplate: '<b>%{x|%b %Y}</b><br>Ensemble: %{y:.1f}%<extra></extra>',
    });
  }

  const shapes = regimeShapes(ENSEMBLE_START);
  const testStart = (D.ensPerf && D.ensPerf.test_start) || '2023-01-01';
  shapes.push({ type: 'line', xref: 'x', yref: 'paper',
    x0: testStart, x1: testStart, y0: 0, y1: 1,
    line: { color: C.muted, width: 1.5, dash: 'dash' } });

  Plotly.newPlot(el, traces, {
    ...PLOT_BASE, shapes,
    annotations: [{ x: testStart, y: 1, xref: 'x', yref: 'paper',
      text: 'TRAIN | TEST', showarrow: false, yanchor: 'bottom',
      font: { size: 10, color: C.muted } }],
    xaxis: { ...AXIS }, yaxis: { ...AXIS, title: { text: 'Cumulative return (%)',
      font: { size: 11, color: C.muted } } },
  }, PLOT_CONFIG);
}

/* ── Sprint History tab ──────────────────────────────────────────────── */
const METRIC_ROWS = [
  ['Full CAGR', 'cagr_full', '%'], ['Full Sharpe', 'sharpe_full', ''],
  ['Test CAGR', 'cagr_test', '%'], ['Test Sharpe', 'sharpe_test', ''],
  ['2022 Max DD', 'max_dd_2022', '%'],
];

const VERDICT_PILL = {
  BASELINE: { c: C.muted, bg: 'rgba(139,148,158,0.15)', bo: 'rgba(139,148,158,0.3)' },
  PASS: { c: C.green, bg: 'rgba(63,185,80,0.15)', bo: 'rgba(63,185,80,0.4)' },
  FAIL: { c: C.red, bg: 'rgba(248,81,73,0.15)', bo: 'rgba(248,81,73,0.4)' },
  REFUTED: { c: C.red, bg: 'rgba(248,81,73,0.15)', bo: 'rgba(248,81,73,0.4)' },
};

function fmtMetric(v, unit) {
  if (v == null) return '—';
  const s = unit === '%' ? Math.abs(v).toFixed(2) + '%' : v.toFixed(3);
  return v < 0 ? '−' + s : s;
}

function renderSprintTable() {
  const el = document.getElementById('sprint-table');
  if (!el || !D.sprints) return;
  const sprints = D.sprints.sprints;

  let thead = '<tr class="sprint-head"><th class="metric-col">Metric</th>';
  for (const s of sprints) {
    const p = VERDICT_PILL[s.verdict] || VERDICT_PILL.BASELINE;
    thead += `<th><div class="sprint-col-head"><span class="sprint-name">${s.name}</span>
      <span class="pill" style="color:${p.c};background:${p.bg};border-color:${p.bo}">${s.verdict}</span>
      <span class="sprint-change">${s.change}</span></div></th>`;
  }
  thead += '</tr>';

  let tbody = '';
  for (const [label, key, unit] of METRIC_ROWS) {
    const vals = sprints.map(s => s.metrics[key]);
    const defined = vals.filter(v => v != null);
    const best = defined.length ? Math.max(...defined) : null; // max_dd is negative → max = least negative
    tbody += `<tr><td class="metric-col">${label}</td>`;
    sprints.forEach((s, i) => {
      const v = vals[i];
      const isBest = v != null && v === best;
      // Sprint 6 test-Sharpe cash-artifact footnote
      const star = (s.id === 'sprint6' && key === 'sharpe_test')
        ? ' <sup class="fn-star">★</sup>' : '';
      const color = v == null ? C.muted
        : (key === 'max_dd_2022' ? (isBest ? C.accentAlt : (v < -34 ? C.red : C.orange))
           : (isBest ? C.accentAlt : (s.verdict === 'FAIL' ? C.red : C.text)));
      tbody += `<td class="num${isBest ? ' best' : ''}" style="color:${color}">${fmtMetric(v, unit)}${star}</td>`;
    });
    tbody += '</tr>';
  }
  el.innerHTML = `<table><thead>${thead}</thead><tbody>${tbody}</tbody></table>`;

  // Footnote (uses real fold diagnostics when available)
  const s6 = sprints.find(s => s.id === 'sprint6');
  const fnEl = document.getElementById('sprint-footnote');
  if (fnEl && s6) {
    const fd = s6.fold_diagnostics || {};
    fnEl.innerHTML = `<span class="fn-star">★</span> Cash artifact — breadth starvation
      left the rolling strategy in cash for much of the test window
      (mean names above threshold ${fd.mean_names_above_threshold ?? '17.9'} vs 24.6 expanding);
      higher Sharpe reflects zero-vol cash, not improved selection. See Breadth tab.`;
  }

  // Verdict notes accordions — VERBATIM from results JSONs
  const notesEl = document.getElementById('verdict-notes');
  if (notesEl) {
    notesEl.innerHTML = sprints.filter(s => s.verdict_notes).map(s => {
      const p = VERDICT_PILL[s.verdict] || VERDICT_PILL.BASELINE;
      return `<div class="note-card">
        <button class="note-toggle" data-note="${s.id}">
          <div class="note-title">
            <span class="pill" style="color:${p.c};background:${p.bg};border-color:${p.bo}">${s.verdict}</span>
            ${s.name} — ${s.change}
          </div><span class="note-arrow">▼</span>
        </button>
        <div class="note-body" id="note-${s.id}" hidden>
          <p>${s.verdict_notes}</p>
        </div></div>`;
    }).join('');
    notesEl.querySelectorAll('.note-toggle').forEach(btn => {
      btn.addEventListener('click', () => {
        const body = document.getElementById(`note-${btn.dataset.note}`);
        body.hidden = !body.hidden;
        btn.querySelector('.note-arrow').textContent = body.hidden ? '▼' : '▲';
      });
    });
  }
}

/* ── Holdings tab ────────────────────────────────────────────────────── */
function initHoldings() {
  const sel = document.getElementById('month-select');
  if (!sel || !D.holdings) return;
  const months = Object.keys(D.holdings.months).sort().reverse();
  sel.innerHTML = months.map(m => `<option value="${m}">${m}</option>`).join('');
  sel.addEventListener('change', () => renderHoldings(sel.value));
  renderHoldings(months[0]);
}

function renderHoldings(month) {
  const grid = document.getElementById('holdings-grid');
  const chip = document.getElementById('regime-chip');
  if (!grid) return;

  const reg = regimeOfMonth(month);
  if (chip) {
    const map = { RISK_OFF: [C.red, 'rgba(248,81,73,0.12)', 'rgba(248,81,73,0.3)'],
                  RISK_ON: [C.green, 'rgba(63,185,80,0.12)', 'rgba(63,185,80,0.3)'],
                  NEUTRAL: [C.accent, 'rgba(88,166,255,0.12)', 'rgba(88,166,255,0.3)'] };
    const [c, bg, bo] = map[reg?.label || 'NEUTRAL'];
    chip.style.color = c; chip.style.background = bg; chip.style.borderColor = bo;
    chip.textContent = `${(reg?.label || 'NEUTRAL').replace('_', '-')} · ${(reg?.multiplier ?? 1.0).toFixed(1)}×`;
  }

  const rows = (D.holdings && D.holdings.months[month]) || [];
  if (!rows.length) {
    grid.innerHTML = `<div class="empty-state">
      <div class="empty-icon">💰</div>
      Strategy in cash — no names above the ${D.holdings?.min_score ?? 0.52} threshold in ${month}.
    </div>`;
    return;
  }
  const min = D.holdings.min_score;
  grid.innerHTML = rows.map(h => {
    const c = sectorColor(h.industry);
    const barW = Math.max(3, Math.round((h.score - min) / 0.20 * 100));
    return `<div class="holding-card" style="border-left-color:${c}">
      <div class="holding-top">
        <span class="holding-ticker">${h.ticker}</span>
        <span class="holding-weight">${(h.weight * 100).toFixed(1)}%</span>
      </div>
      <div class="holding-name" title="${h.name}">${h.name}</div>
      <div class="holding-score-row">
        <div class="score-track"><div class="score-bar" style="background:${c};width:${Math.min(100, barW)}%"></div></div>
        <span class="holding-score">${h.score.toFixed(3)}</span>
      </div></div>`;
  }).join('');
}

/* ── Breadth chart ───────────────────────────────────────────────────── */
function renderBreadthChart() {
  const el = document.getElementById('breadth-chart');
  if (!el || !D.breadth) return;
  const rows = D.breadth.data;
  const x = rows.map(r => r.month + '-01');
  const y = rows.map(r => r.names_above_threshold);
  const colors = rows.map(r => {
    const reg = regimeOfMonth(r.month);
    if (reg?.label === 'RISK_OFF') return 'rgba(248,81,73,0.55)';
    if (reg?.label === 'RISK_ON') return 'rgba(63,185,80,0.55)';
    return 'rgba(88,166,255,0.45)';
  });
  const zeros = rows.filter(r => r.names_above_threshold === 0);

  const traces = [
    { x, y, type: 'bar', marker: { color: colors }, name: 'Names > threshold',
      hovertemplate: '<b>%{x|%b %Y}</b><br>Names above threshold: %{y}<extra></extra>' },
  ];
  if (zeros.length) {
    traces.push({ x: zeros.map(r => r.month + '-01'), y: zeros.map(() => 0.4),
      type: 'scatter', mode: 'markers', name: 'Zero-breadth month',
      marker: { color: C.red, size: 7, symbol: 'circle' },
      hovertemplate: '<b>%{x|%b %Y}</b><br>ZERO names — strategy in cash<extra></extra>' });
  }

  Plotly.newPlot(el, traces, {
    ...PLOT_BASE, showlegend: false, bargap: 0.25,
    shapes: [
      { type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1,
        y0: D.breadth.max_positions, y1: D.breadth.max_positions,
        line: { color: C.muted, width: 1, dash: 'dash' } },
      { type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: 10, y1: 10,
        line: { color: C.orange, width: 1, dash: 'dot' } },
    ],
    annotations: [
      { xref: 'paper', x: 1, y: D.breadth.max_positions, yref: 'y',
        text: 'max portfolio size (20)', showarrow: false, xanchor: 'right',
        yanchor: 'bottom', font: { size: 10, color: C.muted } },
      { xref: 'paper', x: 1, y: 10, yref: 'y', text: 'starvation warning (10)',
        showarrow: false, xanchor: 'right', yanchor: 'bottom',
        font: { size: 10, color: C.orange } },
    ],
    xaxis: { ...AXIS }, yaxis: { ...AXIS, title: { text: `Names above ${D.breadth.min_score}`,
      font: { size: 11, color: C.muted } }, rangemode: 'tozero' },
  }, PLOT_CONFIG);
}

/* ── Macro & Regime chart ────────────────────────────────────────────── */
function renderMacroChart() {
  const el = document.getElementById('macro-chart');
  if (!el || !D.macro) return;
  const rows = D.macro.filter(r => r.date >= '2018-01-01');
  const vixPts = rows.filter(r => r.vix != null);
  const spPts = rows.filter(r => r.yield_spread_10y2y != null);

  const shapes = regimeShapes('2018-01-01');
  // VIX thresholds (top subplot)
  shapes.push(
    { type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: 25, y1: 25,
      line: { color: C.red, width: 1, dash: 'dash' } },
    { type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: 20, y1: 20,
      line: { color: C.orange, width: 1, dash: 'dot' } },
    { type: 'line', xref: 'paper', yref: 'y2', x0: 0, x1: 1, y0: 0, y1: 0,
      line: { color: C.muted, width: 1, dash: 'dash' } },
  );

  const traces = [
    { x: vixPts.map(r => r.date), y: vixPts.map(r => r.vix), name: 'VIX',
      type: 'scatter', mode: 'lines', line: { width: 1.4, color: C.orange },
      hovertemplate: '<b>%{x|%b %d %Y}</b><br>VIX: %{y:.1f}<extra></extra>' },
    { x: spPts.map(r => r.date), y: spPts.map(r => r.yield_spread_10y2y),
      name: '10y−2y spread', type: 'scatter', mode: 'lines',
      line: { width: 1.4, color: C.accent }, xaxis: 'x', yaxis: 'y2',
      fill: 'tozeroy', fillcolor: 'rgba(88,166,255,0.06)',
      hovertemplate: '<b>%{x|%b %d %Y}</b><br>Spread: %{y:.2f}pp<extra></extra>' },
  ];

  Plotly.newPlot(el, traces, {
    ...PLOT_BASE, shapes,
    grid: { rows: 2, columns: 1, pattern: 'independent', roworder: 'top to bottom' },
    xaxis: { ...AXIS, anchor: 'y2' },
    yaxis: { ...AXIS, domain: [0.56, 1], title: { text: 'VIX', font: { size: 11, color: C.muted } } },
    yaxis2: { ...AXIS, domain: [0, 0.44], title: { text: '10y−2y (pp)', font: { size: 11, color: C.muted } } },
    annotations: [
      { xref: 'paper', x: 1, y: 25, yref: 'y', text: 'risk-off 25', showarrow: false,
        xanchor: 'right', yanchor: 'bottom', font: { size: 10, color: C.red } },
      { xref: 'paper', x: 1, y: 20, yref: 'y', text: 'risk-on < 20', showarrow: false,
        xanchor: 'right', yanchor: 'bottom', font: { size: 10, color: C.orange } },
    ],
  }, PLOT_CONFIG);
}

/* ── Breadth caption (real Sprint 6 numbers when available) ──────────── */
function renderBreadthCaption() {
  const el = document.getElementById('breadth-caption');
  if (!el) return;
  const s6 = D.sprints?.sprints.find(s => s.id === 'sprint6');
  const fd = s6?.fold_diagnostics || {};
  const rollingMean = fd.mean_names_above_threshold ?? 17.9;
  const min = D.breadth?.min_score ?? 0.52;
  // Production (expanding) mean from the live export:
  const prodMean = D.breadth
    ? (D.breadth.data.reduce((a, r) => a + r.names_above_threshold, 0) / D.breadth.data.length)
    : 24.6;
  el.innerHTML = `Months where no name clears the <code>${min}</code> model threshold leave
    the strategy in cash. Breadth starvation — not ranking skill — sank the Sprint 6
    rolling-window experiment: mean monthly names above threshold fell from
    <code class="good">${prodMean.toFixed(1)}</code> (expanding, production) to
    <code class="bad">${Number(rollingMean).toFixed(1)}</code> (rolling).
    Verbatim record: <code>sprint6_results.json</code>.`;
}

/* ── Boot ────────────────────────────────────────────────────────────── */
async function boot() {
  await loadAll();
  const stamp = document.getElementById('export-stamp');
  if (stamp && D.perf?.exported_at) stamp.textContent = D.perf.exported_at.slice(0, 16).replace('T', ' ');
  initTabs();
  renderStatCards();
  renderPerfChart();
  renderSprintTable();
  initHoldings();
  renderBreadthCaption();
}

document.addEventListener('DOMContentLoaded', boot);
