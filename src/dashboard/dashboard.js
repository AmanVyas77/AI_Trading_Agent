/* ═══════════════════════════════════════════════════════════════════════
 *  Quant Research Platform — Dashboard JS
 *  ────────────────────────────────────────
 *  Pure JavaScript, no build step. Depends on Plotly.js from CDN.
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── Color Palette ─────────────────────────────────────────────────── */
const C = {
    bg:         '#0d1117',
    surface:    '#161b22',
    surfaceAlt: '#1c2333',
    border:     '#30363d',
    text:       '#c9d1d9',
    textMuted:  '#8b949e',
    accent:     '#58a6ff',
    accentAlt:  '#7ee787',
    red:        '#f85149',
    orange:     '#d29922',
    purple:     '#bc8cff',
    cyan:       '#39d2c0',
    green:      '#3fb950',
    greenFade:  'rgba(63,185,80,0.08)',
    redFade:    'rgba(248,81,73,0.08)',
    plotBg:     '#0d1117',
    plotGrid:   '#21262d',
};

const PLOTLY_LAYOUT_BASE = {
    paper_bgcolor: C.plotBg,
    plot_bgcolor:  C.plotBg,
    font: { color: C.text, family: "'Inter', sans-serif", size: 12 },
    margin: { t: 40, r: 20, b: 50, l: 60 },
    xaxis: { gridcolor: C.plotGrid, zerolinecolor: C.plotGrid },
    yaxis: { gridcolor: C.plotGrid, zerolinecolor: C.plotGrid },
    legend: { bgcolor: 'rgba(0,0,0,0)', font: { size: 11 } },
    hoverlabel: { bgcolor: C.surface, font: { color: C.text, size: 12 } },
};

/* ── Tab Switching ─────────────────────────────────────────────────── */

function initTabs() {
    const tabs = document.querySelectorAll('.tab-btn');
    const panels = document.querySelectorAll('.tab-panel');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            panels.forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            const target = document.getElementById(tab.dataset.tab);
            if (target) target.classList.add('active');

            // Re-render Plotly charts for correct sizing
            setTimeout(() => {
                Plotly.Plots.resize(document.getElementById('perf-chart'));
                Plotly.Plots.resize(document.getElementById('macro-chart'));
                Plotly.Plots.resize(document.getElementById('heatmap-chart'));
            }, 50);
        });
    });
}


/* ── 1. Performance Chart ──────────────────────────────────────────── */

function renderPerformanceChart(perfData) {
    const container = document.getElementById('perf-chart');
    if (!perfData || !perfData.data || perfData.data.length === 0) {
        container.innerHTML = '<p class="empty-msg">No performance data available. Run export_data.py first.</p>';
        return;
    }

    const data = perfData.data;
    const dates = data.map(d => d.date);

    const series = [
        { key: 'quant',       name: 'Quant Strategy',       color: C.accent   },
        { key: 'fundamental', name: 'Fundamental Strategy',  color: C.accentAlt },
        { key: 'xlk',         name: 'XLK (Benchmark)',       color: C.orange   },
        { key: 'spy',         name: 'SPY (Market)',          color: C.purple   },
    ];

    const traces = series
        .filter(s => data.some(d => d[s.key] !== undefined))
        .map(s => ({
            x: dates,
            y: data.map(d => d[s.key] != null ? (d[s.key] * 100) : null),
            name: s.name,
            type: 'scatter',
            mode: 'lines',
            line: { width: 2, color: s.color },
            connectgaps: true,
            hovertemplate: `%{x}<br>${s.name}: %{y:.2f}%<extra></extra>`,
        }));

    // Add train/test period annotation
    const layout = {
        ...PLOTLY_LAYOUT_BASE,
        title: { text: 'Cumulative Returns (%)', font: { size: 16, color: C.text } },
        yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: 'Return (%)' },
        xaxis: {
            ...PLOTLY_LAYOUT_BASE.xaxis,
            type: 'date',
            rangeslider: { visible: false },
        },
        shapes: [{
            type: 'line', x0: '2023-01-01', x1: '2023-01-01',
            y0: 0, y1: 1, yref: 'paper',
            line: { color: C.textMuted, width: 1, dash: 'dash' },
        }],
        annotations: [{
            x: '2019-01-01', y: 1, yref: 'paper', text: 'TRAIN',
            font: { color: C.textMuted, size: 11 }, showarrow: false,
        }, {
            x: '2024-01-01', y: 1, yref: 'paper', text: 'TEST',
            font: { color: C.textMuted, size: 11 }, showarrow: false,
        }],
    };

    Plotly.newPlot(container, traces, layout, { responsive: true, displayModeBar: false });

    /* ── Metrics Cards ─────────────────────────────────────────────── */
    const cardsEl = document.getElementById('perf-cards');
    cardsEl.innerHTML = '';

    series.filter(s => data.some(d => d[s.key] !== undefined)).forEach(s => {
        const vals = data.map(d => d[s.key]).filter(v => v != null);
        if (vals.length === 0) return;

        const totalReturn = vals[vals.length - 1] * 100;
        const years = vals.length / 252;
        const cagr = years > 0 ? ((Math.pow(1 + vals[vals.length - 1], 1 / years) - 1) * 100) : 0;

        // Daily returns for Sharpe / MaxDD
        const dailyRet = [];
        for (let i = 1; i < vals.length; i++) {
            dailyRet.push((1 + vals[i]) / (1 + vals[i - 1]) - 1);
        }
        const mean = dailyRet.reduce((a, b) => a + b, 0) / dailyRet.length;
        const stdDev = Math.sqrt(dailyRet.reduce((a, b) => a + (b - mean) ** 2, 0) / dailyRet.length);
        const sharpe = stdDev > 0 ? (mean / stdDev * Math.sqrt(252)).toFixed(2) : 'N/A';

        let maxDD = 0, peak = -Infinity;
        for (const v of vals) {
            if (v > peak) peak = v;
            const dd = (peak - v) / (1 + peak);
            if (dd > maxDD) maxDD = dd;
        }

        const card = document.createElement('div');
        card.className = 'metric-card';
        card.style.borderTop = `3px solid ${s.color}`;
        card.innerHTML = `
            <h3 style="color:${s.color}">${s.name}</h3>
            <div class="metric-row"><span>Total Return</span><span class="metric-val ${totalReturn >= 0 ? 'positive' : 'negative'}">${totalReturn.toFixed(2)}%</span></div>
            <div class="metric-row"><span>CAGR</span><span class="metric-val">${cagr.toFixed(2)}%</span></div>
            <div class="metric-row"><span>Sharpe</span><span class="metric-val">${sharpe}</span></div>
            <div class="metric-row"><span>Max Drawdown</span><span class="metric-val negative">-${(maxDD * 100).toFixed(2)}%</span></div>
        `;
        cardsEl.appendChild(card);
    });
}


/* ── 2. Universe Table ─────────────────────────────────────────────── */

let _universeData = [];
let _sortCol = 'market_cap';
let _sortAsc = false;

function renderUniverseTable(universeData, quantSignals, fundSignals) {
    _universeData = universeData.map(u => {
        const entry = { ...u };

        // Latest quant score
        if (quantSignals && quantSignals.length > 0) {
            const matches = quantSignals.filter(q => q.ticker === u.ticker);
            entry.quant_score = matches.length > 0
                ? matches[matches.length - 1].quant_score ?? null
                : null;
        }

        // Latest fundamental score
        if (fundSignals && fundSignals.length > 0) {
            const matches = fundSignals.filter(f => f.ticker === u.ticker);
            entry.fund_score = matches.length > 0
                ? matches[matches.length - 1].composite_score ?? null
                : null;
        }

        return entry;
    });

    _sortAndRender();
}

function _sortAndRender() {
    const sorted = [..._universeData].sort((a, b) => {
        let va = a[_sortCol], vb = b[_sortCol];
        if (va == null && vb == null) return 0;
        if (va == null) return 1;
        if (vb == null) return -1;
        if (typeof va === 'string') va = va.toLowerCase();
        if (typeof vb === 'string') vb = vb.toLowerCase();
        if (va < vb) return _sortAsc ? -1 : 1;
        if (va > vb) return _sortAsc ? 1 : -1;
        return 0;
    });

    const container = document.getElementById('universe-table');
    if (sorted.length === 0) {
        container.innerHTML = '<p class="empty-msg">No universe data. Run export_data.py first.</p>';
        return;
    }

    const cols = [
        { key: 'ticker',      label: 'Ticker',     fmt: v => v || '—' },
        { key: 'name',         label: 'Name',       fmt: v => v || '—' },
        { key: 'sector',       label: 'Sector',     fmt: v => v || '—' },
        { key: 'market_cap',   label: 'Market Cap', fmt: v => v ? fmtCap(v) : '—' },
        { key: 'quant_score',  label: 'Quant Score', fmt: v => v != null ? fmtScore(v) : '—' },
        { key: 'fund_score',   label: 'Fund Score',  fmt: v => v != null ? fmtScore(v) : '—' },
    ];

    let html = '<table class="data-table"><thead><tr>';
    cols.forEach(c => {
        const arrow = _sortCol === c.key ? (_sortAsc ? ' ▲' : ' ▼') : '';
        html += `<th data-col="${c.key}" class="sortable">${c.label}${arrow}</th>`;
    });
    html += '</tr></thead><tbody>';

    sorted.forEach(row => {
        html += '<tr>';
        cols.forEach(c => {
            const val = row[c.key];
            let cls = '';
            if (c.key === 'quant_score' || c.key === 'fund_score') {
                cls = val > 0 ? 'positive' : val < 0 ? 'negative' : '';
            }
            html += `<td class="${cls}">${c.fmt(val)}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table>';

    container.innerHTML = html;

    // Attach sort handlers
    container.querySelectorAll('th.sortable').forEach(th => {
        th.addEventListener('click', () => {
            const col = th.dataset.col;
            if (_sortCol === col) {
                _sortAsc = !_sortAsc;
            } else {
                _sortCol = col;
                _sortAsc = true;
            }
            _sortAndRender();
        });
    });
}

function fmtCap(v) {
    if (v >= 1e12) return `$${(v / 1e12).toFixed(1)}T`;
    if (v >= 1e9) return `$${(v / 1e9).toFixed(1)}B`;
    if (v >= 1e6) return `$${(v / 1e6).toFixed(0)}M`;
    return `$${v}`;
}

function fmtScore(v) {
    return v.toFixed(3);
}


/* ── 3. Macro Regime Chart ─────────────────────────────────────────── */

function renderMacroChart(macroData) {
    const container = document.getElementById('macro-chart');
    if (!macroData || macroData.length === 0) {
        container.innerHTML = '<p class="empty-msg">No macro data available.</p>';
        return;
    }

    const dates = macroData.map(d => d.date);

    const panels = [
        { key: 'vix',                label: 'VIX',              color: C.red,       row: 1, col: 1 },
        { key: 'yield_spread_10y2y', label: '10Y-2Y Spread',    color: C.accent,    row: 1, col: 2 },
        { key: 'fed_funds_rate',     label: 'Fed Funds Rate',   color: C.orange,    row: 2, col: 1 },
        { key: 'cpi',               label: 'CPI',              color: C.purple,    row: 2, col: 2 },
    ];

    const traces = panels.map((p, idx) => ({
        x: dates,
        y: macroData.map(d => d[p.key] ?? null),
        name: p.label,
        type: 'scatter',
        mode: 'lines',
        line: { width: 1.5, color: p.color },
        xaxis: `x${idx === 0 ? '' : idx + 1}`,
        yaxis: `y${idx === 0 ? '' : idx + 1}`,
        connectgaps: true,
        hovertemplate: `%{x}<br>${p.label}: %{y:.2f}<extra></extra>`,
    }));

    // Regime shading: risk-off when VIX > 25 OR yield spread < 0
    const shapes = [];
    let riskOffStart = null;
    for (let i = 0; i < macroData.length; i++) {
        const d = macroData[i];
        const isRiskOff = (d.vix != null && d.vix > 25) ||
                          (d.yield_spread_10y2y != null && d.yield_spread_10y2y < 0);
        if (isRiskOff && riskOffStart === null) {
            riskOffStart = d.date;
        } else if (!isRiskOff && riskOffStart !== null) {
            // Add risk-off bands to all subplots
            for (let ax = 0; ax < 4; ax++) {
                shapes.push({
                    type: 'rect',
                    xref: `x${ax === 0 ? '' : ax + 1}`,
                    yref: `y${ax === 0 ? '' : ax + 1}`,
                    x0: riskOffStart, x1: macroData[i - 1].date,
                    y0: 0, y1: 1, yref: `y${ax === 0 ? '' : ax + 1} domain`,
                    fillcolor: 'rgba(248,81,73,0.06)',
                    line: { width: 0 },
                    layer: 'below',
                });
            }
            riskOffStart = null;
        }
    }

    const subplotDomain = (row, col) => {
        const xGap = 0.06;
        const yGap = 0.08;
        const x0 = col === 1 ? 0 : 0.5 + xGap / 2;
        const x1 = col === 1 ? 0.5 - xGap / 2 : 1;
        const y0 = row === 2 ? 0 : 0.5 + yGap / 2;
        const y1 = row === 2 ? 0.5 - yGap / 2 : 1;
        return { xDomain: [x0, x1], yDomain: [y0, y1] };
    };

    const layout = {
        ...PLOTLY_LAYOUT_BASE,
        showlegend: false,
        shapes,
        annotations: panels.map((p, idx) => {
            const { xDomain, yDomain } = subplotDomain(p.row, p.col);
            return {
                x: xDomain[0], y: yDomain[1] + 0.02,
                xref: 'paper', yref: 'paper',
                text: `<b>${p.label}</b>`,
                font: { color: p.color, size: 13 },
                showarrow: false, xanchor: 'left',
            };
        }),
    };

    // Configure subplot axes
    panels.forEach((p, idx) => {
        const { xDomain, yDomain } = subplotDomain(p.row, p.col);
        const xKey = idx === 0 ? 'xaxis' : `xaxis${idx + 1}`;
        const yKey = idx === 0 ? 'yaxis' : `yaxis${idx + 1}`;
        layout[xKey] = {
            domain: xDomain, type: 'date',
            gridcolor: C.plotGrid, zerolinecolor: C.plotGrid,
            showticklabels: p.row === 2,
            tickfont: { size: 10 },
        };
        layout[yKey] = {
            domain: yDomain,
            gridcolor: C.plotGrid, zerolinecolor: C.plotGrid,
            tickfont: { size: 10 },
        };
    });

    Plotly.newPlot(container, traces, layout, { responsive: true, displayModeBar: false });
}


/* ── 4. Factor Heatmap ─────────────────────────────────────────────── */

function renderFactorHeatmap(fundSignals, universeData) {
    const container = document.getElementById('heatmap-chart');
    if (!fundSignals || fundSignals.length === 0) {
        container.innerHTML = '<p class="empty-msg">No fundamental scores available.</p>';
        return;
    }

    // Get the latest quarter
    const allDates = [...new Set(fundSignals.map(d => d.quarter_end))].sort();
    const latestQ = allDates[allDates.length - 1];
    const latest = fundSignals.filter(d => d.quarter_end === latestQ);

    if (latest.length === 0) {
        container.innerHTML = '<p class="empty-msg">No data for latest quarter.</p>';
        return;
    }

    // Factor names — use whatever columns exist beyond ticker/quarter_end/composite_score
    const metaCols = new Set(['ticker', 'quarter_end', 'composite_score',
                              'features_used', 'features_missing']);
    const factorKeys = Object.keys(latest[0]).filter(k => !metaCols.has(k));

    // If no per-factor columns, use composite_score only
    const factors = factorKeys.length > 0
        ? factorKeys
        : ['composite_score'];

    const tickers = latest.map(d => d.ticker).filter(Boolean).slice(0, 40); // cap at 40
    const z = [];
    factors.forEach(f => {
        const row = tickers.map(t => {
            const entry = latest.find(d => d.ticker === t);
            const val = entry ? entry[f] : null;
            return val != null ? Math.max(-3, Math.min(3, val)) : null;
        });
        z.push(row);
    });

    const trace = {
        x: tickers,
        y: factors.map(f => f.replace(/_/g, ' ')),
        z: z,
        type: 'heatmap',
        colorscale: [
            [0,    C.red],
            [0.5,  '#ffffff'],
            [1,    C.green],
        ],
        zmin: -3, zmax: 3,
        hovertemplate: '%{x}<br>%{y}<br>Z-score: %{z:.2f}<extra></extra>',
        colorbar: {
            title: { text: 'Z-score', font: { color: C.text } },
            tickfont: { color: C.text },
            len: 0.6,
        },
    };

    const layout = {
        ...PLOTLY_LAYOUT_BASE,
        title: {
            text: `Factor Z-Scores — ${latestQ}`,
            font: { size: 16, color: C.text },
        },
        xaxis: { tickangle: -45, tickfont: { size: 10 }, gridcolor: C.plotGrid },
        yaxis: { tickfont: { size: 11 }, gridcolor: C.plotGrid, autorange: 'reversed' },
        margin: { ...PLOTLY_LAYOUT_BASE.margin, b: 100, l: 150 },
    };

    Plotly.newPlot(container, [trace], layout, { responsive: true, displayModeBar: false });
}


/* ── Init ──────────────────────────────────────────────────────────── */

async function fetchJSON(url) {
    try {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
        return await resp.json();
    } catch (err) {
        console.warn(`Failed to fetch ${url}: ${err.message}`);
        return null;
    }
}

async function initDashboard() {
    initTabs();

    // Show loading state
    document.querySelectorAll('.chart-container, .table-container').forEach(el => {
        if (!el.innerHTML.trim()) {
            el.innerHTML = '<div class="loader"><div class="spinner"></div><p>Loading data…</p></div>';
        }
    });

    // Fetch all JSON files in parallel
    const [prices, macro, quantSignals, fundSignals, universe, performance] = await Promise.all([
        fetchJSON('./data/prices.json'),
        fetchJSON('./data/macro.json'),
        fetchJSON('./data/quant_signals.json'),
        fetchJSON('./data/fundamental_signals.json'),
        fetchJSON('./data/universe.json'),
        fetchJSON('./data/performance.json'),
    ]);

    // Update header metadata
    if (performance) {
        const tsEl = document.getElementById('export-timestamp');
        const phaseEl = document.getElementById('current-phase');
        if (tsEl && performance.exported_at) {
            const d = new Date(performance.exported_at);
            tsEl.textContent = d.toLocaleString();
        }
        if (phaseEl && performance.phase) {
            const phaseNames = { 1: 'Phase 1 — Quant', 2: 'Phase 2 — Fundamental', 3: 'Phase 3 — Ensemble' };
            phaseEl.textContent = phaseNames[performance.phase] || `Phase ${performance.phase}`;
        }
    }

    // Render each tab
    renderPerformanceChart(performance);
    renderUniverseTable(universe || [], quantSignals || [], fundSignals || []);
    renderMacroChart(macro || []);
    renderFactorHeatmap(fundSignals || [], universe || []);
}

// Boot
document.addEventListener('DOMContentLoaded', initDashboard);
