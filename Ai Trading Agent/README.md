# Quant Finance Research Platform

A three-phase research platform that builds, validates, and combines two independent trading strategies for a ~80-stock US tech universe.

---

## Project Overview

| Phase | Strategy | Key Data | Benchmark |
|-------|----------|----------|-----------|
| 1 — Quant | Price/volume momentum + FRED macro factors | yfinance, FRED | XLK |
| 2 — Fundamental | SEC EDGAR XBRL financials + FinBERT sentiment on earnings transcripts | EDGAR API, Simfin | XLK |
| 3 — Ensemble | XGBoost combiner of Phase 1 + Phase 2 signals | All of the above | SPY, equal-weight universe |

**Data timeline:** Train 2015–2022 · Holdout test 2023–2024 · Live paper trading 2025+

---

## Universe

~80 US-listed tech stocks meeting all of:
- S&P 500 Tech + Semiconductors + Enterprise Software + Mid-cap Growth
- Market cap ≥ $2B, avg daily volume ≥ $5M
- ≥ 8 quarters of XBRL filing history on SEC EDGAR
- NYSE / NASDAQ only, USD reporting
- No IPOs before 2017 in training data

---

## Stack

```
Python 3.11+
├── Data          yfinance · FRED API · SEC EDGAR API · Simfin
├── Storage       SQLite → Postgres (later)  via SQLAlchemy
├── Backtesting   VectorBT
├── ML            XGBoost · scikit-learn · SHAP
├── NLP           FinBERT (HuggingFace transformers)
├── Dashboard     Streamlit + Plotly
└── Research      Obsidian (wiki + RAG source)
```

---

## Folder Structure

```
Ai Trading Agent/
├── config/
│   └── settings.yaml          # all parameters — universe, factors, backtest
├── src/
│   ├── universe/
│   │   └── screener.py        # filters seed tickers → investable universe
│   ├── data/
│   │   └── quant_pipeline.py  # yfinance + FRED → SQLite
│   ├── strategies/
│   │   ├── quant/
│   │   │   └── quant_factors.py   # momentum + macro signals + VectorBT backtest
│   │   ├── fundamental/       # Phase 2: XBRL parser, FinBERT, scorer
│   │   └── ensemble/          # Phase 3: XGBoost combiner
│   ├── ml/                    # feature engineering helpers, model wrappers
│   ├── dashboard/             # Streamlit app
│   └── utils/                 # shared helpers (db, logging, dates)
├── data/
│   ├── raw/
│   │   ├── price_volume/      # gitignored
│   │   ├── edgar_xbrl/        # gitignored
│   │   ├── transcripts/       # gitignored
│   │   └── fred_macro/        # gitignored
│   ├── processed/             # cleaned, merged datasets
│   └── universe/
│       └── universe.csv       # output of screener.py
├── backtests/
│   ├── results/               # serialised VectorBT portfolio objects
│   └── reports/               # HTML tearsheets
├── notebooks/
│   ├── sprint1/               # EDA, factor analysis
│   ├── sprint2/               # XBRL/NLP exploration
│   └── sprint3/               # ensemble tuning
├── tests/
│   ├── unit/
│   └── integration/
├── research_wiki/             # Obsidian vault (strategy notes, paper trails)
├── .env.example               # copy → .env, fill in API keys
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone and set up environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure secrets
cp .env.example .env
# edit .env: add FRED_API_KEY, SEC_USER_AGENT

# 3. Build the investable universe (skip XBRL check for speed during dev)
python -m src.universe.screener --no-xbrl

# 4. Pull price + macro data
python -m src.data.quant_pipeline

# 5. Run quant strategy backtest vs SPY
python -m src.strategies.quant.quant_factors --train-only
```

---

## Sprint Plan

### Sprint 1 — Quant Foundation ✅
- [x] Folder structure and config
- [x] Universe screener (market cap, volume, exchange, XBRL, IPO filters)
- [x] Quant data pipeline (yfinance + FRED → SQLite)
- [x] Quant factors: momentum (1M/3M/6M/12M), volume Z-score, inv-vol, macro regime
- [x] VectorBT backtest vs SPY buy-and-hold (2015–2022 train)
- [ ] Validate on 2023–2024 holdout
- [ ] Sprint 1 EDA notebook

### Sprint 2 — Fundamental Strategy
- [ ] SEC EDGAR XBRL downloader + parser
- [ ] Simfin cross-validation layer
- [ ] Financial feature engineering (valuation, quality, growth)
- [ ] FinBERT sentiment pipeline on earnings call transcripts
- [ ] Fundamental signal scorer + VectorBT backtest vs XLK

### Sprint 3 — Ensemble + Dashboard
- [ ] XGBoost combiner: merge quant + fundamental signals
- [ ] Walk-forward cross-validation (3-year window, quarterly retrain)
- [ ] Streamlit dashboard with Plotly tearsheets
- [ ] Live paper trading loop (2025–present)
- [ ] Obsidian RAG integration

---

## Key Config Parameters

All tunable parameters live in `config/settings.yaml`:

- `universe.*` — screener thresholds
- `timeline.*` — train/test/live date ranges
- `quant_factors.momentum.lookbacks_days` — momentum windows
- `quant_factors.top_n` — portfolio size
- `fred.series` — macro factors to include
- `backtest.*` — capital, commission, slippage

---

## Benchmarks

| Comparison | Benchmark |
|-----------|-----------|
| Phase 1 strategy alpha | XLK |
| Time-in-market cost | SPY buy-and-hold |
| Phase 3 ensemble | SPY + equal-weight screen universe |

---

## Notes

- Raw data directories are gitignored. Run the pipeline to regenerate locally.
- The SQLite DB (`data/quant_research.db`) is also gitignored — too large for Git.
- Phase 2 transcript access TBD — placeholder in config for Seeking Alpha; may substitute with earnings call PDFs from SEC 8-K filings.
