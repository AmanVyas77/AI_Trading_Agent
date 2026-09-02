---
date: 2026-07-24
tags: [llm, sentiment, news, return-predictability, market-efficiency]
category: ai_finance
---

# Can ChatGPT Forecast Stock Price Movements? Return Predictability and LLMs

**Authors:** Lopez-Lira & Tang (2026) — Journal of Financial Economics 184, 104335
**DOI:** 10.1016/j.jfineco.2026.104335

**Supersedes** [[lopez_lira_2023_chatgpt_returns]] (the working-paper draft).
The published version is substantially expanded: GPT-4 rather than GPT-3.5,
12 LLMs compared, a theoretical model, topic-level decomposition, and
adoption-decay evidence. Keep both; cite this one.

## Setup

4,123 US common stocks, Oct 2021 – May 2024 — deliberately **after** the
evaluated model's Sept 2021 knowledge cutoff, to avoid lookahead and
memorisation (Sarkar & Vafa 2024). GPT-4 is prompted to label each headline
positive / negative / neutral for the firm's stock price.

## Key Results

**Initial reaction:** portfolio hit rate 93.3% overnight, 88.8% intraday. Not
tradable — requires advance knowledge of news timing.

**Subsequent drift (tradable):** hit rate 58% / 55%; mean daily return 34 bps /
50 bps; annualised pre-cost Sharpe **2.97 overnight, 2.63 intraday**. Horizon is
1–2 trading days.

**Model-size ladder (drift Sharpe):** GPT-4 **2.97** > GPT-3.5 1.66 >
DistilBART-MNLI 1.26 > GPT-1/GPT-2/Llama2-7b negative. Financial reasoning is
an emergent capacity of scale.

**GPT-4 subsumes commercial sentiment.** With both in the regression, RavenPack's
drift coefficient goes insignificant while GPT-4's holds (0.158, t=5.27).

**Asymmetry:** short leg Sharpe 2.01 vs long leg 0.78. Predictability
concentrated in small caps (GPT-4 × Small interaction 0.404, t=4.55).

**Transaction costs:** +300% cumulative at 5 bps round-trip, +100% at 10 bps,
**unprofitable at 20 bps**. Turnover ~190%/day at full rebalance; a 25% partial
rebalance cuts turnover to ~46%/day and is *more* robust net of costs
(Sharpe 1.34 vs 1.29 at 10 bps).

**Alpha decay:** annualised Sharpe 6.54 (2021Q4) → 3.68 (2022) → 2.33 (2023) →
1.22 (Jan–May 2024), as LLM capability and adoption rose.

**Topic heterogeneity.** Efficiently processed (strong initial alignment, no
significant drift): earnings/revenue, strategic partnerships, clinical trials.
Underreacted (significant Drift×GPT): insider stock transactions 26.3 bps,
dividend announcements 22.3 bps, healthcare conference presentations 34.2 bps.

**Embeddings collapse on small samples.** Supervised embedding models drop to
Sharpe 1.71 on the 26k-observation intraday sample vs GPT-4's 2.63 zero-shot.

## Relevance to Trading Agent

**Do not implement the strategy.** Daily long-short, 1–2 day horizon, ~190%
daily turnover, concentrated in small caps and the short leg, dead at 20 bps.
Our system is monthly, long-only, 54 large-cap tech names — every dimension
that makes their signal work is one we have excluded by design.

**Flag on Sprint 9:** we score all 254k ingested articles with FinBERT, the
weakest arm on their ladder. Monthly cross-sectional aggregation is a different
task from daily drift trading, so this is not disqualifying — but it argues for
a Claude-scored `news_sentiment_llm` second arm rather than assuming FinBERT is
the ceiling. Each arm needs its own pre-committed rule, declared before either
is evaluated.

**Two findings worth carrying:**
- Alpha decay is documented, so a news feature should be expected to decay —
  argues for periodic re-validation, not a one-time PASS.
- Topic-conditional underreaction is a gate on *when* an analyst verdict should
  carry weight. Relevant to the END-GOAL funnel.

**Methodology confirmation:** their post-cutoff sample requirement is the same
hazard already recorded in the funnel's design notes — no historical backtest of
an LLM analyst, forward paper trading only, model version pinned.

See `memory/future_ideas.md` → "Sprint 9 flag — FinBERT is the weakest scorer".
