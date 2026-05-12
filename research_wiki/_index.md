---
date: 2026-05-12
tags: [index, rag, phase-4]
---

# Research Wiki — Index

This vault is the knowledge base for the AI Trading Agent Phase 4 RAG system.
Notes saved here are automatically re-indexed into ChromaDB on the next ingestion run.

## Structure

| Folder | Contents |
|--------|----------|
| `companies/` | Per-ticker research notes and RAG session outputs |
| `research/` | Session Q&A logs and cross-ticker analysis |
| `papers/` | Summaries and annotations of research papers |

## Key Papers in the Corpus

### LLMs for Financial Analysis
- [[papers/kim_2024_financial_statement_llm]] — GPT-4 beats analysts at earnings direction (60.35% vs 52.71%)
- [[papers/kim_2023a_transcripts_risks]] — RAG over earnings calls for corporate risk signals
- [[papers/kim_2023b_bloated_disclosures]] — ChatGPT reduces information overload in filings
- [[papers/cao_2024_man_plus_machine]] — Man + machine outperforms either alone
- [[papers/lopez_lira_2023_chatgpt_returns]] — GPT explains short-term returns from headlines
- [[papers/bybee_2023_ghost_machine]] — GPT macro predictions align with expert surveys
- [[papers/wu_2023_bloomberggpt]] — Domain pre-training alone insufficient; scale matters
- [[papers/bubeck_2023_sparks_agi]] — GPT-4 numerical reasoning failure modes documented
- [[papers/sharma_2024_sycophancy]] — LLMs exhibit sycophancy; co-analyst risk

### RAG & Prompt Engineering
- [[papers/lewis_2020_rag]] — Foundational RAG architecture (NeurIPS 2020)
- [[papers/wei_2022_chain_of_thought]] — Chain-of-thought prompting (+8pp accuracy in finance)
- [[papers/zhang_2023_auto_cot]] — Automatic CoT generalises to structured reasoning

### Benchmarks & Open-Weight Models
- [[papers/xie_2024_finben]] — 42 datasets, 24 financial tasks
- [[papers/xie_2023_pixiu]] — FinMA (Llama fine-tuned for finance)
- [[papers/shah_2022_flue_flang]] — FLUE benchmark for financial NLP
- [[papers/touvron_2023_llama2]] — Llama 2 at 70B approaches GPT-3.5
- [[papers/jiang_2023_mistral]] — Mistral 7B default for local deployment

### Quantitative Fundamentals
- [[papers/ou_penman_1989_financial_prediction]] — Seminal 59-predictor earnings prediction
- [[papers/chen_2022_xbrl_earnings]] — 12,000 XBRL tags ML model
- [[papers/hunt_2022_ml_earnings]] — ML accuracy improvements over time-series
- [[papers/fama_french_2015_five_factor]] — FF5 asset pricing model

### Governance & Human-in-the-Loop
- [[papers/costello_2020_machine_man]] — Human discretion adds value to AI models
- [[papers/dellacqua_2023_jagged_frontier]] — GPT-4 raises knowledge worker output quality
- [[papers/tversky_kahneman_1974_anchoring]] — Anchoring bias in human-AI decisions
- [[papers/bouwman_1987_analyst_decisions]] — How analysts actually make decisions

## How to Use

```bash
# Start a research session
python -m src.rag.cli

# Ingest new papers you've added to research_papers/
python -m src.rag.cli --ingest --sources papers

# Ingest SEC filings for specific tickers
python -m src.rag.cli --ingest --sources edgar --tickers NVDA MSFT

# Check what's indexed
python -m src.rag.cli --stats
```
