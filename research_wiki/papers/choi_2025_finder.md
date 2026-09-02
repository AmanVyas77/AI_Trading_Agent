---
date: 2025-11-15
tags: [rag, retrieval, benchmark, evaluation, financial-qa, 10-k]
category: ai_finance
---

# FinDER: Financial Dataset for Question Answering and Evaluating RAG

**Authors:** Choi, Kwon, Ha, Choi, Kim, Lee, Sohn, Lopez-Lira (2025) — ICAIF '25
**arXiv:** 2504.15800 · **DOI:** 10.1145/3768292.3770361
**Dataset:** https://huggingface.co/datasets/Linq-AI-Research/FinDER

## What It Is

5,703 expert-annotated (query, evidence, answer) triplets over the latest 10-K
filings of 490 S&P 500 companies (index as of 2024-12-31). Queries are sampled
from a real financial Q&A service used by hedge fund analysts, PMs and IB
analysts — so they are short, ambiguous, and dense with jargon and acronyms
("Recent CAGR in MS trading revenue"). 46.4% of queries carry 5+ domain-specific
expressions.

Unlike FinQA / ConvFinQA / FinanceBench, retrieval is a **first-class task**:
no predefined context is handed to the model. Annotation by an IB analyst and a
CPA with cross-validation.

## Key Results

**Retrieval (Context Recall, total):** E5-Mistral **25.95** > GTE 17.83 >
mE5 17.36 > BM25 11.68. Decoder-based dense retrieval wins everywhere; all
neural methods beat sparse.

**Query quality matters as much as model choice.** Expert-rewritten
("well-formed") queries vs raw FinDER queries, precision: E5-Mistral
33.9 → 25.7, GTE 20.2 → 18.1. ~8pp lost to query ambiguity alone.

**LLM reranking works.** Reranking E5-Mistral's top-10 down to top-5:
Claude-3.7-Sonnet F1 **63.05**, GPT-o1 62.90, Qwen-QWQ 61.78,
Deepseek-R1-Distill 60.01. Authors' conclusion: *retrieval sets need not be
perfectly precise — a diverse pool is beneficial, since reasoning-focused
models discern relevant information despite noise.*

**Context ablation (answer correctness), Claude-3.7-Sonnet:**
9.37 (no context) → 33.89 (top-10 retrieved) → **66.48 (perfect context)**.
Same shape for all four generators tested.

**Faithfulness ≫ correctness** across the board (Claude 84.75 faith. vs 28.79
corr. on partial context). Models are consistent with the context they are
given; the context is what is wrong.

## Relevance to Trading Agent

This is the closest published benchmark to the END-GOAL analyst funnel — RAG
over 10-Ks with terse auto-generated queries. Three direct implications:

1. **Retrieval is the binding constraint, not the generator.** With best-case
   context recall at ~26%, roughly half the achievable answer quality is lost
   before the LLM sees anything. Prompt engineering and model selection are
   second-order until retrieval improves.
2. **RAG must never supply numbers to the analyst.** Numbers come from
   `xbrl_facts` and the factor parquets — exact, point-in-time. RAG supplies
   narrative only. `context_builder.py` already injects structured scores;
   this paper argues against expanding RAG's role beyond that.
3. **Our stack is below the paper's floor.** `all-MiniLM-L6-v2` is not even in
   their comparison set; we take top-8 by raw cosine with no reranker and no
   query rewriting. Cheapest wins are LLM reranking and query expansion, both
   in `src/rag/query/`.

## Decision: benchmark DECLINED (2026-08-08)

We are **not running FinDER** as a benchmark. The deciding argument is
decision-relevance: a positive result would have us implement reranking, and a
negative result we would disbelieve as a transfer failure (their baselines strip
HTML to flat paragraphs; our `SECFilingChunker` is section-aware). An experiment
whose outcome does not change the action is not worth ingesting a foreign
multi-GB corpus and standing up a RAGAS judge harness for.

**The paper's value is its findings, above — not its dataset.** Reranking and
query expansion get implemented directly, without benchmark justification.
Retrieval measurement happens in-house, against the funnel's own ~10-20 dossier
question types, after the funnel exists.

Reopen only if the analyst funnel underperforms and retrieval is the prime
suspect — FinDER is the only source of an *absolute* calibration, since
in-house recall@k has no external reference point.

See `memory/future_ideas.md` → "RAG layer — FinDER benchmark DECLINED, three
items kept".

## Caveats

- FinDER ships its own document corpus; gold evidence is specific to their
  10-Ks. It benchmarks the **retrieval stack**, not our pipeline on our data —
  a separate in-house gold set is needed for that.
- 10-K only. Says nothing about news retrieval, which is a different problem
  (already chunk-sized, recency-weighted, needs dedup).
- Baselines use naive preprocessing (strip HTML → paragraphs). Our
  `SECFilingChunker` is section-aware, so absolute numbers are not directly
  comparable — use it for relative config comparison.
