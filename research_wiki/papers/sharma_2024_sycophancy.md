---
date: 2024-01-01
tags: [sycophancy, llm-failure, co-analyst, bias]
category: governance
---

# Towards Understanding Sycophancy in Language Models

**Authors:** Sharma et al. (2024) - ICLR 2024
**arXiv:** 2310.13548

## Core Finding
Five state-of-the-art AI assistants consistently exhibit sycophancy: they adjust outputs to align with user-provided cues even when those cues are logically irrelevant.

## Co-Analyst Risk
If the human analyst frames a question with an implied view, the LLM may simply confirm rather than independently evaluate. Compounding feedback loop when LLM is downstream of a human who has formed a view.

## Mitigation in Our Design
All queries to the RAG agent use raw questions without pre-stated conclusions. Context injected is quantitative data not opinions. Prompt instructs: answer using ONLY the information provided, flag uncertainty explicitly.
