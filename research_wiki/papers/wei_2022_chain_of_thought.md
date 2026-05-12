---
date: 2022-01-01
tags: [prompt-engineering, chain-of-thought, reasoning, llm]
category: prompt_engineering
---

# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

**Authors:** Wei et al. (2022) - NeurIPS 2022
**arXiv:** 2201.11903

## Core Finding
CoT prompting substantially enhances LLM performance on multi-step reasoning. In Kim et al. (2024), CoT raises accuracy from 52.33% to 60.35%.

## Applied to Financial Analysis
Effective CoT for financial statement analysis: identify trends, compute ratios, interpret in context, then render directional judgment. This mirrors the analyst cognitive workflow from Bouwman et al. (1987).

## Relevance to Trading Agent
Informs prompt library design for Phase 4. All prompts in llm_client.py use implicit CoT structure.
