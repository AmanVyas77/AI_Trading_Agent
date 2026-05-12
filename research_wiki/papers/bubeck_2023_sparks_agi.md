---
date: 2023-01-01
tags: [gpt4, numerical-reasoning, failure-modes, llm]
category: ai_finance
---

# Sparks of Artificial General Intelligence: Early Experiments with GPT-4

**Authors:** Bubeck et al. (2023)
**arXiv:** 2303.12712

## Core Finding
Even GPT-4 makes arithmetic errors on multi-step numerical problems, including ratio calculations central to financial statement analysis.

## Failure Modes Documented
Numerical computation errors, temporal confusion (wrong fiscal year), and hallucinated ratios. In Kim et al. (2024), fiscal year identification accuracy was only 2.95% on anonymised statements.

## Design Implication for Trading Agent
The RAG system mitigates hallucination by grounding answers in retrieved text. Numerical computations (ratio checks, score comparisons) should be done by the quantitative layer (Python/pandas), not the LLM.
