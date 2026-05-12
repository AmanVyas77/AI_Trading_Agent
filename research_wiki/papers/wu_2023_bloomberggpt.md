---
date: 2023-01-01
tags: [bloomberg, finance-llm, domain-pretraining, open-weight]
category: ai_finance
---

# BloombergGPT: A Large Language Model for Finance

**Authors:** Wu et al. (2023)
**arXiv:** 2303.17564

## Core Finding
BloombergGPT (50B parameters, trained on Bloomberg terminals, news, and filings) underperforms larger general-purpose models on several NLP benchmarks despite domain-specific training.

## Key Lesson
Domain pre-training alone is insufficient. Instruction tuning and model scale remain critical. This supports our decision to use Mistral/Llama (general purpose, instruction-tuned) rather than a specialised finance model.

## Relevance to Trading Agent
Validates using Ollama with Mistral/Llama for Phase 4 rather than seeking a finance-specific model.
