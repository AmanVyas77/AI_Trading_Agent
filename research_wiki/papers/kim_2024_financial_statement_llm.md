---
date: 2024-01-01
tags: [llm, financial-analysis, earnings-prediction, gpt4]
category: ai_finance
---

# Financial Statement Analysis with Large Language Models

**Authors:** Kim, Muhn, Nikolaev (2024) - University of Chicago Booth
**Source:** SSRN 4450289

## Core Finding
GPT-4 Turbo with CoT prompting achieves 60.35% accuracy predicting directional earnings changes vs 52.71% for consensus analysts. Model identified firm names only 0.07% of the time, ruling out memorisation.

## Key Result for This Project
Long-short portfolios on GPT earnings-direction forecasts yield equal-weighted Sharpe ratios of 3.36 and monthly alphas of up to 84 bps after controlling for FF5 + momentum.

## Critical Limitation
Inputs were anonymised - no MD&A, no footnotes, no earnings-call transcripts. This is the gap the RAG system addresses: injecting the narrative context that professional analysts actually use.

## Complementarity
ANN trained jointly on GPT narrative embeddings + financial statement variables achieves highest recorded accuracy 63.16% F1 66.33%. Motivates our hybrid quant + LLM pipeline.
