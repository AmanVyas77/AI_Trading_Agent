---
date: 2020-01-01
tags: [rag, retrieval-augmented-generation, architecture, nlp]
category: ai_finance
---

# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

**Authors:** Lewis et al. (2020) - NeurIPS 2020
**arXiv:** 2005.11401

## Core Architecture
Combines a parametric language model (generator) with a non-parametric retrieval component that fetches relevant passages from an external document corpus at inference time.

## Why RAG Reduces Hallucination
Outputs are grounded in retrievable evidence rather than model weights alone. Critical for financial analysis where fabricated figures are a first-order risk.

## Relevance to Trading Agent
Foundational architecture for our Phase 4 pipeline. ChromaDB = retrieval component. Ollama/Claude = language model. SEC filings + research papers + Obsidian notes = external corpus.
