"""
Auto-download research papers for Phase 4 RAG ingestion.

Sources:
  - arXiv  : 14 papers downloaded automatically via arXiv API
  - SSRN   : 4 papers — links printed for manual download (free account needed)
  - Paywalled journals: 7 papers flagged for manual addition

Usage:
    python scripts/download_papers.py
    python scripts/download_papers.py --dry-run   # just print what would be downloaded
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from loguru import logger

# ── Output directories ────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR   = PROJECT_ROOT / "research_papers"

DIRS = {
    "ai_finance":         PAPERS_DIR / "ai_finance",
    "fundamentals":       PAPERS_DIR / "fundamentals",
    "prompt_engineering": PAPERS_DIR / "prompt_engineering",
    "governance":         PAPERS_DIR / "governance",
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)


# ── Paper registry ────────────────────────────────────────────────────────────

@dataclass
class Paper:
    key:      str             # short slug for filename
    title:    str
    authors:  str
    year:     int
    category: str             # folder key
    source:   str             # "arxiv", "ssrn", "manual"
    id:       str             # arXiv ID, SSRN abstract ID, or note
    notes:    str = ""


PAPERS = [
    # ── arXiv — auto-downloadable ──────────────────────────────────────────
    Paper(
        key="kim_2024_financial_statement_llm",
        title="Financial Statement Analysis with Large Language Models",
        authors="Kim, Muhn, Nikolaev",
        year=2024,
        category="ai_finance",
        source="ssrn",
        id="4450289",
        notes="Core empirical anchor — GPT-4 vs. analysts on earnings prediction",
    ),
    Paper(
        key="kim_2023a_transcripts_risks",
        title="From Transcripts to Insights: Uncovering Corporate Risks Using Generative AI",
        authors="Kim, Muhn, Nikolaev",
        year=2023,
        category="ai_finance",
        source="arxiv",
        id="2310.17721",
        notes="RAG over earnings-call transcripts for corporate risk signals",
    ),
    Paper(
        key="kim_2023b_bloated_disclosures",
        title="Bloated Disclosures: Can ChatGPT Help Investors Process Information?",
        authors="Kim, Muhn, Nikolaev",
        year=2023,
        category="ai_finance",
        source="ssrn",
        id="4425527",
        notes="LLM-assisted disclosure processing; supports RAG design rationale",
    ),
    Paper(
        key="lopez_lira_2023_chatgpt_returns",
        title="Can ChatGPT Forecast Stock Price Movements?",
        authors="Lopez-Lira, Tang",
        year=2023,
        category="ai_finance",
        source="arxiv",
        id="2304.07619",
        notes="GPT explains short-term returns from headlines",
    ),
    Paper(
        key="cao_2024_man_plus_machine",
        title="From Man vs. Machine to Man + Machine: The Art and AI of Stock Analyses",
        authors="Cao, Jiang, Wang, Yang",
        year=2024,
        category="ai_finance",
        source="ssrn",
        id="3840838",
        notes="Man+machine outperforms either alone — core motivation for hybrid workflow",
    ),
    Paper(
        key="bybee_2023_ghost_machine",
        title="The Ghost in the Machine: Generating Beliefs with Large Language Models",
        authors="Bybee",
        year=2023,
        category="ai_finance",
        source="arxiv",
        id="2305.02823",
        notes="GPT macro predictions align with expert surveys",
    ),
    Paper(
        key="wu_2023_bloomberggpt",
        title="BloombergGPT: A Large Language Model for Finance",
        authors="Wu et al.",
        year=2023,
        category="ai_finance",
        source="arxiv",
        id="2303.17564",
        notes="Domain pre-training alone insufficient; instruction tuning matters",
    ),
    Paper(
        key="xie_2024_finben",
        title="FinBen: A Holistic Financial Benchmark for Large Language Models",
        authors="Xie et al.",
        year=2024,
        category="ai_finance",
        source="arxiv",
        id="2402.12659",
        notes="42 datasets, 24 financial tasks — benchmark for model evaluation",
    ),
    Paper(
        key="xie_2023_pixiu",
        title="PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance",
        authors="Xie et al.",
        year=2023,
        category="ai_finance",
        source="arxiv",
        id="2306.05443",
        notes="FinMA (Llama fine-tuned); numerical reasoning weakness documented",
    ),
    Paper(
        key="shah_2022_flue_flang",
        title="When FLUE Meets FLANG: Benchmarks and Large Pre-trained Language Model for Finance",
        authors="Shah et al.",
        year=2022,
        category="ai_finance",
        source="arxiv",
        id="2211.00083",
        notes="FLUE benchmark for financial NLP",
    ),
    Paper(
        key="lewis_2020_rag",
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        authors="Lewis et al.",
        year=2020,
        category="ai_finance",
        source="arxiv",
        id="2005.11401",
        notes="Foundational RAG architecture paper (NeurIPS 2020)",
    ),
    Paper(
        key="bubeck_2023_sparks_agi",
        title="Sparks of Artificial General Intelligence: Early Experiments with GPT-4",
        authors="Bubeck et al.",
        year=2023,
        category="ai_finance",
        source="arxiv",
        id="2303.12712",
        notes="Documents numerical reasoning errors — key failure mode",
    ),
    Paper(
        key="sharma_2024_sycophancy",
        title="Towards Understanding Sycophancy in Language Models",
        authors="Sharma et al.",
        year=2024,
        category="ai_finance",
        source="arxiv",
        id="2310.13548",
        notes="Sycophancy failure mode — critical for co-analyst deployment",
    ),
    Paper(
        key="wei_2022_chain_of_thought",
        title="Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        authors="Wei et al.",
        year=2022,
        category="prompt_engineering",
        source="arxiv",
        id="2201.11903",
        notes="Foundational CoT paper; 60.35% vs 52.33% accuracy gap in finance tasks",
    ),
    Paper(
        key="zhang_2023_auto_cot",
        title="Automatic Chain of Thought Prompting in Large Language Models",
        authors="Zhang et al.",
        year=2023,
        category="prompt_engineering",
        source="arxiv",
        id="2210.11610",
        notes="Auto-CoT generalises across structured reasoning tasks (ICLR 2023)",
    ),
    Paper(
        key="touvron_2023_llama2",
        title="Llama 2: Open Foundation and Fine-Tuned Chat Models",
        authors="Touvron et al.",
        year=2023,
        category="ai_finance",
        source="arxiv",
        id="2307.09288",
        notes="Open-weight model approaching GPT-3.5 at 70B — supports Ollama deployment",
    ),
    Paper(
        key="jiang_2023_mistral",
        title="Mistral 7B",
        authors="Jiang et al.",
        year=2023,
        category="ai_finance",
        source="arxiv",
        id="2310.06825",
        notes="Default model for Ollama local deployment in Phase 4",
    ),
    # ── PAYWALLED — add manually if you have library access ───────────────
    Paper(
        key="bouwman_1987_analyst_decisions",
        title="How Do Financial Analysts Make Decisions?",
        authors="Bouwman, Frishkoff, Frishkoff",
        year=1987,
        category="fundamentals",
        source="manual",
        id="Accounting, Organizations and Society 12(1):1–29",
        notes="PAYWALLED — foundational analyst cognition study; add PDF manually if available",
    ),
    Paper(
        key="chen_2022_xbrl_earnings",
        title="Predicting Future Earnings Changes Using Machine Learning and Detailed Financial Data",
        authors="Chen, Cho, Dou, Lev",
        year=2022,
        category="fundamentals",
        source="manual",
        id="Journal of Accounting Research 60(2):467–515",
        notes="PAYWALLED — 12,000 XBRL tags ML model; benchmark for GPT comparison",
    ),
    Paper(
        key="costello_2020_machine_man",
        title="Machine + Man: A Field Experiment on the Role of Discretion in Augmenting AI-Based Lending Models",
        authors="Costello, Down, Mehta",
        year=2020,
        category="governance",
        source="manual",
        id="Journal of Accounting and Economics 70(2-3):101360",
        notes="PAYWALLED — human-in-the-loop design template",
    ),
    Paper(
        key="fama_french_2015_five_factor",
        title="A Five-Factor Asset Pricing Model",
        authors="Fama, French",
        year=2015,
        category="fundamentals",
        source="manual",
        id="Journal of Financial Economics 116(1):1–22",
        notes="PAYWALLED — FF5 model; used as benchmark in Kim et al. (2024). Often freely available via Google Scholar.",
    ),
    Paper(
        key="hunt_2022_ml_earnings",
        title="Improving Earnings Predictions and Abnormal Returns with Machine Learning",
        authors="Hunt, Myers, Myers",
        year=2022,
        category="fundamentals",
        source="manual",
        id="Accounting Horizons 36(1):131–149",
        notes="PAYWALLED — ML accuracy improvements over time-series models",
    ),
    Paper(
        key="ou_penman_1989_financial_prediction",
        title="Financial Statement Analysis and the Prediction of Stock Returns",
        authors="Ou, Penman",
        year=1989,
        category="fundamentals",
        source="manual",
        id="Journal of Accounting and Economics 11(4):295–329",
        notes="PAYWALLED — seminal 59-predictor logistic regression paper. Often freely available.",
    ),
    Paper(
        key="tversky_kahneman_1974_anchoring",
        title="Judgment Under Uncertainty: Heuristics and Biases",
        authors="Tversky, Kahneman",
        year=1974,
        category="governance",
        source="manual",
        id="Science 185(4157):1124–1131",
        notes="PAYWALLED — anchoring bias; foundational for co-analyst sycophancy risk",
    ),
    Paper(
        key="dellacqua_2023_jagged_frontier",
        title="Navigating the Jagged Technological Frontier: Field Experimental Evidence on AI and Knowledge Worker Productivity",
        authors="Dell'Acqua et al.",
        year=2023,
        category="governance",
        source="manual",
        id="HBS Working Paper 24-013 — try: https://www.hbs.edu/faculty/Pages/item.aspx?num=64700",
        notes="Working paper — check HBS website or SSRN for free PDF",
    ),
]


# ── Download functions ────────────────────────────────────────────────────────

ARXIV_PDF_URL = "https://arxiv.org/pdf/{id}.pdf"
SSRN_PDF_URL  = "https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID{id}.pdf"

HEADERS = {
    "User-Agent": "QuantResearchPlatform/1.0 (amanvyas1505@gmail.com)"
}


def download_arxiv(paper: Paper, dry_run: bool = False) -> bool:
    dest = DIRS[paper.category] / f"{paper.key}.pdf"
    if dest.exists():
        logger.info(f"  [SKIP] {paper.key} already downloaded")
        return True

    url = ARXIV_PDF_URL.format(id=paper.id)
    if dry_run:
        print(f"  [DRY RUN] Would download: {url}")
        return True

    try:
        logger.info(f"  Downloading {paper.key} from arXiv…")
        r = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        r.raise_for_status()
        dest.write_bytes(r.content)
        logger.success(f"  ✓ {dest.name} ({len(r.content) // 1024} KB)")
        time.sleep(3)  # be kind to arXiv servers
        return True
    except Exception as exc:
        logger.warning(f"  ✗ Failed to download {paper.key}: {exc}")
        return False


def print_ssrn_instructions(paper: Paper):
    print(f"\n  [SSRN] {paper.title}")
    print(f"         Authors: {paper.authors} ({paper.year})")
    print(f"         URL: https://ssrn.com/abstract={paper.id}")
    print(f"         Save as: {DIRS[paper.category] / (paper.key + '.pdf')}")
    print(f"         Note: {paper.notes}")


def print_manual_instructions(paper: Paper):
    print(f"\n  [MANUAL] {paper.title}")
    print(f"           Authors: {paper.authors} ({paper.year})")
    print(f"           Source: {paper.id}")
    print(f"           Save as: {DIRS[paper.category] / (paper.key + '.pdf')}")
    print(f"           Note: {paper.notes}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download research papers for RAG pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without downloading")
    args = parser.parse_args()

    arxiv_papers  = [p for p in PAPERS if p.source == "arxiv"]
    ssrn_papers   = [p for p in PAPERS if p.source == "ssrn"]
    manual_papers = [p for p in PAPERS if p.source == "manual"]

    print(f"\n{'='*60}")
    print(f"  Research Paper Downloader — Phase 4 RAG")
    print(f"  {len(arxiv_papers)} arXiv (auto) | {len(ssrn_papers)} SSRN | {len(manual_papers)} manual")
    print(f"{'='*60}\n")

    # ── arXiv downloads ───────────────────────────────────────────────────────
    print(f"\n[1/3] Downloading arXiv papers automatically…")
    success, failed = 0, 0
    for p in arxiv_papers:
        ok = download_arxiv(p, dry_run=args.dry_run)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\n  arXiv: {success} downloaded, {failed} failed")

    # ── SSRN instructions ─────────────────────────────────────────────────────
    print(f"\n[2/3] SSRN papers — download manually (free account required):")
    for p in ssrn_papers:
        print_ssrn_instructions(p)

    # ── Paywalled / manual instructions ──────────────────────────────────────
    print(f"\n[3/3] Paywalled / working papers — add PDFs manually if available:")
    for p in manual_papers:
        print_manual_instructions(p)

    print(f"\n{'='*60}")
    print("  Next step: run ingestion to embed all downloaded papers:")
    print("    python -m src.rag.cli --ingest --sources papers")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
