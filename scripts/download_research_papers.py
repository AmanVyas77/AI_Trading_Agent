#!/usr/bin/env python3
"""
Download curated research papers for the AI Trading Agent platform.

Downloads academic PDFs from free public URLs into organized subfolders
under research_papers/. Papers behind paywalls or SSRN login walls are
flagged for manual download.

Usage (from project root):
    python scripts/download_research_papers.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import requests

# ── Resolve project root relative to this script ─────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = PROJECT_ROOT / "research_papers"

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Paper:
    filename: str          # relative path under research_papers/
    url: str               # download URL (empty string for manual-only)
    manual: bool = False   # if True, skip auto-download
    manual_note: str = ""  # instruction shown in the manual-download summary


# ── Paper registry ────────────────────────────────────────────────────────────

PAPERS: List[Paper] = [
    # ── REGIME DETECTION ──────────────────────────────────────────────────────
    Paper(
        filename="regime_detection/hamilton_1989_regime_switching.pdf",
        url="https://econweb.ucsd.edu/~jhamilto/palgrav1.pdf",
    ),
    Paper(
        filename="regime_detection/ang_timmermann_2012_regime_changes.pdf",
        url="https://www.nber.org/system/files/working_papers/w17182/w17182.pdf",
    ),
    Paper(
        filename="regime_detection/ang_bekaert_2002_regime_switches_rates.pdf",
        url="https://business.columbia.edu/sites/default/files-efs/citation_file_upload/"
            "Journal%20of%20Business%20and%20Economic%20Statistics%2020,%20April%202002,%20163.pdf",
    ),
    Paper(
        filename="regime_detection/hansen_2024_vix_yield_curve_recessions.pdf",
        url="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3943982",
        manual=True,
        manual_note="SSRN — requires free account. Visit the URL and click 'Download This Paper'.",
    ),
    Paper(
        filename="regime_detection/estrella_mishkin_1996_yield_curve_recessions.pdf",
        url="https://www.newyorkfed.org/medialibrary/media/research/current_issues/ci2-7.pdf",
    ),
    Paper(
        filename="regime_detection/regimefolio_2025_vix_sector_optimization.pdf",
        url="https://arxiv.org/pdf/2510.14986",
    ),

    # ── FACTOR INVESTING ──────────────────────────────────────────────────────
    Paper(
        filename="factor_investing/fama_french_1993_three_factor.pdf",
        url="https://www.bauer.uh.edu/rsusmel/phd/Fama-French_JFE93.pdf",
    ),
    Paper(
        filename="factor_investing/asness_2019_quality_minus_junk.pdf",
        url="http://www.econ.yale.edu/~shiller/behfin/2013_04-10/asness-frazzini-pedersen.pdf",
    ),
    Paper(
        filename="factor_investing/piotroski_2000_f_score.pdf",
        url="https://www.semanticscholar.org/paper/Value-Investing%3A-The-Use-of-Historical-Financial-to-Piotroski/0559e92e06dae21e77ea79d79417b8a1d40be772",
        manual=True,
        manual_note="No canonical free URL. Try Semantic Scholar or Google Scholar.",
    ),
    Paper(
        filename="factor_investing/ou_penman_1989_financial_statement_returns.pdf",
        url="https://www.semanticscholar.org/paper/Financial-statement-analysis-and-the-prediction-of-Ou-Penman/",
        manual=True,
        manual_note="No canonical free URL. Try Semantic Scholar or Google Scholar.",
    ),

    # ── INTEREST RATES ────────────────────────────────────────────────────────
    Paper(
        filename="interest_rates/born_2024_firms_investment_sensitivity.pdf",
        url="https://www.benjaminborn.de/files/BBM_Interest_Oct2025.pdf",
    ),
    Paper(
        filename="interest_rates/chen_2025_rate_volatility_bankruptcy.pdf",
        url="https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0317185&type=printable",
    ),
    Paper(
        filename="interest_rates/kim_2024_rate_sensitivity_growth_stocks.pdf",
        url="https://business.columbia.edu/sites/default/files-efs/citation_file_upload/"
            "Interest%20rate%20sensitivity%20and%20growth_Nov13_2024.pdf",
    ),

    # ── NLP & SENTIMENT ───────────────────────────────────────────────────────
    Paper(
        filename="nlp_sentiment/loughran_mcdonald_2011_financial_sentiment.pdf",
        url="https://www.uts.edu.au/globalassets/sites/default/files/adg_cons2015_loughran-mcdonald-je-2011.pdf",
    ),
    Paper(
        filename="nlp_sentiment/yang_2020_finbert.pdf",
        url="https://arxiv.org/pdf/2006.08097",
    ),
    Paper(
        filename="nlp_sentiment/wan_2021_sentiment_network_markets.pdf",
        url="https://www.nature.com/articles/s41598-021-82338-6.pdf",
    ),
    Paper(
        filename="nlp_sentiment/costola_2020_ml_sentiment_covid.pdf",
        url="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690922",
        manual=True,
        manual_note="SSRN — requires free account. Visit the URL and click 'Download This Paper'.",
    ),

    # ── QUANT MODELING ────────────────────────────────────────────────────────
    Paper(
        filename="quant_modeling/mienye_2024_deep_learning_finance_survey.pdf",
        url="https://www.mdpi.com/2673-2688/5/4/101/pdf",
    ),
    Paper(
        filename="quant_modeling/arratia_2021_sentiment_mechanics_statistics.pdf",
        url="https://link.springer.com/content/pdf/10.1007/978-3-030-66891-4_9.pdf",
    ),
]

# ── HTTP config ───────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}

TIMEOUT = 30        # seconds
DELAY   = 3         # seconds between downloads (be polite)
PDF_MAGIC = b"%PDF"

# ── Download logic ────────────────────────────────────────────────────────────

def download_paper(paper: Paper) -> str:
    """
    Attempt to download a single paper.

    Returns one of:
        "downloaded"  — new file saved successfully
        "exists"      — file already present, skipped
        "not_pdf"     — response was not a PDF; file not saved
        "error:<msg>" — request or I/O error
    """
    dest = PAPERS_DIR / paper.filename
    if dest.exists():
        return "exists"

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        resp = requests.get(paper.url, headers=HEADERS, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return f"error:{exc}"

    # Stream into a buffer so we can validate before writing
    chunks: list[bytes] = []
    try:
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            if chunk:
                chunks.append(chunk)
    except requests.RequestException as exc:
        return f"error:stream interrupted — {exc}"

    data = b"".join(chunks)

    # Validate: check Content-Type header first, then fall back to magic bytes
    content_type = resp.headers.get("Content-Type", "").lower()
    is_pdf_content_type = "application/pdf" in content_type
    is_pdf_magic = data[:4] == PDF_MAGIC

    if not is_pdf_content_type and not is_pdf_magic:
        return "not_pdf"

    dest.write_bytes(data)
    return "downloaded"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    auto_papers   = [p for p in PAPERS if not p.manual]
    manual_papers = [p for p in PAPERS if p.manual]

    downloaded: List[Paper] = []
    existed:    List[Paper] = []
    failed:     List[tuple[Paper, str]] = []  # (paper, reason)

    total_auto = len(auto_papers)

    print(f"\n{'=' * 64}")
    print("  Research Paper Downloader")
    print(f"  {total_auto} auto-download | {len(manual_papers)} manual")
    print(f"{'=' * 64}\n")

    for i, paper in enumerate(auto_papers, 1):
        short = paper.filename.split("/")[-1]
        print(f"  [{i}/{total_auto}] {short} ... ", end="", flush=True)

        result = download_paper(paper)

        if result == "downloaded":
            size_kb = (PAPERS_DIR / paper.filename).stat().st_size // 1024
            print(f"✓ downloaded ({size_kb} KB)")
            downloaded.append(paper)
            if i < total_auto:
                time.sleep(DELAY)
        elif result == "exists":
            print("⊘ already exists, skipped")
            existed.append(paper)
        elif result == "not_pdf":
            print("✗ response is not a PDF")
            failed.append((paper, "Response was not a valid PDF"))
        else:
            reason = result.removeprefix("error:")
            print(f"✗ {reason}")
            failed.append((paper, reason))

    # ── Manual downloads block ────────────────────────────────────────────────
    if manual_papers:
        print(f"\n{'=' * 64}")
        print("  === MANUAL DOWNLOADS REQUIRED ===")
        print("  The following papers could not be downloaded automatically.")
        print("  Please download them manually and save to the paths shown:")
        print(f"{'=' * 64}")
        for paper in manual_papers:
            dest = PAPERS_DIR / paper.filename
            status = "  ✓ already saved" if dest.exists() else "  ✗ not yet saved"
            print(f"\n  {dest}")
            print(f"    URL:  {paper.url}")
            print(f"    Note: {paper.manual_note}")
            print(f"    Status: {status}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print("  SUMMARY")
    print(f"{'=' * 64}")
    print(f"  Total attempted (auto):     {total_auto}")
    print(f"  Successfully downloaded:    {len(downloaded)}")
    print(f"  Already existed (skipped):  {len(existed)}")
    print(f"  Failed:                     {len(failed)}")
    if failed:
        for paper, reason in failed:
            print(f"    - {paper.filename}: {reason}")
    print(f"  Manual downloads required:  {len(manual_papers)}")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
