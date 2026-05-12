"""
Obsidian writer — saves LLM research answers back into research_wiki/ as
structured markdown notes with YAML frontmatter.

Notes are written to:
  research_wiki/research/   — for general questions
  research_wiki/companies/  — when a ticker is detected in the question

Each note includes:
  - YAML frontmatter: date, ticker (if any), tags, sources
  - The original question
  - The LLM's answer
  - A sources section with citations

Usage:
    from src.rag.output.obsidian_writer import ObsidianWriter
    writer = ObsidianWriter()
    path = writer.save(question="...", answer="...", sources=[...], ticker="NVDA")
"""

from __future__ import annotations

import re
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from loguru import logger

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT      = Path(__file__).resolve().parents[3]
RESEARCH_WIKI_DIR = PROJECT_ROOT / "research_wiki"


class ObsidianWriter:
    """
    Writes RAG session outputs as Obsidian-compatible markdown notes.

    Parameters
    ----------
    vault_dir : root of the research_wiki vault
    """

    def __init__(self, vault_dir: Optional[Path] = None):
        self.vault_dir = vault_dir or RESEARCH_WIKI_DIR

    def save(
        self,
        question: str,
        answer: str,
        sources: Optional[List[Dict]] = None,
        ticker: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Path:
        """
        Write a Q&A session to a markdown note.

        Returns the path of the written file.
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M")

        # Decide output directory
        if ticker:
            out_dir = self.vault_dir / "companies"
            filename = f"{ticker.upper()}_{date_str}_{time_str}.md"
        else:
            out_dir = self.vault_dir / "research"
            slug = self._slugify(question)[:40]
            filename = f"{date_str}_{time_str}_{slug}.md"

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename

        content = self._render_note(
            question=question,
            answer=answer,
            sources=sources or [],
            ticker=ticker,
            tags=tags or [],
            date_str=date_str,
        )

        out_path.write_text(content, encoding="utf-8")
        logger.success(f"Saved to research_wiki: {out_path.relative_to(self.vault_dir)}")
        return out_path

    def append_to_company_note(
        self,
        ticker: str,
        question: str,
        answer: str,
        sources: Optional[List[Dict]] = None,
    ) -> Path:
        """
        Append a new Q&A block to a persistent per-ticker note.
        Creates the note if it doesn't exist.
        """
        out_dir  = self.vault_dir / "companies"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ticker.upper()}.md"

        if not out_path.exists():
            # Bootstrap the note with a frontmatter header
            header = self._render_frontmatter(
                ticker=ticker,
                tags=["company-note", ticker.lower()],
                date_str=datetime.now().strftime("%Y-%m-%d"),
            )
            out_path.write_text(header + f"\n# {ticker} Research Notes\n\n", encoding="utf-8")

        # Append the new Q&A block
        block = self._render_qa_block(
            question=question,
            answer=answer,
            sources=sources or [],
        )
        with out_path.open("a", encoding="utf-8") as f:
            f.write("\n" + block)

        logger.success(f"Appended to {out_path.relative_to(self.vault_dir)}")
        return out_path

    # ── Rendering helpers ─────────────────────────────────────────────────────

    def _render_note(
        self,
        question: str,
        answer: str,
        sources: List[Dict],
        ticker: Optional[str],
        tags: List[str],
        date_str: str,
    ) -> str:
        auto_tags = ["rag-output", date_str]
        if ticker:
            auto_tags.append(ticker.lower())
        all_tags = list(dict.fromkeys(auto_tags + tags))  # dedup, preserve order

        frontmatter = self._render_frontmatter(
            ticker=ticker,
            tags=all_tags,
            date_str=date_str,
        )

        return (
            frontmatter
            + f"\n## Question\n{question}\n\n"
            + f"## Answer\n{answer}\n\n"
            + self._render_sources_section(sources)
        )

    def _render_qa_block(self, question: str, answer: str, sources: List[Dict]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        return (
            f"---\n### {now}\n\n"
            f"**Q:** {question}\n\n"
            f"**A:** {answer}\n\n"
            + self._render_sources_section(sources)
        )

    @staticmethod
    def _render_frontmatter(
        ticker: Optional[str],
        tags: List[str],
        date_str: str,
    ) -> str:
        fm: Dict = {"date": date_str, "tags": tags}
        if ticker:
            fm["ticker"] = ticker.upper()
        return "---\n" + yaml.dump(fm, default_flow_style=False, allow_unicode=True) + "---\n"

    @staticmethod
    def _render_sources_section(sources: List[Dict]) -> str:
        if not sources:
            return ""
        lines = ["## Sources\n"]
        for i, s in enumerate(sources, start=1):
            citation = s.get("citation", f"Source {i}")
            lines.append(f"{i}. {citation}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _slugify(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[\s_-]+", "-", text)
        return text.strip("-")
