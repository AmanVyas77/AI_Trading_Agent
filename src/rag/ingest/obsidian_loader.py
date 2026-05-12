"""
Obsidian / research_wiki markdown loader for Phase 4 RAG pipeline.

Scans the research_wiki/ directory, reads every .md file, strips YAML
frontmatter, and returns Chunk objects ready for embedding.

Notes are treated as single chunks — they are typically short enough that
splitting them would lose the connective context that makes Obsidian valuable.

Usage:
    from src.rag.ingest.obsidian_loader import ObsidianLoader
    loader = ObsidianLoader()
    chunks = loader.load_all()
"""

from __future__ import annotations

import re
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from loguru import logger

from src.rag.ingest.chunker import ObsidianChunker, Chunk

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT     = Path(__file__).resolve().parents[3]
RESEARCH_WIKI_DIR = PROJECT_ROOT / "research_wiki"


# ── Loader ────────────────────────────────────────────────────────────────────

class ObsidianLoader:
    """
    Loads .md files from the research_wiki/ vault.

    YAML frontmatter (if present) is parsed into metadata and stripped from
    the embedded text.  Wikilinks ([[note]]) are preserved as plain text so
    the LLM can follow conceptual connections.

    Parameters
    ----------
    vault_dir : root of the Obsidian vault (default: research_wiki/)
    """

    def __init__(self, vault_dir: Optional[Path] = None):
        self.vault_dir = vault_dir or RESEARCH_WIKI_DIR
        self.chunker = ObsidianChunker()

    # ── Public API ────────────────────────────────────────────────────────────

    def load_all(self) -> List[Chunk]:
        """Load every .md file in the vault and return chunks."""
        md_paths = sorted(self.vault_dir.rglob("*.md"))

        if not md_paths:
            logger.info(f"No markdown files found in {self.vault_dir} yet.")
            return []

        logger.info(f"ObsidianLoader: found {len(md_paths)} note(s).")
        all_chunks: List[Chunk] = []

        for md_path in md_paths:
            try:
                chunks = self.load_note(md_path)
                all_chunks.extend(chunks)
            except Exception as exc:
                logger.warning(f"  Failed to load {md_path.name}: {exc}")

        logger.success(f"ObsidianLoader: {len(all_chunks)} chunks from {len(md_paths)} notes.")
        return all_chunks

    def load_note(self, md_path: Path) -> List[Chunk]:
        """Parse a single .md file and return its chunk(s)."""
        raw = md_path.read_text(encoding="utf-8", errors="replace")
        frontmatter, body = self._split_frontmatter(raw)

        # Resolve relative path within the vault for a stable ID
        try:
            rel_path = md_path.relative_to(self.vault_dir)
        except ValueError:
            rel_path = md_path

        # Infer sub-section from parent folder
        section = rel_path.parts[0] if len(rel_path.parts) > 1 else "root"

        metadata: Dict[str, Any] = {
            "source":    "obsidian",
            "filepath":  str(rel_path),
            "note_name": md_path.stem,
            "section":   section,
        }

        # Merge any frontmatter fields into metadata
        if frontmatter:
            for key in ("ticker", "date", "tags", "category", "author"):
                if key in frontmatter:
                    metadata[key] = frontmatter[key]

        return self.chunker.chunk(body, metadata)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _split_frontmatter(text: str):
        """
        Returns (frontmatter_dict_or_None, body_string).
        Handles both --- delimited YAML and notes with no frontmatter.
        """
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, flags=re.DOTALL)
        if not match:
            return None, text

        try:
            fm = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            fm = {}

        body = text[match.end():].strip()
        return fm, body
