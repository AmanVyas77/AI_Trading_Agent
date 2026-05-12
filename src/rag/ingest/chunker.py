"""
Smart document chunking for different source types.

- SECFilingChunker  : splits by named section (Risk Factors, MD&A, etc.) then
                      applies a sliding-window within each section.
- ResearchPaperChunker : paragraph-aware sliding window.
- ObsidianChunker   : treats each note as a single chunk (notes are typically short).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Chunk:
    text: str
    metadata: Dict = field(default_factory=dict)


# ── SEC Filing Chunker ────────────────────────────────────────────────────────

class SECFilingChunker:
    """
    Splits SEC filings by 10-K/10-Q section headings, then applies a
    sliding-window within each section to keep chunks under chunk_size words.
    """

    # Regex patterns for the most analytically valuable 10-K sections
    SECTION_PATTERNS: Dict[str, str] = {
        "business":              r"item\s+1[\.\s]+business\b",
        "risk_factors":          r"item\s+1a[\.\s]+risk\s+factors",
        "mda":                   r"item\s+7[\.\s]+management.{0,10}discussion",
        "quantitative_risk":     r"item\s+7a[\.\s]+quantitative",
        "financial_statements":  r"item\s+8[\.\s]+financial\s+statements",
        "controls":              r"item\s+9a[\.\s]+controls",
    }

    def __init__(self, chunk_size: int = 1_000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: Dict) -> List[Chunk]:
        sections = self._extract_sections(text)
        chunks: List[Chunk] = []

        if sections:
            for section_name, section_text in sections.items():
                section_meta = {**metadata, "section": section_name}
                chunks.extend(self._sliding_window(section_text, section_meta))
        else:
            # No section headers found — fall back to plain sliding window
            chunks = self._sliding_window(text, metadata)

        return chunks

    def _extract_sections(self, text: str) -> Dict[str, str]:
        text_lower = text.lower()
        boundaries: List[tuple] = []

        for name, pattern in self.SECTION_PATTERNS.items():
            for m in re.finditer(pattern, text_lower):
                boundaries.append((m.start(), name))

        if not boundaries:
            return {}

        boundaries.sort(key=lambda x: x[0])
        sections: Dict[str, str] = {}

        for i, (start, name) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            section_text = text[start:end].strip()
            if len(section_text.split()) >= 50:
                sections[name] = section_text

        return sections

    def _sliding_window(self, text: str, metadata: Dict) -> List[Chunk]:
        words = text.split()
        chunks: List[Chunk] = []
        step = self.chunk_size - self.overlap

        for i in range(0, len(words), step):
            window = words[i: i + self.chunk_size]
            if len(window) < 40:
                break
            chunks.append(Chunk(
                text=" ".join(window),
                metadata={**metadata, "chunk_index": len(chunks), "word_offset": i},
            ))

        return chunks


# ── Research Paper Chunker ────────────────────────────────────────────────────

class ResearchPaperChunker:
    """
    Paragraph-aware chunker for academic PDFs.  Accumulates paragraphs until
    the chunk_size word budget is reached, then emits and resets (keeping
    the last paragraph as overlap context).
    """

    def __init__(self, chunk_size: int = 800, overlap_paragraphs: int = 1):
        self.chunk_size = chunk_size
        self.overlap_paragraphs = overlap_paragraphs

    def chunk(self, text: str, metadata: Dict) -> List[Chunk]:
        # Split on blank lines; filter noise
        raw_paras = re.split(r"\n\s*\n", text)
        paras = [p.strip() for p in raw_paras if len(p.strip().split()) >= 20]

        chunks: List[Chunk] = []
        buffer: List[str] = []
        buffer_words = 0

        for para in paras:
            wc = len(para.split())

            if buffer_words + wc > self.chunk_size and buffer:
                chunks.append(Chunk(
                    text=" ".join(buffer),
                    metadata={**metadata, "chunk_index": len(chunks)},
                ))
                # Keep trailing paragraphs for overlap
                buffer = buffer[-self.overlap_paragraphs:]
                buffer_words = sum(len(p.split()) for p in buffer)

            buffer.append(para)
            buffer_words += wc

        if buffer:
            chunks.append(Chunk(
                text=" ".join(buffer),
                metadata={**metadata, "chunk_index": len(chunks)},
            ))

        return chunks


# ── Obsidian Note Chunker ─────────────────────────────────────────────────────

class ObsidianChunker:
    """
    Treats each Obsidian note as a single chunk.  Strips YAML frontmatter
    before embedding so the model sees only the note body.
    """

    MIN_WORDS = 30

    def chunk(self, text: str, metadata: Dict) -> List[Chunk]:
        # Remove YAML frontmatter block
        body = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL).strip()

        if len(body.split()) < self.MIN_WORDS:
            return []

        return [Chunk(text=body, metadata={**metadata, "chunk_index": 0})]
