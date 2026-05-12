"""
Context builder — assembles the full prompt context for the LLM.

Combines:
  1. Retrieved document chunks (citations + text)
  2. Live factor scores from Phase 2/3 parquet files (if a ticker is detected)
  3. Current macro regime from the macro_series DB table
  4. The user's question

This is the key bridge between the quantitative strategy layer and the RAG
research assistant — the LLM answers questions with both narrative evidence
AND quantitative context injected automatically.

Usage:
    from src.rag.query.context_builder import ContextBuilder
    cb = ContextBuilder()
    prompt = cb.build(question="What are NVDA's key risks?", results=chunks)
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from loguru import logger

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT     = Path(__file__).resolve().parents[3]
DB_PATH          = PROJECT_ROOT / "data" / "quant_research.db"
SCORES_PATH      = PROJECT_ROOT / "data" / "processed" / "ensemble_scores.parquet"
FUND_SCORES_PATH = PROJECT_ROOT / "data" / "processed" / "fundamental_scores.parquet"


# ── ContextBuilder ────────────────────────────────────────────────────────────

class ContextBuilder:
    """
    Builds a structured LLM prompt context block by merging:
      - Retrieved document chunks
      - Quantitative factor scores for any mentioned ticker
      - Current macro regime snapshot
    """

    MAX_CHUNKS      = 6      # chunks to include in context
    MAX_CHUNK_WORDS = 300    # truncate very long chunks

    def __init__(self):
        self._scores_cache: Optional[pd.DataFrame]    = None
        self._fund_cache:   Optional[pd.DataFrame]    = None
        self._macro_cache:  Optional[Dict]             = None

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        question: str,
        results: List[Dict],
        ticker: Optional[str] = None,
    ) -> str:
        """
        Assemble the full prompt to send to the LLM.

        Parameters
        ----------
        question : the user's original question
        results  : list of retrieval result dicts (from Retriever.retrieve)
        ticker   : if provided, inject factor scores for this ticker
        """
        sections: List[str] = []

        # 1. Macro regime context
        macro_block = self._macro_context()
        if macro_block:
            sections.append(macro_block)

        # 2. Ticker factor scores (if applicable)
        if ticker:
            scores_block = self._scores_context(ticker.upper())
            if scores_block:
                sections.append(scores_block)

        # 3. Retrieved document chunks
        if results:
            doc_block = self._documents_context(results)
            sections.append(doc_block)

        # 4. Instruction header + question
        context_str = "\n\n".join(sections)

        prompt = (
            "You are a rigorous financial research analyst. "
            "Answer the question using ONLY the information provided in the context below. "
            "When you cite a source, include the [Source N] label. "
            "If the context does not contain enough information to answer, say so explicitly "
            "rather than speculating.\n\n"
            f"{context_str}\n\n"
            f"QUESTION: {question}\n\n"
            "ANSWER (cite sources, be specific, flag any uncertainty):"
        )

        return prompt

    # ── Block builders ────────────────────────────────────────────────────────

    def _documents_context(self, results: List[Dict]) -> str:
        lines = ["--- RETRIEVED SOURCES ---"]
        for i, r in enumerate(results[: self.MAX_CHUNKS], start=1):
            citation = r.get("citation", f"Source {i}")
            text = r["text"]
            # Truncate very long chunks
            words = text.split()
            if len(words) > self.MAX_CHUNK_WORDS:
                text = " ".join(words[: self.MAX_CHUNK_WORDS]) + " […]"
            lines.append(f"\n[Source {i}] {citation}\n{text}")
        return "\n".join(lines)

    def _scores_context(self, ticker: str) -> str:
        """Inject the most recent ensemble + fundamental scores for a ticker."""
        lines = [f"--- QUANTITATIVE FACTOR SCORES: {ticker} ---"]
        added = False

        # Ensemble scores
        ens = self._load_ensemble_scores()
        if ens is not None and ticker in ens.index.get_level_values("ticker"):
            ticker_ens = ens.xs(ticker, level="ticker").sort_index()
            if not ticker_ens.empty:
                latest = ticker_ens.iloc[-1]
                date   = ticker_ens.index[-1]
                lines.append(f"Ensemble XGBoost score (as of {str(date)[:10]}): {latest.get('ensemble_score', float('nan')):.3f}")
                added = True

        # Fundamental scores
        fund = self._load_fund_scores()
        if fund is not None:
            if "ticker" in fund.columns:
                row = fund[fund["ticker"] == ticker]
            elif ticker in fund.index.get_level_values("ticker"):
                row = fund.xs(ticker, level="ticker")
            else:
                row = pd.DataFrame()

            if not row.empty:
                latest_fund = row.iloc[-1]
                score_cols = [c for c in row.columns if c not in ("ticker", "date", "period_end", "quarter_end")]
                scores = {c: round(float(latest_fund[c]), 3) for c in score_cols
                          if pd.notna(latest_fund.get(c))}
                if scores:
                    lines.append("Fundamental factor scores (latest quarter):")
                    for k, v in scores.items():
                        lines.append(f"  {k}: {v}")
                    added = True

        return "\n".join(lines) if added else ""

    def _macro_context(self) -> str:
        """Inject latest macro regime snapshot from the DB."""
        macro = self._load_macro()
        if not macro:
            return ""

        lines = ["--- MACRO REGIME SNAPSHOT ---"]
        for k, v in macro.items():
            lines.append(f"  {k}: {v}")

        # Simple regime classification
        vix = macro.get("vix")
        spread = macro.get("yield_spread_10y2y")
        if vix is not None and spread is not None:
            if vix > 25 or spread < 0:
                regime = "RISK-OFF (VIX elevated or yield curve inverted)"
            elif vix < 18 and spread > 0.5:
                regime = "RISK-ON (low volatility, positive curve)"
            else:
                regime = "NEUTRAL"
            lines.append(f"  → Current regime: {regime}")

        return "\n".join(lines)

    # ── Data loaders (lazy, cached) ───────────────────────────────────────────

    def _load_ensemble_scores(self) -> Optional[pd.DataFrame]:
        if self._scores_cache is not None:
            return self._scores_cache
        try:
            df = pd.read_parquet(SCORES_PATH)
            if "ticker" in df.columns and "date" in df.columns:
                df = df.set_index(["date", "ticker"])
            self._scores_cache = df
            return df
        except Exception as e:
            logger.debug(f"Could not load ensemble scores: {e}")
            return None

    def _load_fund_scores(self) -> Optional[pd.DataFrame]:
        if self._fund_cache is not None:
            return self._fund_cache
        try:
            df = pd.read_parquet(FUND_SCORES_PATH)
            self._fund_cache = df
            return df
        except Exception as e:
            logger.debug(f"Could not load fundamental scores: {e}")
            return None

    def _load_macro(self) -> Optional[Dict]:
        if self._macro_cache is not None:
            return self._macro_cache
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql(
                "SELECT * FROM macro_series ORDER BY date DESC LIMIT 7",
                conn,
                parse_dates=["date"],
            )
            conn.close()
            if df.empty:
                return None
            # Pivot: one row per series, take the most recent value
            if "series_name" in df.columns and "value" in df.columns:
                latest = df.sort_values("date").groupby("series_name")["value"].last()
                macro = {k.lower().replace(" ", "_"): round(v, 4) for k, v in latest.items()}
            else:
                # Fallback: assume wide format
                macro = df.iloc[0].dropna().to_dict()
            self._macro_cache = macro
            return macro
        except Exception as e:
            logger.debug(f"Could not load macro data: {e}")
            return None
