"""
Phase 4a RAG Research Assistant — Interactive CLI

Start a research session:
    python -m src.rag.cli

First-time setup (ingest everything):
    python -m src.rag.cli --ingest

Ingest only specific sources:
    python -m src.rag.cli --ingest --sources papers
    python -m src.rag.cli --ingest --sources edgar --tickers NVDA MSFT AAPL

Non-interactive single query:
    python -m src.rag.cli --query "What are NVDA's export control risks?"

Check what's in the vector store:
    python -m src.rag.cli --stats
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

# ── Rich for pretty terminal output ──────────────────────────────────────────
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.rule import Rule
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from loguru import logger

# Silence loguru during interactive sessions (show only warnings+)
logger.remove()
logger.add(sys.stderr, level="WARNING")

# ── Pipeline imports ──────────────────────────────────────────────────────────
from src.rag.embed.embedder import Embedder
from src.rag.store.chroma_store import ChromaStore
from src.rag.query.retriever import Retriever
from src.rag.query.context_builder import ContextBuilder
from src.rag.query.llm_client import LLMClient
from src.rag.output.obsidian_writer import ObsidianWriter

# ── Console ───────────────────────────────────────────────────────────────────

console = Console() if HAS_RICH else None


def _print(msg: str, style: str = ""):
    if console:
        console.print(msg, style=style)
    else:
        print(msg)

def _rule(title: str = ""):
    if console:
        console.print(Rule(title, style="dim"))
    else:
        print(f"\n{'─'*60} {title}")

def _panel(content: str, title: str = "", style: str = "cyan"):
    if console:
        console.print(Panel(content, title=title, border_style=style))
    else:
        print(f"[{title}]\n{content}\n")


# ── Ticker detection ──────────────────────────────────────────────────────────

# Load universe tickers for detection
_UNIVERSE_PATH = Path(__file__).resolve().parents[2] / "data" / "universe" / "universe.csv"

def _detect_ticker(text: str) -> Optional[str]:
    """Return the first universe ticker mentioned in the text (uppercase)."""
    try:
        import pandas as pd
        tickers = pd.read_csv(_UNIVERSE_PATH)["ticker"].str.upper().tolist()
    except Exception:
        tickers = []

    text_upper = text.upper()
    for t in sorted(tickers, key=len, reverse=True):  # longest first to avoid partial matches
        if re.search(rf"\b{re.escape(t)}\b", text_upper):
            return t
    return None


# ── Ingestion helpers ─────────────────────────────────────────────────────────

def run_ingest(
    store: ChromaStore,
    sources: List[str],
    tickers: Optional[List[str]] = None,
    forms: Optional[List[str]] = None,
    limit: int = 3,
):
    """Ingest one or more source types into ChromaDB."""
    from src.rag.ingest.pdf_loader import PDFLoader
    from src.rag.ingest.obsidian_loader import ObsidianLoader
    from src.rag.ingest.edgar_downloader import EDGARDownloader

    for source in sources:
        if source in ("papers", "all"):
            _rule("Ingesting research papers")
            loader = PDFLoader()
            chunks = loader.load_all()
            added = store.add_chunks(chunks)
            _print(f"  ✓ Papers: {added} new chunks added", style="green")

        if source in ("obsidian", "notes", "all"):
            _rule("Ingesting Obsidian notes")
            loader = ObsidianLoader()
            chunks = loader.load_all()
            added = store.add_chunks(chunks)
            _print(f"  ✓ Obsidian notes: {added} new chunks added", style="green")

        if source in ("edgar", "filings", "all"):
            _rule("Ingesting SEC filings")
            dl = EDGARDownloader(forms=forms or ["10-K", "8-K"], limit=limit)
            if tickers:
                total = 0
                for t in tickers:
                    _print(f"  Downloading {t}…")
                    chunks = dl.download_and_chunk(t)
                    added = store.add_chunks(chunks)
                    total += added
                    _print(f"    {t}: {added} new chunks", style="dim")
                _print(f"  ✓ EDGAR ({', '.join(tickers)}): {total} new chunks added", style="green")
            else:
                results = dl.download_universe()
                all_chunks = [c for chunks in results.values() for c in chunks]
                added = store.add_chunks(all_chunks)
                _print(f"  ✓ EDGAR (all 54 tickers): {added} new chunks added", style="green")


# ── Interactive session ────────────────────────────────────────────────────────

def interactive_session(
    retriever: Retriever,
    ctx_builder: ContextBuilder,
    llm: LLMClient,
    writer: ObsidianWriter,
):
    """Run the interactive REPL."""
    _panel(
        "[bold]AI Trading Agent — Research Assistant[/bold]\n"
        "Ask anything about SEC filings, research papers, or your notes.\n"
        "Type [bold]!save[/bold] after an answer to save it to Obsidian.\n"
        "Type [bold]quit[/bold] or [bold]exit[/bold] to end the session.\n"
        "Tip: mention a ticker (e.g. NVDA) to get factor scores injected automatically.",
        title="Phase 4a RAG",
        style="bold cyan",
    )

    last_question = ""
    last_answer   = ""
    last_sources:  List = []

    while True:
        _rule()
        try:
            if console:
                question = Prompt.ask("[bold green]You[/bold green]").strip()
            else:
                question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            _print("\nGoodbye.", style="dim")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            _print("Goodbye.", style="dim")
            break

        # Save last answer to Obsidian
        if question.lower() == "!save" and last_answer:
            path = writer.save(
                question=last_question,
                answer=last_answer,
                sources=last_sources,
                ticker=_detect_ticker(last_question),
            )
            _print(f"  Saved → {path}", style="green")
            continue

        # Stats command
        if question.lower() in ("!stats", "stats"):
            stats = retriever.store.stats()
            for coll, count in stats.items():
                _print(f"  {coll}: {count:,} documents")
            continue

        # ── Retrieve + generate ───────────────────────────────────────────────
        ticker = _detect_ticker(question)

        _print(
            f"  Searching…" + (f" (ticker detected: {ticker})" if ticker else ""),
            style="dim",
        )

        results = retriever.retrieve(query=question, n=8, ticker=ticker)

        if not results:
            _print(
                "  No relevant documents found. Try ingesting more sources first "
                "(run with --ingest).",
                style="yellow",
            )
            continue

        # Show source summary
        _print(f"  Retrieved {len(results)} source(s):", style="dim")
        for i, r in enumerate(results[:6], 1):
            score = 1 - r["distance"]  # similarity score
            _print(f"    [{i}] {r['citation']}  ({score:.2f})", style="dim")

        # Build prompt and stream response
        prompt = ctx_builder.build(
            question=question,
            results=results,
            ticker=ticker,
        )

        _rule("Answer")
        answer_parts: List[str] = []

        try:
            for token in llm.stream(prompt):
                if console:
                    console.print(token, end="", markup=False)
                else:
                    print(token, end="", flush=True)
                answer_parts.append(token)
        except Exception as exc:
            _print(f"\n[ERROR] {exc}", style="red")
            continue

        print()  # newline after streaming

        last_question = question
        last_answer   = "".join(answer_parts)
        last_sources  = results

        # Prompt to save
        try:
            if console:
                save = Confirm.ask(
                    "[dim]Save this answer to Obsidian?[/dim]",
                    default=False,
                )
            else:
                save = input("Save to Obsidian? [y/N] ").strip().lower() == "y"
        except (EOFError, KeyboardInterrupt):
            save = False

        if save:
            path = writer.save(
                question=last_question,
                answer=last_answer,
                sources=last_sources,
                ticker=ticker,
            )
            _print(f"  Saved → {path}", style="green")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 4a RAG Research Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ingest",   action="store_true", help="Ingest documents into vector store")
    parser.add_argument("--sources",  nargs="*", default=["all"],
                        help="Sources to ingest: papers, obsidian, edgar, all (default: all)")
    parser.add_argument("--tickers",  nargs="*", help="Specific tickers for EDGAR ingestion")
    parser.add_argument("--forms",    nargs="*", default=["10-K", "8-K"])
    parser.add_argument("--limit",    type=int, default=3, help="Filings per ticker (default: 3)")
    parser.add_argument("--query",    type=str, help="Single non-interactive query")
    parser.add_argument("--stats",    action="store_true", help="Print vector store stats and exit")
    parser.add_argument("--backend",  type=str, help="LLM backend: ollama or anthropic")
    parser.add_argument("--model",    type=str, help="Model name override")
    args = parser.parse_args()

    # ── Initialise shared components ──────────────────────────────────────────
    embedder  = Embedder()
    store     = ChromaStore(embedder=embedder)
    retriever = Retriever(store)
    ctx       = ContextBuilder()
    llm       = LLMClient(backend=args.backend, model=args.model)
    writer    = ObsidianWriter()

    # ── Stats mode ────────────────────────────────────────────────────────────
    if args.stats:
        _print("\nVector store contents:", style="bold")
        for coll, count in store.stats().items():
            _print(f"  {coll}: {count:,} documents")
        return

    # ── Ingest mode ───────────────────────────────────────────────────────────
    if args.ingest:
        run_ingest(
            store=store,
            sources=args.sources,
            tickers=args.tickers,
            forms=args.forms,
            limit=args.limit,
        )
        _print("\nIngestion complete. Run without --ingest to start querying.", style="green")
        return

    # ── Single query mode ─────────────────────────────────────────────────────
    if args.query:
        ticker  = _detect_ticker(args.query)
        results = retriever.retrieve(query=args.query, n=8, ticker=ticker)
        prompt  = ctx.build(question=args.query, results=results, ticker=ticker)
        for token in llm.stream(prompt):
            print(token, end="", flush=True)
        print()
        return

    # ── Interactive session ───────────────────────────────────────────────────
    interactive_session(
        retriever=retriever,
        ctx_builder=ctx,
        llm=llm,
        writer=writer,
    )


if __name__ == "__main__":
    main()
