# Phase 4a Code Review — Claude Code Prompt

Copy and paste everything below this line into a Claude Code session from the project root.

---

## Context: What This Project Is

This is an AI Trading Agent with 4 phases:
- **Phase 1**: Quant momentum strategy (CAGR 18.12%, Sharpe 0.75)
- **Phase 2**: Fundamental strategy via SEC EDGAR XBRL (CAGR 42.86% OOS, Sharpe 1.43)
- **Phase 3**: XGBoost ensemble (OOS CAGR 27.6%, Sharpe 1.29, but -45.3% max DD in full period due to no regime gate)
- **Phase 4a** (just built): RAG research assistant — LLM over SEC filings + research papers + Obsidian notes

**Python**: 3.11, venv at `.venv/`
**DB**: `data/quant_research.db` (SQLite)
**Config**: `config/settings.yaml`

## What Was Just Built (Phase 4a)

The following files were created in this session. Please review each one:

### New files to review:

```
src/rag/__init__.py
src/rag/ingest/chunker.py
src/rag/ingest/edgar_downloader.py
src/rag/ingest/pdf_loader.py
src/rag/ingest/obsidian_loader.py
src/rag/embed/embedder.py
src/rag/store/chroma_store.py
src/rag/query/retriever.py
src/rag/query/context_builder.py
src/rag/query/llm_client.py
src/rag/output/obsidian_writer.py
src/rag/cli.py
scripts/download_papers.py
research_wiki/_index.md
research_wiki/papers/*.md  (7 stub notes)
requirements.txt           (added: chromadb, sentence-transformers, ollama, pdfplumber, arxiv, rich)
config/settings.yaml       (added rag: section at the bottom)
```

## Review Tasks — Please Do All of These

### 1. Read every new file
Read all files listed above in full. Do not skim.

### 2. Check for import errors and broken references
- Verify all `from src.rag.X import Y` paths are correct relative to the project root
- Check that `PROJECT_ROOT = Path(__file__).resolve().parents[N]` has the correct N for each file's depth:
  - `src/rag/ingest/*.py` → parents[3] should reach project root
  - `src/rag/embed/*.py` → parents[3]
  - `src/rag/store/*.py` → parents[3]
  - `src/rag/query/*.py` → parents[3]
  - `src/rag/output/*.py` → parents[3]
  - `src/rag/cli.py` → parents[2]
  - `scripts/download_papers.py` → parents[1]

### 3. Check interface consistency
- `chunker.py` exports `Chunk`, `SECFilingChunker`, `ResearchPaperChunker`, `ObsidianChunker`
- `edgar_downloader.py` imports `SECFilingChunker, Chunk` from chunker — verify the import path and that it passes correct metadata dict fields
- `pdf_loader.py` imports `ResearchPaperChunker, Chunk` — same check
- `obsidian_loader.py` imports `ObsidianChunker, Chunk` — same check
- `chroma_store.py` imports `Chunk` from ingest/chunker and `Embedder` from embed/embedder — check both
- `retriever.py` imports `ChromaStore, ALL_COLLECTIONS` from chroma_store — verify ALL_COLLECTIONS is exported
- `context_builder.py` — check that the parquet file paths and DB path match actual files in `data/processed/` and `data/`
- `cli.py` imports all modules — trace every import and confirm it resolves

### 4. Check the ChromaDB query filter logic
In `chroma_store.py`, the `query()` method passes a `where` filter to ChromaDB.
ChromaDB (v0.5+) uses `$and`/`$or` operators for compound filters. Verify:
- Single-field filters use the correct format: `{"field": {"$eq": "value"}}`
- Multi-field filters use: `{"$and": [{"field1": {...}}, {"field2": {...}}]}`
- The `_build_where()` method in `retriever.py` constructs these correctly

### 5. Check the ChromaDB API compatibility
`chroma_store.py` uses `chromadb.PersistentClient`. Verify this is the correct API for chromadb>=0.5.0.
In older versions (<0.4) the API was `chromadb.Client(Settings(...))`. The new API changed.
Also check: `collection.get(ids=ids)` — confirm this is valid for deduplication.

### 6. Check the Ollama client
In `llm_client.py`, the Ollama streaming call uses:
```python
stream = ollama.chat(model=..., messages=[...], stream=True, options={...})
for chunk in stream:
    token = chunk.get("message", {}).get("content", "")
```
Verify this matches the `ollama` Python library API (v0.3+). The response format may be different — it might be `chunk["message"]["content"]` or need `.message.content` (object attribute, not dict key).

### 7. Check the Anthropic streaming backend
In `llm_client.py` `_stream_anthropic()`:
```python
with client.messages.stream(...) as stream:
    for text in stream.text_stream:
```
Verify this is the correct streaming API for `anthropic` SDK (v0.25+). The `.text_stream` attribute should yield string tokens directly.

### 8. Check settings.yaml RAG section
The new `rag:` section was appended to `config/settings.yaml`. Verify:
- YAML is syntactically valid (no indentation errors)
- The `_load_rag_config()` function in `embedder.py` and `llm_client.py` correctly reads `cfg.get("rag", {})` and then `.get("embedding_model")` etc.

### 9. Check context_builder.py data loading
- `_load_ensemble_scores()`: reads `data/processed/ensemble_scores.parquet` and sets index to `["date", "ticker"]`. Verify the parquet actually has these columns (from Phase 3).
- `_load_macro()`: queries `SELECT * FROM macro_series ORDER BY date DESC LIMIT 7`. Verify the `macro_series` table schema matches the pivot logic — does it have `series` and `value` columns, or is it wide format?
- The macro pivot logic: `df.sort_values("date").groupby("series")["value"].last()` — this assumes long format. If the DB has wide format, this will fail silently.

### 10. Verify the CLI argument flow
In `cli.py`, trace through the `main()` function:
- `--ingest` calls `run_ingest()` which imports `PDFLoader`, `ObsidianLoader`, `EDGARDownloader` — all inside the function body (lazy imports). Confirm this is intentional and works.
- The `interactive_session()` function references `_detect_ticker()` which reads `universe.csv`. Confirm path is correct.
- The `!save` and `!stats` commands in the REPL — confirm the string comparison logic is correct.

### 11. Check for any Python 3.11 compatibility issues
- `from __future__ import annotations` is used throughout — confirm no issues with lazy evaluation of type hints in these files.
- `@dataclass` usage in `chunker.py` — confirm `Chunk` dataclass with `field(default_factory=dict)` is correct.

### 12. Dry-run syntax check
Run this in the terminal from the project root:
```bash
python -c "
import ast, pathlib, sys
errors = []
for f in pathlib.Path('src/rag').rglob('*.py'):
    try:
        ast.parse(f.read_text())
    except SyntaxError as e:
        errors.append(f'{f}: {e}')
for f in ['scripts/download_papers.py']:
    try:
        ast.parse(pathlib.Path(f).read_text())
    except SyntaxError as e:
        errors.append(f'{f}: {e}')
if errors:
    print('SYNTAX ERRORS FOUND:')
    for e in errors: print(e)
    sys.exit(1)
else:
    print('All files pass syntax check.')
"
```

### 13. Check requirements.txt additions
Confirm the following were added correctly:
```
chromadb>=0.5.0
sentence-transformers>=3.0.0
ollama>=0.3.0
pdfplumber>=0.11.0
arxiv>=2.1.0
rich>=13.7.0
```

### 14. Report findings in this format

For each issue found, report:
```
FILE: src/rag/store/chroma_store.py
LINE: ~85
ISSUE: ChromaDB query filter format incorrect for compound filters
SEVERITY: HIGH / MEDIUM / LOW
FIX: [specific code change]
```

Then apply all HIGH and MEDIUM severity fixes directly.

For LOW severity issues (style, minor improvements), list them but don't apply unless asked.

## Expected Clean State After Review

When finished, the following should work without errors:
```bash
# Syntax check passes
python -c "import ast; [ast.parse(open(f).read()) for f in __import__('pathlib').Path('src/rag').rglob('*.py')]"

# Import check (requires deps installed)
python -c "from src.rag.ingest.chunker import Chunk, SECFilingChunker; print('chunker OK')"
python -c "from src.rag.store.chroma_store import ChromaStore; print('store OK')"
python -c "from src.rag.query.llm_client import LLMClient; print('llm_client OK')"
python -c "from src.rag.cli import main; print('cli OK')"

# Stats (no ingestion needed, just initialises store)
python -m src.rag.cli --stats
```
