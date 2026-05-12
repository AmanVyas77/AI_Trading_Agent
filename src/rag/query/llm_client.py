"""
LLM client — abstracts the generation backend so it can be swapped without
touching the rest of the pipeline.

Supported backends (set via TRADING_AGENT_LLM_BACKEND env var or config):
  - "ollama"    : local Ollama server (default; free, private, runs on your Mac)
  - "anthropic" : Claude API via Anthropic SDK (set ANTHROPIC_API_KEY)

For Ollama setup:
    brew install ollama           # or download from ollama.com
    ollama pull mistral           # ~4 GB; or "llama3.1" for a larger model
    ollama serve                  # starts the local server (port 11434)

Usage:
    from src.rag.query.llm_client import LLMClient
    client = LLMClient()
    for token in client.stream("Your prompt here"):
        print(token, end="", flush=True)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Optional

import yaml
from loguru import logger

# ── Config ────────────────────────────────────────────────────────────────────

_SETTINGS_PATH = Path(__file__).resolve().parents[3] / "config" / "settings.yaml"


def _load_rag_config() -> dict:
    try:
        with open(_SETTINGS_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("rag", {})
    except Exception:
        return {}


# ── LLMClient ─────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Thin wrapper around Ollama (default) or the Anthropic API.

    Parameters
    ----------
    backend    : "ollama" or "anthropic" (default: from env/config or "ollama")
    model      : model identifier (e.g. "mistral", "llama3.1", "claude-sonnet-4-6")
    temperature: generation temperature (lower = more factual)
    """

    OLLAMA_DEFAULT_MODEL     = "mistral"
    ANTHROPIC_DEFAULT_MODEL  = "claude-sonnet-4-6"

    def __init__(
        self,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ):
        cfg = _load_rag_config()

        self.backend     = backend or os.getenv("TRADING_AGENT_LLM_BACKEND") or cfg.get("llm_backend", "ollama")
        self.temperature = temperature
        self.model       = model or cfg.get("llm_model") or self._default_model()

        logger.info(f"LLMClient: backend={self.backend}, model={self.model}")

    def _default_model(self) -> str:
        return (
            self.ANTHROPIC_DEFAULT_MODEL
            if self.backend == "anthropic"
            else self.OLLAMA_DEFAULT_MODEL
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Yield response tokens one at a time for streaming output.
        Falls back to yielding the full response as a single chunk if the
        backend doesn't support streaming.
        """
        if self.backend == "anthropic":
            yield from self._stream_anthropic(prompt)
        else:
            yield from self._stream_ollama(prompt)

    def generate(self, prompt: str) -> str:
        """Return the full response as a string (non-streaming)."""
        return "".join(self.stream(prompt))

    # ── Ollama backend ────────────────────────────────────────────────────────

    def _stream_ollama(self, prompt: str) -> Generator[str, None, None]:
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "ollama Python client not installed.\n"
                "Run: pip install ollama\n"
                "Also install the Ollama app: https://ollama.com/download"
            )

        try:
            stream = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options={"temperature": self.temperature, "num_ctx": 4096},
            )
            for chunk in stream:
                token = chunk.message.content
                if token:
                    yield token
        except Exception as exc:
            # Provide a helpful error if Ollama isn't running
            if "connection" in str(exc).lower() or "refused" in str(exc).lower():
                raise RuntimeError(
                    "Cannot connect to Ollama.\n"
                    "Start it with: ollama serve\n"
                    f"Then pull the model: ollama pull {self.model}"
                ) from exc
            raise

    # ── Anthropic backend ─────────────────────────────────────────────────────

    def _stream_anthropic(self, prompt: str) -> Generator[str, None, None]:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic SDK not installed. Run: pip install anthropic")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set.\n"
                "Export it: export ANTHROPIC_API_KEY=sk-ant-..."
            )

        client = anthropic.Anthropic(api_key=api_key)

        with client.messages.stream(
            model=self.model,
            max_tokens=2048,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text
