"""
LLM client abstraction.

Why wrap Ollama's API instead of calling it directly from core.py
-------------------------------------------------------------------
Companies almost never hardcode one model provider's SDK into their
agent logic. They put a thin interface in between (this is literally
what LiteLLM, Vercel's AI SDK, and most in-house "model gateway" layers
are) so that:

  - swapping models/providers is a config change, not a rewrite
  - you can point the exact same agent loop at a cheap local model for
    development and a hosted model for production
  - you can substitute a scripted fake model in tests (see `FakeLLM`
    below) instead of making real network calls in CI

`LLMClient` is the interface. `OllamaClient` is the real implementation,
talking to Ollama's OpenAI-compatible tool-calling API
(POST /api/chat with a `tools` array). `FakeLLM` is a test double that
plays back a scripted list of responses -- it's how tests/test_core.py
exercises the full agent loop (including the guardrail stop condition)
without a GPU, a model download, or a running Ollama server.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

import urllib.error
import urllib.request


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = ""


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class OllamaClient:
    """Talks to a local Ollama server's /api/chat endpoint.

    Requires Ollama to be installed and running (`ollama serve`, usually
    started automatically by the Ollama app/service) with a tool-calling
    capable model pulled, e.g.:

        ollama pull llama3.2
        ollama pull qwen2.5:7b-instruct   # also tool-capable, a bit stronger

    See README.md for setup. This client intentionally has no fallback
    to a cloud API -- the point of this project is that inference stays
    on your machine.
    """

    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434", timeout: float = 120.0):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"could not reach Ollama at {self.host} -- is `ollama serve` running? "
                f"(underlying error: {e})"
            ) from e

        message = body.get("message", {})
        content = message.get("content", "") or ""
        raw_tool_calls = message.get("tool_calls") or []
        tool_calls = [
            ToolCall(
                name=tc["function"]["name"],
                arguments=tc["function"].get("arguments", {}) or {},
                id=str(tc.get("id", "")),
            )
            for tc in raw_tool_calls
        ]
        return LLMResponse(content=content, tool_calls=tool_calls)

    def embed(self, text: str, model: str = "nomic-embed-text") -> list[float]:
        """Separate endpoint for embeddings, used by VectorMemory. A
        different (usually much smaller) model is normally used for
        embeddings than for chat -- pull it once with
        `ollama pull nomic-embed-text`."""
        payload = {"model": model, "input": text}
        req = urllib.request.Request(
            f"{self.host}/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"could not reach Ollama at {self.host} for embeddings -- "
                f"have you run `ollama pull {model}`? (underlying error: {e})"
            ) from e
        embeddings = body.get("embeddings")
        if embeddings:
            return embeddings[0]
        return body.get("embedding", [])


class FakeLLM:
    """Deterministic test double. Construct with a list of `LLMResponse`
    objects; each call to `.chat()` returns the next one. Lets
    tests/test_core.py exercise the real agent loop -- tool dispatch,
    guardrail stop conditions, tracing -- with zero network calls."""

    def __init__(self, scripted_responses: list[LLMResponse]):
        self._responses = list(scripted_responses)
        self.calls: list[dict] = []  # for assertions in tests

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools})
        if not self._responses:
            return LLMResponse(content="(FakeLLM ran out of scripted responses)")
        return self._responses.pop(0)
