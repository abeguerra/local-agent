"""
Two kinds of memory, because "agent memory" is actually two different
engineering problems that companies solve differently:

1. ConversationBuffer -- short-term, in-process memory of the current
   conversation. This is just "the message list," trimmed so it doesn't
   grow the context window (and the per-call cost/latency) without
   bound. This is the memory every chat product has.

2. VectorMemory -- long-term, persisted memory: a local store of text
   chunks + embeddings, searched by semantic similarity. This is the
   RAG ("retrieval-augmented generation") pattern that knowledge-base
   and support agents use in production (and that products like
   MemGPT/Letta or Zep generalize into "agent memory" more broadly):
   instead of stuffing every document into the prompt, you embed once,
   store the vectors, and retrieve only the relevant chunks per query.

Both are intentionally minimal -- no external vector database. The
embedding call goes to Ollama's `/api/embed` endpoint, and similarity
search is a plain cosine-similarity loop over rows in a local SQLite
table. That's a legitimate production pattern at small-to-medium scale
(thousands of chunks); companies reach for Pinecone/Weaviate/pgvector
mainly for scale and concurrent-write reasons, not because the
underlying math is different.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# 1. Short-term conversation memory
# ---------------------------------------------------------------------------

@dataclass
class ConversationBuffer:
    system_prompt: str
    max_messages: int = 40  # crude token-budget guardrail; see docs/ARCHITECTURE.md
    _messages: list[dict] = field(default_factory=list)

    def add(self, role: str, content: str | None = None, **extra) -> None:
        msg = {"role": role}
        if content is not None:
            msg["content"] = content
        msg.update(extra)
        self._messages.append(msg)
        self._trim()

    def _trim(self) -> None:
        # Keep the system prompt implicit (added separately in as_list())
        # and drop the oldest turns once we're over budget. A production
        # system would summarize instead of dropping -- see
        # docs/ARCHITECTURE.md for why that's the next step up.
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages :]

    def as_list(self) -> list[dict]:
        return [{"role": "system", "content": self.system_prompt}, *self._messages]


# ---------------------------------------------------------------------------
# 2. Long-term vector memory (local RAG)
# ---------------------------------------------------------------------------

EmbedFn = Callable[[str], list[float]]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorMemory:
    """A tiny local vector store backed by SQLite.

    Rows are (id, source, text, embedding-as-json). This is intentionally
    the simplest thing that can be called a vector database -- enough to
    demonstrate chunk -> embed -> store -> retrieve, without pulling in
    faiss/chroma/pgvector. Swapping the storage/search internals for one
    of those later is a drop-in change because the public interface
    (`add_text`, `search`) doesn't need to change.
    """

    def __init__(self, db_path: str | Path, embed_fn: EmbedFn):
        self.db_path = Path(db_path)
        self.embed_fn = embed_fn
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "source TEXT, text TEXT, embedding TEXT)"
        )
        self._conn.commit()

    def add_text(self, source: str, text: str, chunk_size: int = 500) -> int:
        """Chunk a document and embed+store each chunk. Returns the
        number of chunks stored."""
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)] or [text]
        for chunk in chunks:
            if not chunk.strip():
                continue
            embedding = self.embed_fn(chunk)
            self._conn.execute(
                "INSERT INTO chunks (source, text, embedding) VALUES (?, ?, ?)",
                (source, chunk, json.dumps(embedding)),
            )
        self._conn.commit()
        return len(chunks)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_embedding = self.embed_fn(query)
        rows = self._conn.execute("SELECT source, text, embedding FROM chunks").fetchall()
        scored = []
        for source, text, embedding_json in rows:
            embedding = json.loads(embedding_json)
            score = _cosine(query_embedding, embedding)
            scored.append({"source": source, "text": text, "score": score})
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
