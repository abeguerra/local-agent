"""
A RAG tool: lets the agent search a local notes folder by meaning
rather than exact keyword match. This is the same shape as every
"chat with your docs" / internal-knowledge-bot product: embed once at
startup, retrieve top-k relevant chunks per question, hand those chunks
to the model as context instead of the whole corpus.

This module is wired up lazily (`init_notes_tool`) rather than at import
time, because building the vector index requires a live embedding
model -- we don't want importing `agent.tools.builtin` to require a
running Ollama server for tools that don't need one.
"""

from __future__ import annotations

from pathlib import Path

from agent.memory import VectorMemory
from agent.tools import ToolError, registry

_memory: VectorMemory | None = None


def init_notes_tool(memory: VectorMemory, notes_dir: str | Path) -> int:
    """Index every .md/.txt file under notes_dir into `memory`. Returns
    the number of chunks indexed. Call this once at startup."""
    global _memory
    _memory = memory
    notes_dir = Path(notes_dir)
    total = 0
    if memory.count() == 0:  # don't re-index on every run
        for path in sorted(notes_dir.glob("*.md")) + sorted(notes_dir.glob("*.txt")):
            text = path.read_text(encoding="utf-8")
            total += memory.add_text(source=path.name, text=text)
    return total


@registry.register(
    name="search_notes",
    description="Semantically search the local notes knowledge base and return the most relevant snippets. Use this before answering questions about the notes content instead of guessing.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "what to search for"},
            "top_k": {"type": "integer", "description": "how many snippets to return, default 3"},
        },
        "required": ["query"],
    },
)
def search_notes(query: str, top_k: int = 3) -> dict:
    if _memory is None:
        raise ToolError("notes search is not initialized (no embedding model configured)")
    results = _memory.search(query, top_k=top_k)
    return {"query": query, "results": results}
