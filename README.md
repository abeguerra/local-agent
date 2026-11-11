# local-agent

A small, fully-local tool-calling AI agent, built as a learning project
for understanding how companies actually architect AI agents in
production. It runs entirely on your machine against a local model
through [Ollama](https://ollama.com) — no API keys, no cloud calls, no
hidden framework magic. Every file is short enough to read end to end,
and every design choice in it is explained in terms of the real pattern
it stands in for. Start with `docs/ARCHITECTURE.md` for that mapping;
this file is just setup and a tour.

## What's actually here

- a tool-calling agent loop (`agent/core.py`) — the same ReAct-style
  loop under the hood of OpenAI's Assistants API, Claude's tool use,
  and LangChain's `AgentExecutor`
- a model-agnostic LLM client (`agent/llm.py`) with a real Ollama
  backend and a scripted fake for tests — the same reason companies
  put LiteLLM or an in-house gateway between their agent code and a
  specific provider's SDK
- a tool registry (`agent/tools/`) using JSON Schema, the same shape
  OpenAI/Ollama tool-calling and Anthropic's MCP use
- two kinds of memory (`agent/memory.py`): a short-term conversation
  buffer, and a local vector store for retrieval-augmented search over
  a notes folder
- guardrails (`agent/guardrails.py`): a loop-limit, a tool allowlist,
  and a confirm-before-destructive-action hook
- structured JSONL tracing (`agent/trace.py`) — a minimal local
  stand-in for what LangSmith/Langfuse/Helicone give you in production,
  following the shape of OpenTelemetry's GenAI span conventions
- 25 offline tests (`tests/`) that exercise the whole loop — including
  the guardrail stop condition — using a fake LLM, no network required

## Setup

**1. Install Ollama** (this runs the model locally):

```bash
curl -fsSL https://ollama.com/install.sh | sh     # Linux
# or: brew install ollama                          # macOS
# or download the installer from ollama.com         # Windows
```

**2. Pull a tool-calling-capable model.** Not every local model
supports structured tool calls — these do:

```bash
ollama pull llama3.2              # ~2GB, good default
# ollama pull qwen2.5:7b-instruct # ~4.7GB, noticeably stronger reasoning
ollama pull nomic-embed-text      # small embedding model, powers search_notes
```

**3. Install Python dev dependencies** (stdlib only for runtime; pytest
for tests):

```bash
pip install -r requirements.txt
```

**4. Run the tests** (no Ollama required — they use a fake LLM):

```bash
python3 -m pytest tests/ -v
```

**5. Make sure Ollama is running, then start the agent:**

```bash
ollama serve &      # if it isn't already running as a background service
python3 cli.py
```

## Example session

```
Local agent ready. Tools: calculator, get_current_time, list_workspace_files, read_file, write_file, search_notes
Type 'exit' to quit, 'trace' to show the last run's trace.

you> what's (14 * 3) + today's weekday?
agent> 14 * 3 is 42. Today is Thursday.

you> trace
[01] USER        "what's (14 * 3) + today's weekday?"
[02] LLM ->      sending 2 messages, 6 tools available
[03] LLM <-      wants to call: calculator, get_current_time
[04] TOOL ->     calculator({'expression': '14 * 3'})
[05] TOOL <-     calculator -> ok  (0ms)
[06] TOOL ->     get_current_time({})
[07] TOOL <-     get_current_time -> ok  (0ms)
[08] LLM ->      sending 4 messages, 6 tools available
[09] LLM <-      final answer (41 chars)
[10] FINAL       "14 * 3 is 42. Today is Thursday."

you> save a file called plan.txt with "step 1: ship it"
  [confirm] agent wants to call write_file({'path': 'plan.txt', 'content': 'step 1: ship it'})
  allow? [y/N] y
agent> Done — I saved that to plan.txt.
```

That confirmation prompt is the `confirm_before` guardrail in
`agent/guardrails.py` firing on `write_file`. Change which tools require
it, or turn it off, in `GuardrailConfig`.

## Project layout

```
agent/
  core.py           the agent loop
  llm.py            LLM client abstraction (Ollama + FakeLLM for tests)
  memory.py         ConversationBuffer + VectorMemory (local RAG)
  guardrails.py     loop limit, tool allowlist, confirmation hook
  trace.py          JSONL structured tracing + pretty-printer
  tools/
    __init__.py     the tool registry (JSON Schema based)
    builtin.py      calculator, get_current_time, file read/write/list
    notes_search.py the RAG search tool over data/notes/
cli.py               interactive REPL
tests/               25 offline tests using FakeLLM, no network needed
data/notes/          sample markdown notes, indexed by search_notes
workspace/           sandbox directory read_file/write_file are confined to
docs/ARCHITECTURE.md the deep dive: what each piece maps to in real systems
```

## Where to go next

`docs/ARCHITECTURE.md` walks through every module and names the real
framework, product, or published pattern each one is a minimal version
of, plus concrete next steps for extending this toward what a company
would actually ship (multi-agent orchestration, real moderation models,
a proper vector database, streaming responses, and so on).

A good way to use this project for learning: pick one module, read the
docstring at the top of the file (every file has one explaining *why*
it's built this way, not just what it does), then read the matching
section of `docs/ARCHITECTURE.md`, then go look at how a real framework
(LangGraph, the OpenAI Agents SDK, CrewAI) solves the same problem and
compare.
