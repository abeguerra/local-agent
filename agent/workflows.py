"""
Workflows: tool calls your code makes, not the model.

`agent/core.py` is an *agent*: the model chooses which tools to call,
and in what order, at runtime. A *workflow* (Anthropic's term, see
docs/ARCHITECTURE.md) fixes that order in code. The model is still
used, but only for the step that actually needs language -- here,
writing an answer -- and it gets no tools at all for that step.

The sample below is a "notes Q&A" workflow, a fixed four-step chain:

    1. get_current_time       code calls it   -> date for the file name
    2. search_notes(question) code calls it   -> the context (this is
                                                 plain RAG: retrieval
                                                 always happens, the
                                                 model isn't asked)
    3. llm.chat(tools=[])     one model call  -> the answer
    4. write_file(...)        code calls it   -> saved to workspace/answers/

Compared with asking the agent "answer from my notes and save it":

  - it always searches and always saves -- a small local model can't
    forget to, or call write_file with a made-up path
  - it costs exactly one model call, not one per tool round-trip
  - if search finds nothing, *code* decides to stop before the model
    is called at all, instead of hoping the model admits it doesn't
    know

The trade-off is flexibility: it can only do this one job. That's the
usual reason to reach for a workflow first and an agent loop only when
the steps genuinely can't be known ahead of time.

Every tool call here still goes through `run_tool` from core.py, so
the allowlist, the confirm-before-write_file prompt, and the trace
apply exactly as they do in the agent loop. Bypassing them just because
"our code made this call, not the model" is how a workflow ends up
writing files nobody approved.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

from agent.core import run_tool
from agent.guardrails import Guardrails
from agent.llm import LLMClient
from agent.tools import ToolRegistry
from agent.trace import Trace

ANSWER_PROMPT = """You answer questions using only the notes provided in the user's \
message. If the notes don't contain the answer, say so plainly instead of guessing. \
Be concise, and mention which note file the answer came from."""


@dataclass
class NotesAnswer:
    question: str
    answer: str
    sources: list[str] = field(default_factory=list)
    saved_path: str | None = None
    # Why the workflow stopped early or skipped a step, if it did.
    problem: str = ""


class NotesQAWorkflow:
    name = "notes_qa"

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        guardrails: Guardrails | None = None,
        confirm_callback=None,  # same contract as Agent's
        top_k: int = 3,
        min_score: float = 0.3,  # below this, a snippet is treated as "not relevant"
    ):
        self.llm = llm
        self.tools = tools
        self.guardrails = guardrails or Guardrails()
        self.confirm_callback = confirm_callback or (lambda name, args: True)
        self.top_k = top_k
        self.min_score = min_score

    def _tool(self, name: str, arguments: dict, trace: Trace):
        return run_tool(
            name,
            arguments,
            tools=self.tools,
            guardrails=self.guardrails,
            confirm_callback=self.confirm_callback,
            trace=trace,
        )

    def run(self, question: str, trace: Trace | None = None, save: bool = True) -> NotesAnswer:
        trace = trace or Trace.start("data/traces")
        self.guardrails.check_input(question)
        trace.log("workflow_start", workflow=self.name, question=question)
        result = self._run(question, trace, save)
        trace.log("workflow_end", workflow=self.name, saved_path=result.saved_path, problem=result.problem)
        return result

    def _run(self, question: str, trace: Trace, save: bool) -> NotesAnswer:
        # Step 1: the date, for the saved file's name and header.
        now = self._tool("get_current_time", {}, trace)
        timestamp = now.value["iso"] if now.ok else dt.datetime.now().isoformat(timespec="seconds")

        # Step 2: retrieval. Not optional, not the model's call.
        search = self._tool("search_notes", {"query": question, "top_k": self.top_k}, trace)
        if not search.ok:
            return NotesAnswer(question, answer="I couldn't search the notes.", problem=search.error)

        hits = [h for h in search.value["results"] if h["score"] >= self.min_score]
        if not hits:
            # A code-side stopping condition: with nothing relevant to
            # ground on, calling the model would only invite a guess.
            return NotesAnswer(question, answer="Nothing in the notes matches that question.", problem="no relevant notes")

        # Step 3: the only model call, with no tools offered.
        context = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        messages = [
            {"role": "system", "content": ANSWER_PROMPT},
            {"role": "user", "content": f"Notes:\n\n{context}\n\nQuestion: {question}"},
        ]
        trace.log("llm_call_start", n_messages=len(messages), n_tools=0)
        response = self.llm.chat(messages, tools=[])
        trace.log("llm_call_end", tool_calls=[], n_chars=len(response.content))
        answer = response.content.strip()

        sources = sorted({h["source"] for h in hits})
        result = NotesAnswer(question, answer=answer, sources=sources)
        if not save:
            return result

        # Step 4: save. write_file is in confirm_before by default, so
        # the human still gets asked -- the workflow can't skip that.
        path = f"answers/{timestamp[:10]}-{_slug(question)}.md"
        body = (
            f"# {question}\n\n_{timestamp}_\n\n{answer}\n\n"
            "## Sources\n\n" + "".join(f"- {s}\n" for s in sources)
        )
        saved = self._tool("write_file", {"path": path, "content": body}, trace)
        if saved.ok:
            result.saved_path = path
        else:
            result.problem = saved.error
        return result


def _slug(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "question"
