"""
Structured tracing for the agent.

Why this exists
----------------
In production, the #1 thing that separates a demo agent from a company's
real deployment is *observability*. When an agent misbehaves, you need to
replay exactly what the model saw, what it decided, and what each tool
returned -- in order. Companies buy or build tools for this (LangSmith,
Langfuse, Helicone, Arize Phoenix, Datadog LLM Observability) and the
emerging standard shape is OpenTelemetry's GenAI semantic conventions:
one span per LLM call, one span per tool call, nested under a trace per
request/session.

This module is a minimal, dependency-free version of that idea: every
step of a run is appended as one JSON object per line (JSONL) to a trace
file. JSONL is used deliberately -- it's append-only and crash-safe (if
the process dies mid-run, every line written so far is still valid JSON
you can read back), which matters for a long-running autonomous loop.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Trace:
    """One trace per agent run (i.e. per user turn / session)."""

    path: Path
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    _step: int = 0

    @classmethod
    def start(cls, trace_dir: str | Path) -> "Trace":
        trace_dir = Path(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_id = uuid.uuid4().hex[:12]
        path = trace_dir / f"{int(time.time())}_{trace_id}.jsonl"
        return cls(path=path, trace_id=trace_id)

    def log(self, event_type: str, **payload: Any) -> None:
        """Append one structured event. Never raises -- tracing must not
        be able to crash the agent it's observing."""
        self._step += 1
        record = {
            "trace_id": self.trace_id,
            "step": self._step,
            "ts": time.time(),
            "event": event_type,
            **payload,
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            pass  # observability must be best-effort, never fatal

    def events(self) -> list[dict]:
        """Read back every event logged so far (used by tests and the
        pretty-printer below)."""
        if not self.path.exists():
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out


def pretty_print(path: str | Path) -> str:
    """Render a trace file as a human-readable timeline. This is the
    cheap, local equivalent of opening a run in LangSmith/Langfuse's UI."""
    path = Path(path)
    if not path.exists():
        return f"(no trace found at {path})"

    lines = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            step = rec.get("step")
            event = rec.get("event")
            if event == "user_message":
                lines.append(f"[{step:02d}] USER        {rec.get('content')!r}")
            elif event == "llm_call_start":
                lines.append(f"[{step:02d}] LLM ->      sending {rec.get('n_messages')} messages, {rec.get('n_tools')} tools available")
            elif event == "llm_call_end":
                if rec.get("tool_calls"):
                    names = ", ".join(tc["name"] for tc in rec["tool_calls"])
                    lines.append(f"[{step:02d}] LLM <-      wants to call: {names}")
                else:
                    lines.append(f"[{step:02d}] LLM <-      final answer ({rec.get('n_chars')} chars)")
            elif event == "tool_call_start":
                lines.append(f"[{step:02d}] TOOL ->     {rec.get('name')}({rec.get('arguments')})")
            elif event == "tool_call_end":
                status = "ok" if not rec.get("error") else f"ERROR: {rec.get('error')}"
                dur = rec.get("duration_ms", 0)
                lines.append(f"[{step:02d}] TOOL <-     {rec.get('name')} -> {status}  ({dur}ms)")
            elif event == "guardrail_block":
                lines.append(f"[{step:02d}] GUARDRAIL   blocked {rec.get('name')}: {rec.get('reason')}")
            elif event == "loop_limit_hit":
                lines.append(f"[{step:02d}] GUARDRAIL   max_steps ({rec.get('max_steps')}) reached, stopping")
            elif event == "workflow_start":
                lines.append(f"[{step:02d}] WORKFLOW    {rec.get('workflow')} started: {rec.get('question')!r}")
            elif event == "workflow_end":
                outcome = f"saved {rec['saved_path']}" if rec.get("saved_path") else (rec.get("problem") or "done")
                lines.append(f"[{step:02d}] WORKFLOW    {rec.get('workflow')} finished: {outcome}")
            elif event == "final_answer":
                lines.append(f"[{step:02d}] FINAL       {rec.get('content')!r}")
            else:
                lines.append(f"[{step:02d}] {event}  {rec}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m agent.trace <trace_file.jsonl>")
        raise SystemExit(1)
    print(pretty_print(sys.argv[1]))
