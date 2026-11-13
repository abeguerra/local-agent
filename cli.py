#!/usr/bin/env python3
"""
Interactive REPL for the local agent.

Run with:  python cli.py
Requires Ollama running locally with a tool-calling model pulled --
see README.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.core import SYSTEM_PROMPT, Agent
from agent.guardrails import Guardrails, GuardrailConfig
from agent.llm import OllamaClient
from agent.memory import VectorMemory
from agent.skills import load_skills, register_skill_tool, skills_prompt_section
from agent.tools import registry
from agent.tools import builtin  # noqa: F401  -- registers calculator/time/file tools
from agent.tools.notes_search import init_notes_tool
from agent.trace import Trace, pretty_print
from agent.workflows import NotesQAWorkflow

ROOT = Path(__file__).resolve().parent


def confirm_callback(name: str, arguments: dict) -> bool:
    print(f"\n  [confirm] agent wants to call {name}({arguments})")
    answer = input("  allow? [y/N] ").strip().lower()
    return answer == "y"


def main() -> None:
    # Override with e.g. LOCAL_AGENT_MODEL=qwen2.5:7b-instruct for more
    # reliable multi-step tool use (pull it first with `ollama pull`).
    llm = OllamaClient(model=os.environ.get("LOCAL_AGENT_MODEL", "llama3.2"))

    # Wire up the local notes RAG tool. If embeddings aren't available
    # (model not pulled yet), carry on without it -- the other tools
    # still work. This graceful-degradation pattern (don't let one
    # unavailable dependency take down the whole agent) matters once you
    # have more than one or two tools.
    try:
        vector_memory = VectorMemory(ROOT / "data" / "notes_index.sqlite3", embed_fn=llm.embed)
        n = init_notes_tool(vector_memory, ROOT / "data" / "notes")
        if n:
            print(f"(indexed {n} note chunks for search_notes)")
    except Exception as e:  # noqa: BLE001 -- best-effort setup, not core functionality
        print(f"(notes search unavailable: {e})")

    # Skills: only names + descriptions go into the system prompt; the
    # agent pulls a skill's full instructions in with load_skill when a
    # request matches. See agent/skills.py.
    skills = load_skills(ROOT / "skills")
    if skills:
        register_skill_tool(registry, skills)
        print(f"(skills available: {', '.join(skills)})")

    guardrails = Guardrails(GuardrailConfig(max_steps=8))
    agent = Agent(
        llm=llm,
        tools=registry,
        guardrails=guardrails,
        system_prompt=SYSTEM_PROMPT + skills_prompt_section(skills),
        confirm_callback=confirm_callback,
    )
    notes_qa = NotesQAWorkflow(llm=llm, tools=registry, guardrails=guardrails, confirm_callback=confirm_callback)

    print("Local agent ready. Tools:", ", ".join(registry.names()))
    print("Type 'exit' to quit, 'trace' to show the last run's trace,")
    print("or '/notes <question>' to run the notes Q&A workflow instead of the agent.\n")

    trace = None
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input.lower() == "trace":
            if trace:
                print(pretty_print(trace.path))
            else:
                print("(no run yet)")
            continue

        trace = Trace.start(ROOT / "data" / "traces")
        try:
            if user_input.startswith("/notes "):
                result = notes_qa.run(user_input.removeprefix("/notes ").strip(), trace=trace)
                answer = result.answer
                if result.saved_path:
                    answer += f"\n(saved to workspace/{result.saved_path})"
                elif result.problem:
                    answer += f"\n({result.problem})"
            else:
                answer = agent.run(user_input, trace=trace)
        except ConnectionError as e:
            print(f"\n[error] {e}")
            continue
        print(f"\nagent> {answer}\n")


if __name__ == "__main__":
    main()
