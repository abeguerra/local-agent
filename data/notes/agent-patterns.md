# Notes on agent design patterns

Workflows are systems where LLM calls and tools are orchestrated through
predefined code paths. Agents are systems where the LLM dynamically
directs its own process and tool usage, keeping control over how a task
gets done.

The core building block under both is the "augmented LLM": a model
given retrieval, tools, and memory. Everything else is composition.

Named workflow patterns: prompt chaining (decompose a task into a fixed
sequence of LLM calls), routing (classify input, then dispatch to a
specialized path), parallelization (split into independent subtasks run
concurrently, or run the same task multiple times and vote), orchestrator
-workers (a central LLM plans subtasks at runtime and delegates them),
and evaluator-optimizer (one LLM generates, another critiques, in a
loop until the critic is satisfied).

Autonomous agents differ from all of the above: they only make sense
once a task's success criteria are clear, and they get real feedback
from their environment (tool results, file contents, test runs) at
every step instead of guessing blind.
