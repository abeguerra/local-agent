"""
Guardrails: the boring code that keeps an autonomous loop from hurting you.

Anthropic's own write-up on agent design is blunt about this: autonomy
"means higher costs, and the potential for compounding errors," and
recommends "stopping conditions (such as a maximum number of iterations)
to maintain control," plus sandboxing, human checkpoints, and extensive
testing before trusting an agent with real actions.

Every company running agents in production has some version of the
checks below. None of this is exotic -- that's the point. The failure
mode this guards against is not "the model is evil," it's "the model
gets into a retry loop and calls a tool 400 times" or "the model decides
to delete a file nobody asked it to touch."
"""

from __future__ import annotations

from dataclasses import dataclass, field


class GuardrailViolation(Exception):
    """Raised when a proposed action is not allowed to proceed."""


@dataclass
class GuardrailConfig:
    # Loop control -- the single most important guardrail for an
    # autonomous tool-calling agent. Without this, a confused model can
    # call tools forever, burning tokens and money.
    max_steps: int = 8

    # Tool allowlist. Production systems almost always run with a
    # narrower tool set than "everything the model could theoretically
    # call" -- least privilege, same as any other software.
    allowed_tools: set[str] | None = None  # None = all registered tools

    # Tools that must be confirmed before they execute. This is the
    # cheap, local version of the "pause for human feedback at
    # checkpoints" pattern -- here it's a yes/no callback instead of a
    # full human-in-the-loop UI, but the principle (don't let the model
    # unilaterally take an irreversible action) is the same one that
    # production systems enforce before sending an email, charging a
    # card, or deleting data.
    confirm_before: set[str] = field(default_factory=lambda: {"write_file"})

    # Hard cap on a single user message, to stop obviously-broken input
    # (e.g. a pasted 500k-character log file) from blowing the context
    # window or the cost budget on one turn.
    max_input_chars: int = 8000


class Guardrails:
    def __init__(self, config: GuardrailConfig | None = None):
        self.config = config or GuardrailConfig()

    def check_input(self, text: str) -> None:
        if len(text) > self.config.max_input_chars:
            raise GuardrailViolation(
                f"input is {len(text)} chars, over the {self.config.max_input_chars}-char limit"
            )

    def check_tool_allowed(self, tool_name: str) -> None:
        allowed = self.config.allowed_tools
        if allowed is not None and tool_name not in allowed:
            raise GuardrailViolation(f"tool '{tool_name}' is not in the allowlist")

    def needs_confirmation(self, tool_name: str) -> bool:
        return tool_name in self.config.confirm_before

    def step_budget_exceeded(self, step: int) -> bool:
        return step >= self.config.max_steps
