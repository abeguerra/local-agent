"""
The agent loop itself.

This implements the "autonomous agent" pattern from Anthropic's agent
design guidance: the model is given tools and, in a loop, decides which
ones to call based on what comes back, until it decides it has enough
to answer or a guardrail stops it. This is the same basic loop under
the hood of OpenAI's Assistants/Responses API, Claude's tool use loop,
LangChain's AgentExecutor, and most "ReAct"-style agents: there is no
secret sauce here, which is itself the lesson -- the loop is simple;
the engineering effort in production systems goes into the tools, the
guardrails, and the observability around it, not the loop.

    1. Add the user's message to the conversation.
    2. Call the model with the full message list + available tool
       schemas.
    3a. If the model's response has no tool calls -> that's the final
        answer. Return it.
    3b. If it has tool calls -> for each one: check guardrails, execute
        the tool, append the tool's result to the conversation as a
        "tool" role message (this is the "ground truth from the
        environment" step), and go back to step 2.
    4. If step 2 happens `max_steps` times without a final answer,
       stop anyway and say so. This is the loop-limit guardrail --
       without it, a model that keeps calling tools never returns
       control to the user.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.guardrails import Guardrails, GuardrailViolation
from agent.llm import LLMClient, LLMResponse
from agent.memory import ConversationBuffer
from agent.tools import ToolError, ToolRegistry
from agent.trace import Trace

SYSTEM_PROMPT = """You are a careful local assistant running entirely on the user's \
own machine. You have access to a small set of tools. Use a tool whenever it would \
give you real information instead of a guess (the current time, arithmetic, file \
contents, or notes search). Tools cannot be called inside other tools' arguments: \
call one tool, read its result, then use that value in your next call. \
After you have enough information, answer the user \
directly and concisely. If a tool errors, read the error and either fix your \
arguments or explain the problem to the user -- do not repeat the same failing call."""


@dataclass
class StepResult:
    final: bool
    content: str = ""


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        guardrails: Guardrails | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        confirm_callback=None,  # Callable[[str, dict], bool] -- see needs_confirmation below
    ):
        self.llm = llm
        self.tools = tools
        self.guardrails = guardrails or Guardrails()
        self.memory = ConversationBuffer(system_prompt=system_prompt)
        self.confirm_callback = confirm_callback or (lambda name, args: True)

    def run(self, user_message: str, trace: Trace | None = None) -> str:
        trace = trace or Trace.start("data/traces")

        self.guardrails.check_input(user_message)
        self.memory.add("user", user_message)
        trace.log("user_message", content=user_message)

        tool_schemas = self.tools.schemas(self.guardrails.config.allowed_tools)

        for step in range(1, self.guardrails.config.max_steps + 1):
            messages = self.memory.as_list()
            trace.log("llm_call_start", n_messages=len(messages), n_tools=len(tool_schemas))

            response: LLMResponse = self.llm.chat(messages, tool_schemas)

            trace.log(
                "llm_call_end",
                tool_calls=[{"name": tc.name, "arguments": tc.arguments} for tc in response.tool_calls],
                n_chars=len(response.content),
            )

            if not response.tool_calls:
                self.memory.add("assistant", response.content)
                trace.log("final_answer", content=response.content)
                return response.content

            # The model wants to call one or more tools. Ollama (and most
            # tool-calling APIs) allow several tool calls per turn; we
            # execute them in order and feed every result back before
            # calling the model again.
            self.memory.add("assistant", response.content or None, tool_calls=[
                {"function": {"name": tc.name, "arguments": tc.arguments}} for tc in response.tool_calls
            ])

            for tool_call in response.tool_calls:
                result_text = self._execute_tool(tool_call.name, tool_call.arguments, trace)
                self.memory.add("tool", result_text)

        trace.log("loop_limit_hit", max_steps=self.guardrails.config.max_steps)
        giveup_message = (
            f"I stopped after {self.guardrails.config.max_steps} steps without reaching a final answer "
            "(loop-limit guardrail). This usually means the task needs breaking into smaller steps."
        )
        self.memory.add("assistant", giveup_message)
        return giveup_message

    def _execute_tool(self, name: str, arguments: dict, trace: Trace) -> str:
        import time

        trace.log("tool_call_start", name=name, arguments=arguments)
        start = time.monotonic()
        try:
            self.guardrails.check_tool_allowed(name)
            if self.guardrails.needs_confirmation(name):
                if not self.confirm_callback(name, arguments):
                    raise GuardrailViolation(f"user declined to confirm tool call '{name}'")
            tool = self.tools.get(name)
            result = tool.call(arguments)
            duration_ms = int((time.monotonic() - start) * 1000)
            trace.log("tool_call_end", name=name, result=result, duration_ms=duration_ms)
            return str(result)
        except (ToolError, GuardrailViolation) as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            trace.log("tool_call_end", name=name, error=str(e), duration_ms=duration_ms)
            # Errors go back to the MODEL as a tool result (not raised to
            # the caller) so it can self-correct -- e.g. retry with fixed
            # arguments, or explain the limitation to the user. This is
            # the "ground truth from the environment" feedback loop.
            return f"ERROR: {e}"
