import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.core import Agent
from agent.guardrails import Guardrails, GuardrailConfig
from agent.llm import FakeLLM, LLMResponse, ToolCall
from agent.tools import ToolRegistry
from agent.trace import Trace


def make_registry():
    reg = ToolRegistry()

    @reg.register(
        name="calculator",
        description="add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    )
    def calculator(a: int, b: int):
        return {"result": a + b}

    return reg


def test_agent_answers_directly_with_no_tool_calls(tmp_path):
    llm = FakeLLM([LLMResponse(content="2 + 2 is 4.")])
    agent = Agent(llm=llm, tools=make_registry())
    trace = Trace.start(tmp_path)

    answer = agent.run("what is 2+2", trace=trace)

    assert answer == "2 + 2 is 4."
    assert len(llm.calls) == 1
    events = [e["event"] for e in trace.events()]
    assert events[0] == "user_message"
    assert "final_answer" in events


def test_agent_executes_tool_call_then_returns_final_answer(tmp_path):
    llm = FakeLLM(
        [
            LLMResponse(content="", tool_calls=[ToolCall(name="calculator", arguments={"a": 3, "b": 4})]),
            LLMResponse(content="3 + 4 is 7."),
        ]
    )
    agent = Agent(llm=llm, tools=make_registry())
    trace = Trace.start(tmp_path)

    answer = agent.run("add 3 and 4", trace=trace)

    assert answer == "3 + 4 is 7."
    assert len(llm.calls) == 2
    # the tool result must have been fed back to the model as context
    # for the second call
    second_call_messages = llm.calls[1]["messages"]
    assert any(m["role"] == "tool" and "7" in m["content"] for m in second_call_messages)


def test_agent_feeds_tool_error_back_to_model_for_self_correction(tmp_path):
    llm = FakeLLM(
        [
            # bad arguments -> ToolError -> fed back as a "tool" message
            LLMResponse(content="", tool_calls=[ToolCall(name="calculator", arguments={"a": 3})]),
            LLMResponse(content="sorry, I need both numbers."),
        ]
    )
    agent = Agent(llm=llm, tools=make_registry())
    trace = Trace.start(tmp_path)

    answer = agent.run("add 3 and ???", trace=trace)

    assert answer == "sorry, I need both numbers."
    second_call_messages = llm.calls[1]["messages"]
    assert any(m["role"] == "tool" and "ERROR" in m["content"] for m in second_call_messages)


def test_guardrail_stops_infinite_tool_loop(tmp_path):
    # A model that NEVER stops calling tools -- this is exactly the
    # failure mode max_steps exists to contain.
    infinite_tool_calls = LLMResponse(
        content="", tool_calls=[ToolCall(name="calculator", arguments={"a": 1, "b": 1})]
    )
    llm = FakeLLM([infinite_tool_calls] * 20)
    guardrails = Guardrails(GuardrailConfig(max_steps=3))
    agent = Agent(llm=llm, tools=make_registry(), guardrails=guardrails)
    trace = Trace.start(tmp_path)

    answer = agent.run("loop forever", trace=trace)

    assert "stopped after 3 steps" in answer
    assert len(llm.calls) == 3  # never exceeded the budget
    events = [e["event"] for e in trace.events()]
    assert "loop_limit_hit" in events


def test_tool_allowlist_blocks_disallowed_tool(tmp_path):
    llm = FakeLLM(
        [
            LLMResponse(content="", tool_calls=[ToolCall(name="calculator", arguments={"a": 1, "b": 1})]),
            LLMResponse(content="ok, I cannot use that tool."),
        ]
    )
    guardrails = Guardrails(GuardrailConfig(allowed_tools={"some_other_tool"}))
    agent = Agent(llm=llm, tools=make_registry(), guardrails=guardrails)
    trace = Trace.start(tmp_path)

    answer = agent.run("add 1 and 1", trace=trace)

    assert answer == "ok, I cannot use that tool."
    second_call_messages = llm.calls[1]["messages"]
    assert any(m["role"] == "tool" and "not in the allowlist" in m["content"] for m in second_call_messages)


def test_confirmation_callback_can_block_a_tool_call(tmp_path):
    llm = FakeLLM(
        [
            LLMResponse(content="", tool_calls=[ToolCall(name="calculator", arguments={"a": 1, "b": 1})]),
            LLMResponse(content="ok, skipping that."),
        ]
    )
    guardrails = Guardrails(GuardrailConfig(confirm_before={"calculator"}))
    agent = Agent(
        llm=llm,
        tools=make_registry(),
        guardrails=guardrails,
        confirm_callback=lambda name, args: False,  # user says no
    )
    trace = Trace.start(tmp_path)

    answer = agent.run("add 1 and 1", trace=trace)

    assert answer == "ok, skipping that."
    second_call_messages = llm.calls[1]["messages"]
    assert any(m["role"] == "tool" and "declined" in m["content"] for m in second_call_messages)
