import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.guardrails import Guardrails, GuardrailConfig
from agent.llm import FakeLLM, LLMResponse
from agent.tools import ToolError, ToolRegistry
from agent.trace import Trace
from agent.workflows import NotesQAWorkflow


def make_registry(hits=None, search_error=None):
    """Stand-ins for get_current_time / search_notes / write_file, so the
    workflow runs with no Ollama, no embeddings and no real disk writes."""
    reg = ToolRegistry()
    reg.written = {}

    @reg.register(name="get_current_time", description="now", parameters={"type": "object", "properties": {}})
    def get_current_time():
        return {"iso": "2026-11-13T09:30:00", "weekday": "Friday", "weekday_number": 5}

    @reg.register(name="search_notes", description="search")
    def search_notes(query: str, top_k: int = 3):
        if search_error:
            raise ToolError(search_error)
        return {"query": query, "results": hits or []}

    @reg.register(name="write_file", description="write")
    def write_file(path: str, content: str):
        reg.written[path] = content
        return {"path": path, "bytes_written": len(content)}

    return reg


HIT = {"source": "guardrails.md", "text": "max_steps bounds the agent loop.", "score": 0.8}


def test_workflow_calls_tools_in_order_with_a_single_tool_free_llm_call(tmp_path):
    reg = make_registry(hits=[HIT])
    llm = FakeLLM([LLMResponse(content="max_steps bounds the loop (guardrails.md).")])
    trace = Trace.start(tmp_path)

    result = NotesQAWorkflow(llm=llm, tools=reg).run("What does max_steps do?", trace=trace)

    # the code, not the model, chose these tools and their order
    tool_calls = [e["name"] for e in trace.events() if e["event"] == "tool_call_start"]
    assert tool_calls == ["get_current_time", "search_notes", "write_file"]

    # exactly one model call, offered no tools, given the retrieved notes
    assert len(llm.calls) == 1
    assert llm.calls[0]["tools"] == []
    assert "max_steps bounds the agent loop." in llm.calls[0]["messages"][-1]["content"]

    assert result.answer == "max_steps bounds the loop (guardrails.md)."
    assert result.sources == ["guardrails.md"]
    assert result.saved_path == "answers/2026-11-13-what-does-max-steps-do.md"
    saved = reg.written[result.saved_path]
    assert "# What does max_steps do?" in saved
    assert "- guardrails.md" in saved


def test_workflow_stops_before_the_model_when_no_note_is_relevant(tmp_path):
    reg = make_registry(hits=[{**HIT, "score": 0.1}])
    llm = FakeLLM([LLMResponse(content="should never be used")])

    result = NotesQAWorkflow(llm=llm, tools=reg).run("unrelated question", trace=Trace.start(tmp_path))

    assert llm.calls == []
    assert result.saved_path is None
    assert result.problem == "no relevant notes"
    assert reg.written == {}


def test_workflow_reports_search_failure_without_calling_the_model(tmp_path):
    reg = make_registry(search_error="notes search is not initialized")
    llm = FakeLLM([])

    result = NotesQAWorkflow(llm=llm, tools=reg).run("anything", trace=Trace.start(tmp_path))

    assert llm.calls == []
    assert "not initialized" in result.problem


def test_workflow_tool_calls_still_go_through_confirmation(tmp_path):
    # write_file is in confirm_before by default; declining must stop the save
    reg = make_registry(hits=[HIT])
    llm = FakeLLM([LLMResponse(content="an answer")])
    workflow = NotesQAWorkflow(llm=llm, tools=reg, confirm_callback=lambda name, args: False)

    result = workflow.run("What does max_steps do?", trace=Trace.start(tmp_path))

    assert result.answer == "an answer"
    assert result.saved_path is None
    assert "declined" in result.problem
    assert reg.written == {}


def test_workflow_tool_calls_still_respect_the_allowlist(tmp_path):
    reg = make_registry(hits=[HIT])
    llm = FakeLLM([LLMResponse(content="an answer")])
    guardrails = Guardrails(GuardrailConfig(allowed_tools={"get_current_time", "search_notes"}))

    result = NotesQAWorkflow(llm=llm, tools=reg, guardrails=guardrails).run("q", trace=Trace.start(tmp_path))

    assert result.saved_path is None
    assert "not in the allowlist" in result.problem


def test_save_false_skips_write_file(tmp_path):
    reg = make_registry(hits=[HIT])
    llm = FakeLLM([LLMResponse(content="an answer")])
    trace = Trace.start(tmp_path)

    result = NotesQAWorkflow(llm=llm, tools=reg).run("q", trace=trace, save=False)

    assert result.saved_path is None
    assert "write_file" not in [e.get("name") for e in trace.events()]
