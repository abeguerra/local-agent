import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.trace import Trace, pretty_print


def test_trace_writes_jsonl_and_reads_back(tmp_path):
    trace = Trace.start(tmp_path)
    trace.log("user_message", content="hi")
    trace.log("final_answer", content="hello")

    events = trace.events()
    assert len(events) == 2
    assert events[0]["event"] == "user_message"
    assert events[0]["step"] == 1
    assert events[1]["event"] == "final_answer"
    assert events[1]["step"] == 2
    assert all(e["trace_id"] == trace.trace_id for e in events)


def test_trace_file_is_crash_safe_append_only(tmp_path):
    trace = Trace.start(tmp_path)
    trace.log("a")
    raw = trace.path.read_text().strip().splitlines()
    assert len(raw) == 1  # one complete JSON object already on disk
    trace.log("b")
    raw = trace.path.read_text().strip().splitlines()
    assert len(raw) == 2


def test_pretty_print_handles_missing_file(tmp_path):
    out = pretty_print(tmp_path / "nope.jsonl")
    assert "no trace found" in out


def test_pretty_print_renders_known_event_types(tmp_path):
    trace = Trace.start(tmp_path)
    trace.log("user_message", content="2+2?")
    trace.log("llm_call_start", n_messages=1, n_tools=1)
    trace.log("llm_call_end", tool_calls=[{"name": "calculator", "arguments": {}}], n_chars=0)
    trace.log("tool_call_start", name="calculator", arguments={"expression": "2+2"})
    trace.log("tool_call_end", name="calculator", result={"result": 4}, duration_ms=1)
    trace.log("final_answer", content="4")

    out = pretty_print(trace.path)
    assert "USER" in out
    assert "calculator" in out
    assert "FINAL" in out
