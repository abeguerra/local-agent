import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from agent.core import SYSTEM_PROMPT, Agent
from agent.llm import FakeLLM, LLMResponse, ToolCall
from agent.skills import SkillError, load_skills, parse_skill, register_skill_tool, skills_prompt_section
from agent.tools import ToolError, ToolRegistry
from agent.trace import Trace

ROOT = Path(__file__).resolve().parent.parent


def write_skill(skills_dir: Path, folder: str, text: str) -> Path:
    path = skills_dir / folder / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    return path


GREETING = """---
name: greeting
description: How to greet the user.
---
Say "ahoy" and nothing else.
"""


def test_parse_skill_reads_frontmatter_and_instructions(tmp_path):
    skill = parse_skill(write_skill(tmp_path, "greeting", GREETING))
    assert skill.name == "greeting"
    assert skill.description == "How to greet the user."
    assert skill.instructions == 'Say "ahoy" and nothing else.'


@pytest.mark.parametrize(
    "text, message",
    [
        ("no frontmatter at all", "must start"),
        ("---\nname: x\ndescription: y\nnever closed", "never closed"),
        ("---\nname: x\n---\nbody", "missing 'description'"),
        ("---\nname: x\ndescription: y\n---\n", "no instructions"),
    ],
)
def test_malformed_skill_fails_loudly(tmp_path, text, message):
    with pytest.raises(SkillError, match=message):
        parse_skill(write_skill(tmp_path, "bad", text))


def test_duplicate_skill_names_are_rejected(tmp_path):
    write_skill(tmp_path, "a", GREETING)
    write_skill(tmp_path, "b", GREETING)
    with pytest.raises(SkillError, match="duplicate"):
        load_skills(tmp_path)


def test_prompt_lists_descriptions_but_not_instructions(tmp_path):
    write_skill(tmp_path, "greeting", GREETING)
    section = skills_prompt_section(load_skills(tmp_path))
    assert "- greeting: How to greet the user." in section
    assert "ahoy" not in section  # progressive disclosure: body stays out until loaded


def test_load_skill_tool_returns_instructions_and_rejects_unknown_names(tmp_path):
    write_skill(tmp_path, "greeting", GREETING)
    reg = ToolRegistry()
    register_skill_tool(reg, load_skills(tmp_path))

    loaded = reg.get("load_skill").call({"name": "greeting"})
    assert isinstance(loaded, str)  # plain text, so newlines survive str() in the agent loop
    assert 'Say "ahoy" and nothing else.' in loaded
    with pytest.raises(ToolError, match="Available skills: greeting"):
        reg.get("load_skill").call({"name": "farewell"})


def test_agent_loads_a_skill_and_sees_its_instructions_on_the_next_call(tmp_path):
    write_skill(tmp_path / "skills", "greeting", GREETING)
    skills = load_skills(tmp_path / "skills")
    reg = ToolRegistry()
    register_skill_tool(reg, skills)
    llm = FakeLLM(
        [
            LLMResponse(content="", tool_calls=[ToolCall(name="load_skill", arguments={"name": "greeting"})]),
            LLMResponse(content="ahoy"),
        ]
    )
    agent = Agent(llm=llm, tools=reg, system_prompt=SYSTEM_PROMPT + skills_prompt_section(skills))

    answer = agent.run("hi there", trace=Trace.start(tmp_path))

    assert answer == "ahoy"
    first, second = llm.calls
    assert "greeting: How to greet the user." in first["messages"][0]["content"]
    assert not any("ahoy" in str(m.get("content")) for m in first["messages"])
    assert any(m["role"] == "tool" and "ahoy" in m["content"] for m in second["messages"])


def test_bundled_skills_parse():
    skills = load_skills(ROOT / "skills")
    assert {"notes-answer", "meeting-notes"} <= set(skills)
