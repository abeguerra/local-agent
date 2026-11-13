"""
Skills: instructions the agent loads only when it needs them.

A skill is a folder with a SKILL.md file: YAML-style frontmatter giving
a `name` and a one-line `description`, then markdown instructions. It
is the same shape as Anthropic's Agent Skills (Claude Code, the Claude
apps, the Agent SDK), minus scripts and bundled resource files.

The point is *progressive disclosure*. Putting every procedure the agent
might ever need into the system prompt makes every call longer and
buries the instructions that matter for this request. Instead:

  1. the system prompt lists only each skill's name and description
     (a line or two per skill, cheap no matter how many there are)
  2. when a request matches, the model calls the `load_skill` tool
  3. the full instructions come back as a tool result, so they enter
     the conversation only for the tasks that need them

Compare with agent/workflows.py. A workflow is code: the steps run in
order because Python runs them. A skill is text: the steps run only if
the model reads and follows them. skills/notes-answer/SKILL.md
describes the same job as NotesQAWorkflow, so you can try both and see
the difference -- the skill copes with oddly phrased requests, the
workflow never skips a step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.tools import ToolError, ToolRegistry


@dataclass
class Skill:
    name: str
    description: str
    instructions: str
    path: Path


class SkillError(Exception):
    """A SKILL.md file is malformed. Raised at startup, not to the model --
    a broken skill is the developer's bug, not something to negotiate."""


def parse_skill(path: str | Path) -> Skill:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillError(f"{path}: must start with a '---' frontmatter block")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise SkillError(f"{path}: frontmatter block is never closed with '---'") from None

    # Deliberately tiny "key: value" parser -- enough for name/description,
    # and it keeps the project's stdlib-only runtime.
    meta = {}
    for line in lines[1:end]:
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip("\"'")

    for key in ("name", "description"):
        if not meta.get(key):
            raise SkillError(f"{path}: frontmatter is missing '{key}'")

    instructions = "\n".join(lines[end + 1 :]).strip()
    if not instructions:
        raise SkillError(f"{path}: has no instructions after the frontmatter")
    return Skill(name=meta["name"], description=meta["description"], instructions=instructions, path=path)


def load_skills(skills_dir: str | Path) -> dict[str, Skill]:
    """Read every <skills_dir>/<folder>/SKILL.md. Missing dir = no skills."""
    skills: dict[str, Skill] = {}
    skills_dir = Path(skills_dir)
    if not skills_dir.is_dir():
        return skills
    for skill_file in sorted(skills_dir.glob("*/SKILL.md")):
        skill = parse_skill(skill_file)
        if skill.name in skills:
            raise SkillError(f"duplicate skill name '{skill.name}' in {skill_file} and {skills[skill.name].path}")
        skills[skill.name] = skill
    return skills


def skills_prompt_section(skills: dict[str, Skill]) -> str:
    """The part of the system prompt that advertises skills: names and
    descriptions only, never the instructions themselves."""
    if not skills:
        return ""
    listing = "\n".join(f"- {s.name}: {s.description}" for s in skills.values())
    # Small models readily mistake a skill name for a tool name and try to
    # "call" it directly, so spell out that skills are reached only
    # through load_skill.
    return (
        "\n\nSkills are step-by-step instructions for specific kinds of tasks. They are NOT tools "
        "and cannot be called directly. When a request matches one of the skills below, first call "
        'the load_skill tool with {"name": "<skill name>"}, then follow the instructions it returns '
        "using your normal tools.\n" + listing
    )


def register_skill_tool(registry: ToolRegistry, skills: dict[str, Skill]) -> None:
    @registry.register(
        name="load_skill",
        description=(
            "Load the full step-by-step instructions for one of the skills listed in your system prompt. "
            "Call this before starting a task that matches a skill's description."
        ),
        parameters={
            "type": "object",
            # The enum puts the valid names into the tool schema itself,
            # where the model looks when filling in arguments.
            "properties": {"name": {"type": "string", "enum": sorted(skills), "description": "the skill to load"}},
            "required": ["name"],
        },
    )
    def load_skill(name: str) -> str:
        skill = skills.get(name)
        if skill is None:
            raise ToolError(f"no skill named '{name}'. Available skills: {', '.join(skills) or '(none)'}")
        # Plain text, not a dict: the agent str()s tool results, and a
        # dict's repr would show the markdown as one line full of literal
        # "\n"s -- much harder for a small model to follow.
        return (
            f"Instructions for the '{skill.name}' skill. Follow them now, step by step, "
            f"making each tool call they describe:\n\n{skill.instructions}"
        )
