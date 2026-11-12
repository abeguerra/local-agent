"""
Concrete tools the agent can call.

Two design rules worth noticing, because companies get burned when they
skip them:

1. No `eval()`, ever. The calculator below parses a restricted AST
   instead of calling Python's `eval`, because `eval` on model-generated
   text is a code-execution vulnerability -- the model is an untrusted
   input source, same as a user-submitted form field.

2. All filesystem access is sandboxed to one directory (`workspace/`)
   and every path is resolved and checked to still be inside it. This
   is the local, single-file version of the "sandboxed environments"
   guidance in Anthropic's agent guidance, and of what every coding
   agent (Claude Code, Devin, Cursor's agent mode, etc.) does in some
   form: the model gets a tool, not a shell with your full filesystem.
"""

from __future__ import annotations

import ast
import datetime as dt
import operator
from pathlib import Path

from agent.tools import ToolError, registry

WORKSPACE = Path(__file__).resolve().parent.parent.parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Calculator -- safe expression evaluation, no eval()
# ---------------------------------------------------------------------------

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ToolError("only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, (ast.Call, ast.Name, ast.Attribute)):
        # Small models often try to nest one tool inside another, e.g.
        # "42 + get_current_time().weekday()". Say plainly what to do
        # instead -- a raw AST dump just makes them conclude the other
        # tool doesn't exist.
        raise ToolError(
            "the calculator only accepts numbers and arithmetic operators -- no function calls or names. "
            "Tools cannot be called inside other tools' arguments: call the other tool on its own first, "
            "then pass its result here as a number."
        )
    raise ToolError(f"unsupported expression: {ast.dump(node)}")


@registry.register(
    name="calculator",
    description="Evaluate a numeric arithmetic expression, e.g. '(3 + 4) * 2 / 7'. Supports + - * / % ** and parentheses only.",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "the arithmetic expression to evaluate"}},
        "required": ["expression"],
    },
)
def calculator(expression: str) -> dict:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
    except (SyntaxError, ZeroDivisionError, ToolError) as e:
        raise ToolError(f"could not evaluate '{expression}': {e}") from e
    return {"expression": expression, "result": result}


# ---------------------------------------------------------------------------
# Date/time -- models are notoriously unreliable at knowing "now"
# ---------------------------------------------------------------------------

@registry.register(
    name="get_current_time",
    description=(
        "Get the current date and time, including the weekday name and its ISO number (1 = Monday ... 7 = Sunday). "
        "Use this instead of guessing -- you do not otherwise know what day it is."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
)
def get_current_time() -> dict:
    now = dt.datetime.now()
    return {
        "iso": now.isoformat(timespec="seconds"),
        "weekday": now.strftime("%A"),
        "weekday_number": now.isoweekday(),  # 1 = Monday ... 7 = Sunday
    }


# ---------------------------------------------------------------------------
# Sandboxed filesystem tools
# ---------------------------------------------------------------------------

def _resolve_in_workspace(relative_path: str) -> Path:
    candidate = (WORKSPACE / relative_path).resolve()
    if WORKSPACE not in candidate.parents and candidate != WORKSPACE:
        raise ToolError(f"path '{relative_path}' escapes the sandboxed workspace directory -- not allowed")
    return candidate


@registry.register(
    name="list_workspace_files",
    description="List files in the agent's sandboxed workspace directory (optionally under a subfolder).",
    parameters={
        "type": "object",
        "properties": {"subfolder": {"type": "string", "description": "subfolder to list, default '.'"}},
        "required": [],
    },
)
def list_workspace_files(subfolder: str = ".") -> dict:
    target = _resolve_in_workspace(subfolder)
    if not target.exists():
        raise ToolError(f"no such folder: {subfolder}")
    files = sorted(p.relative_to(WORKSPACE).as_posix() for p in target.rglob("*") if p.is_file())
    return {"workspace": str(WORKSPACE), "files": files}


@registry.register(
    name="read_file",
    description="Read a text file from the agent's sandboxed workspace directory.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "path relative to the workspace root"}},
        "required": ["path"],
    },
)
def read_file(path: str) -> dict:
    target = _resolve_in_workspace(path)
    if not target.is_file():
        raise ToolError(f"no such file: {path}")
    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise ToolError(f"'{path}' is not a text file: {e}") from e
    return {"path": path, "content": content}


@registry.register(
    name="write_file",
    description=(
        "Write text to a file in the agent's sandboxed workspace directory, creating or overwriting it. "
        "This is a destructive-ish action -- the agent runtime may ask the human to confirm before this runs."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "path relative to the workspace root"},
            "content": {"type": "string", "description": "the full text content to write"},
        },
        "required": ["path", "content"],
    },
)
def write_file(path: str, content: str) -> dict:
    target = _resolve_in_workspace(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"path": path, "bytes_written": len(content.encode("utf-8"))}
