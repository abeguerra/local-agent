"""
Tool registry.

Why a registry instead of just "a bunch of python functions"
--------------------------------------------------------------
Every production tool-calling system -- OpenAI's function calling,
Anthropic's tool use, Ollama's tool support, and Anthropic's Model
Context Protocol (MCP) -- describes tools to the model the same way:
a name, a natural-language description, and a JSON Schema for the
arguments. The model never sees your Python code; it only ever sees
this schema, and it only ever "calls" a tool by emitting structured
JSON that matches it. Your code is responsible for validating that
JSON and executing the real function.

That's what `@tool` below captures: it turns a plain Python function
into (a) a JSON-Schema tool definition to hand the LLM and (b) a safe
callable the agent loop can invoke by name with validated arguments.

This is also exactly the shape MCP servers expose tools in, which is
why a registry like this is a natural stepping stone to writing an
MCP server later -- the schema format doesn't change, only how it's
transported.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints


class ToolError(Exception):
    """Raised for a tool-usage error that the MODEL should see and can
    recover from (e.g. bad arguments, file not found) -- as opposed to
    a GuardrailViolation, which the model never gets to negotiate with."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema
    func: Callable[..., Any]

    def to_schema(self) -> dict:
        """OpenAI/Ollama-style tool-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def call(self, arguments: dict) -> Any:
        try:
            return self.func(**arguments)
        except TypeError as e:
            raise ToolError(f"bad arguments for '{self.name}': {e}") from e


_PY_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, parameters: dict | None = None):
        """Decorator. If `parameters` is omitted, it is inferred from the
        function's type hints (required = no default value)."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            schema = parameters or _infer_schema(func)
            self._tools[name] = Tool(name=name, description=description, parameters=schema, func=func)
            return func

        return decorator

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"no such tool: '{name}'")
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self, allowed: set[str] | None = None) -> list[dict]:
        """The list you hand to the LLM client's `tools=` parameter."""
        tools = self._tools.values()
        if allowed is not None:
            tools = [t for t in tools if t.name in allowed]
        return [t.to_schema() for t in tools]


def _infer_schema(func: Callable[..., Any]) -> dict:
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    properties = {}
    required = []
    for pname, param in sig.parameters.items():
        py_type = hints.get(pname, str)
        properties[pname] = {"type": _PY_TO_JSON_TYPE.get(py_type, "string")}
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    return {"type": "object", "properties": properties, "required": required}


# A single process-wide registry, the same pattern most of these
# frameworks use (a global default registry you can still override).
registry = ToolRegistry()
