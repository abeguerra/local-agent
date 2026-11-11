import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from agent.guardrails import Guardrails, GuardrailConfig, GuardrailViolation


def test_input_length_guardrail():
    g = Guardrails(GuardrailConfig(max_input_chars=10))
    g.check_input("short")
    with pytest.raises(GuardrailViolation):
        g.check_input("this is way too long for the limit")


def test_tool_allowlist():
    g = Guardrails(GuardrailConfig(allowed_tools={"calculator"}))
    g.check_tool_allowed("calculator")
    with pytest.raises(GuardrailViolation):
        g.check_tool_allowed("write_file")


def test_allowlist_none_means_everything_allowed():
    g = Guardrails(GuardrailConfig(allowed_tools=None))
    g.check_tool_allowed("anything_goes")  # should not raise


def test_confirmation_set():
    g = Guardrails(GuardrailConfig(confirm_before={"write_file"}))
    assert g.needs_confirmation("write_file") is True
    assert g.needs_confirmation("calculator") is False


def test_step_budget():
    g = Guardrails(GuardrailConfig(max_steps=3))
    assert g.step_budget_exceeded(2) is False
    assert g.step_budget_exceeded(3) is True
