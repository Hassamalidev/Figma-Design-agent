"""The tool-calling probe: does it correctly tell a working model from a broken one?

Tool calling is the ONE hard requirement on the generator, and the failure it
produces reads like an agent bug: every step "replies with text instead of
calling the tool" and every step fails. This probe is what turns that into a
one-line answer, so it has to be right about all four outcomes.

No network -- every client here is a fake.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.tool_probe import PROBE_TOOL, probe_tool_calling


def call(name="render_ui", arguments=None):
    return SimpleNamespace(
        id="c1",
        function=SimpleNamespace(
            name=name,
            arguments=arguments if isinstance(arguments, str) else json.dumps(arguments or {}),
        ),
    )


GOOD_SPEC = {"spec": {"kind": "section", "name": "Sign in",
                      "children": [{"kind": "text", "value": "Welcome back"}]}}


class Fake:
    """A model that returns whatever it was constructed with."""

    def __init__(self, message=None, error=None):
        self._message, self._error = message, error
        self.tools_offered = None

    def complete(self, messages, tools=None):
        self.tools_offered = tools
        if self._error:
            raise self._error
        return self._message


def test_a_working_model_passes():
    result = probe_tool_calling(Fake(SimpleNamespace(content=None, tool_calls=[call(arguments=GOOD_SPEC)])), "m")

    assert result.status == "ok" and result.usable
    assert "correctly" in result.detail


def test_a_model_that_answers_in_prose_fails():
    """The whole point. A `tools` capability flag is a claim, not a fact."""
    result = probe_tool_calling(Fake(SimpleNamespace(content="I would build a sign-in section.", tool_calls=None)), "m")

    assert result.status == "no_tools" and not result.usable
    assert "prose" in result.detail
    assert "I would build" in result.reply


def test_a_call_with_no_spec_is_a_failure_not_a_pass():
    """This exact shape cost a real run half its tool budget: `render_ui` called
    twenty-five times with the tree sent as the arguments object."""
    result = probe_tool_calling(
        Fake(SimpleNamespace(content=None, tool_calls=[call(arguments={"kind": "section"})])), "m"
    )

    assert result.status == "no_tools"
    assert "instead of a `spec` object" in result.detail


def test_a_model_that_calls_the_tool_but_ignores_the_task_fails():
    """A pass has to mean the arguments were sane, or the probe measures only
    that the endpoint has a tool-calling code path."""
    result = probe_tool_calling(
        Fake(SimpleNamespace(content=None, tool_calls=[
            call(arguments={"spec": {"kind": "section", "children": []}})])), "m"
    )

    assert result.status == "no_tools" and "ignored what it was asked" in result.detail


def test_unparseable_arguments_fail():
    result = probe_tool_calling(
        Fake(SimpleNamespace(content=None, tool_calls=[call(arguments="{not json")])), "m"
    )

    assert result.status == "no_tools" and "valid JSON" in result.detail


def test_a_recovered_call_is_reported_as_weaker_than_a_native_one():
    """`llm.py` salvages a call printed as text. It works -- and it costs a
    round trip on every step -- so it must not be scored the same as a model
    that calls the tool properly."""
    recovered = SimpleNamespace(
        content=None, tool_calls=[call(arguments=GOOD_SPEC)], recovered_from_text=True
    )

    result = probe_tool_calling(Fake(recovered), "m")

    assert result.status == "recovered" and result.usable
    assert "round trip" in result.detail


@pytest.mark.parametrize(
    "error, expected",
    [
        (RuntimeError("Error code: 429 - rate limit exceeded"), "rate limited"),
        (RuntimeError("Error code: 403 - requires a subscription"), "paid subscription"),
        (RuntimeError("Error code: 410 - model was retired"), "retired"),
        (RuntimeError("Error code: 404 - model not found"), "no such model"),
        (RuntimeError("Error code: 401 - invalid api key"), "key was rejected"),
    ],
)
def test_each_endpoint_failure_is_reported_as_the_thing_to_do_about_it(error, expected):
    """"429" and "403" call for completely different actions: wait, versus pick
    a different model. Reporting the raw exception makes the user work that out."""
    result = probe_tool_calling(Fake(error=error), "m")

    assert result.status == "error" and not result.usable
    assert expected in result.detail


def test_the_probe_asks_for_the_shape_the_agent_really_uses():
    """A toy string parameter is easier than what the run needs. The probe has
    to measure the real thing: one required OBJECT argument."""
    parameters = PROBE_TOOL[0]["function"]["parameters"]

    assert parameters["required"] == ["spec"]
    assert parameters["properties"]["spec"]["type"] == "object"


def test_the_probe_never_raises_whatever_the_endpoint_does():
    """It is a diagnostic. Crashing while diagnosing is not an option."""
    for bad in (Fake(error=ValueError("boom")), Fake(SimpleNamespace(content=None, tool_calls=[]))):
        assert probe_tool_calling(bad, "m").status in ("error", "no_tools")
