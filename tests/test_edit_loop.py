"""The edit pipeline end to end, with a fake model and a fake Figma.

The fake bridge answers the harness-authored reads (the canvas inventory, the
file's styles, the layout gate) automatically, so each test only scripts the
model-driven calls it actually cares about -- the same rule the create-mode
tests follow.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from agent import edit_loop, inventory
from bridge.protocol import Response

PAGE = {
    "pageId": "0:1",
    "pageName": "Nexora",
    "selection": [],
    "roots": [{
        "id": "1:2", "name": "Login", "type": "FRAME", "x": 0, "y": 0, "width": 1440,
        "height": 900, "visible": True, "layoutMode": "VERTICAL",
        "fill": {"r": 1, "g": 1, "b": 1}, "children": [
            {"id": "1:3", "name": "Auth Card", "type": "FRAME", "x": 0, "y": 0, "width": 440,
             "height": 520, "visible": True, "layoutMode": "VERTICAL",
             "fill": {"r": 1, "g": 1, "b": 1}, "children": [
                 {"id": "1:4", "name": "Heading", "type": "TEXT", "x": 0, "y": 0, "width": 300,
                  "height": 40, "visible": True, "characters": "Welcome back", "fontSize": 32,
                  "fill": {"r": 0.07, "g": 0.09, "b": 0.15}, "children": []},
                 {"id": "1:9", "name": "Button / Log in", "type": "FRAME", "x": 0, "y": 60,
                  "width": 372, "height": 48, "visible": True,
                  "fill": {"r": 0.9, "g": 0.9, "b": 0.9}, "children": [
                      {"id": "1:10", "name": "Text", "type": "TEXT", "x": 0, "y": 0,
                       "width": 60, "height": 20, "visible": True, "characters": "Log in",
                       "fontSize": 15, "fill": {"r": 0.07, "g": 0.09, "b": 0.15},
                       "children": []}]}]}]}],
}

STYLES = {
    "paintStyles": ["color/primary", "color/card", "color/main-text", "color/secondary-text"],
    "paintColors": {
        "color/primary": {"r": 0.424, "g": 0.361, "b": 0.906},
        "color/card": {"r": 1, "g": 1, "b": 1},
        "color/main-text": {"r": 0.067, "g": 0.094, "b": 0.153},
        "color/secondary-text": {"r": 0.42, "g": 0.447, "b": 0.502},
    },
    "textStyles": ["Display", "Heading", "Body", "Caption", "Button"],
}


class FakeModelClient:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        return self._replies.pop(0) if self._replies else message(content="done")


def message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def edit_call(call_id: str, edits):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name="edit_ui", arguments=json.dumps({"edits": edits})),
    )


class FakeBridge:
    """Answers the harness's own reads; everything else must be scripted."""

    def __init__(self, page=None, styles=None, applied=None, layout_tree=None):
        self._page = page if page is not None else PAGE
        self._styles = styles if styles is not None else STYLES
        # What the fake Figma reports back per edit_ui script, in order.
        self._applied = list(applied or [])
        self._layout_tree = layout_tree
        self.sent: list = []
        self.edit_scripts: list[str] = []

    def send(self, request, timeout: float = 30.0) -> Response:
        self.sent.append(request)
        if request.type == "screenshot":
            return Response(id=request.id, ok=True, image_base64="PNG")
        code = request.code or ""
        if "selection" in code and "roots" in code:
            return Response(id=request.id, ok=True, result=dict(self._page, createdNodeIds=[]))
        if "paintColors" in code:
            return Response(id=request.id, ok=True, result=dict(self._styles, createdNodeIds=[]))
        if "function describe" in code:
            tree = self._layout_tree if self._layout_tree is not None else self._page["roots"][0]
            return Response(id=request.id, ok=True, result={"createdNodeIds": [], "tree": tree})
        if "appliedEdits" in code:
            self.edit_scripts.append(code)
            outcome = self._applied.pop(0) if self._applied else {"applied": ["ok"], "failed": []}
            if isinstance(outcome, str):
                return Response(id=request.id, ok=False, error=outcome)
            return Response(
                id=request.id, ok=True,
                result={
                    "createdNodeIds": outcome.get("created", []),
                    "appliedEdits": outcome.get("applied", []),
                    "failedEdits": outcome.get("failed", []),
                },
            )
        raise AssertionError(f"FakeBridge got an unexpected script:\n{code[:200]}")


def targets_of(script: str) -> list[str]:
    """Node ids an edit script actually resolved to."""
    return re.findall(r'getNodeByIdAsync\("([^"]+)"\)', script)


# ---- the happy path -------------------------------------------------------


def test_a_colour_change_reaches_the_right_node():
    llm = FakeModelClient([
        message(content='["Make the login button use the accent colour."]'),
        message(tool_calls=[edit_call("c1", [{"op": "set_fill", "target": "1:9", "color": "accent"}])]),
        message(content="Done."),
    ])
    bridge = FakeBridge(applied=[{"applied": ["set_fill on 1:9"], "failed": []}])

    result = edit_loop.run("make the login button purple", bridge, llm, max_retries=2)

    assert result.success
    assert targets_of(bridge.edit_scripts[0]) == ["1:9"]
    assert '"color/primary"' in bridge.edit_scripts[0]


def test_the_file_s_own_styles_are_adopted_not_replaced():
    """Creating a fresh palette would mean "make it purple" introduces a second,
    slightly different purple that nothing else references."""
    llm = FakeModelClient([
        message(content='["Recolour the button."]'),
        message(tool_calls=[edit_call("c1", [{"op": "set_fill", "target": "1:9", "color": "accent"}])]),
        message(content="Done."),
    ])
    bridge = FakeBridge()

    edit_loop.run("make it purple", bridge, llm, max_retries=1)

    # No token-creation script was ever sent.
    assert not any("createVariableCollection" in (r.code or "") for r in bridge.sent)
    step_prompt = llm.calls[-1]["messages"][1]["content"]
    assert "color/primary" in step_prompt and "color/secondary-text" in step_prompt


def test_the_canvas_listing_is_in_the_prompt_with_real_ids():
    """The ids are the whole game: an edit targets what the model can name."""
    llm = FakeModelClient([
        message(content='["Change the heading."]'),
        message(tool_calls=[edit_call("c1", [{"op": "set_text", "target": "1:4", "value": "Hi"}])]),
        message(content="Done."),
    ])

    edit_loop.run("change the heading", FakeBridge(), llm, max_retries=1)

    prompt = llm.calls[-1]["messages"][1]["content"]
    for node_id in ("1:2", "1:3", "1:4", "1:9"):
        assert node_id in prompt


def test_a_figma_selection_is_treated_as_the_answer_to_which_one():
    page = dict(PAGE, selection=["1:9"])
    llm = FakeModelClient([
        message(content='["Recolour the selected button."]'),
        message(tool_calls=[edit_call("c1", [{"op": "set_fill", "target": "1:9", "color": "accent"}])]),
        message(content="Done."),
    ])

    edit_loop.run("make this purple", FakeBridge(page=page), llm, max_retries=1)

    prompt = llm.calls[-1]["messages"][1]["content"]
    assert "SELECTED" in prompt and "1:9" in prompt


def test_the_canvas_is_re_read_between_steps():
    """An edit changes what the ids mean -- a `replace` makes the old id dead --
    so a stale listing would target a node that is gone."""
    llm = FakeModelClient([
        message(content='["Recolour the button.", "Shorten the heading."]'),
        message(tool_calls=[edit_call("c1", [{"op": "set_fill", "target": "1:9", "color": "accent"}])]),
        message(content="Done."),
        message(tool_calls=[edit_call("c2", [{"op": "set_text", "target": "1:4", "value": "Hi"}])]),
        message(content="Done."),
    ])
    bridge = FakeBridge()

    edit_loop.run("recolour and shorten", bridge, llm, max_retries=1)

    reads = [r for r in bridge.sent if "roots" in (r.code or "") and "selection" in (r.code or "")]
    assert len(reads) >= 3  # once up front, once per step


# ---- what must not happen -------------------------------------------------


def test_edit_mode_is_never_offered_a_tool_that_builds_a_screen():
    """`render_ui` would turn "make the button purple" into a second copy of the
    whole screen."""
    llm = FakeModelClient([
        message(content='["Recolour the button."]'),
        message(tool_calls=[edit_call("c1", [{"op": "set_fill", "target": "1:9", "color": "accent"}])]),
        message(content="Done."),
    ])

    edit_loop.run("make it purple", FakeBridge(), llm, max_retries=1)

    offered = {t["function"]["name"] for t in llm.calls[-1]["tools"]}
    assert offered == {"edit_ui", "get_metadata", "get_screenshot"}


def test_an_empty_file_is_refused_rather_than_built_into():
    """Edit mode changing nothing is correct; edit mode quietly becoming create
    mode is how a user who asked for a tweak gets a new screen."""
    llm = FakeModelClient([])
    bridge = FakeBridge(page={"pageName": "Blank", "selection": [], "roots": []})

    result = edit_loop.run("make the button purple", bridge, llm, max_retries=1)

    assert not result.success
    assert any("empty" in w for w in result.warnings)
    assert llm.calls == []   # not one model call was spent on it


def test_a_hallucinated_id_is_rejected_before_anything_reaches_figma():
    llm = FakeModelClient([
        message(content='["Recolour the button."]'),
        message(tool_calls=[edit_call("c1", [{"op": "set_fill", "target": "9:99", "color": "accent"}])]),
        message(tool_calls=[edit_call("c2", [{"op": "set_fill", "target": "1:9", "color": "accent"}])]),
        message(content="Done."),
    ])
    bridge = FakeBridge()

    result = edit_loop.run("make it purple", bridge, llm, max_retries=1)

    assert len(bridge.edit_scripts) == 1                 # only the good one ran
    assert targets_of(bridge.edit_scripts[0]) == ["1:9"]
    assert result.success


def test_an_edit_that_partly_fails_is_reported_as_partly_applied():
    """The batch is deliberately not atomic, so pretending nothing happened
    would send the user looking for changes that are really there."""
    llm = FakeModelClient([
        message(content='["Recolour both buttons."]'),
        message(tool_calls=[edit_call("c1", [
            {"op": "set_fill", "target": "1:9", "color": "accent"},
            {"op": "set_text", "target": "1:10", "value": "Sign in"}])]),
        message(tool_calls=[edit_call("c2", [{"op": "set_text", "target": "1:10", "value": "Sign in"}])]),
        message(tool_calls=[edit_call("c3", [{"op": "set_text", "target": "1:10", "value": "Sign In"}])]),
    ])
    bridge = FakeBridge(applied=[
        {"applied": ["set_fill on 1:9"], "failed": ["set_text on 1:10: not a text node"]},
        {"applied": [], "failed": ["set_text on 1:10: not a text node"]},
        {"applied": [], "failed": ["set_text on 1:10: not a text node"]},
    ])

    result = edit_loop.run("recolour and relabel", bridge, llm, max_retries=3)

    assert not result.success
    assert any("only partly applied" in w for w in result.warnings)


def test_a_retry_is_told_exactly_which_edit_failed():
    llm = FakeModelClient([
        message(content='["Relabel the button."]'),
        message(tool_calls=[edit_call("c1", [{"op": "set_text", "target": "1:9", "value": "Sign in"}])]),
        message(tool_calls=[edit_call("c2", [{"op": "set_text", "target": "1:10", "value": "Sign in"}])]),
        message(content="Done."),
    ])
    bridge = FakeBridge(applied=[
        {"applied": [], "failed": ["set_text on 1:9: not a text node"]},
        {"applied": ["set_text on 1:10"], "failed": []},
    ])

    result = edit_loop.run("relabel the button", bridge, llm, max_retries=2)

    retry_prompt = llm.calls[-1]["messages"][1]["content"]
    assert "not a text node" in retry_prompt
    assert result.success


def test_replying_with_text_and_no_call_is_a_failure_not_a_success():
    llm = FakeModelClient([
        message(content='["Recolour the button."]'),
        message(content="I would set the fill to the accent colour."),
        message(content="I would set the fill to the accent colour."),
    ])
    bridge = FakeBridge()

    result = edit_loop.run("make it purple", bridge, llm, max_retries=2)

    assert not result.success
    assert bridge.edit_scripts == []


def test_losing_figma_mid_edit_keeps_the_changes_that_landed():
    class DyingBridge(FakeBridge):
        """Figma goes away once the edit has landed, during the final review."""

        def send(self, request, timeout: float = 30.0):
            code = request.code or ""
            if self.edit_scripts and "roots" in code and "selection" in code:
                raise TimeoutError("Timed out waiting for plugin response")
            return super().send(request, timeout)

    llm = FakeModelClient([
        message(content='["Recolour the button."]'),
        message(tool_calls=[edit_call("c1", [{"op": "set_fill", "target": "1:9", "color": "accent"}])]),
        message(content="Done."),
    ])

    result = edit_loop.run("make it purple", DyingBridge(), llm, max_retries=1)

    assert "1:9" in result.created_node_ids
    assert any("ended early" in w for w in result.warnings)
    assert not result.success
