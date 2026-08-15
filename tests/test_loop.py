"""The loop with a FakeModelClient (scripted tool calls) and a FakeBridge
(canned exec/metadata/screenshot results). Proves loop logic, retries, and
termination -- no network or a running Figma required.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from agent import loop
from agent.state import RunState
from bridge.protocol import Response


def message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def tool_call(call_id: str, name: str, arguments: dict):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


class FakeModelClient:
    """Returns canned messages in call order, regardless of the messages/tools passed in."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        assert self._responses, "FakeModelClient ran out of scripted responses"
        return self._responses.pop(0)


ROOT_ID = "0:root"


class FakeBridge:
    """Returns canned Responses per request type, popped in order. No sockets.

    Scripts the harness itself authors (canvas inspection, the root frame) are
    answered automatically so each test only has to script the model-driven
    calls it actually cares about.
    """

    CLEAN_TREE = {
        "id": ROOT_ID, "name": "Root", "type": "FRAME", "x": 0, "y": 0,
        "width": 1440, "height": 400, "visible": True, "layoutMode": "VERTICAL",
        "children": [
            {
                "id": "sec:1", "name": "Section", "type": "FRAME", "x": 0, "y": 0,
                "width": 1440, "height": 400, "visible": True, "layoutMode": "VERTICAL",
                "children": [
                    {
                        "id": "t:1", "name": "Heading", "type": "TEXT", "x": 0, "y": 0,
                        "width": 600, "height": 40, "visible": True, "layoutMode": None,
                        "children": [], "characters": "Hello", "fontSize": 32,
                    }
                ],
            }
        ],
    }

    def __init__(
        self,
        responses_by_type: dict[str, list[Response]],
        existing_nodes: list | None = None,
        layout_tree: dict | None = None,
        still_unbound: list | None = None,
    ):
        self._responses_by_type = {k: list(v) for k, v in responses_by_type.items()}
        self._existing_nodes = existing_nodes or []
        self._layout_tree = layout_tree or FakeBridge.CLEAN_TREE
        self._still_unbound = still_unbound or []
        self.sent: list = []

    def send(self, request, timeout: float = 30.0) -> Response:
        self.sent.append(request)
        queue = self._responses_by_type.get(request.type, [])
        if request.type == "exec" and self._harness_response(request) is not None:
            # Harness-authored script. A test may still script its own answer
            # (the binding audit does); otherwise serve the default.
            if self._is_audit(request.code or "") and queue:
                return queue.pop(0)
            return self._harness_response(request)
        if request.type == "screenshot" and not queue:
            # Screenshots are harness-driven (visual gate + final review), so
            # tests only script them when they assert on the image itself.
            return Response(id=request.id, ok=True, image_base64="")
        assert queue, f"FakeBridge ran out of scripted '{request.type}' responses"
        return queue.pop(0)

    @staticmethod
    def _is_audit(code: str) -> bool:
        return "unboundFillNodeIds" in code

    @staticmethod
    def _is_token_script(code: str) -> bool:
        return "createVariableCollection('Tokens')" in code

    @staticmethod
    def _is_text_style_script(code: str) -> bool:
        return "getLocalTextStylesAsync" in code and "createTextStyle" in code

    @staticmethod
    def _is_placeholder(code: str) -> bool:
        return "TODO" in code

    @staticmethod
    def _is_font_script(code: str) -> bool:
        return "listAvailableFontsAsync" in code

    @staticmethod
    def _is_layout_script(code: str) -> bool:
        return "function describe" in code

    @staticmethod
    def _is_bind_script(code: str) -> bool:
        return "stillUnbound" in code

    @staticmethod
    def _is_hug_fix(code: str) -> bool:
        return "primaryAxisSizingMode = 'AUTO'" in code and "createFrame" not in code

    def _harness_response(self, request) -> Response | None:
        code = request.code or ""
        if code == loop.INSPECT_SCRIPT:
            return Response(
                id=request.id,
                ok=True,
                result={"createdNodeIds": [], "topLevelNodes": self._existing_nodes},
            )
        # Match the root-frame script precisely. Matching loosely on
        # "counterAxisSizingMode + createFrame" also swallowed every render_ui
        # script, so section builds silently never reached the bridge.
        if "figma.currentPage.appendChild(frame)" in code:
            return Response(id=request.id, ok=True, result={"createdNodeIds": [ROOT_ID]})
        if FakeBridge._is_token_script(code):
            return Response(id=request.id, ok=True, result={"createdNodeIds": [], "tokenNames": ["color/accent"]})
        if FakeBridge._is_text_style_script(code):
            return Response(id=request.id, ok=True, result={"createdNodeIds": [], "textStyleNames": ["Heading"]})
        if FakeBridge._is_placeholder(code):
            return Response(id=request.id, ok=True, result={"createdNodeIds": ["ph:1"]})
        if FakeBridge._is_font_script(code):
            return Response(
                id=request.id, ok=True,
                result={"createdNodeIds": [], "interStyles": ["Regular", "Semi Bold", "Bold"]},
            )
        if FakeBridge._is_layout_script(code):
            # A clean tree by default, so the visual gate passes unless a test
            # deliberately supplies a broken one.
            return Response(id=request.id, ok=True, result={"createdNodeIds": [], "tree": self._layout_tree})
        if FakeBridge._is_hug_fix(code):
            return Response(id=request.id, ok=True, result={"createdNodeIds": [], "changed": False})
        if FakeBridge._is_bind_script(code):
            return Response(
                id=request.id, ok=True,
                result={"createdNodeIds": [], "boundNodes": [], "stillUnbound": self._still_unbound},
            )
        if FakeBridge._is_audit(code):
            return Response(id=request.id, ok=True, result={"createdNodeIds": [], "unboundFillNodeIds": []})
        return None

    def model_exec_requests(self) -> list:
        """Only the execs the MODEL asked for -- excludes harness-authored ones."""
        return [r for r in self.sent if r.type == "exec" and self._harness_response(r) is None]


def test_successful_single_step_run():
    llm = FakeModelClient(
        [
            message(content="A clean red rectangle on a white background."),  # enhance_instruction
            message(content='["Create a red rectangle"]'),  # planner
            message(
                tool_calls=[
                    tool_call("call-1", "execute_figma_js", {"code": "return {createdNodeIds: ['1:2']}"})
                ]
            ),  # step: asks to run a script
            message(content="done"),  # step: no more tool calls -> step complete
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [
                Response(id="e1", ok=True, result={"createdNodeIds": ["1:2"]}),
                Response(id="a1", ok=True, result={"createdNodeIds": [], "unboundFillNodeIds": []}),  # binding audit
            ],
            "metadata": [Response(id="m1", ok=True, result={"id": "1:2", "name": "Rectangle", "type": "RECTANGLE"})],
            "screenshot": [Response(id="s1", ok=True, image_base64="iVBORw0KG...")],
        }
    )

    result = loop.run("a red rectangle", bridge, llm, max_retries=3, max_steps=10)

    assert result.success is True
    assert result.created_node_ids == [ROOT_ID, "1:2"]
    assert result.failed_steps == []
    assert result.final_screenshot_base64 == "iVBORw0KG..."
    assert result.warnings == []


def _one_step_run(bridge):
    llm = FakeModelClient(
        [
            message(content="accent: #0066FF"),  # enhance_instruction -> the palette
            message(content='["Create a red rectangle"]'),  # planner
            message(
                tool_calls=[
                    tool_call("call-1", "execute_figma_js", {"code": "return {createdNodeIds: ['1:2']}"})
                ]
            ),
            message(content="done"),
        ]
    )
    return loop.run("a red rectangle", bridge, llm, max_retries=3, max_steps=10)


def test_hardcoded_fills_are_rebound_to_tokens_not_just_reported():
    """Warning about a hardcoded colour doesn't make a design dynamic --
    rebinding it does. Only what matches no token is reported."""
    bridge = FakeBridge(
        {
            "exec": [Response(id="e1", ok=True, result={"createdNodeIds": ["1:2"]})],
            "metadata": [Response(id="m1", ok=True, result={"id": "1:2"})],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    result = _one_step_run(bridge)

    # The rebinding pass ran against the real palette parsed from the brief.
    bind = [r for r in bridge.sent if r.type == "exec" and "stillUnbound" in (r.code or "")]
    assert len(bind) == 1
    assert "color/accent" in bind[0].code
    assert result.success is True
    assert result.warnings == []  # nothing left unbound -> nothing to complain about


def test_only_colours_matching_no_token_are_reported():
    bridge = FakeBridge(
        {
            "exec": [Response(id="e1", ok=True, result={"createdNodeIds": ["1:2"]})],
            "metadata": [Response(id="m1", ok=True, result={"id": "1:2"})],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        },
        still_unbound=["1:2"],
    )

    result = _one_step_run(bridge)

    assert result.success is True  # a one-off colour doesn't fail the run
    assert any("one-off colour" in w for w in result.warnings)


def test_node_ids_are_never_double_counted():
    """A correcting retry returns ids we already have; they used to be added
    twice, so the audit reported the same node as two nodes."""
    state = RunState(instruction="x")
    state.add_node_ids(["9:30", "9:31"])
    state.add_node_ids(["9:30", "9:32"])  # 9:30 seen again after a retry

    assert state.created_node_ids == ["9:30", "9:31", "9:32"]


def test_failed_step_exhausts_retries_and_is_recorded():
    llm = FakeModelClient(
        [
            message(content="A clean red rectangle on a white background."),  # enhance_instruction
            message(content='["Create a red rectangle"]'),  # planner
            message(
                tool_calls=[tool_call("call-1", "execute_figma_js", {"code": "figma.bogus()"})]
            ),  # only retry attempt (max_retries=1)
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [
                Response(id="e1", ok=False, error="figma.bogus is not a function"),
            ],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    result = loop.run("a red rectangle", bridge, llm, max_retries=1, max_steps=10)

    assert result.success is False
    assert result.failed_steps == ["Create a red rectangle"]
    assert result.created_node_ids == [ROOT_ID]


def test_planner_falls_back_when_model_returns_unparseable_json():
    llm = FakeModelClient(
        [
            message(content="A clean red rectangle on a white background."),  # enhance_instruction
            message(content="I refuse to write JSON today."),  # planner: unparseable
            message(tool_calls=[tool_call("c1", "execute_figma_js", {"code": "return {createdNodeIds:['1:2']}"})]),
            message(content="done"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [
                Response(id="e1", ok=True, result={"createdNodeIds": ["1:2"]}),
                Response(id="a1", ok=True, result={"createdNodeIds": [], "unboundFillNodeIds": []}),
            ],
            "metadata": [Response(id="m1", ok=True, result={"id": "1:2"})],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    result = loop.run("anything", bridge, llm, max_retries=1, max_steps=10)

    # Falls back to a single default step rather than crashing the run, and
    # that fallback step is a real, runnable step.
    assert result.success is True
    assert result.created_node_ids == [ROOT_ID, "1:2"]


def test_enhanced_brief_feeds_into_planning_not_the_raw_instruction():
    llm = FakeModelClient(
        [
            message(content="EXPANDED BRIEF TEXT"),  # enhance_instruction
            message(content='["Create a red rectangle"]'),  # planner
            message(content="done"),  # step: no tool calls, finishes immediately
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    loop.run("a red rectangle", bridge, llm, max_retries=1, max_steps=10)

    planning_prompt = llm.calls[1]["messages"][-1]["content"]  # calls[0] was enhance_instruction
    assert "EXPANDED BRIEF TEXT" in planning_prompt


def test_step_that_never_runs_a_script_is_not_counted_as_done():
    """Real failure from a live run: the model printed the script as chat text
    instead of calling the tool, and the step was still marked 'done' despite
    nothing reaching the canvas."""
    llm = FakeModelClient(
        [
            message(content="brief"),  # enhance_instruction
            message(content='["Create a button component"]'),  # planner
            message(content="Here is the code you should run: ```js ... ```"),  # attempt 1: text only
            message(content="Here it is again as text"),  # attempt 2 (max_retries=2)
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    result = loop.run("a button", bridge, llm, max_retries=2, max_steps=10)

    assert result.success is False
    assert result.failed_steps == ["Create a button component"]


def test_step_is_done_once_a_script_actually_ran():
    llm = FakeModelClient(
        [
            message(content="brief"),  # enhance_instruction
            message(content='["Create a button component"]'),  # planner
            message(tool_calls=[tool_call("c1", "execute_figma_js", {"code": "return {createdNodeIds:['1:2']}"})]),
            message(content="All done."),  # no tool calls, but a script DID run
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [
                Response(id="e1", ok=True, result={"createdNodeIds": ["1:2"]}),
                Response(id="a1", ok=True, result={"createdNodeIds": [], "unboundFillNodeIds": []}),
            ],
            "metadata": [Response(id="m1", ok=True, result={"id": "1:2"})],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    result = loop.run("a button", bridge, llm, max_retries=2, max_steps=10)

    assert result.success is True
    assert result.created_node_ids == [ROOT_ID, "1:2"]


def test_a_malformed_tool_call_does_not_crash_the_run():
    """`Run crashed: 'code'` in a live run -- a call with no `code` argument
    must be reported back to the model, not raised."""
    llm = FakeModelClient(
        [
            message(content="brief"),  # enhance_instruction
            message(content='["Create a button"]'),  # planner
            message(tool_calls=[tool_call("c1", "execute_figma_js", {})]),  # no 'code'
            message(content="giving up"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    result = loop.run("a button", bridge, llm, max_retries=1, max_steps=10)  # must not raise

    assert result.success is False


def test_root_frame_is_created_by_the_harness_and_given_to_every_step():
    """A live run failed the root frame 3x (FILL is never legal on a child of
    the PAGE), and all six 'append into root frame' steps then had no parent.
    The harness owns this frame now, and every step is told its id."""
    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Build the hero"]'),
            message(content="done"),
        ]
    )
    bridge = FakeBridge({"screenshot": [Response(id="s1", ok=True, image_base64="")]})

    loop.run("a 1440px wide landing page", bridge, llm, max_retries=1, max_steps=10)

    # The root frame script ran before planning...
    root_scripts = [r for r in bridge.sent if r.type == "exec" and "counterAxisSizingMode" in (r.code or "")]
    assert len(root_scripts) == 1
    assert "1440" in root_scripts[0].code
    assert "VERTICAL" in root_scripts[0].code

    # ...and the step prompt carries its id, so no step has to search for it.
    step_prompt = next(
        m["content"] for m in llm.calls[-1]["messages"] if m["role"] == "user"
    )
    assert ROOT_ID in step_prompt
    assert "getNodeByIdAsync" in step_prompt


def test_existing_root_frame_is_reused_not_duplicated():
    """Re-running on a file you already designed must CONTINUE it. A hardcoded
    x/y meant every run stamped a second root frame on the first."""
    existing = [
        {
            "id": "9:1", "name": "Northwind — Page", "type": "FRAME",
            "x": 200, "y": 200, "width": 1440, "height": 3000, "layoutMode": "VERTICAL",
            "children": [
                {"id": "9:2", "name": "Nav Bar", "type": "FRAME"},
                {"id": "9:3", "name": "Hero", "type": "FRAME"},
            ],
        }
    ]
    llm = FakeModelClient(
        [message(content="brief"), message(content='["Add the footer"]'), message(content="done")]
    )
    bridge = FakeBridge(
        {"screenshot": [Response(id="s1", ok=True, image_base64="")]},
        existing_nodes=existing,
    )

    loop.run("a 1440px landing page", bridge, llm, max_retries=1, max_steps=10)

    # No new root frame was created at all.
    assert [r for r in bridge.sent if r.type == "exec" and "counterAxisSizingMode" in (r.code or "")] == []

    # The planner was told what already exists, so it plans only the gap.
    planning_prompt = llm.calls[1]["messages"][-1]["content"]
    assert "CONTINUATION" in planning_prompt
    assert "Nav Bar" in planning_prompt and "Hero" in planning_prompt

    # And so was the step itself.
    step_prompt = next(m["content"] for m in llm.calls[-1]["messages"] if m["role"] == "user")
    assert "9:1" in step_prompt
    assert "Do not recreate" in step_prompt


def test_a_new_root_frame_is_placed_clear_of_existing_content():
    """If there's nothing reusable, build beside the existing art -- not on it."""
    existing = [
        {"id": "1:1", "name": "Old sketch", "type": "RECTANGLE",
         "x": 0, "y": 0, "width": 800, "height": 600, "layoutMode": None, "children": []}
    ]
    llm = FakeModelClient(
        [message(content="brief"), message(content='["Build it"]'), message(content="done")]
    )
    bridge = FakeBridge(
        {"screenshot": [Response(id="s1", ok=True, image_base64="")]},
        existing_nodes=existing,
    )

    loop.run("a landing page", bridge, llm, max_retries=1, max_steps=10)

    root_script = [r for r in bridge.sent if r.type == "exec" and "counterAxisSizingMode" in (r.code or "")][0]
    assert "frame.x = 1000;" in root_script.code  # 800 wide + 200 gap, clear of the sketch


def test_a_non_autolayout_frame_is_not_mistaken_for_our_root():
    """Only auto-layout frames can safely receive appended sections."""
    plain = [{"id": "2:1", "name": "Moodboard", "type": "FRAME", "x": 0, "y": 0,
              "width": 1440, "height": 900, "layoutMode": "NONE", "children": []}]

    assert loop._find_reusable_root(plain, 1440) is None
    assert loop._find_reusable_root([], 1440) is None


def test_tokens_are_created_by_the_harness_and_offered_to_every_step():
    """Token creation was the most-failed step in live runs, and it's entirely
    mechanical -- the harness does it, and hands the names to the model."""
    llm = FakeModelClient(
        [message(content="accent: #0066FF"), message(content='["Add the hero"]'), message(content="done")]
    )
    bridge = FakeBridge({"screenshot": [Response(id="s1", ok=True, image_base64="")]})

    loop.run("a landing page", bridge, llm, max_retries=1, max_steps=10)

    codes = [r.code for r in bridge.sent if r.type == "exec"]
    assert any("createVariableCollection('Tokens')" in c for c in codes)
    assert any("createTextStyle" in c for c in codes)

    step_prompt = next(m["content"] for m in llm.calls[-1]["messages"] if m["role"] == "user")
    assert "color/accent" in step_prompt
    assert "setFillStyleIdAsync" in step_prompt
    assert "do not create any style or variable" in step_prompt


def test_a_failed_section_step_leaves_a_labelled_placeholder():
    """Rather than a hole in the page, the harness drops a visible TODO block."""
    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Add Footer section to root frame"]'),
            message(tool_calls=[tool_call("c1", "execute_figma_js", {"code": "boom()"})]),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [Response(id="e1", ok=False, error="boom is not a function")],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    result = loop.run("a landing page", bridge, llm, max_retries=1, max_steps=10)

    placeholders = [r for r in bridge.sent if r.type == "exec" and "TODO" in (r.code or "")]
    assert len(placeholders) == 1
    assert "Footer" in placeholders[0].code
    assert "ph:1" in result.created_node_ids  # the placeholder is real, tracked output
    assert result.failed_steps  # still reported as failed -- we don't hide it


def test_the_visual_gate_blocks_a_step_that_looks_broken():
    """A step can 'succeed' structurally and still be visibly wrong. The gate
    must not let that compound into the next step."""
    broken = {
        "id": ROOT_ID, "name": "Root", "type": "FRAME", "x": 0, "y": 0,
        "width": 1440, "height": 400, "visible": True, "layoutMode": "VERTICAL",
        "children": [
            {  # a headline collapsed to nothing -- invisible in the render
                "id": "t:1", "name": "Headline", "type": "TEXT", "x": 0, "y": 0,
                "width": 0, "height": 0, "visible": True, "layoutMode": None,
                "children": [], "characters": "Schedule smarter", "fontSize": 40,
            }
        ],
    }
    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Add Hero section to root frame"]'),
            message(tool_calls=[tool_call("c1", "execute_figma_js", {"code": "one"})]),
            message(content="done"),          # model thinks it finished...
            message(tool_calls=[tool_call("c2", "execute_figma_js", {"code": "fix"})]),
            message(content="fixed"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [
                Response(id="e1", ok=True, result={"createdNodeIds": ["t:1"]}),
                Response(id="e2", ok=True, result={"createdNodeIds": ["t:1"]}),
            ],
            "metadata": [Response(id="m1", ok=True, result={}), Response(id="m2", ok=True, result={})],
        },
        layout_tree=broken,
    )

    result = loop.run("a landing page", bridge, llm, max_retries=2, max_steps=10)

    # The gate ran, saw the collapsed headline, and spent the retry correcting it.
    assert any("collapsed" in d for d in result.layout_defects)
    corrections = [c for c in llm.calls if any(
        "CORRECTING THE PREVIOUS ATTEMPT" in m.get("content", "")
        for m in c["messages"] if m["role"] == "user"
    )]
    assert corrections, "the defect list should be fed back as a correction brief"
    # ...and that brief must name the node to fix and forbid a rebuild.
    brief = corrections[0]["messages"][1]["content"]
    assert "t:1" in brief
    assert "collapsed" in brief
    assert "DO NOT START OVER" in brief


def test_visual_gate_can_be_turned_off():
    """The Settings toggle must change real behaviour, not just the UI."""
    broken = {
        "id": ROOT_ID, "name": "Root", "type": "FRAME", "x": 0, "y": 0,
        "width": 1440, "height": 400, "visible": True, "layoutMode": "VERTICAL",
        "children": [{
            "id": "t:1", "name": "Headline", "type": "TEXT", "x": 0, "y": 0,
            "width": 0, "height": 0, "visible": True, "layoutMode": None,
            "children": [], "characters": "hi", "fontSize": 40,
        }],
    }

    def run_with(gate: bool):
        llm = FakeModelClient([
            message(content="brief"),
            message(content='["Add Hero section to root frame"]'),
            message(tool_calls=[tool_call("c1", "execute_figma_js", {"code": "one"})]),
            message(content="done"),
            message(tool_calls=[tool_call("c2", "execute_figma_js", {"code": "fix"})]),
            message(content="fixed"),
        ])
        bridge = FakeBridge(
            {
                "exec": [
                    Response(id="e1", ok=True, result={"createdNodeIds": ["t:1"]}),
                    Response(id="e2", ok=True, result={"createdNodeIds": ["t:1"]}),
                ],
                "metadata": [Response(id="m", ok=True, result={}) for _ in range(4)],
            },
            layout_tree=broken,
        )
        return loop.run("x", bridge, llm, max_retries=2, max_steps=10, visual_gate=gate), bridge

    on, bridge_on = run_with(True)
    off, bridge_off = run_with(False)

    # With the gate on, the broken layout is caught and retried.
    assert on.layout_defects
    # With it off, no layout read happens during the steps at all.
    layout_reads = [r for r in bridge_off.sent if r.type == "exec" and "function describe" in (r.code or "")]
    assert len(layout_reads) == 1  # only the final review, never the per-step gate


def test_runtime_font_discovery_reaches_the_step_prompt():
    """A live run died on 'Inter SemiBold' -- the real style is 'Semi Bold'."""
    llm = FakeModelClient(
        [message(content="brief"), message(content='["Add Hero to root frame"]'), message(content="done")]
    )
    bridge = FakeBridge({})

    loop.run("a landing page", bridge, llm, max_retries=1, max_steps=10)

    step_prompt = next(m["content"] for m in llm.calls[-1]["messages"] if m["role"] == "user")
    assert "Semi Bold" in step_prompt
    assert "never guess" in step_prompt
    # And known-good exemplars are always present.
    assert "resize BEFORE" in step_prompt or "appendChild" in step_prompt


def test_no_placeholder_for_token_or_component_steps():
    """A fake component is worse than none, and tokens already exist."""
    assert loop._is_section_step("Add Footer section to root frame") is True
    assert loop._is_section_step("Append Hero into the root frame") is True
    assert loop._is_section_step("Create color styles for the palette") is False
    assert loop._is_section_step("Create spacing variables") is False
    assert loop._is_section_step("Create Button component") is False


def test_section_labels_are_readable():
    assert loop._section_label("Add Footer section to root frame") == "Footer section"
    assert loop._section_label("Create Hero section: headline, CTA") == "Hero section"
    assert loop._section_label("Append the Features grid into root frame") == "Features grid"


def test_root_width_follows_an_explicit_pixel_width():
    assert loop._root_width("a 1440px wide landing page") == 1440
    assert loop._root_width("a 375 px mobile sign-in screen") == 375
    assert loop._root_width("a landing page") == loop.DEFAULT_ROOT_WIDTH
    # A stray small number (icon size) must not be mistaken for the page width.
    assert loop._root_width("cards with a 48px icon") == loop.DEFAULT_ROOT_WIDTH


def test_known_errors_get_a_targeted_fix_hint():
    """Generic docs weren't enough for the errors the model kept repeating."""
    fill = loop.augment_with_error("", "in set_layoutSizingHorizontal: FILL can only be set on children of auto-layout frames")
    assert "appendChild" in fill

    font = loop.augment_with_error("", 'The font "Inter SemiBold" could not be loaded')
    assert "Semi Bold" in font

    chaining = loop.augment_with_error("", "unexpected token in expression: '?'")
    assert "optional chaining" in chaining

    unknown = loop.augment_with_error("", "some totally novel error")
    assert "some totally novel error" in unknown  # still fed back, just no hint


def test_node_ids_are_normalized():
    """A live run returned ids with trailing commas ('S:44e4...,'), which then
    failed every lookup. Also tolerate a comma-joined string."""
    assert loop._normalize_node_ids(["1:2,", " 1:3 "]) == ["1:2", "1:3"]
    assert loop._normalize_node_ids("1:2,1:3") == ["1:2", "1:3"]
    assert loop._normalize_node_ids(None) == []
    assert loop._normalize_node_ids([]) == []
    assert loop._normalize_node_ids([None, 5, "1:4"]) == ["1:4"]


def test_style_ids_are_not_verified_as_nodes():
    """Style ids (S:...) aren't nodes -- verifying them wasted a round trip and
    produced a bogus warning for every style created."""
    state = RunState(instruction="x")
    bridge = FakeBridge({})  # any bridge call would raise "ran out of responses"

    loop.validate_creation(["S:abc", "S:def"], state, bridge)

    assert state.warnings == []
    assert bridge.sent == []


def test_query_docs_budget_stops_a_search_loop():
    """One live step burned all 6 turns on back-to-back query_docs calls."""
    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Do the thing"]'),
            *[
                message(tool_calls=[tool_call(f"c{i}", "query_docs", {"query": "how"})])
                for i in range(loop.MAX_TOOL_TURNS_PER_STEP)
            ],
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    loop.run("x", bridge, llm, max_retries=1, max_steps=10)

    # Only the allowed number of searches actually reached the docs tool;
    # the rest were refused with a nudge instead of being served.
    doc_calls = [c for c in llm.calls if c["tools"]]
    assert len(doc_calls) <= loop.MAX_TOOL_TURNS_PER_STEP


def test_identical_repeated_scripts_run_only_once():
    """Live runs looped: one step ran the same createPaintStyle script 8 times,
    another created 7 duplicate footer frames. The repeat must be refused
    rather than silently re-executed."""
    same = {"code": "const s = figma.createPaintStyle(); return {createdNodeIds: []}"}
    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Create the color styles"]'),
            *[
                message(tool_calls=[tool_call(f"c{i}", "execute_figma_js", same)])
                for i in range(4)
            ],
            message(content="done"),
        ]
    )
    bridge = FakeBridge(
        {
            # Only TWO exec responses queued: the inspect script plus ONE real
            # run of the model's script. If the repeats were executed, the fake
            # bridge would raise "ran out of scripted 'exec' responses".
            "exec": [
                Response(id="e1", ok=True, result={"createdNodeIds": []}),
            ],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    loop.run("colors", bridge, llm, max_retries=1, max_steps=10)

    exec_requests = bridge.model_exec_requests()
    assert len(exec_requests) == 1  # the repeated script ran exactly once


def test_different_scripts_in_one_step_all_run():
    """The repeat guard must not block legitimately different scripts."""
    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Build it"]'),
            message(tool_calls=[tool_call("c1", "execute_figma_js", {"code": "one"})]),
            message(tool_calls=[tool_call("c2", "execute_figma_js", {"code": "two"})]),
            message(content="done"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [
                Response(id="e1", ok=True, result={"createdNodeIds": ["1:1"]}),
                Response(id="e2", ok=True, result={"createdNodeIds": ["1:2"]}),
                Response(id="a1", ok=True, result={"createdNodeIds": [], "unboundFillNodeIds": []}),
            ],
            "metadata": [Response(id="m1", ok=True, result={}), Response(id="m2", ok=True, result={})],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    result = loop.run("x", bridge, llm, max_retries=1, max_steps=10)

    assert result.created_node_ids == [ROOT_ID, "1:1", "1:2"]


def test_enhance_falls_back_to_raw_instruction_when_model_returns_nothing():
    llm = FakeModelClient(
        [
            message(content=""),  # enhance_instruction: empty reply
            message(content='["Create a red rectangle"]'),  # planner
            message(content="done"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    loop.run("a red rectangle", bridge, llm, max_retries=1, max_steps=10)

    planning_prompt = llm.calls[1]["messages"][-1]["content"]
    assert "a red rectangle" in planning_prompt


def test_build_step_prompt_receives_the_brief_and_the_plan_outline():
    """The brief used to be consumed by the planner and then thrown away, so the
    step that made every visual decision never saw it. Guard that end to end.
    """
    brief = "Palette: Deep Navy (#0B1F3A). Sections: hero then footer. Tone: calm, nautical."
    llm = FakeModelClient(
        [
            message(content=brief),  # enhance_instruction
            message(content='["Add the hero section into the root frame.", '
                            '"Add the footer section into the root frame."]'),  # planner
            message(tool_calls=[tool_call("c1", "execute_figma_js", {"code": "one"})]),
            message(content="done"),
            message(tool_calls=[tool_call("c2", "execute_figma_js", {"code": "two"})]),
            message(content="done"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [
                Response(id="e1", ok=True, result={"createdNodeIds": ["1:1"]}),
                Response(id="e2", ok=True, result={"createdNodeIds": ["1:2"]}),
                Response(id="a1", ok=True, result={"createdNodeIds": [], "unboundFillNodeIds": []}),
            ],
            "metadata": [Response(id="m1", ok=True, result={}), Response(id="m2", ok=True, result={})],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    loop.run("a sailing school landing page", bridge, llm, max_retries=1, max_steps=10)

    # calls[0] = enhance, calls[1] = plan, calls[2] = first build step.
    # messages[1] is the step's user message -- not [-1], because converse_step
    # keeps appending assistant/tool turns to the same list the fake recorded.
    first_step_prompt = llm.calls[2]["messages"][1]["content"]
    assert "a sailing school landing page" in first_step_prompt   # the raw request
    assert "#0B1F3A" in first_step_prompt                          # the brief's palette
    assert "nautical" in first_step_prompt                         # the brief's tone
    assert ">>> THIS STEP: Add the hero" in first_step_prompt      # where this step sits

    # The second step must know the first one is already on the page.
    second_step_prompt = llm.calls[4]["messages"][1]["content"]
    assert "[already built] Add the hero" in second_step_prompt
    assert ">>> THIS STEP: Add the footer" in second_step_prompt


def test_a_defect_in_an_earlier_section_does_not_fail_a_later_step():
    """Regression: the gate used to read the whole root frame, so one bad
    section early in the run failed every step after it -- each burning its
    full retry budget and ending in a TODO placeholder.
    """
    dirty_page = {
        "id": ROOT_ID, "name": "Root", "type": "FRAME", "x": 0, "y": 0,
        "width": 1440, "height": 800, "visible": True, "layoutMode": "VERTICAL",
        "children": [
            {  # built by step 1, and broken
                "id": "1:1", "name": "Broken Hero", "type": "FRAME", "x": 0, "y": 0,
                "width": 0, "height": 0, "visible": True, "layoutMode": "VERTICAL",
                "children": [],
            },
            {  # built by step 2, and fine
                "id": "1:2", "name": "Good Footer", "type": "FRAME", "x": 0, "y": 400,
                "width": 1440, "height": 400, "visible": True, "layoutMode": "VERTICAL",
                "children": [
                    {
                        "id": "t:2", "name": "Fine", "type": "TEXT", "x": 0, "y": 0,
                        "width": 600, "height": 40, "visible": True, "layoutMode": None,
                        "children": [], "characters": "Hello", "fontSize": 32,
                    }
                ],
            },
        ],
    }

    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Add the hero section into the root frame.", '
                            '"Add the footer section into the root frame."]'),
            message(tool_calls=[tool_call("c1", "execute_figma_js", {"code": "hero"})]),
            message(content="done"),
            message(tool_calls=[tool_call("c2", "execute_figma_js", {"code": "footer"})]),
            message(content="done"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [
                Response(id="e1", ok=True, result={"createdNodeIds": ["1:1"]}),
                Response(id="e2", ok=True, result={"createdNodeIds": ["1:2"]}),
                Response(id="a1", ok=True, result={"createdNodeIds": [], "unboundFillNodeIds": []}),
            ],
            "metadata": [Response(id="m1", ok=True, result={}), Response(id="m2", ok=True, result={})],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        },
        layout_tree=dirty_page,
    )

    result = loop.run("x", bridge, llm, max_retries=1, max_steps=10)

    # Step 2 is clean on its own nodes, so it must pass despite the broken page.
    assert "Add the footer section into the root frame." not in result.failed_steps
    # Step 1 genuinely is broken, so it must still be caught.
    assert "Add the hero section into the root frame." in result.failed_steps
    # ...and the whole-page problem still surfaces in the final review.
    assert result.layout_defects


def test_repeat_guard_survives_across_retries():
    """A step used to build seven identical footers: each retry was a brand new
    conversation, so the identical-call guard reset with it. The guard is now
    owned by run_step and spans every attempt at the step.
    """
    broken = {
        "id": ROOT_ID, "name": "Root", "type": "FRAME", "x": 0, "y": 0,
        "width": 1440, "height": 400, "visible": True, "layoutMode": "VERTICAL",
        "children": [
            {
                "id": "1:1", "name": "Footer", "type": "FRAME", "x": 0, "y": 0,
                "width": 0, "height": 0, "visible": True, "layoutMode": None, "children": [],
            }
        ],
    }
    same_script = {"code": "build the footer"}
    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Add the footer section into the root frame."]'),
            message(tool_calls=[tool_call("c1", "execute_figma_js", same_script)]),
            message(content="done"),
            message(tool_calls=[tool_call("c2", "execute_figma_js", same_script)]),  # identical
            message(content="done"),
            message(tool_calls=[tool_call("c3", "execute_figma_js", same_script)]),  # identical
            message(content="done"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [Response(id="e1", ok=True, result={"createdNodeIds": ["1:1"]})],
            "metadata": [Response(id="m1", ok=True, result={})],
        },
        layout_tree=broken,
    )

    loop.run("a landing page", bridge, llm, max_retries=3, max_steps=10)

    # Three attempts, one distinct script: it must only ever have reached Figma once.
    assert len(bridge.model_exec_requests()) == 1


def test_a_retry_after_a_failed_script_still_repairs_what_landed():
    """A step can run script A successfully and then have script B throw. A is
    on the canvas, so the retry must be told about it instead of rebuilding.
    """
    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Add the hero section into the root frame."]'),
            message(tool_calls=[tool_call("c1", "execute_figma_js", {"code": "frame"})]),
            message(tool_calls=[tool_call("c2", "execute_figma_js", {"code": "text"})]),
            message(tool_calls=[tool_call("c3", "execute_figma_js", {"code": "fixed text"})]),
            message(content="done"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [
                Response(id="e1", ok=True, result={"createdNodeIds": ["1:1"]}),   # landed
                Response(id="e2", ok=False, error="Property lineHeight failed validation"),
                Response(id="e3", ok=True, result={"createdNodeIds": ["1:2"]}),
                Response(id="a1", ok=True, result={"createdNodeIds": [], "unboundFillNodeIds": []}),
            ],
            "metadata": [Response(id="m1", ok=True, result={}), Response(id="m2", ok=True, result={})],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    loop.run("a landing page", bridge, llm, max_retries=2, max_steps=10)

    retry_prompt = llm.calls[4]["messages"][1]["content"]
    assert "CORRECTING THE PREVIOUS ATTEMPT" in retry_prompt
    assert "1:1" in retry_prompt                        # the frame that survived
    assert "lineHeight" in retry_prompt                 # why the attempt stopped


def test_a_first_attempt_is_never_framed_as_a_repair():
    """Nothing is on the canvas yet, so the build instruction must stand alone."""
    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Add the hero section into the root frame."]'),
            message(tool_calls=[tool_call("c1", "execute_figma_js", {"code": "one"})]),
            message(content="done"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [
                Response(id="e1", ok=True, result={"createdNodeIds": ["sec:1"]}),
                Response(id="a1", ok=True, result={"createdNodeIds": [], "unboundFillNodeIds": []}),
            ],
            "metadata": [Response(id="m1", ok=True, result={})],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    loop.run("a landing page", bridge, llm, max_retries=2, max_steps=10)

    first_prompt = llm.calls[2]["messages"][1]["content"]
    assert "CORRECTING THE PREVIOUS ATTEMPT" not in first_prompt
    assert "accomplishes this step" in first_prompt


def test_sections_built_in_this_run_are_reported_to_later_steps():
    """existing_sections used to be filled in only when reusing a root frame,
    so on a fresh run every step was told the page was empty -- and duly built
    its own copy of what was already there.
    """
    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Add the hero section into the root frame.", '
                            '"Add the footer section into the root frame."]'),
            message(tool_calls=[tool_call("c1", "execute_figma_js", {"code": "hero"})]),
            message(content="done"),
            message(tool_calls=[tool_call("c2", "execute_figma_js", {"code": "footer"})]),
            message(content="done"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [
                Response(id="e1", ok=True, result={"createdNodeIds": ["1:1"]}),
                Response(id="e2", ok=True, result={"createdNodeIds": ["1:2"]}),
                Response(id="a1", ok=True, result={"createdNodeIds": [], "unboundFillNodeIds": []}),
            ],
            "metadata": [
                # the real Figma name is read back and reused verbatim
                Response(id="m1", ok=True, result={"id": "1:1", "name": "Hero", "type": "FRAME"}),
                Response(id="m2", ok=True, result={"id": "1:2", "name": "Footer", "type": "FRAME"}),
            ],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    loop.run("a landing page", bridge, llm, max_retries=1, max_steps=10)

    first_prompt = llm.calls[2]["messages"][1]["content"]
    second_prompt = llm.calls[4]["messages"][1]["content"]

    assert "already contains these sections" not in first_prompt  # nothing built yet
    assert "already contains these sections" in second_prompt
    assert "Hero" in second_prompt
    assert "They are FINISHED" in second_prompt


def test_a_placeholder_counts_as_an_occupied_section():
    """The TODO frame takes up the slot, so later steps must be told about it."""
    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Add the hero section into the root frame.", '
                            '"Add the footer section into the root frame."]'),
            message(content="I cannot do this"),   # step 1: never runs a script -> fails
            message(tool_calls=[tool_call("c2", "execute_figma_js", {"code": "footer"})]),
            message(content="done"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [
                Response(id="e2", ok=True, result={"createdNodeIds": ["1:2"]}),
                Response(id="a1", ok=True, result={"createdNodeIds": [], "unboundFillNodeIds": []}),
            ],
            "metadata": [Response(id="m2", ok=True, result={"id": "1:2", "name": "Footer", "type": "FRAME"})],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    loop.run("a landing page", bridge, llm, max_retries=1, max_steps=10)

    second_prompt = llm.calls[3]["messages"][1]["content"]
    assert "TODO — hero" in second_prompt


def test_only_tokens_that_really_exist_are_offered_to_the_model():
    """When the token script failed, the prompt still listed every colour from
    the brief. Every generated script then looked its style up, found nothing,
    and silently hardcoded a fill instead -- a whole run of untokenised colour
    that no gate could see.
    """
    state = RunState(instruction="x")
    state.token_names = ["color/navy"]          # only ONE token actually landed
    loop.describe_usable_palette(state, [("navy", "#0B1F3A"), ("sand", "#E8DCC8")])

    offered = [name for name, _, _ in state.palette_info]
    assert offered == ["color/navy"]
    # A pairing needs two real tokens; one cannot make a legal pair.
    assert state.readable_pairings == []


def test_losing_every_token_is_reported_as_a_warning():
    state = RunState(instruction="x")
    state.token_names = []
    loop.describe_usable_palette(state, [("navy", "#0B1F3A")])

    assert state.palette_info == []
    assert any("cannot be token-backed" in w for w in state.warnings)


def test_surviving_tokens_still_get_roles_and_pairings():
    state = RunState(instruction="x")
    state.token_names = ["color/navy", "color/sand"]
    loop.describe_usable_palette(state, [("navy", "#0B1F3A"), ("sand", "#E8DCC8")])

    assert len(state.palette_info) == 2
    assert state.readable_pairings
    assert state.warnings == []


def test_a_step_may_only_append_one_section_to_the_root():
    """From a real dashboard run: step 3 appended the nav bar to the root frame
    on three consecutive turns of the SAME attempt. Each script was worded
    slightly differently, so the identical-call guard let all three through and
    the page ended up with three stacked nav bars.
    """
    first = {"code": f'const root = await figma.getNodeByIdAsync("{ROOT_ID}");\nroot.appendChild(nav);'}
    reworded = {"code": f'const root = await figma.getNodeByIdAsync("{ROOT_ID}");\n// nav bar\nroot.appendChild(navBar);'}
    third = {"code": f'const r = await figma.getNodeByIdAsync("{ROOT_ID}");\nr.appendChild(bar);'}

    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Add the top navigation bar section into the root frame."]'),
            message(tool_calls=[tool_call("c1", "execute_figma_js", first)]),
            message(tool_calls=[tool_call("c2", "execute_figma_js", reworded)]),
            message(tool_calls=[tool_call("c3", "execute_figma_js", third)]),
            message(content="done"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [
                Response(id="e1", ok=True, result={"createdNodeIds": ["1:50"]}),
                Response(id="a1", ok=True, result={"createdNodeIds": [], "unboundFillNodeIds": []}),
            ],
            "metadata": [Response(id="m1", ok=True, result={"id": "1:50", "name": "Nav Bar", "type": "FRAME"})],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    loop.run("a dashboard", bridge, llm, max_retries=1, max_steps=10)

    # Only the first append reached Figma; the other two were refused.
    assert len(bridge.model_exec_requests()) == 1


def test_the_second_append_refusal_tells_the_model_how_to_finish():
    """A bare refusal would just make it try a fourth wording."""
    assert "SECOND copy" in loop._SECOND_APPEND_REFUSAL
    assert "NO tool call" in loop._SECOND_APPEND_REFUSAL or "no tool call" in loop._SECOND_APPEND_REFUSAL


def test_building_inside_your_own_section_is_not_blocked():
    """Only the ROOT frame is capped. Parenting into the section this step just
    created is normal work and must still go through."""
    assert not loop._appends_to_root(
        "execute_figma_js", {"code": 'const s = await figma.getNodeByIdAsync("1:50"); s.appendChild(t);'}, ROOT_ID
    )
    assert loop._appends_to_root(
        "execute_figma_js", {"code": f'const r = await figma.getNodeByIdAsync("{ROOT_ID}"); r.appendChild(s);'}, ROOT_ID
    )
    # Reading the root without appending is fine.
    assert not loop._appends_to_root(
        "execute_figma_js", {"code": f'const r = await figma.getNodeByIdAsync("{ROOT_ID}"); return r.width;'}, ROOT_ID
    )


def test_a_repair_attempt_may_not_append_to_the_root_at_all():
    """In repair mode the section already exists, so any root append is a duplicate."""
    broken = {
        "id": ROOT_ID, "name": "Root", "type": "FRAME", "x": 0, "y": 0,
        "width": 1440, "height": 400, "visible": True, "layoutMode": "VERTICAL",
        "children": [{
            "id": "1:50", "name": "Nav", "type": "FRAME", "x": 0, "y": 0,
            "width": 0, "height": 0, "visible": True, "layoutMode": None, "children": [],
        }],
    }
    append = {"code": f'const r = await figma.getNodeByIdAsync("{ROOT_ID}"); r.appendChild(nav);'}

    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Add the nav section into the root frame."]'),
            message(tool_calls=[tool_call("c1", "execute_figma_js", append)]),
            message(content="done"),
            # retry: tries to rebuild by appending to the root again
            message(tool_calls=[tool_call("c2", "execute_figma_js", dict(append, code=append["code"] + " // v2"))]),
            message(content="done"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [Response(id="e1", ok=True, result={"createdNodeIds": ["1:50"]})],
            "metadata": [Response(id="m1", ok=True, result={"id": "1:50", "name": "Nav", "type": "FRAME"})],
        },
        layout_tree=broken,
    )

    loop.run("a dashboard", bridge, llm, max_retries=2, max_steps=10)

    assert len(bridge.model_exec_requests()) == 1


def test_errors_seen_in_the_dashboard_run_all_map_to_a_targeted_fix():
    """Every error below burned at least one whole attempt in a live run with
    no hint attached, so the model just tried another guess."""
    real_errors = [
        "not a function",
        "in appendChild: Cannot move node. Reparenting would create a component inside a component",
        "in set_layoutSizingHorizontal: FILL can only be set on children of auto-layout frames",
        "in set_layoutSizingVertical: HUG can only be set on auto-layout frames or text children of auto-layout frames",
    ]
    for error in real_errors:
        augmented = loop.augment_with_error("", error)
        assert "FIX:" in augmented, f"no targeted hint for: {error}"


# ---- the vision critic --------------------------------------------------

class FakeCritic:
    """A vision critic returning canned JSON critiques."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    def complete(self, messages, tools=None):
        self.calls += 1
        reply = self._replies.pop(0) if self._replies else "[]"
        return SimpleNamespace(content=reply, tool_calls=None)


def _one_section_run(critic_llm, max_retries=2, layout_tree=None):
    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Add the hero section into the root frame."]'),
            message(tool_calls=[tool_call("c1", "execute_figma_js", {"code": "hero"})]),
            message(content="done"),
            message(tool_calls=[tool_call("c2", "execute_figma_js", {"code": "fix hero"})]),
            message(content="fixed"),
            message(tool_calls=[tool_call("c3", "execute_figma_js", {"code": "fix again"})]),
            message(content="fixed"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [
                Response(id="e1", ok=True, result={"createdNodeIds": ["sec:1"]}),
                Response(id="e2", ok=True, result={"createdNodeIds": ["sec:1"]}),
                Response(id="e3", ok=True, result={"createdNodeIds": ["sec:1"]}),
                Response(id="a1", ok=True, result={"createdNodeIds": [], "unboundFillNodeIds": []}),
            ],
            "metadata": [Response(id=f"m{i}", ok=True, result={"id": "sec:1", "name": "Hero"}) for i in range(4)],
            "screenshot": [Response(id=f"s{i}", ok=True, image_base64="PNG") for i in range(8)],
        },
        layout_tree=layout_tree,
    )
    result = loop.run("a landing page", bridge, llm, max_retries=max_retries,
                      max_steps=10, critic_llm=critic_llm)
    return result, bridge


def test_minor_visual_notes_never_fail_a_step():
    """A vision model always has polish suggestions. Gating on them would fail
    essentially every step and bury the page in TODO placeholders."""
    critic_llm = FakeCritic([json.dumps([
        {"severity": "minor", "element": "Hero", "problem": "could use more breathing room"},
        {"severity": "minor", "element": "Heading", "problem": "hierarchy could be stronger"},
    ])])
    result, _ = _one_section_run(critic_llm)

    assert result.failed_steps == []
    assert result.success is True
    assert critic_llm.calls == 1  # passed first time; no retry needed


def test_a_blocking_visual_defect_does_trigger_a_repair():
    critic_llm = FakeCritic([
        json.dumps([{"severity": "blocking", "element": "Headline", "problem": "unreadable on the navy fill"}]),
        "[]",  # the repair fixed it
    ])
    result, bridge = _one_section_run(critic_llm)

    assert result.failed_steps == []
    assert critic_llm.calls == 2
    assert len(bridge.model_exec_requests()) == 2  # built, then repaired


def test_an_unfixable_visual_complaint_keeps_the_section_rather_than_a_placeholder():
    """Geometry defects are facts; 'it could look better' is judgement. Trading
    a real section for an empty TODO frame over judgement is a regression."""
    critic_llm = FakeCritic([json.dumps(
        [{"severity": "blocking", "element": "Hero", "problem": "still looks unbalanced"}]
    )] * 5)
    result, bridge = _one_section_run(critic_llm, max_retries=2)

    assert result.failed_steps == []                      # NOT marked failed
    assert not any("TODO" in (r.code or "") for r in bridge.sent if r.type == "exec")
    assert any("unresolved visual notes" in w for w in result.warnings)


def test_a_geometry_defect_still_falls_back_to_a_placeholder():
    """The contrast case: a 0x0 node renders nothing, and that IS a failure."""
    broken = {
        "id": ROOT_ID, "name": "Root", "type": "FRAME", "x": 0, "y": 0,
        "width": 1440, "height": 400, "visible": True, "layoutMode": "VERTICAL",
        "children": [{
            "id": "sec:1", "name": "Hero", "type": "FRAME", "x": 0, "y": 0,
            "width": 0, "height": 0, "visible": True, "layoutMode": None, "children": [],
        }],
    }
    result, bridge = _one_section_run(FakeCritic(["[]"] * 5), max_retries=2, layout_tree=broken)

    assert result.failed_steps
    assert any("TODO" in (r.code or "") for r in bridge.sent if r.type == "exec")


def test_no_critic_configured_means_no_images_are_ever_sent():
    """A text-only endpoint 400s on an image; discovering that costs a round
    trip per step. If no vision model is configured, don't try at all."""
    result, bridge = _one_section_run(None)

    assert result.failed_steps == []
    # The only screenshot is the final one for the dashboard.
    assert len([r for r in bridge.sent if r.type == "screenshot"]) == 1


def test_a_run_that_builds_nothing_is_not_reported_as_success():
    """The history showed runs marked "Success -- 0 nodes". A green tick on an
    empty page destroys trust in every other number the tool reports.
    """
    from agent.state import StepResult

    state = RunState(instruction="a dashboard")
    assert state.succeeded() is False          # nothing built, nothing failed
    assert state.result().success is False

    # A step that ran but touched nothing is still not a success.
    state.record_step_result("x", StepResult("x", ok=True, created_node_ids=[]))
    assert state.succeeded() is False

    state.record_step_result("y", StepResult("y", ok=True, created_node_ids=["1:9"]))
    assert state.succeeded() is True


def test_a_page_of_todo_placeholders_is_not_a_success():
    """Placeholders keep the layout readable; they are not built sections."""
    from agent.state import StepResult

    state = RunState(instruction="a dashboard")
    state.mark_failed("build the sidebar")
    state.record_step_result(
        "build the sidebar",
        StepResult("build the sidebar", ok=False, created_node_ids=["ph:1"],
                   summary="exhausted retries (placeholder added)"),
    )
    state.record_section("TODO — sidebar")

    assert state.built_section_count() == 0
    assert state.result().success is False


def test_render_ui_builds_a_whole_section_in_one_call():
    """The model describes WHAT; the harness writes the Figma code. Every API
    mistake it used to make -- FILL before append, collapsed text, unloaded
    fonts -- is unreachable through this path.
    """
    spec = {"kind": "section", "name": "Sign in", "children": [
        {"kind": "text", "style": "Heading", "value": "Welcome back"},
        {"kind": "input", "label": "Email", "placeholder": "you@co.com"},
        {"kind": "button", "label": "Sign in", "variant": "primary"},
    ]}
    llm = FakeModelClient([
        message(content="brief"),
        message(content='["Add the sign-in section into the root frame."]'),
        message(tool_calls=[tool_call("c1", "render_ui", {"spec": spec})]),
        message(content="done"),
    ])
    bridge = FakeBridge({
        "exec": [
            Response(id="e1", ok=True, result={"createdNodeIds": ["1:1", "1:2", "1:3"]}),
            Response(id="a1", ok=True, result={"createdNodeIds": [], "unboundFillNodeIds": []}),
        ],
        "metadata": [Response(id="m1", ok=True, result={"id": "1:1", "name": "Sign in"})],
        "screenshot": [Response(id="s1", ok=True, image_base64="")],
    })

    result = loop.run("a sign-in screen", bridge, llm, max_retries=1, max_steps=10)

    assert result.success is True
    sent = bridge.model_exec_requests()
    assert len(sent) == 1
    code = sent[0].code
    assert "loadFontAsync" in code            # fonts handled for the model
    assert "textAutoResize = 'WIDTH_AND_HEIGHT'" in code   # text cannot collapse
    assert "setFillStyleIdAsync" in code      # colour is token-backed


def test_an_invalid_spec_never_reaches_figma():
    """A bad spec must fail as a readable message, not as half a section."""
    llm = FakeModelClient([
        message(content="brief"),
        message(content='["Add the hero section into the root frame."]'),
        message(tool_calls=[tool_call("c1", "render_ui", {"spec": {"kind": "carousel"}})]),
        message(tool_calls=[tool_call("c2", "render_ui", {"spec": {"kind": "section", "children": []}})]),
        message(content="done"),
    ])
    bridge = FakeBridge({
        "exec": [
            Response(id="e1", ok=True, result={"createdNodeIds": ["1:1"]}),
            Response(id="a1", ok=True, result={"createdNodeIds": [], "unboundFillNodeIds": []}),
        ],
        "metadata": [Response(id="m1", ok=True, result={"id": "1:1", "name": "Hero"})],
        "screenshot": [Response(id="s1", ok=True, image_base64="")],
    })

    loop.run("a landing page", bridge, llm, max_retries=1, max_steps=10)

    # The invalid spec was rejected in Python; only the valid one ran.
    assert len(bridge.model_exec_requests()) == 1
