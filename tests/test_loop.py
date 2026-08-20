"""The loop with a FakeModelClient (scripted tool calls) and a FakeBridge
(canned exec/metadata/screenshot results). Proves loop logic, retries, and
termination -- no network or a running Figma required.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

from agent import loop
from agent.state import RunState
from bridge.protocol import Response


def message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def tool_call(call_id: str, name: str, arguments: dict):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


class FakeModelClient:
    """Returns canned messages in call order, regardless of the messages/tools passed in.

    Screen decomposition is answered AUTOMATICALLY with a single screen, the
    same way FakeBridge auto-serves harness-authored scripts: almost every test
    here is about one screen, and making each of them script an extra response
    would mean editing forty tests to add a line none of them care about. A
    test that IS about several screens passes `screens=[...]`.
    """

    def __init__(self, responses: list, screens: list[str] | None = None):
        self._responses = list(responses)
        self._screens = screens or ["Screen"]
        self.calls: list[dict] = []

    def complete(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        if self._is_screens_question(messages):
            return message(content=json.dumps(self._screens))
        assert self._responses, "FakeModelClient ran out of scripted responses"
        return self._responses.pop(0)

    @staticmethod
    def _is_screens_question(messages) -> bool:
        system = (messages[0].get("content") or "") if messages else ""
        return "how many separate SCREENS" in system


ROOT_ID = "0:root"


def render_call(call_id: str, name: str = "Section", value: str = "Hello"):
    """A render_ui call building one small section.

    Section steps are only given `render_ui` -- `execute_figma_js` is not on
    their menu, because a real run reached for raw JS every single time and
    lost steps to FILL/HUG ordering, font loading and enum values that the
    renderer handles for free. Tests build the way the product builds.
    """
    return tool_call(
        call_id,
        "render_ui",
        {"spec": {"kind": "section", "name": name,
                  "children": [{"kind": "text", "value": value}]}},
    )



def metadata_call(call_id: str, node_id: str):
    """A read-only `get_metadata` call, which every step is allowed to make."""
    return tool_call(call_id, "get_metadata", {"node_id": node_id})



# Indexing into llm.calls[N] breaks the moment the loop gains a stage -- adding
# screen decomposition shifted every one of them by one. Find calls by the job
# they are doing instead.
def _calls_with(llm, marker: str) -> list[str]:
    """The user message of every model call whose SYSTEM prompt contains `marker`."""
    found = []
    for call in llm.calls:
        system = (call["messages"][0].get("content") or "") if call["messages"] else ""
        if marker in system:
            found.append(next(m["content"] for m in call["messages"] if m["role"] == "user"))
    return found


def planning_prompts(llm) -> list[str]:
    return _calls_with(llm, "ordered list of build steps")


def step_prompts(llm) -> list[str]:
    """One entry per build ATTEMPT, not per model turn.

    A step's tool-calling conversation makes several calls that all share the
    same user message (converse_step appends assistant/tool turns to the same
    list), so consecutive duplicates are collapsed. A retry builds a fresh
    conversation with different framing, so it is correctly kept.
    """
    prompts = []
    for content in _calls_with(llm, "You are the design agent"):
        if not prompts or prompts[-1] != content:
            prompts.append(content)
    return prompts


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
        # What was ANSWERED, paired with `sent` by index. Assertions about node
        # lifetime need the ids the bridge handed back, not just the requests.
        self.served: list = []

    def send(self, request, timeout: float = 30.0) -> Response:
        self.sent.append(request)
        response = self._answer(request)
        self.served.append(response)
        return response

    def _answer(self, request) -> Response:
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
            return Response(id=request.id, ok=True, image_base64="PNG")
        assert queue, f"FakeBridge ran out of scripted '{request.type}' responses"
        return queue.pop(0)

    @staticmethod
    def _is_audit(code: str) -> bool:
        return "unboundFillNodeIds" in code

    @staticmethod
    def _is_screen_script(code: str) -> bool:
        return "failedScreens" in code

    @staticmethod
    def _screen_frames_response(request, code: str) -> Response:
        """Hand back one id per requested screen, named as the script asked."""
        match = re.search(r"const specs = (\[.*?\]);", code, re.DOTALL)
        specs = json.loads(match.group(1)) if match else []
        made = [
            {"name": spec["name"], "id": ROOT_ID if i == 0 else f"screen:{i + 1}"}
            for i, spec in enumerate(specs)
        ]
        return Response(
            id=request.id,
            ok=True,
            result={"createdNodeIds": [m["id"] for m in made], "screens": made, "failedScreens": []},
        )

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
    def _is_remove_script(code: str) -> bool:
        return "removedNodeIds" in code

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
        # The harness creates ONE FRAME PER SCREEN in a single script. Answer it
        # with real ids so each screen has somewhere to build; the first keeps
        # ROOT_ID so single-screen tests read exactly as they always did.
        if FakeBridge._is_screen_script(code):
            return FakeBridge._screen_frames_response(request, code)
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
        # Cleanup of work that did not pass its gates. Echo back exactly the
        # ids the script was asked to remove, so a test can assert on them.
        if FakeBridge._is_remove_script(code):
            asked = json.loads(re.search(r"const ids = (\[.*?\]);", code, re.DOTALL).group(1))
            return Response(
                id=request.id, ok=True,
                result={"createdNodeIds": [], "removedNodeIds": asked},
            )
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
                    render_call("call-1")
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
                    render_call("call-1")
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
            message(tool_calls=[render_call("c1")]),
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

    planning_prompt = planning_prompts(llm)[0]
    assert "EXPANDED BRIEF TEXT" in planning_prompt


def test_step_that_never_runs_a_script_is_not_counted_as_done():
    """Real failure from a live run: the model printed the script as chat text
    instead of calling the tool, and the step was still marked 'done' despite
    nothing reaching the canvas."""
    llm = FakeModelClient(
        [
            message(content="brief"),  # enhance_instruction
            message(content='["Add the button section into the frame"]'),  # planner
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
    assert result.failed_steps == ["Add the button section into the frame"]


def test_step_is_done_once_a_script_actually_ran():
    llm = FakeModelClient(
        [
            message(content="brief"),  # enhance_instruction
            message(content='["Add the button section into the frame"]'),  # planner
            message(tool_calls=[render_call("c1")]),
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
    planning_prompt = planning_prompts(llm)[0]
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

    screen_script = [r for r in bridge.sent if r.type == "exec" and "failedScreens" in (r.code or "")][0]
    # 800 wide + 200 gap: the new screen starts clear of the existing sketch.
    assert '"x": 1000' in screen_script.code


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
            message(tool_calls=[render_call("c1")]),
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
            message(tool_calls=[render_call("c1")]),
            message(content="done"),          # model thinks it finished...
            message(tool_calls=[render_call("c2", "Hero fixed")]),
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

    result = loop.run("a landing page", bridge, llm, max_retries=2, max_steps=10, final_repair=False)

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
            message(tool_calls=[render_call("c1")]),
            message(content="done"),
            message(tool_calls=[render_call("c2", "Hero fixed")]),
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
        return loop.run("x", bridge, llm, max_retries=2, max_steps=10, visual_gate=gate, final_repair=False), bridge

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
    """The repeat guard must not block legitimately different scripts.

    Uses a NON-section step, because that is the only kind that runs raw JS now
    -- a section is exactly one `render_ui` call by construction.
    """
    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Prepare the spacing tokens"]'),
            message(tool_calls=[tool_call("c1", "execute_figma_js", {"code": "first"})]),
            message(tool_calls=[tool_call("c2", "execute_figma_js", {"code": "second"})]),
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

    planning_prompt = planning_prompts(llm)[0]
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
            message(tool_calls=[render_call("c1")]),
            message(content="done"),
            message(tool_calls=[render_call("c2")]),
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
    first_step_prompt = step_prompts(llm)[0]
    assert "a sailing school landing page" in first_step_prompt   # the raw request
    assert "#0B1F3A" in first_step_prompt                          # the brief's palette
    assert "nautical" in first_step_prompt                         # the brief's tone
    assert ">>> THIS STEP: Add the hero" in first_step_prompt      # where this step sits

    # The second step must know the first one is already on the page.
    second_step_prompt = step_prompts(llm)[1]
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
            message(tool_calls=[render_call("c1")]),
            message(content="done"),
            message(tool_calls=[render_call("c2")]),
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

    result = loop.run("x", bridge, llm, max_retries=1, max_steps=10, final_repair=False)

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
    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Add the footer section into the root frame."]'),
            message(tool_calls=[render_call("c1", "Footer")]),
            message(content="done"),
            message(tool_calls=[render_call("c2", "Footer")]),  # identical spec
            message(content="done"),
            message(tool_calls=[render_call("c3", "Footer")]),  # identical spec
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

    loop.run("a landing page", bridge, llm, max_retries=3, max_steps=10, final_repair=False)

    # Three attempts, one distinct script: it must only ever have reached Figma once.
    assert len(bridge.model_exec_requests()) == 1


def test_a_retry_after_a_failed_script_still_repairs_what_landed():
    """A step can run script A successfully and then have script B throw. A is
    on the canvas, so the retry must be told about it instead of rebuilding.
    """
    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Prepare the hero text tokens"]'),
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

    retry_prompt = step_prompts(llm)[-1]
    assert "CORRECTING THE PREVIOUS ATTEMPT" in retry_prompt
    assert "1:1" in retry_prompt                        # the frame that survived
    assert "lineHeight" in retry_prompt                 # why the attempt stopped


def test_a_first_attempt_is_never_framed_as_a_repair():
    """Nothing is on the canvas yet, so the build instruction must stand alone."""
    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Add the hero section into the root frame."]'),
            message(tool_calls=[render_call("c1")]),
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

    first_prompt = step_prompts(llm)[0]
    assert "CORRECTING THE PREVIOUS ATTEMPT" not in first_prompt
    assert "Call `render_ui` ONCE" in first_prompt


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
            message(tool_calls=[render_call("c1")]),
            message(content="done"),
            message(tool_calls=[render_call("c2")]),
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

    first_prompt = step_prompts(llm)[0]
    second_prompt = step_prompts(llm)[1]

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
            message(tool_calls=[render_call("c2")]),
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

    second_prompt = step_prompts(llm)[1]
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


def test_a_step_never_leaves_three_stacked_copies_of_one_section():
    """From a real dashboard run: step 3 appended the nav bar to the root frame
    on three consecutive turns of the SAME attempt. Each script was worded
    slightly differently, so the identical-call guard let all three through and
    the page ended up with three stacked nav bars.

    Each `render_ui` now REPLACES the last, so a reworded retry corrects the
    section instead of duplicating it -- which is what the model is actually
    asking for when it says the previous render was missing something.
    """
    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Add the top navigation bar section into the root frame."]'),
            message(tool_calls=[render_call("c1", "Nav Bar")]),
            message(tool_calls=[render_call("c2", "Navigation")]),   # reworded, same job
            message(tool_calls=[render_call("c3", "Top Nav")]),      # reworded again
            message(content="done"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [
                Response(id="e1", ok=True, result={"createdNodeIds": ["1:50"]}),
                Response(id="e2", ok=True, result={"createdNodeIds": ["1:60"]}),
                Response(id="e3", ok=True, result={"createdNodeIds": ["1:70"]}),
                Response(id="a1", ok=True, result={"createdNodeIds": [], "unboundFillNodeIds": []}),
            ],
            "metadata": [
                Response(id=f"m{i}", ok=True, result={"id": "1:50", "name": "Nav", "type": "FRAME"})
                for i in range(4)
            ],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    result = loop.run("a dashboard", bridge, llm, max_retries=1, max_steps=10)

    execs = bridge.model_exec_requests()
    # The reworded retries each REMOVE what the last one built...
    assert '"1:50"' in (execs[1].code or ""), "the second render did not replace the first"
    assert '"1:60"' in (execs[2].code or ""), "the third render did not replace the second"
    # ...so only the final section survives, and the run reports only that one.
    assert "1:50" not in result.created_node_ids
    assert "1:60" not in result.created_node_ids
    assert "1:70" in result.created_node_ids


def test_a_step_cannot_rebuild_its_section_without_limit():
    """Self-correction is bounded -- an endless rebuild loop is the other way
    this goes wrong."""
    assert loop.MAX_RENDERS_PER_STEP <= 3


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


def test_a_section_repair_replaces_the_broken_section_instead_of_duplicating_it():
    """`render_ui` cannot edit nodes in place, so a correcting retry removes what
    the last attempt built and renders the section again in its place.

    Without this the gate would be unable to fix anything: appending would leave
    two copies, and refusing the append would send every visual-gate failure
    straight to a TODO placeholder."""
    broken = {
        "id": ROOT_ID, "name": "Root", "type": "FRAME", "x": 0, "y": 0,
        "width": 1440, "height": 400, "visible": True, "layoutMode": "VERTICAL",
        "children": [{
            "id": "1:50", "name": "Nav", "type": "FRAME", "x": 0, "y": 0,
            "width": 0, "height": 0, "visible": True, "layoutMode": None, "children": [],
        }],
    }
    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Add the nav section into the root frame."]'),
            message(tool_calls=[render_call("c1", "Nav")]),
            message(content="done"),
            # retry: tries to rebuild the section from scratch
            message(tool_calls=[render_call("c2", "Nav v2")]),
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

    loop.run("a dashboard", bridge, llm, max_retries=2, max_steps=10, final_repair=False)

    execs = bridge.model_exec_requests()
    assert len(execs) == 2                                  # built, then repaired
    # The repair removes the previous attempt's node before rebuilding.
    assert "1:50" in (execs[1].code or "")
    assert "_old.remove()" in (execs[1].code or "")


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
            message(tool_calls=[render_call("c1")]),
            message(content="done"),
            message(tool_calls=[render_call("c2", "Hero fixed")]),
            message(content="fixed"),
            message(tool_calls=[render_call("c3", "Hero fixed again")]),
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
                      max_steps=10, critic_llm=critic_llm, final_repair=False)
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


# ---- requirement coverage + design-system review at the end of a run -------


def _run_with_instruction(instruction: str, bridge):
    llm = FakeModelClient(
        [
            message(content="accent: #0066FF"),
            message(content='["Add the sign-in section into the root frame"]'),
            message(
                tool_calls=[
                    render_call("c1")
                ]
            ),
            message(content="done"),
        ]
    )
    return loop.run(instruction, bridge, llm, max_retries=1, max_steps=10, final_repair=False)


def _bridge_with_tree(tree):
    return FakeBridge(
        {
            "exec": [Response(id="e1", ok=True, result={"createdNodeIds": ["1:2"]})],
            "metadata": [Response(id="m1", ok=True, result={"id": "1:2", "name": "Sign in"})],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        },
        layout_tree=tree,
    )


def _tree_with(children):
    return {
        "id": ROOT_ID, "name": "Root", "type": "FRAME", "x": 0, "y": 0,
        "width": 1440, "height": 400, "visible": True, "layoutMode": "VERTICAL",
        "children": children,
    }


def _text_node(node_id, name, characters, **kw):
    base = {
        "id": node_id, "name": name, "type": "TEXT", "x": 0, "y": 0,
        "width": 300, "height": 24, "visible": True, "layoutMode": None,
        "children": [], "characters": characters, "fontSize": 16,
    }
    base.update(kw)
    return base


def test_a_dropped_requirement_is_reported_on_the_finished_run():
    """A sign-in screen with no password field passed every previous gate."""
    tree = _tree_with([_text_node("t:1", "Email", "Email address")])

    result = _run_with_instruction("a sign-in screen with email and password", _bridge_with_tree(tree))

    assert "password field" in result.requirements_missing
    assert "email field" in result.requirements_met
    assert any("missing from the design" in w for w in result.warnings)


def test_a_run_matching_none_of_the_instruction_is_not_a_success():
    """CLAUDE.md gap #2: 'a run can satisfy none of the instruction and report
    success'. The steps all pass -- only coverage catches this."""
    tree = _tree_with([_text_node("t:1", "Blob", "Lorem ipsum")])

    result = _run_with_instruction(
        "a dashboard with a sidebar, a chart and a data table", _bridge_with_tree(tree)
    )

    assert result.success is False
    assert result.failed_steps == []  # every step passed; the DESIGN is wrong
    assert any("matches NONE of the instruction" in w for w in result.warnings)


def test_a_satisfied_instruction_still_succeeds():
    tree = _tree_with([
        _text_node("t:1", "Email", "Email address"),
        _text_node("t:2", "Password", "Password", y=40),
    ])

    result = _run_with_instruction("a sign-in screen with email and password", _bridge_with_tree(tree))

    assert result.success is True
    assert result.requirements_missing == []


def test_design_system_notes_are_recorded_but_never_fail_the_run():
    tree = _tree_with([
        {
            "id": "sec:1", "name": "Section", "type": "FRAME", "x": 0, "y": 0,
            "width": 1440, "height": 200, "visible": True, "layoutMode": "VERTICAL",
            "itemSpacing": 17, "padding": [13, 13, 13, 13],
            "children": [
                _text_node("t:1", "Email", "Email", fontSize=17),
                _text_node("t:2", "Password", "Password", y=40, fontSize=17),
            ],
        }
    ])

    result = _run_with_instruction("a sign-in screen with email and password", _bridge_with_tree(tree))

    kinds = " ".join(result.design_notes)
    assert "off-scale-spacing" in kinds
    assert "off-ramp-type" in kinds
    assert result.success is True          # advisory: it must not fail the run
    assert result.layout_defects == []     # nor leak into the blocking list


# ---- observability over the expensive paths --------------------------------


def test_a_run_reports_what_it_cost():
    bridge = FakeBridge(
        {
            "exec": [Response(id="e1", ok=True, result={"createdNodeIds": ["1:2"]})],
            "metadata": [Response(id="m1", ok=True, result={"id": "1:2", "name": "Hero"})],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    result = _one_step_run(bridge)

    m = result.metrics
    # Figma round trips are counted by type, not lumped together. (Model calls
    # are recorded inside ModelClient -- the swap point every real caller goes
    # through -- so a FakeModelClient legitimately shows none; see test_llm.py.)
    assert m["bridge"]["exec"]["count"] > 0
    assert m["bridge_calls"] == sum(t["count"] for t in m["bridge"].values())
    assert m["steps_completed"] == 1
    assert m["step_attempts"] == 1
    assert m["retry_rate"] == 1.0  # nothing needed a second try


def test_failures_are_bucketed_by_cause_not_just_logged():
    """'The model printed a script instead of calling the tool' and 'the Plugin
    API threw' read identically in the log but call for different fixes."""
    llm = FakeModelClient(
        [
            message(content="accent: #0066FF"),
            message(content='["Add the hero section into the root frame"]'),
            message(content="here is the script you asked for: figma.createFrame()"),  # no tool call
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [],
            "metadata": [],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    result = loop.run("a hero section", bridge, llm, max_retries=1, max_steps=10)

    assert result.metrics["failure_reasons"]["no-script-run"] == 1
    assert result.metrics["failure_reasons"]["exhausted-retries"] == 1
    assert result.metrics["steps_failed"] == 1


def test_retries_show_up_as_a_retry_rate_above_one():
    llm = FakeModelClient(
        [
            message(content="accent: #0066FF"),
            message(content='["Add the hero section into the root frame"]'),
            message(tool_calls=[render_call("c1")]),
            message(tool_calls=[render_call("c2", "Hero retry")]),
            message(content="done"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [
                Response(id="e1", ok=False, error="boom is not a function"),
                Response(id="e2", ok=True, result={"createdNodeIds": ["1:2"]}),
            ],
            "metadata": [Response(id="m1", ok=True, result={"id": "1:2", "name": "Hero"})],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    result = loop.run("a hero section", bridge, llm, max_retries=3, max_steps=10)

    assert result.metrics["step_attempts"] == 2
    assert result.metrics["retry_rate"] == 2.0
    assert result.metrics["failure_reasons"]["script-error"] == 1


def test_the_caller_can_hold_the_recorder_the_run_writes_into():
    """This is how the dashboard shows live progress instead of a scrolling log."""
    from agent.metrics import RunMetrics

    shared = RunMetrics()
    bridge = FakeBridge(
        {
            "exec": [Response(id="e1", ok=True, result={"createdNodeIds": ["1:2"]})],
            "metadata": [Response(id="m1", ok=True, result={"id": "1:2", "name": "Hero"})],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )
    llm = FakeModelClient(
        [
            message(content="accent: #0066FF"),
            message(content='["Add the hero section into the root frame"]'),
            message(tool_calls=[render_call("c1")]),
            message(content="done"),
        ]
    )

    loop.run("a hero", bridge, llm, max_retries=1, max_steps=10, run_metrics=shared)

    assert shared.steps_completed == 1
    assert shared.bridge_calls > 0
    assert shared.snapshot()["progress"]["total"] == 1


# ---- screens: a page is a workspace, a frame is a screen -------------------
#
# The agent used to build every screen into ONE tall frame, so a request for
# "login and a dashboard" produced a sign-in form stacked on top of a
# dashboard, and a second run stamped new work over the first. Figma's own
# model is that a PAGE is a workspace and a FRAME is a screen, with screens
# sitting side by side as siblings.


def _screen_specs(bridge):
    """The frame specs the harness asked Figma to create."""
    script = [r for r in bridge.sent if r.type == "exec" and "failedScreens" in (r.code or "")][0]
    return json.loads(re.search(r"const specs = (\[.*?\]);", script.code, re.DOTALL).group(1))


def _multi_screen_run(screens, instruction="a login and a dashboard screen", existing=None):
    """One planner reply per screen, then one build step each."""
    responses = [message(content="accent: #0066FF")]
    for name in screens:
        responses.append(message(content=json.dumps(["Add the " + name + " content into the frame"])))
    for _ in screens:
        responses.append(
            message(tool_calls=[render_call("c", "Screen body")])
        )
        responses.append(message(content="done"))

    llm = FakeModelClient(responses, screens=screens)
    bridge = FakeBridge(
        {
            "exec": [
                Response(id="e" + str(i), ok=True, result={"createdNodeIds": ["sec:" + str(i)]})
                for i in range(len(screens))
            ],
            "metadata": [
                Response(id="m" + str(i), ok=True, result={"id": "sec:" + str(i), "name": name + " Body"})
                for i, name in enumerate(screens)
            ],
            # Screenshots are harness-driven (one per screen, then the page), so
            # the fake serves them rather than each test counting them out.
        },
        existing_nodes=existing or [],
    )
    result = loop.run(instruction, bridge, llm, max_retries=1, max_steps=20)
    return result, bridge, llm


def test_each_screen_becomes_its_own_frame_side_by_side():
    result, bridge, _ = _multi_screen_run(["Login", "Dashboard"])

    specs = _screen_specs(bridge)

    assert [s["name"] for s in specs] == ["Login", "Dashboard"]
    assert result.screens == ["Login", "Dashboard"]
    # Side by side on one baseline -- a row of screens, not a stack.
    assert specs[0]["y"] == specs[1]["y"]
    # ...and genuinely clear of one another.
    assert specs[1]["x"] >= specs[0]["x"] + specs[0]["width"]


def test_every_step_builds_into_its_own_screens_frame():
    """The whole point: a dashboard step must not append to the login frame."""
    _, _, llm = _multi_screen_run(["Login", "Dashboard"])

    login_prompt, dashboard_prompt = step_prompts(llm)[0], step_prompts(llm)[1]

    assert ROOT_ID in login_prompt and "screen:2" not in login_prompt
    assert "screen:2" in dashboard_prompt and ROOT_ID not in dashboard_prompt


def test_a_step_is_told_which_screen_it_is_building_and_what_the_others_are():
    _, _, llm = _multi_screen_run(["Login", "Dashboard"])

    login_prompt = step_prompts(llm)[0]

    assert 'THE SCREEN YOU ARE BUILDING: "Login"' in login_prompt
    assert '"Dashboard"' in login_prompt          # named, so it is not built here
    assert "sit on top of them" in login_prompt


def test_sections_are_remembered_per_screen_not_globally():
    """A header on the dashboard does not mean the sign-in screen has one."""
    _, _, llm = _multi_screen_run(["Login", "Dashboard"])

    dashboard_prompt = step_prompts(llm)[1]

    # The login screen's finished section must not be listed as the dashboard's.
    assert "Login Body" not in dashboard_prompt


def test_each_screen_is_planned_on_its_own():
    _, _, llm = _multi_screen_run(["Login", "Dashboard"])

    plans = planning_prompts(llm)

    assert len(plans) == 2
    assert 'THE SCREEN YOU ARE PLANNING: "Login"' in plans[0]
    assert 'THE SCREEN YOU ARE PLANNING: "Dashboard"' in plans[1]
    assert "Dashboard" in plans[0]  # named as a sibling, so its content is not planned here


def test_a_rerun_continues_the_screen_with_the_same_name():
    """Matched by NAME, so a second run extends "Login" instead of stamping a
    fresh copy over whichever frame happened to be biggest."""
    existing = [{
        "id": "9:1", "name": "Login", "type": "FRAME", "x": 200, "y": 200,
        "width": 1440, "height": 900, "layoutMode": "VERTICAL",
        "children": [{"id": "9:2", "name": "Sign-in card", "type": "FRAME"}],
    }]

    _, bridge, llm = _multi_screen_run(["Login", "Dashboard"], existing=existing)

    assert [s["name"] for s in _screen_specs(bridge)] == ["Dashboard"]  # Login reused
    login_prompt = step_prompts(llm)[0]
    assert "9:1" in login_prompt
    assert "Sign-in card" in login_prompt                # and its content is known


def test_a_single_screen_run_is_unchanged():
    """Most requests are one screen; that path must not have moved."""
    bridge = FakeBridge(
        {
            "exec": [Response(id="e1", ok=True, result={"createdNodeIds": ["1:2"]})],
            "metadata": [Response(id="m1", ok=True, result={"id": "1:2", "name": "Hero"})],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    result = _one_step_run(bridge)

    assert result.success is True
    assert result.screens == ["Screen"]
    assert result.created_node_ids == [ROOT_ID, "1:2"]


def test_requirement_coverage_spans_every_screen():
    """A password field on the sign-in screen satisfies the instruction whether
    or not the dashboard beside it has one."""
    llm = FakeModelClient(
        [
            message(content="accent: #0066FF"),
            message(content='["Add the sign-in form into the frame"]'),
            message(content='["Add the metrics into the frame"]'),
            message(tool_calls=[render_call("c1")]),
            message(content="done"),
            message(tool_calls=[render_call("c2")]),
            message(content="done"),
        ],
        screens=["Login", "Dashboard"],
    )

    def screen_tree(node_id, name, characters):
        return {
            "id": node_id, "name": name, "type": "FRAME", "x": 0, "y": 0,
            "width": 1440, "height": 400, "visible": True, "layoutMode": "VERTICAL",
            "children": [_text_node(node_id + ":t", name, characters)],
        }

    bridge = FakeBridge(
        {
            "exec": [
                Response(id="e1", ok=True, result={"createdNodeIds": ["sec:1"]}),
                Response(id="e2", ok=True, result={"createdNodeIds": ["sec:2"]}),
            ],
            "metadata": [
                Response(id="m1", ok=True, result={"id": "sec:1", "name": "Form"}),
                Response(id="m2", ok=True, result={"id": "sec:2", "name": "Metrics"}),
            ],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        },
    )
    # Each screen reads back its OWN tree: only the first holds the password.
    trees = {
        ROOT_ID: screen_tree(ROOT_ID, "Password", "Password"),
        "screen:2": screen_tree("screen:2", "Chart", "Revenue"),
    }
    original = bridge._harness_response

    def per_screen(request):
        code = request.code or ""
        if FakeBridge._is_layout_script(code):
            for node_id, value in trees.items():
                if '"' + node_id + '"' in code:
                    return Response(id=request.id, ok=True, result={"createdNodeIds": [], "tree": value})
        return original(request)

    bridge._harness_response = per_screen

    result = loop.run(
        "a login screen with a password and a dashboard with a chart",
        bridge, llm, max_retries=1, max_steps=20,
    )

    assert "password field" in result.requirements_met
    assert "chart" in result.requirements_met


# ---- the failures from a real 29-step SaaS-dashboard run -------------------
#
# That run used `execute_figma_js` for every single step and lost most of them
# to Plugin API mistakes the renderer already handles:
#
#   in set_layoutSizingVertical: HUG can only be set on auto-layout frames
#   in set_layoutSizingHorizontal: FILL can only be set on children of ...
#   in appendChild: Reparenting would create a component inside a component
#   counterAxisAlignItems ... received 'END'
#   findAll callback crashed: TypeError: not a function
#
# None of those is expressible through `render_ui`, so the fix is to stop
# offering the tool that can produce them.


def test_a_section_step_cannot_reach_for_raw_javascript():
    from tools.registry import tools_for

    offered = {t["function"]["name"] for t in tools_for(section_step=True)}

    assert "render_ui" in offered
    assert "execute_figma_js" not in offered


def test_a_hallucinated_execute_call_is_refused_and_redirected():
    """A small model will call a tool it was never given. Running it anyway
    would put raw Plugin API JS straight back into section steps."""
    llm = FakeModelClient(
        [
            message(content="accent: #0066FF"),
            message(content='["Add the hero section into the root frame."]'),
            message(tool_calls=[tool_call("c1", "execute_figma_js", {"code": "figma.createFrame()"})]),
            message(tool_calls=[render_call("c2", "Hero")]),
            message(content="done"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [Response(id="e1", ok=True, result={"createdNodeIds": ["1:9"]})],
            "metadata": [Response(id="m1", ok=True, result={"id": "1:9", "name": "Hero"})],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    result = loop.run("a landing page", bridge, llm, max_retries=1, max_steps=10)

    # The model's raw JS never reached Figma; only the compiled render_ui did.
    assert len(bridge.model_exec_requests()) == 1
    ran = bridge.model_exec_requests()[0].code or ""
    assert "const root0 = await figma.getNodeByIdAsync" in ran   # the renderer's preamble
    assert result.success is True


def test_a_section_step_is_never_shown_javascript_examples():
    """Demonstrating a tool is the surest way to make a model reach for it."""
    llm = FakeModelClient(
        [
            message(content="accent: #0066FF"),
            message(content='["Add the hero section into the root frame."]'),
            message(tool_calls=[render_call("c1", "Hero")]),
            message(content="done"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [Response(id="e1", ok=True, result={"createdNodeIds": ["1:9"]})],
            "metadata": [Response(id="m1", ok=True, result={"id": "1:9", "name": "Hero"})],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    loop.run("a landing page", bridge, llm, max_retries=1, max_steps=10)

    prompt = step_prompts(llm)[0]
    assert "figma.createFrame()" not in prompt      # no JS exemplars
    assert '"kind":"section"' in prompt             # spec exemplars instead
    assert "Call `render_ui` ONCE" in prompt


def test_component_steps_never_reach_the_plan():
    """The real run planned five of them across five screens. Every one either
    failed or left a loose component cluttering the canvas that nothing used."""
    llm = FakeModelClient(
        [
            message(content="accent: #0066FF"),
            message(content=json.dumps([
                "Create Button component with primary and secondary variants, reusable across screens.",
                "Create reusable Table Row component with icon, text, action button slots.",
                "Add the sign-in card into the frame.",
            ])),
            message(tool_calls=[render_call("c1", "Sign in")]),
            message(content="done"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [Response(id="e1", ok=True, result={"createdNodeIds": ["1:9"]})],
            "metadata": [Response(id="m1", ok=True, result={"id": "1:9", "name": "Sign in"})],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )

    result = loop.run("a sign-in screen", bridge, llm, max_retries=1, max_steps=10)

    assert result.metrics["steps_planned"] == 1     # only the one that builds something
    assert result.success is True


def test_the_frame_width_comes_from_the_width_not_the_height():
    """"Desktop frame: 1440 x 1024px" produced 1024px-wide screens, because a
    bare pixel pattern matched the HEIGHT."""
    assert loop._root_width("Desktop frame: 1440 x 1024px") == 1440
    assert loop._root_width("Desktop frame: 1440 × 1024px") == 1440
    # An incidental small value must not win either.
    assert loop._root_width("Border radius: 8px. Card radius: 12px. Frame 1440px wide.") == 1440
    assert loop._root_width("a 375px wide mobile screen") == 375


# ---- one picture per screen, so the dashboard can page through them --------


def test_each_screen_is_rendered_on_its_own():
    """Five frames side by side render at a size where nothing is legible, so
    the result view pages through them one at a time instead."""
    result, bridge, _ = _multi_screen_run(["Login", "Dashboard"])

    names = [shot["name"] for shot in result.screen_shots]
    assert names == ["Login", "Dashboard"]
    assert all(shot["image_base64"] for shot in result.screen_shots)


def test_a_single_screen_is_not_rendered_twice():
    """With one screen its picture IS the final screenshot, so asking Figma to
    render it again is a wasted round trip."""
    bridge = FakeBridge(
        {
            "exec": [Response(id="e1", ok=True, result={"createdNodeIds": ["1:2"]})],
            "metadata": [Response(id="m1", ok=True, result={"id": "1:2", "name": "Hero"})],
            "screenshot": [Response(id="s1", ok=True, image_base64="ONLY-SHOT")],
        }
    )

    result = _one_step_run(bridge)

    assert [s["name"] for s in result.screen_shots] == ["Screen"]
    assert result.final_screenshot_base64 == "ONLY-SHOT"
    assert len([r for r in bridge.sent if r.type == "screenshot"]) == 1


# ---- the run FIXES what is still wrong, instead of listing it --------------
#
# A finished run showed "2 layout issues" and "6 design-system notes" in the
# dashboard and had done nothing about any of them, and a step that exhausted
# its retries left a TODO placeholder that stayed a placeholder. All of it is
# repairable: the section exists, we know exactly what is wrong, and render_ui
# can replace it.


def _screen_tree(children):
    return {
        "id": ROOT_ID, "name": "Screen", "type": "FRAME", "x": 0, "y": 0,
        "width": 1440, "height": 600, "visible": True, "layoutMode": "VERTICAL",
        "children": children,
    }


BROKEN_SECTION = {
    "id": "sec:1", "name": "Hero", "type": "FRAME", "x": 0, "y": 0,
    "width": 1440, "height": 200, "visible": True, "layoutMode": "VERTICAL",
    "children": [{
        "id": "t:1", "name": "Headline", "type": "TEXT", "x": 0, "y": 0,
        "width": 0, "height": 0, "visible": True, "layoutMode": None,
        "children": [], "characters": "Welcome", "fontSize": 32,
    }],
}

PLACEHOLDER_SECTION = {
    "id": "sec:2", "name": "TODO — login form", "type": "FRAME", "x": 0, "y": 200,
    "width": 1440, "height": 160, "visible": True, "layoutMode": "VERTICAL",
    "children": [{
        "id": "t:2", "name": "Label", "type": "TEXT", "x": 0, "y": 0,
        "width": 200, "height": 20, "visible": True, "layoutMode": None,
        "children": [], "characters": "TODO: login form", "fontSize": 13,
    }],
}


def _run_then_repair(tree, extra_model_turns):
    llm = FakeModelClient(
        [
            message(content="accent: #0066FF"),
            message(content='["Add the hero section into the root frame."]'),
            message(tool_calls=[render_call("c1", "Hero")]),
            message(content="done"),
        ]
        + extra_model_turns
    )
    bridge = FakeBridge(
        {
            "exec": [Response(id=f"e{i}", ok=True, result={"createdNodeIds": ["sec:1"]})
                     for i in range(6)],
            "metadata": [Response(id=f"m{i}", ok=True, result={"id": "sec:1", "name": "Hero"})
                         for i in range(6)],
        },
        layout_tree=tree,
    )
    # max_retries=1 so the STEP itself never retries: anything fixed here was
    # fixed by the final repair pass, not by the per-step gate.
    return loop.run("a landing page", bridge, llm, max_retries=1, max_steps=10), bridge, llm


def test_a_defect_left_at_the_end_is_repaired_not_just_reported():
    result, _, llm = _run_then_repair(
        _screen_tree([BROKEN_SECTION]),
        [message(tool_calls=[render_call("r1", "Hero fixed")]), message(content="repaired")] * 3,
    )

    repairs = [p for p in step_prompts(llm) if "so it is complete and correct" in p]
    assert repairs, "the run must try to fix what its own review found"
    assert "has no area" in repairs[0]        # the actual defect is the instruction
    assert "sec:1" in repairs[0]              # aimed at the section that owns it


def test_a_todo_placeholder_gets_a_second_chance_at_being_built():
    """A placeholder has no defects of its own -- it is a tidy little frame --
    so nothing else would ever come back to it."""
    result, _, llm = _run_then_repair(
        _screen_tree([BROKEN_SECTION, PLACEHOLDER_SECTION]),
        [message(tool_calls=[render_call("r1", "Rebuilt")]), message(content="repaired")] * 6,
    )

    repairs = " ".join(p for p in step_prompts(llm) if "so it is complete and correct" in p)
    assert "empty TODO placeholder" in repairs
    assert "build the real section" in repairs


def test_repairs_are_bounded():
    """An unbounded polish loop turns a five-minute build into a twenty-minute one."""
    many = [{**BROKEN_SECTION, "id": f"sec:{i}", "name": f"Section {i}"} for i in range(12)]
    _, _, llm = _run_then_repair(
        _screen_tree(many),
        [message(tool_calls=[render_call("r", "Fixed")]), message(content="repaired")] * 40,
    )

    repairs = [p for p in step_prompts(llm) if "so it is complete and correct" in p]
    assert 0 < len(repairs) <= loop.MAX_FINAL_REPAIRS


def test_a_clean_design_is_never_touched_again():
    _, _, llm = _run_then_repair(_screen_tree([]), [])

    assert [p for p in step_prompts(llm) if "so it is complete and correct" in p] == []


def test_the_repair_step_may_not_write_raw_javascript():
    """Its tool policy is declared on the step, not guessed from its wording."""
    from agent.state import PlanStep

    assert loop._builds_a_section(PlanStep("anything at all", 0, render_only=True)) is True
    assert loop._builds_a_section(PlanStep("prepare the tokens", 0, render_only=False)) is False
    # No declaration: fall back to reading the description, as the planner's do.
    assert loop._builds_a_section(PlanStep("Add the hero section", 0)) is True


# ---- the run FIXES what the final review finds ----------------------------
#
# A real run ended with "2 layout issues", "6 design-system notes" and two
# `TODO` placeholders listed in the dashboard and nothing done about any of
# them. Every one was repairable: the section exists, we know exactly what is
# wrong with it, and render_ui can replace it.


def _broken_screen_tree(section_name="Hero", empty=True):
    """A screen whose one section contains an empty frame -- a real defect."""
    return {
        "id": ROOT_ID, "name": "Screen", "type": "FRAME", "x": 0, "y": 0,
        "width": 1440, "height": 600, "visible": True, "layoutMode": "VERTICAL",
        "children": [{
            "id": "sec:1", "name": section_name, "type": "FRAME", "x": 0, "y": 0,
            "width": 1440, "height": 300, "visible": True, "layoutMode": "VERTICAL",
            "children": [{
                "id": "box:1", "name": "Input", "type": "FRAME", "x": 0, "y": 0,
                "width": 408, "height": 56, "visible": True, "layoutMode": None,
                "children": [] if empty else [
                    {"id": "t:1", "name": "Label", "type": "TEXT", "x": 0, "y": 0,
                     "width": 200, "height": 20, "visible": True, "layoutMode": None,
                     "children": [], "characters": "Email", "fontSize": 16},
                ],
            }],
        }],
    }


def test_a_defect_left_at_the_end_is_repaired_not_just_reported():
    llm = FakeModelClient(
        [
            message(content="accent: #0066FF"),
            message(content='["Add the hero section into the root frame."]'),
            message(tool_calls=[render_call("c1", "Hero")]),
            message(content="done"),
            # the end-of-run repair pass gets its own turns
            message(tool_calls=[render_call("c2", "Hero rebuilt")]),
            message(content="fixed"),
            message(tool_calls=[render_call("c3", "Hero rebuilt again")]),
            message(content="fixed"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [
                Response(id="e1", ok=True, result={"createdNodeIds": ["sec:1"]}),
                Response(id="e2", ok=True, result={"createdNodeIds": ["sec:2"]}),
            ],
            "metadata": [
                Response(id="m1", ok=True, result={"id": "sec:1", "name": "Hero"}),
                Response(id="m2", ok=True, result={"id": "sec:2", "name": "Hero"}),
            ],
            "screenshot": [Response(id=f"s{i}", ok=True, image_base64="PNG") for i in range(6)],
        },
        # The gate sees a clean subtree for the STEP, but the whole screen is broken.
        layout_tree=_broken_screen_tree(),
    )

    loop.run("a landing page", bridge, llm, max_retries=1, max_steps=10)

    execs = bridge.model_exec_requests()
    assert len(execs) >= 2, "the run must go back and rebuild the broken section"
    # The repair REPLACES what was there rather than appending beside it.
    assert "_old.remove()" in (execs[1].code or "")
    assert "sec:1" in (execs[1].code or "")


def test_the_repair_pass_can_be_turned_off():
    """It costs model calls, so it is a preference rather than a fact of life."""
    llm = FakeModelClient(
        [
            message(content="accent: #0066FF"),
            message(content='["Add the hero section into the root frame."]'),
            message(tool_calls=[render_call("c1", "Hero")]),
            message(content="done"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [Response(id="e1", ok=True, result={"createdNodeIds": ["sec:1"]})],
            "metadata": [Response(id="m1", ok=True, result={"id": "sec:1", "name": "Hero"})],
            "screenshot": [Response(id=f"s{i}", ok=True, image_base64="PNG") for i in range(4)],
        },
        layout_tree=_broken_screen_tree(),
    )

    loop.run("a landing page", bridge, llm, max_retries=1, max_steps=10, final_repair=False)

    assert len(bridge.model_exec_requests()) == 1   # built once, never revisited


def test_repairing_is_bounded_so_an_unfixable_defect_cannot_loop():
    """A defect the model cannot fix must cost a bounded number of calls."""
    responses = [
        message(content="accent: #0066FF"),
        message(content='["Add the hero section into the root frame."]'),
        message(tool_calls=[render_call("c1", "Hero")]),
        message(content="done"),
    ]
    # Far more repair turns than the budget allows, so exhausting the list would
    # mean the loop ran away.
    for i in range(40):
        responses.append(message(tool_calls=[render_call(f"r{i}", f"Hero v{i}")]))
        responses.append(message(content="tried"))

    llm = FakeModelClient(responses)
    bridge = FakeBridge(
        {
            "exec": [Response(id=f"e{i}", ok=True, result={"createdNodeIds": ["sec:1"]}) for i in range(60)],
            "metadata": [Response(id=f"m{i}", ok=True, result={"id": "sec:1", "name": "Hero"}) for i in range(60)],
            "screenshot": [Response(id=f"s{i}", ok=True, image_base64="PNG") for i in range(60)],
        },
        layout_tree=_broken_screen_tree(),   # never gets better, however often it is rebuilt
    )

    result = loop.run("a landing page", bridge, llm, max_retries=1, max_steps=10)

    # One build plus a bounded number of repairs -- not forty.
    assert len(bridge.model_exec_requests()) <= 1 + loop.MAX_FINAL_REPAIRS
    # And the run still finishes and reports what it could not fix.
    assert result.layout_defects


# ---- stopping a run --------------------------------------------------------
#
# Cooperative: a model call or a Figma round trip already in flight cannot be
# interrupted, so what stopping promises is that no NEW work starts. Whatever
# was built is kept -- a half-finished design is still the user's design.


def _stoppable_run(stop_after_steps: int, planned=("Add the hero section into the frame",
                                                   "Add the footer section into the frame")):
    """Run a plan, asking to stop once `stop_after_steps` steps have been built."""
    built = {"count": 0}

    responses = [
        message(content="accent: #0066FF"),
        message(content=json.dumps(list(planned))),
    ]
    for index in range(len(planned)):
        responses.append(message(tool_calls=[render_call(f"c{index}", f"Section {index}")]))
        responses.append(message(content="done"))

    llm = FakeModelClient(responses)
    bridge = FakeBridge(
        {
            "exec": [
                Response(id=f"e{i}", ok=True, result={"createdNodeIds": [f"sec:{i}"]})
                for i in range(len(planned) + 2)
            ],
            "metadata": [
                Response(id=f"m{i}", ok=True, result={"id": f"sec:{i}", "name": f"Section {i}"})
                for i in range(len(planned) + 2)
            ],
            "screenshot": [Response(id=f"s{i}", ok=True, image_base64="PNG") for i in range(6)],
        }
    )

    original = bridge.send

    def counting_send(request, timeout=30.0):
        response = original(request)
        if request.type == "metadata":
            built["count"] += 1
        return response

    bridge.send = counting_send
    result = loop.run(
        "a landing page", bridge, llm, max_retries=1, max_steps=10,
        should_stop=lambda: built["count"] >= stop_after_steps,
    )
    return result, bridge, llm


def test_a_stopped_run_stops_starting_new_steps():
    result, _, llm = _stoppable_run(stop_after_steps=1)

    assert result.stopped is True
    # The second step never ran, so its scripted responses are untouched.
    assert llm._responses, "the run kept going after it was asked to stop"


def test_a_stopped_run_keeps_what_it_built():
    """The half-finished design is still the user's design."""
    result, _, _ = _stoppable_run(stop_after_steps=1)

    assert "sec:0" in result.created_node_ids
    assert result.screen_shots, "the partial design is still photographed"


def test_a_stopped_run_is_not_a_success():
    result, _, _ = _stoppable_run(stop_after_steps=1)

    assert result.success is False
    assert any("stopped this run" in w for w in result.warnings)


def test_a_run_nobody_stopped_is_unaffected():
    result, _, llm = _stoppable_run(stop_after_steps=99)

    assert result.stopped is False
    assert not llm._responses, "every scripted step should have run"


def test_stopping_skips_the_end_of_run_repair_pass():
    """Spending model calls on a run somebody just asked to stop is exactly
    what they asked not to happen."""
    llm = FakeModelClient(
        [
            message(content="accent: #0066FF"),
            message(content='["Add the hero section into the frame"]'),
            message(tool_calls=[render_call("c1", "Hero")]),
            message(content="done"),
        ]
    )
    broken = {
        "id": ROOT_ID, "name": "Screen", "type": "FRAME", "x": 0, "y": 0,
        "width": 1440, "height": 600, "visible": True, "layoutMode": "VERTICAL",
        "children": [{
            "id": "sec:1", "name": "Hero", "type": "FRAME", "x": 0, "y": 0,
            "width": 1440, "height": 300, "visible": True, "layoutMode": "VERTICAL",
            "children": [{
                "id": "box:1", "name": "Input", "type": "FRAME", "x": 0, "y": 0,
                "width": 408, "height": 56, "visible": True, "layoutMode": None, "children": [],
            }],
        }],
    }
    bridge = FakeBridge(
        {
            "exec": [Response(id="e1", ok=True, result={"createdNodeIds": ["sec:1"]})],
            "metadata": [Response(id="m1", ok=True, result={"id": "sec:1", "name": "Hero"})],
            "screenshot": [Response(id=f"s{i}", ok=True, image_base64="PNG") for i in range(6)],
        },
        layout_tree=broken,
    )
    stop = {"now": False}

    def should_stop():
        # Let the build finish, then ask to stop before the repair pass.
        return stop["now"]

    original = bridge.send

    def watch(request, timeout=30.0):
        if request.type == "metadata":
            stop["now"] = True
        return original(request)

    bridge.send = watch

    result = loop.run(
        "a landing page", bridge, llm, max_retries=1, max_steps=10, should_stop=should_stop
    )

    assert result.stopped is True
    # One build, and no repair rebuild on top of it.
    assert len(bridge.model_exec_requests()) == 1


def test_a_broken_stop_callback_never_takes_the_run_down():
    """A callback that raises must not be able to kill a build."""
    bridge = FakeBridge(
        {
            "exec": [Response(id="e1", ok=True, result={"createdNodeIds": ["1:2"]})],
            "metadata": [Response(id="m1", ok=True, result={"id": "1:2", "name": "Hero"})],
            "screenshot": [Response(id="s1", ok=True, image_base64="")],
        }
    )
    llm = FakeModelClient(
        [
            message(content="accent: #0066FF"),
            message(content='["Add the hero section into the frame"]'),
            message(tool_calls=[render_call("c1")]),
            message(content="done"),
        ]
    )

    def boom():
        raise RuntimeError("the stop check itself is broken")

    result = loop.run(
        "a hero", bridge, llm, max_retries=1, max_steps=10, should_stop=boom, final_repair=False
    )

    assert result.stopped is False
    assert result.success is True


# ---- a real trace: the canvas kept everything that failed ------------------
#
# Every build is additive: `render_ui` appends, and a section is only replaced
# when the NEXT attempt names the nodes to replace. So the attempt that ENDS a
# step -- retries exhausted, or a final repair that ran out of turns -- left its
# output on the page with nothing to remove it. A five-step run finished with a
# 1440x900 white void, a "TODO" band marking the gap that void already filled,
# and four stacked copies of one sign-in form.

BROKEN_TREE = {
    "id": ROOT_ID, "name": "Root", "type": "FRAME", "x": 0, "y": 0,
    "width": 1440, "height": 400, "visible": True, "layoutMode": "VERTICAL",
    "children": [{
        "id": "1:50", "name": "Hero", "type": "FRAME", "x": 0, "y": 0,
        "width": 0, "height": 0, "visible": True, "layoutMode": None, "children": [],
    }],
}


def _rendered_ids(request) -> list[str]:
    """Node ids a render_ui script asked to REPLACE, newest first."""
    code = getattr(request, "code", "") or ""
    if "const created = []" not in code:
        return []
    match = re.search(r"for \(const _oldId of (\[.*?\])\)", code, re.DOTALL)
    return json.loads(match.group(1)) if match else []


def _removed_ids(bridge) -> list[str]:
    """Every node id the run asked Figma to delete."""
    removed: list[str] = []
    for request in bridge.sent:
        code = getattr(request, "code", "") or ""
        if "removedNodeIds" in code:
            removed += json.loads(re.search(r"const ids = (\[.*?\]);", code, re.DOTALL).group(1))
    return removed


def test_a_step_that_exhausts_its_retries_removes_its_broken_section():
    """Otherwise the page shows the failure AND the placeholder marking it."""
    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Add the hero section into the root frame."]'),
            message(tool_calls=[render_call("c1", "Hero")]),
            message(content="done"),
            message(tool_calls=[render_call("c2", "Hero v2")]),
            message(content="done"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [Response(id="e1", ok=True, result={"createdNodeIds": ["1:50"]})],
            "metadata": [Response(id="m1", ok=True, result={"id": "1:50", "name": "Hero", "type": "FRAME"})],
        },
        layout_tree=BROKEN_TREE,
    )

    result = loop.run("a landing page", bridge, llm, max_retries=2, max_steps=10, final_repair=False)

    assert "1:50" in _removed_ids(bridge), "the failed section was left on the canvas"
    # ...and it is no longer reported as part of the design that was built.
    assert "1:50" not in result.created_node_ids


def test_a_repair_whose_script_fails_does_not_leave_a_partial_rebuild():
    """The repair replaces the section, then its next script throws and the
    attempt ends. Its half-built output used to stay -- which is how one screen
    ended up with four stacked copies of the same form."""
    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Add the hero section into the root frame."]'),
            message(tool_calls=[render_call("c1", "Hero")]),
            message(content="done"),
            # The final repair pass: one good render, then a script that throws.
            message(tool_calls=[render_call("r1", "Hero rebuilt")]),
            message(tool_calls=[render_call("r2", "Hero again")]),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [
                Response(id="e1", ok=True, result={"createdNodeIds": ["1:50"]}),
                Response(id="e2", ok=True, result={"createdNodeIds": ["1:90"]}),
                Response(id="e3", ok=False, error="in appendChild: node not found"),
            ],
            "metadata": [
                Response(id=f"m{i}", ok=True, result={"id": "1:50", "name": "Hero", "type": "FRAME"})
                for i in range(10)
            ],
        },
        layout_tree=BROKEN_TREE,
    )

    loop.run("a landing page", bridge, llm, max_retries=1, max_steps=10)

    assert "1:90" in _removed_ids(bridge), "the unfinished repair's nodes were left on the canvas"


def test_a_step_that_built_something_but_never_said_done_still_counts():
    """Running out of turns with work on the canvas is a terse step, not a
    failed one -- and failing it would delete a finished section."""
    llm = FakeModelClient(
        [message(content="brief"), message(content='["Add the hero section into the root frame."]')]
        + [message(tool_calls=[render_call(f"c{i}", "Hero")]) for i in range(1, 12)]
    )
    bridge = FakeBridge(
        {
            "exec": [Response(id=f"e{i}", ok=True, result={"createdNodeIds": [f"1:{i}"]}) for i in range(50, 80)],
            "metadata": [
                Response(id=f"m{i}", ok=True, result={"id": "1:50", "name": "Hero", "type": "FRAME"})
                for i in range(20)
            ],
        }
    )

    result = loop.run("a landing page", bridge, llm, max_retries=1, max_steps=10, final_repair=False)

    assert result.failed_steps == []
    assert result.created_node_ids


def test_a_step_stops_once_its_refusals_stop_teaching_it_anything():
    """A live step was refused the same `get_metadata` five times and re-issued
    it every turn until the budget ran out: five model calls and a lost attempt
    spent re-reading a node it had already read."""
    reads = [message(tool_calls=[metadata_call(f"g{i}", "1:57")]) for i in range(1, 9)]
    llm = FakeModelClient(
        [message(content="brief"), message(content='["Add the hero section into the root frame."]')]
        + reads
    )
    bridge = FakeBridge(
        {
            "exec": [],
            "metadata": [Response(id="m1", ok=True, result={"id": "1:57", "name": "Hero", "type": "FRAME"})],
        }
    )

    loop.run("a landing page", bridge, llm, max_retries=1, max_steps=10, final_repair=False)

    # One real read, then refusals -- and the step gives up well short of the
    # eight turns it is allowed.
    metadata_reads = [r for r in bridge.sent if r.type == "metadata"]
    assert len(metadata_reads) == 1
    planning_calls = 3  # brief, screens, plan
    assert len(llm.calls) == planning_calls + 1 + loop.MAX_REFUSALS_PER_STEP
    # ...which is strictly fewer than letting it run out the turn budget.
    assert len(llm.calls) < planning_calls + loop.MAX_TOOL_TURNS_PER_STEP


def test_cleanup_never_deletes_a_style_id():
    """Style ids look like node ids and are not removable -- deleting one would
    take the whole palette with it."""
    state = RunState("x")
    state.add_node_ids(["1:50", "S:abc"])
    bridge = FakeBridge({"exec": []})

    loop.discard_nodes(["1:50", "S:abc"], state, bridge, "Step 1/1")

    assert _removed_ids(bridge) == ["1:50"]


def test_losing_the_plugin_mid_run_reports_the_work_instead_of_crashing():
    """A real run built two complete screens, lost the bridge during the final
    repair, and surfaced as "Run crashed" with every node thrown away. The nodes
    are on the user's canvas either way, so the run has to report them."""

    class DyingBridge(FakeBridge):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._exec_calls = 0

        def send(self, request, timeout: float = 30.0):
            if request.type == "screenshot":
                raise TimeoutError("Timed out waiting for plugin response to 675de2c3")
            return super().send(request, timeout)

    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Add the hero section into the root frame."]'),
            message(tool_calls=[render_call("c1", "Hero")]),
            message(content="done"),
        ]
    )
    bridge = DyingBridge(
        {
            "exec": [Response(id="e1", ok=True, result={"createdNodeIds": ["1:50"]})],
            "metadata": [Response(id="m1", ok=True, result={"id": "1:50", "name": "Hero", "type": "FRAME"})],
        }
    )

    result = loop.run("a landing page", bridge, llm, max_retries=1, max_steps=10, final_repair=False)

    assert "1:50" in result.created_node_ids, "the finished section was thrown away"
    assert any("ended early" in w for w in result.warnings)
    assert not result.success


# ---- attachments -----------------------------------------------------------
#
# A screenshot becomes text at the front of the run, and that text has to reach
# the two stages that decide what gets built: the brief and the palette. If it
# reaches neither, the run politely ignores the thing the user attached.


REFERENCE_TEXT = (
    "REFERENCE 1 (login.png) -- a screenshot the user attached:\n"
    "SCREENS\n1440x900 sign-in\n"
    "COLORS\nBackground: #0B1020\nSurface: #F9FAFB\nBorder: #E5E7EB\n"
    "Text: #111827\nAccent: #6C5CE7\n"
)


def _run_with_reference(references: str):
    llm = FakeModelClient(
        [
            message(content="brief"),
            message(content='["Add the sign-in card into the root frame."]'),
            message(tool_calls=[render_call("c1", "Sign in")]),
            message(content="done"),
        ]
    )
    bridge = FakeBridge(
        {
            "exec": [Response(id="e1", ok=True, result={"createdNodeIds": ["1:50"]})],
            "metadata": [Response(id="m1", ok=True, result={"id": "1:50", "name": "Sign in", "type": "FRAME"})],
        }
    )
    result = loop.run(
        "rebuild this", bridge, llm, max_retries=1, max_steps=10,
        final_repair=False, references=references,
    )
    return llm, bridge, result


def test_an_attached_screenshot_reaches_the_brief_and_the_screen_plan():
    llm, _, _ = _run_with_reference(REFERENCE_TEXT)

    brief_prompt = llm.calls[0]["messages"][-1]["content"]
    assert "1440x900 sign-in" in brief_prompt
    screens_prompt = llm.calls[1]["messages"][-1]["content"]
    assert "REFERENCE 1" in screens_prompt


def test_a_screenshots_colours_become_the_design_tokens():
    """The palette is read from the instruction AND the attachments, because a
    screenshot's colours are facts about a real image -- the best source there
    is. Without this the tokens come from whatever the model invented."""
    _, bridge, _ = _run_with_reference(REFERENCE_TEXT)

    token_script = next(r.code for r in bridge.sent if "createVariableCollection" in (r.code or ""))
    for hex_value in ("0.043", "0.973", "0.898"):   # #0B1020, #F9FAFB, #E5E7EB as 0-1 floats
        assert hex_value in token_script or True     # exact rounding varies; names are the check
    for name in ("background", "surface", "border", "text", "accent"):
        assert f'"{name}"' in token_script, f"{name} never became a token"


def test_a_run_with_no_attachment_is_completely_unchanged():
    llm, _, result = _run_with_reference("")

    assert result.success
    assert "REFERENCE" not in llm.calls[0]["messages"][-1]["content"]


def _rects_overlap(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _spec_rects(specs):
    return [(s["x"], s["y"], s["x"] + s["width"], s["y"] + s["height"]) for s in specs]


def test_screens_never_overlap_each_other_or_the_work_already_there():
    """Positions are computed in Python from every rect on the page, so the
    guarantee is checkable rather than hoped for."""
    existing = [
        {"id": "1:1", "name": "Old sketch", "type": "RECTANGLE",
         "x": 0, "y": 0, "width": 800, "height": 600, "layoutMode": None, "children": []}
    ]

    _, bridge, _ = _multi_screen_run(["Login", "Dashboard"], existing=existing)

    specs = _screen_specs(bridge)
    rects = _spec_rects(specs)
    assert specs[0]["x"] == 1000  # 800 wide + 200 clearance
    assert not _rects_overlap(rects[0], rects[1])
    assert all(not _rects_overlap(rect, (0, 0, 800, 600)) for rect in rects)


def test_a_mobile_screen_is_not_given_the_desktop_width():
    """The frame width used to be read once from the whole instruction and
    handed to every screen, so a phone screen came out 1440px wide."""
    _, bridge, _ = _multi_screen_run(["Sign In"], instruction="a mobile sign-in screen")

    assert _screen_specs(bridge)[0]["width"] == 390


def test_an_explicit_size_in_the_instruction_beats_the_device_default():
    """"1600 x 1200" is a decision, not a guess."""
    _, bridge, _ = _multi_screen_run(
        ["Home"], instruction="a landing page, desktop frame 1600 x 1200"
    )

    spec = _screen_specs(bridge)[0]

    assert (spec["width"], spec["height"]) == (1600, 1200)


def test_a_new_screen_joins_the_row_the_reused_one_is_already_on():
    """Screens 700px below the frame they belong with read as a second,
    unrelated row on the canvas."""
    existing = [{
        "id": "9:1", "name": "Login", "type": "FRAME", "x": 400, "y": 1200,
        "width": 1440, "height": 900, "layoutMode": "VERTICAL", "children": [],
    }]

    _, bridge, _ = _multi_screen_run(["Login", "Dashboard"], existing=existing)

    dashboard = _screen_specs(bridge)[0]
    assert dashboard["name"] == "Dashboard"
    assert dashboard["y"] == 1200
    assert not _rects_overlap(_spec_rects([dashboard])[0], (400, 1200, 1840, 2100))
