"""Screen decomposition: how many Figma FRAMES a request needs.

In Figma a page is a workspace and a frame is a screen, so "a login and a
dashboard" is two sibling frames side by side -- not two sections stacked into
one frame, and not two Figma pages.

The dangerous failure here is INVENTING screens: every extra screen is a whole
frame of empty canvas and its own planning call, so most of these tests are
about what must NOT become a screen.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from agent import planner


class FakeLLM:
    def __init__(self, content):
        self._content = content

    def complete(self, messages, tools=None):
        return SimpleNamespace(content=self._content, tool_calls=None)


class ExplodingLLM:
    def complete(self, messages, tools=None):
        raise RuntimeError("endpoint down")


def names(content):
    return [s.name for s in planner.plan_screens("an app", "brief", FakeLLM(content))]


# ---- what becomes a screen ------------------------------------------------


def test_several_screens_become_several_frames():
    assert names('["Login", "Sign Up", "Dashboard"]') == ["Login", "Sign Up", "Dashboard"]


def test_one_screen_is_the_normal_answer():
    assert names('["Landing Page"]') == ["Landing Page"]


def test_numbering_is_stripped():
    assert names('["1. Login", "2) Sign Up"]') == ["Login", "Sign Up"]


# ---- what must NOT become a screen ----------------------------------------


def test_sections_are_never_promoted_to_screens():
    """A hero is part of a screen. Building it as its own top-level frame
    scatters one screen's parts across the canvas."""
    assert names('["Landing Page", "Hero", "Footer", "Nav"]') == ["Landing Page"]


def test_duplicates_are_collapsed():
    assert names('["Login", "login", "LOGIN"]') == ["Login"]


def test_the_screen_count_is_capped():
    many = json.dumps([f"Screen {i}" for i in range(20)])

    assert len(names(many)) == planner.MAX_SCREENS


def test_absurdly_long_names_are_dropped():
    assert names(json.dumps(["Login", "x" * 200])) == ["Login"]


# ---- never take the run down ----------------------------------------------


def test_unparseable_output_falls_back_to_a_single_screen():
    assert len(names("I think you want a login screen and a dashboard.")) == 1


def test_an_empty_list_falls_back_to_a_single_screen():
    assert len(names("[]")) == 1


def test_a_failing_endpoint_falls_back_to_a_single_screen():
    """A screen list is a convenience. Losing the run over it is not a trade
    worth making."""
    screens = planner.plan_screens("a dashboard", "brief", ExplodingLLM())

    assert len(screens) == 1
    assert screens[0].frame_id is None


def test_a_quoted_product_name_names_the_single_screen():
    screens = planner.plan_screens('a landing page for "Northwind"', "brief", FakeLLM("[]"))

    assert screens[0].name == "Northwind"


# ---- per-screen planning ---------------------------------------------------


def test_a_plan_is_scoped_to_one_screen_and_told_about_its_siblings():
    from agent.state import RunState

    captured = {}

    class RecordingLLM:
        def complete(self, messages, tools=None):
            captured["user"] = next(m["content"] for m in messages if m["role"] == "user")
            return SimpleNamespace(content='["Add the sign-in card"]', tool_calls=None)

    state = RunState(instruction="x")
    planner.make_plan("brief", state, RecordingLLM(), screen="Login", other_screens=["Dashboard"])

    assert 'THE SCREEN YOU ARE PLANNING: "Login"' in captured["user"]
    assert "Dashboard" in captured["user"]
    assert "do not plan their content" in captured["user"]


def test_a_single_screen_plan_carries_no_screen_framing():
    """Most requests are one screen; that prompt must not grow noise."""
    from agent.state import RunState

    captured = {}

    class RecordingLLM:
        def complete(self, messages, tools=None):
            captured["user"] = next(m["content"] for m in messages if m["role"] == "user")
            return SimpleNamespace(content='["Add the hero"]', tool_calls=None)

    planner.make_plan("brief", RunState(instruction="x"), RecordingLLM())

    assert "THE SCREEN YOU ARE PLANNING" not in captured["user"]


# ---- a real trace: a split screen came out as stacked bands ----------------


def test_a_side_by_side_screen_is_collapsed_into_one_step():
    """A screen frame is a VERTICAL auto-layout, so every step appends a
    full-width band beneath the last. Planning "add the left panel" then "add
    the right panel" produced the form UNDERNEATH the artwork -- and no gate can
    see it, because both bands are individually well-formed."""
    plan = [
        "Add the left visual panel with gradient background and glowing shapes, into the frame.",
        "Add the right auth panel form with logo, heading, and inputs, into the frame.",
        "Add the right auth panel supporting text, divider and prompt, into the frame.",
    ]

    collapsed = planner._collapse_side_by_side(plan, "Login")

    assert len(collapsed) == 1
    for region in ("left visual panel", "right auth panel form", "supporting text"):
        assert region in collapsed[0], f"{region!r} was dropped, not merged"


def test_a_stacked_scrolling_page_is_left_alone():
    """Bands really are stacked on a landing page -- collapsing them would undo
    the one case where splitting a screen is correct."""
    plan = [
        "Add the hero section into the frame.",
        "Add the features section into the frame.",
        "Add the footer into the frame.",
    ]

    assert planner._collapse_side_by_side(plan, "Landing") == plan


def test_a_single_step_screen_is_never_rewritten():
    plan = ["Add the sign-in card with email and password, into the frame."]
    assert planner._collapse_side_by_side(plan, "Login") == plan
