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


# ---- screens come back described, not just named --------------------------
#
# A bare name told the next stage nothing: the plan for "Dashboard" was written
# from a brief about the whole design, and its frame width came from a single
# scan of the instruction that every screen shared.


def _screens(content, instruction="an app"):
    return planner.plan_screens(instruction, "brief", FakeLLM(content))


def test_a_screen_carries_its_purpose_and_device():
    screens = _screens(json.dumps([
        {"name": "Login", "purpose": "Sign in with email and password.", "device": "mobile"}
    ]))

    assert screens[0].purpose == "Sign in with email and password."
    assert screens[0].device == "mobile"
    assert screens[0].width == 390


def test_a_bare_list_of_names_is_still_accepted():
    """Small models answer the easy way. The one question every run depends on
    must not fail because the richer shape was insisted on."""
    screens = _screens('["Login", "Dashboard"]')

    assert [s.name for s in screens] == ["Login", "Dashboard"]
    assert all(s.device == "desktop" and s.width == 1440 for s in screens)


def test_a_section_is_dropped_however_it_is_spelled():
    """"Hero" was filtered and "Hero Section" was not, so the same mistake
    became a top-level frame whenever the model wrote the noun out."""
    kept = [s.name for s in _screens(
        '["Landing Page", "Hero Section", "Navigation Bar", "Footer Area", "Side Bar"]'
    )]

    assert kept == ["Landing Page"]


def test_a_page_is_not_mistaken_for_a_section():
    """"Settings Page" is a screen. Stripping the noun would leave "Settings",
    which is why "page" is not a section suffix."""
    kept = [s.name for s in _screens('["Settings Page", "Profile Page"]')]

    assert kept == ["Settings Page", "Profile Page"]


def test_the_device_comes_from_the_request_when_the_model_omits_it():
    screens = _screens('["Sign In"]', instruction="a mobile sign-in screen")

    assert screens[0].device == "mobile"
    assert screens[0].width == 390


def test_one_mobile_screen_does_not_make_every_screen_mobile():
    """The width used to be read once from the whole instruction, so a phone
    screen asked for beside a desktop one came out 1440px wide."""
    screens = _screens(json.dumps([
        {"name": "Marketing Site", "purpose": "Desktop landing page.", "device": "desktop"},
        {"name": "App Preview", "purpose": "The phone app.", "device": "mobile"},
    ]))

    assert [s.width for s in screens] == [1440, 390]


def test_a_fallback_screen_is_still_sized_from_the_request():
    screens = planner.plan_screens("a mobile checkout screen", "brief", ExplodingLLM())

    assert screens[0].width == 390


# ---- build order: the order of the steps IS the order of the screen --------
#
# Every section step appends to the bottom of a vertical auto-layout frame, so
# a plan that names the footer first builds the footer at the top -- and no
# gate can see it, because each band is individually well formed.


def test_a_footer_planned_first_is_built_last():
    plan = ["Add the footer into the frame.", "Add the hero into the frame."]

    assert planner.order_steps(plan) == [plan[1], plan[0]]


def test_a_nav_bar_planned_last_is_built_first():
    plan = ["Add the pricing table into the frame.", "Add the nav bar with logo and links."]

    assert planner.order_steps(plan) == [plan[1], plan[0]]


def test_steps_the_harness_cannot_place_keep_the_order_the_model_chose():
    """Reordering a plan we do not understand is a worse failure than the one
    being fixed, so the sort is stable and only clear cases move."""
    plan = [
        "Add the features grid into the frame.",
        "Add the testimonials into the frame.",
        "Add the pricing into the frame.",
    ]

    assert planner.order_steps(plan) == plan


def test_a_footer_mentioned_inside_a_step_does_not_move_it():
    """"...and a footer link" is not a footer section. Ranking on it would sort
    the sign-in card below the real footer."""
    plan = [
        "Add the sign-in card with email, password and a footer link, into the frame.",
        "Add the footer into the frame.",
    ]

    assert planner.order_steps(plan) == plan


# ---- the step budget merges the tail, it never drops it -------------------


def test_a_plan_within_its_budget_is_untouched():
    plan = ["Add the hero into the frame.", "Add the footer into the frame."]

    assert planner.fit_steps(plan, 3) == plan


def test_an_overlong_plan_keeps_every_region():
    """A five-band landing page capped at three used to ship without its
    testimonials or its footer -- and every step it did run passed, so nothing
    downstream could notice the bottom of the page was missing."""
    plan = [
        "Add the hero into the frame.",
        "Add the features into the frame.",
        "Add the pricing into the frame.",
        "Add the testimonials into the frame.",
        "Add the footer into the frame.",
    ]

    fitted = planner.fit_steps(plan, 3)

    assert len(fitted) == 3
    merged = " ".join(fitted)
    for region in ("hero", "features", "pricing", "testimonials", "footer"):
        assert region in merged, f"{region!r} was dropped, not merged"


def test_a_budget_of_one_still_carries_the_whole_screen():
    plan = ["Add the hero into the frame.", "Add the footer into the frame."]

    fitted = planner.fit_steps(plan, 1)

    assert len(fitted) == 1
    assert "hero" in fitted[0] and "footer" in fitted[0]


def test_a_screens_purpose_reaches_its_own_plan():
    """The brief describes the WHOLE design, so without this the plan for the
    dashboard was written from a document that mostly talks about signing in."""
    from agent.state import RunState

    captured = {}

    class RecordingLLM:
        def complete(self, messages, tools=None):
            captured["user"] = next(m["content"] for m in messages if m["role"] == "user")
            return SimpleNamespace(content='["Add the metrics row"]', tool_calls=None)

    planner.make_plan(
        "brief", RunState(instruction="x"), RecordingLLM(),
        screen="Dashboard", screen_purpose="Revenue metrics, a chart and recent orders.",
    )

    assert "Revenue metrics, a chart and recent orders." in captured["user"]


def test_a_device_word_meant_for_one_screen_does_not_size_the_others():
    """"a login, a phone dashboard and settings" made the SETTINGS screen
    390px wide, because "phone" was about the screen beside it. When other
    screens declared a device, the run's own dominant device is the better
    guess than a word from the request."""
    screens = _screens(
        json.dumps([
            {"name": "Login", "purpose": "Sign in.", "device": "desktop"},
            {"name": "Mobile Dashboard", "purpose": "Metrics on a phone.", "device": "mobile"},
            {"name": "Settings Page", "purpose": "Profile and billing.", "device": ""},
        ]),
        instruction="a login, a phone dashboard and settings",
    )

    assert [(s.name, s.width) for s in screens] == [
        ("Login", 1440), ("Mobile Dashboard", 390), ("Settings Page", 1440)
    ]


def test_a_uniformly_mobile_request_still_sizes_every_screen_as_a_phone():
    """The fallback only ignores the request when some screen declared a
    device. A phone app that declared nothing must still be a phone app."""
    screens = _screens(
        '["Sign In", "Feed", "Profile"]', instruction="a mobile app: sign in, feed and profile"
    )

    assert {s.width for s in screens} == {390}
