"""Prototype wiring: what gets linked, and -- mostly -- what must not.

The dangerous failure here is not a missing link. It is a WRONG one: a heading
that becomes a navigation link, a label that swallows its own button's click,
or a batch that wires half the page together because one word matched. So most
of this file is about what `auto_link` refuses to do.
"""
from __future__ import annotations

import json

from agent import interactions
from agent.state import Screen


def screens(*names) -> list[Screen]:
    return [Screen(name=name, frame_id=f"f:{i}") for i, name in enumerate(names, start=1)]


def candidate(node_id, label, screen_id="f:1", name="Button / x", type_="FRAME", **kw):
    return interactions.Candidate(
        id=node_id, name=name, type=type_, label=label, screen_id=screen_id, **kw
    )


# -- what the harness wires by itself ---------------------------------------


def test_a_button_naming_a_screen_is_linked_to_it():
    design = screens("Login", "Dashboard")

    links = interactions.auto_link([candidate("1:9", "Dashboard")], design)

    assert len(links) == 1
    assert links[0].destination_id == "f:2"
    assert links[0].action == "navigate"


def test_a_label_that_means_a_screen_in_other_words_is_linked():
    """"Sign in" opens the dashboard. That is a convention, not a coincidence,
    and it is string matching rather than judgement -- so Python does it."""
    design = screens("Login", "Dashboard")

    links = interactions.auto_link([candidate("1:9", "Sign in")], design)

    assert [link.destination_name for link in links] == ["Dashboard"]


def test_a_hint_never_invents_a_screen_that_does_not_exist():
    """"Sign in" means the dashboard only when there IS a dashboard."""
    design = screens("Login", "Terms of Service")

    assert interactions.auto_link([candidate("1:9", "Sign in")], design) == []


def test_back_goes_back_without_naming_a_destination():
    design = screens("Login", "Sign Up")

    links = interactions.auto_link(
        [candidate("1:9", "Back", screen_id="f:2")], design
    )

    assert links[0].action == "back"
    assert links[0].destination_id == ""


def test_a_heading_that_merely_mentions_a_screen_is_not_a_link():
    """"Welcome back to your dashboard" is prose. Linking it would make the
    page heading navigate, which no one would ever click."""
    design = screens("Login", "Dashboard")
    heading = candidate(
        "1:4", "Welcome back to your dashboard", name="Heading", type_="TEXT"
    )

    assert interactions.auto_link([heading], design) == []


def test_the_button_wins_over_the_text_inside_it():
    """A Button frame and its label both say "Sign in". Wiring both means the
    label swallows the click and the button's own reaction never fires."""
    design = screens("Login", "Dashboard")
    button = candidate("1:9", "Sign in")
    label = candidate("1:10", "Sign in", name="Text", type_="TEXT", path=("f:1", "1:9"))

    links = interactions.auto_link([button, label], design)

    assert [link.source_id for link in links] == ["1:9"]


def test_a_node_that_is_already_wired_is_never_touched():
    """It was wired by the model while building, or by the user. Both know
    more about intent than a name match does."""
    design = screens("Login", "Dashboard")
    already = candidate("1:9", "Dashboard", wired=True)

    assert interactions.auto_link([already], design) == []


def test_a_child_of_an_already_wired_node_is_left_alone_too():
    design = screens("Login", "Dashboard")
    parent = candidate("1:9", "Dashboard", wired=True)
    child = candidate("1:10", "Dashboard", type_="TEXT", name="Text", path=("f:1", "1:9"))

    assert interactions.auto_link([parent, child], design) == []


def test_nothing_links_to_the_screen_it_is_already_on():
    """A nav item labelled "Home" on the Home screen is a state, not a link."""
    design = screens("Home", "Shop")
    on_home = candidate("1:9", "Home", screen_id="f:1", name="Nav / Home")

    assert interactions.auto_link([on_home], design) == []


def test_a_plain_decorative_frame_is_not_clickable():
    design = screens("Login", "Dashboard")
    box = candidate("1:9", "Dashboard", name="Illustration")

    assert interactions.auto_link([box], design) == []


def test_wiring_is_capped():
    design = screens("Login", "Dashboard")
    many = [candidate(f"1:{i}", "Dashboard") for i in range(200)]

    assert len(interactions.auto_link(many, design)) <= interactions.MAX_LINKS


# -- the reaction itself -----------------------------------------------------


def test_a_navigate_reaction_has_the_shape_the_plugin_api_wants():
    link = interactions.Link(source_id="1:9", destination_id="f:2", label="Sign in")

    payload = interactions.reaction(link)

    assert payload["trigger"] == {"type": "ON_CLICK"}
    action = payload["actions"][0]
    assert action["type"] == "NODE"
    assert action["navigation"] == "NAVIGATE"
    assert action["destinationId"] == "f:2"
    assert action["transition"]["type"] == "DISSOLVE"


def test_back_carries_no_transition():
    """BACK replays whatever brought you here -- the typings give it no
    transition field at all, and inventing one fails validation."""
    payload = interactions.reaction(interactions.Link(source_id="1:9", action="back"))

    assert payload["actions"] == [{"type": "BACK"}]


def test_an_unknown_trigger_or_transition_falls_back_rather_than_failing():
    assert interactions.normalize_trigger("ON_CLICK") == "click"
    assert interactions.normalize_trigger("wiggle") == interactions.DEFAULT_TRIGGER
    assert interactions.normalize_transition("fade") == "dissolve"
    assert interactions.normalize_transition("teleport") == interactions.DEFAULT_TRANSITION


def test_an_instant_transition_is_expressible_as_null():
    link = interactions.Link(source_id="1:9", destination_id="f:2", transition="instant")

    assert interactions.reaction(link)["actions"][0]["transition"] is None


# -- resolving a screen name -------------------------------------------------


def test_a_screen_name_is_resolved_loosely_but_never_wildly():
    mapping = {"Dashboard": "f:2", "Sign Up": "f:3"}

    assert interactions.resolve_screen("dashboard", mapping) == "f:2"
    assert interactions.resolve_screen("Dashboard screen", mapping) == "f:2"
    assert interactions.resolve_screen("Checkout", mapping) == ""
    assert interactions.resolve_screen("", mapping) == ""


# -- the model's own plan, checked ------------------------------------------


def test_an_invented_node_id_never_reaches_figma():
    design = screens("Login", "Dashboard")
    known = [candidate("1:9", "Continue")]

    plan = interactions.parse_link_plan(
        json.dumps([{"id": "9:99", "to": "Dashboard"}]), known, design
    )

    assert plan.links == []
    assert "9:99" in plan.rejected[0]


def test_an_invented_screen_is_rejected_with_a_reason():
    design = screens("Login", "Dashboard")
    known = [candidate("1:9", "Continue")]

    plan = interactions.parse_link_plan(
        json.dumps([{"id": "1:9", "to": "Checkout"}]), known, design
    )

    assert plan.links == []
    assert "Checkout" in plan.rejected[0]


def test_a_link_to_its_own_screen_is_rejected():
    design = screens("Login", "Dashboard")
    known = [candidate("1:9", "Continue", screen_id="f:1")]

    plan = interactions.parse_link_plan(
        json.dumps([{"id": "1:9", "to": "Login"}]), known, design
    )

    assert plan.links == []


def test_a_good_plan_survives_prose_and_code_fences():
    design = screens("Login", "Dashboard")
    known = [candidate("1:9", "Continue")]

    plan = interactions.parse_link_plan(
        'Sure!\n```json\n[{"id": "1:9", "to": "Dashboard"}]\n```',
        known,
        design,
    )

    assert [link.destination_id for link in plan.links] == ["f:2"]


def test_an_unparseable_reply_is_no_links_rather_than_an_exception():
    design = screens("Login", "Dashboard")

    plan = interactions.parse_link_plan("I could not work that out.", [], design)

    assert plan.links == []


# -- what still has no way in ------------------------------------------------


def test_the_first_screen_is_never_reported_as_unreachable():
    """It is the entry point. A home screen nothing links INTO is normal."""
    design = screens("Login", "Dashboard")

    stranded = interactions.unreachable(design, [])

    assert [s.name for s in stranded] == ["Dashboard"]


def test_a_screen_with_an_inbound_link_is_reachable():
    design = screens("Login", "Dashboard")
    links = [interactions.Link(source_id="1:9", destination_id="f:2")]

    assert interactions.unreachable(design, links) == []


# -- the scripts -------------------------------------------------------------


def test_the_apply_script_wraps_each_link_separately():
    """Not atomic, deliberately: one stale id must not discard nineteen good
    interactions -- the same reason agent/editor.py is not atomic."""
    script = interactions.build_apply_script(
        [
            interactions.Link(source_id="1:9", label="a", destination_id="f:2"),
            interactions.Link(source_id="1:10", label="b", destination_id="f:2"),
        ]
    )

    assert script.count("try {") == 1  # one loop, one try -- per iteration
    assert "for (const link of links)" in script
    assert "failed.push" in script


def test_the_flow_script_merges_rather_than_replacing_existing_starting_points():
    """Assigning flowStartingPoints overwrites the whole list. A re-run must
    not throw away flows the user set up by hand."""
    script = interactions.build_flow_script([{"id": "f:1", "name": "Login", "start": True}])

    assert "page.flowStartingPoints" in script
    assert "wanted.concat(kept)" in script


def test_a_tall_screen_is_told_to_scroll():
    script = interactions.build_flow_script(
        [{"id": "f:1", "name": "Landing", "start": True, "scrolls": True}]
    )

    assert "overflowDirection = 'VERTICAL'" in script
    assert '"scrolls": true' in script


def test_the_scripts_are_valid_javascript():
    from tests.test_scaffold import compiles_as_async_body

    assert compiles_as_async_body(
        interactions.build_candidates_script([{"id": "f:1", "name": "Login"}])
    )
    assert compiles_as_async_body(
        interactions.build_apply_script(
            [interactions.Link(source_id="1:9", label="Sign in", destination_id="f:2")]
        )
    )
    assert compiles_as_async_body(
        interactions.build_flow_script([{"id": "f:1", "name": "Login", "start": True}])
    )


def test_the_candidate_reader_prefers_the_outermost_node():
    """Breadth-first, so a Button frame is seen before the text inside it --
    which is what makes the "button wins" rule above work on real output."""
    script = interactions.build_candidates_script([{"id": "f:1", "name": "Login"}])

    assert "queue.shift()" in script


# ---- the auth pair, which is how most designs start -------------------------
#
# A Login and a Sign Up screen link to each other through one sentence each,
# and the sentence is longer than a button label. Missing them means a
# two-screen design where nothing is clickable at all.


def test_the_sign_up_sentence_links_to_the_sign_up_screen():
    design = screens("Login", "Sign Up")
    prompt = candidate(
        "1:20", "Don't have an account? Create one", type_="TEXT", name="Text"
    )

    links = interactions.auto_link([prompt], design)

    assert [link.destination_name for link in links] == ["Sign Up"]


def test_the_sign_in_sentence_links_back_to_the_login_screen():
    design = screens("Login", "Sign Up")
    prompt = candidate(
        "1:30", "Already have an account? Sign in", screen_id="f:2",
        type_="TEXT", name="Text",
    )

    links = interactions.auto_link([prompt], design)

    assert [link.destination_name for link in links] == ["Login"]


def test_a_sentence_ending_in_a_screen_NAME_is_still_prose():
    """The false positive the tail rule has to avoid: only an ACTION tail
    counts, so a heading is never turned into navigation."""
    design = screens("Login", "Dashboard")
    heading = candidate(
        "1:4", "Welcome back to your dashboard", type_="TEXT", name="Heading"
    )

    assert interactions.auto_link([heading], design) == []


def test_sign_in_prefers_where_signing_in_takes_you():
    """Same words, two meanings, decided by what exists: on a login screen with
    a dashboard beside it, "Sign In" submits."""
    design = screens("Login", "Sign Up", "Dashboard")
    button = candidate("1:9", "Sign In")

    links = interactions.auto_link([button], design)

    assert [link.destination_name for link in links] == ["Dashboard"]


def test_sign_in_falls_back_to_the_login_screen_when_there_is_nowhere_to_land():
    design = screens("Sign Up", "Login")
    link_text = candidate("1:9", "Sign in", screen_id="f:1")

    links = interactions.auto_link([link_text], design)

    assert [link.destination_name for link in links] == ["Login"]


def test_the_sign_in_button_on_the_login_screen_links_nowhere_in_an_auth_pair():
    """There is no dashboard to land on, and linking it to Sign Up would be
    wrong. No link is the right answer."""
    design = screens("Login", "Sign Up")
    button = candidate("1:9", "Sign In", screen_id="f:1")

    assert interactions.auto_link([button], design) == []
