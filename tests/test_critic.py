"""The visual gate's deterministic half.

These are the defects a text-only model cannot see and metadata alone doesn't
announce: collapsed text, overflow, overlap, blank regions. Pure geometry, so
it is exactly testable -- and it must never invent a defect in a clean tree.
"""
from __future__ import annotations

import json

from agent import critic


def frame(**kw):
    base = {
        "id": "1:1", "name": "Frame", "type": "FRAME", "x": 0, "y": 0,
        "width": 100, "height": 100, "visible": True, "layoutMode": None, "children": [],
    }
    base.update(kw)
    return base


def text(**kw):
    base = {
        "id": "1:2", "name": "Text", "type": "TEXT", "x": 0, "y": 0,
        "width": 100, "height": 20, "visible": True, "layoutMode": None,
        "children": [], "characters": "Hello", "fontSize": 16,
    }
    base.update(kw)
    return base


def kinds(tree):
    return sorted({d.kind for d in critic.find_layout_defects(tree)})


def test_a_clean_layout_reports_nothing():
    tree = frame(
        name="Root", width=1440, height=400, layoutMode="VERTICAL",
        children=[
            frame(id="2:1", name="Nav", width=1440, height=80, children=[text(id="2:2", width=200)]),
            frame(id="2:3", name="Hero", y=80, width=1440, height=320,
                  children=[text(id="2:4", width=600, height=48, fontSize=40)]),
        ],
    )

    assert critic.find_layout_defects(tree) == []


def test_collapsed_text_is_caught():
    """The classic Figma trap: a TEXT node collapses to ~0px and vanishes."""
    tree = frame(children=[text(id="3:1", name="Headline", width=0, height=0)])

    assert "collapsed" in kinds(tree)


def test_a_sliver_wide_text_node_is_caught():
    tree = frame(children=[text(id="3:2", name="Label", width=3, height=20)])

    assert "collapsed-text" in kinds(tree)


def test_text_shorter_than_its_font_is_clipped():
    tree = frame(children=[text(id="3:3", name="Title", width=300, height=10, fontSize=32)])

    assert "clipped-text" in kinds(tree)


def test_children_escaping_their_parent_are_caught():
    tree = frame(
        name="Card", width=200, height=100,
        children=[frame(id="4:1", name="Overflowing", x=150, y=0, width=200, height=50)],
    )

    defects = critic.find_layout_defects(tree)
    assert any(d.kind == "overflow" for d in defects)
    assert "Overflowing" in str(defects[0])


def test_stacked_autolayout_content_is_one_defect_not_one_per_child():
    """A live run reported the SAME clipping problem 28 times, once per stacked
    section. In auto-layout Figma owns child positions, so this is one bug: the
    frame is FIXED and no longer hugging."""
    tree = frame(
        name="Generated Page", width=1440, height=2233, layoutMode="VERTICAL",
        children=[
            frame(id="a", name="Modal A", y=2235, width=1440, height=1),
            frame(id="b", name="Modal B", y=2236, width=1440, height=1),
            frame(id="c", name="Modal C", y=2237, width=375, height=812),
        ],
    )

    defects = critic.find_layout_defects(tree)
    clipped = [d for d in defects if d.kind == "clipped-content"]
    assert len(clipped) == 1
    assert "overflow" not in kinds(tree)          # not blamed on the children
    assert "primaryAxisSizingMode" in clipped[0].detail   # says how to fix it


def test_an_autolayout_frame_that_fits_its_content_is_clean():
    tree = frame(
        name="Root", width=1440, height=400, layoutMode="VERTICAL",
        children=[
            frame(id="a", name="Nav", y=0, width=1440, height=80, children=[text(id="t", width=200)]),
            frame(id="b", name="Hero", y=80, width=1440, height=320, children=[text(id="t2", width=400)]),
        ],
    )

    assert "clipped-content" not in kinds(tree)


def test_defects_are_deduplicated_and_capped():
    """Twelve identically-broken siblings should not produce twelve lines."""
    tree = frame(
        name="Root", width=800, height=600, layoutMode="NONE",
        children=[text(id=f"t{i}", name="Label", width=0, height=0) for i in range(12)],
    )

    defects = critic.find_layout_defects(tree)
    assert len(defects) <= critic.MAX_DEFECTS
    assert len([d for d in defects if d.kind == "collapsed"]) <= critic.MAX_PER_KIND


def test_overlapping_siblings_are_caught_only_without_auto_layout():
    overlapping = [
        frame(id="5:1", name="A", x=0, y=0, width=100, height=50),
        frame(id="5:2", name="B", x=0, y=10, width=100, height=50),
    ]
    loose = frame(name="Root", width=200, height=200, layoutMode="NONE", children=overlapping)
    assert "overlap" in kinds(loose)

    # Auto layout positions children itself -- reporting overlap there would be
    # a false positive, since the coordinates are managed by Figma.
    managed = frame(name="Root", width=200, height=200, layoutMode="VERTICAL", children=overlapping)
    assert "overlap" not in kinds(managed)


def test_a_two_pixel_nudge_is_not_reported():
    """Rounding shouldn't produce noise -- only real visual overlap counts."""
    tree = frame(
        name="Root", width=300, height=200, layoutMode="NONE",
        children=[
            frame(id="6:1", name="A", x=0, y=0, width=100, height=50),
            frame(id="6:2", name="B", x=98, y=0, width=100, height=50),
        ],
    )

    assert "overlap" not in kinds(tree)


def test_empty_and_hidden_regions_are_caught():
    assert "empty-frame" in kinds(frame(children=[frame(id="7:1", name="Blank", width=300, height=200)]))
    assert "invisible" in kinds(frame(children=[text(id="7:2", name="Ghost", visible=False)]))
    assert "empty-text" in kinds(frame(children=[text(id="7:3", name="Blank label", characters="  ")]))


def test_tiny_empty_frames_are_ignored():
    """A 24x24 empty frame is an icon placeholder, not a blank region."""
    tree = frame(children=[frame(id="8:1", name="Icon", width=24, height=24)])

    assert "empty-frame" not in kinds(tree)


# ---- model critique plumbing -------------------------------------------


def test_clean_reply_means_no_defects():
    assert critic.parse_critique("CLEAN") == []
    assert critic.parse_critique("  clean  ") == []
    assert critic.parse_critique("") == []


def test_defect_list_is_parsed_and_capped():
    reply = json.dumps([
        {"severity": "blocking", "element": "Headline", "problem": "overlaps the nav"},
        {"severity": "blocking", "element": "Footer text", "problem": "is unreadable"},
        {"severity": "minor", "element": "Cards", "problem": "are uneven"},
    ])
    defects = critic.parse_critique(reply)

    assert [d.element for d in defects] == ["Headline", "Footer text", "Cards"]
    assert [d.severity for d in defects] == ["blocking", "blocking", "minor"]

    many = json.dumps([{"severity": "minor", "element": f"e{i}", "problem": "x"} for i in range(20)])
    assert len(critic.parse_critique(many)) == critic.MAX_VISUAL_DEFECTS


def test_only_blocking_defects_can_fail_a_step():
    """The old parser treated every line as blocking, so a vision model -- which
    always finds something to say about a work in progress -- would have failed
    essentially every step and left the page full of TODO placeholders.
    """
    reply = json.dumps([
        {"severity": "blocking", "element": "Email label", "problem": "is unreadable on the navy fill"},
        {"severity": "minor", "element": "Card", "problem": "could use more breathing room"},
        {"severity": "minor", "element": "Heading", "problem": "hierarchy could be stronger"},
    ])
    blocking = critic.blocking_only(critic.parse_critique(reply))

    assert blocking == ["[visual] Email label: is unreadable on the navy fill"]


def test_an_unknown_severity_is_treated_as_minor():
    """Ambiguity must never block -- the safe default is 'record it, keep going'."""
    reply = json.dumps([{"severity": "critical!!", "element": "x", "problem": "y"}])
    assert critic.parse_critique(reply)[0].severity == "minor"
    assert critic.blocking_only(critic.parse_critique(reply)) == []


def test_unparseable_critique_blocks_nothing():
    """A critic that rambles instead of answering must not fail the step."""
    assert critic.parse_critique("I think the design looks quite nice overall!") == []
    assert critic.parse_critique("") == []


def test_a_fenced_json_reply_is_still_parsed():
    reply = '```json\n[{"severity":"blocking","element":"Hero","problem":"text is clipped"}]\n```'
    assert len(critic.parse_critique(reply)) == 1


def test_critique_message_carries_both_signals():
    """Metadata is structural truth, the screenshot is visual truth -- send both."""
    messages = critic.build_critique_messages('{"a":1}', "BASE64PNG")

    content = messages[-1]["content"]
    assert any(c.get("type") == "image_url" for c in content)
    assert any("metadata" in str(c.get("text", "")) for c in content)


def test_critique_message_without_a_screenshot_omits_the_image():
    messages = critic.build_critique_messages('{"a":1}', None)

    assert all(c.get("type") != "image_url" for c in messages[-1]["content"])


# ---- scoping the gate to one step's own nodes -----------------------------

def _page_with_a_broken_earlier_section() -> dict:
    """A page where section A is broken (collapsed text) and section B is fine."""
    return {
        "id": "0:root", "name": "Root", "type": "FRAME", "x": 0, "y": 0,
        "width": 1440, "height": 800, "visible": True, "layoutMode": "VERTICAL",
        "children": [
            {
                "id": "sec:a", "name": "Broken Hero", "type": "FRAME", "x": 0, "y": 0,
                "width": 1440, "height": 400, "visible": True, "layoutMode": "VERTICAL",
                "children": [
                    {
                        "id": "t:a", "name": "Collapsed", "type": "TEXT", "x": 0, "y": 0,
                        "width": 0, "height": 0, "visible": True, "layoutMode": None,
                        "children": [], "characters": "Invisible", "fontSize": 32,
                    }
                ],
            },
            {
                "id": "sec:b", "name": "Good Footer", "type": "FRAME", "x": 0, "y": 400,
                "width": 1440, "height": 400, "visible": True, "layoutMode": "VERTICAL",
                "children": [
                    {
                        "id": "t:b", "name": "Fine", "type": "TEXT", "x": 0, "y": 0,
                        "width": 600, "height": 40, "visible": True, "layoutMode": None,
                        "children": [], "characters": "Hello", "fontSize": 32,
                    }
                ],
            },
        ],
    }


def test_scoped_gate_ignores_a_defect_in_another_section():
    """The whole point: step B must not fail because step A left a defect."""
    tree = _page_with_a_broken_earlier_section()

    assert critic.find_layout_defects(tree)  # unscoped: the page is dirty
    assert critic.find_layout_defects(tree, scope_ids=["sec:b"]) == []


def test_scoped_gate_still_catches_a_defect_inside_its_own_subtree():
    tree = _page_with_a_broken_earlier_section()
    defects = critic.find_layout_defects(tree, scope_ids=["sec:a"])
    assert [d.kind for d in defects] == ["collapsed"]
    assert defects[0].node_name == "Collapsed"


def test_unknown_scope_ids_report_nothing_rather_than_the_whole_page():
    """An id that isn't in the tree must not silently fall back to page scope."""
    tree = _page_with_a_broken_earlier_section()
    assert critic.find_layout_defects(tree, scope_ids=["nope:1"]) == []


def test_a_node_and_its_own_child_are_not_walked_twice():
    tree = _page_with_a_broken_earlier_section()
    defects = critic.find_layout_defects(tree, scope_ids=["sec:a", "t:a"])
    assert len(defects) == 1


def test_text_wrapping_one_character_per_line_is_caught():
    """From a real canvas: "Recent Transactions" rendered as a vertical column
    of letters. Unmissable to a human, invisible to every check -- 20px is not
    "collapsed", and the node's height was large rather than small.
    """
    tree = frame(
        name="Table Card", width=1376, height=400,
        children=[text(id="9:1", name="Recent Transactions", characters="Recent Transactions",
                       width=20, height=260, fontSize=18)],
    )

    defects = critic.find_layout_defects(tree)
    assert "vertical-text" in {d.kind for d in defects}
    assert "one character per line" in str(defects[0])


def test_normal_wrapped_paragraphs_are_not_flagged():
    """A wide paragraph is tall too -- it must not look like the same bug."""
    tree = frame(
        name="Card", width=600, height=200,
        children=[text(id="9:2", name="Body", characters="A" * 200,
                       width=520, height=120, fontSize=16)],
    )

    assert "vertical-text" not in {d.kind for d in critic.find_layout_defects(tree)}


def test_a_short_narrow_label_is_not_flagged():
    """A one-word label in a narrow column is legitimate."""
    tree = frame(
        name="Card", width=200, height=100,
        children=[text(id="9:3", name="Qty", characters="Qty", width=30, height=20, fontSize=13)],
    )

    assert "vertical-text" not in {d.kind for d in critic.find_layout_defects(tree)}


# ---- contrast: the defect geometry is structurally blind to ----------------


def fill(hex_value: str) -> dict:
    """A solid fill in the 0-1 form the Plugin API stores and the read returns."""
    h = hex_value.lstrip("#")
    return {c: int(h[i : i + 2], 16) / 255 for c, i in (("r", 0), ("g", 2), ("b", 4))}


def test_invisible_text_is_caught_even_though_the_geometry_is_perfect():
    """The exact failure CLAUDE.md 6a describes: right size, right place, unreadable.

    Every existing check passes this tree -- the node has area, fits its parent,
    overlaps nothing and holds real characters.
    """
    tree = frame(
        name="Card", width=400, height=200, fill=fill("#FFFFFF"),
        children=[text(id="9:1", name="Body", width=300, height=24, fill=fill("#F2F2F2"))],
    )

    assert "contrast" in kinds(tree)
    assert critic.find_layout_defects(tree) != []


def test_readable_text_reports_nothing():
    tree = frame(
        name="Card", width=400, height=200, fill=fill("#FFFFFF"), tokenBacked=True,
        children=[text(id="9:2", width=300, height=24, fill=fill("#101828"),
                       tokenBacked=True)],
    )

    assert critic.find_layout_defects(tree) == []
    assert critic.find_design_defects(tree) == []


def test_a_near_miss_on_AA_is_advisory_not_blocking():
    """Legible but non-compliant must not burn a step's retries -- and must not
    end up replacing a working section with a TODO placeholder."""
    # ~3.9:1 against white: below AA (4.5) but far above unreadable (3.0).
    tree = frame(
        name="Card", width=400, height=200, fill=fill("#FFFFFF"),
        children=[text(id="9:3", width=300, height=24, fill=fill("#949494"))],
    )

    assert "contrast" not in kinds(tree)
    assert "contrast-aa" in {d.kind for d in critic.find_design_defects(tree)}


def test_large_text_is_held_to_the_lower_wcag_bar():
    """3.4:1 fails AA for body copy but passes it for a 48px display heading."""
    grey = fill("#949494")
    heading = frame(
        name="Card", width=400, height=200, fill=fill("#FFFFFF"),
        children=[text(id="9:4", width=300, height=56, fontSize=48, fill=grey)],
    )
    body = frame(
        name="Card", width=400, height=200, fill=fill("#FFFFFF"),
        children=[text(id="9:5", width=300, height=24, fontSize=16, fill=grey)],
    )

    assert "contrast-aa" not in {d.kind for d in critic.find_design_defects(heading)}
    assert "contrast-aa" in {d.kind for d in critic.find_design_defects(body)}


def test_background_is_inherited_from_the_nearest_filled_ancestor():
    """A transparent wrapper between the text and the fill must not hide the defect."""
    tree = frame(
        name="Section", width=400, height=200, fill=fill("#101828"),
        children=[
            frame(id="9:6", name="Wrapper", width=400, height=100, fill=None,
                  children=[text(id="9:7", width=300, height=24, fill=fill("#1A2438"))]),
        ],
    )

    assert "contrast" in kinds(tree)


def test_text_with_no_resolvable_background_is_left_alone():
    """Over an image or an unfilled page there is no single background colour.
    Inventing one would produce confident nonsense."""
    tree = frame(
        name="Section", width=400, height=200, fill=None,
        children=[text(id="9:8", width=300, height=24, fill=fill("#FFFFFF"))],
    )

    assert "contrast" not in kinds(tree)
    assert "contrast-aa" not in {d.kind for d in critic.find_design_defects(tree)}


# ---- design-system adherence (advisory, never gating) ---------------------


def test_off_scale_spacing_is_reported_but_never_blocks():
    tree = frame(
        name="Section", layoutMode="VERTICAL", itemSpacing=17, padding=[13, 13, 13, 13],
        children=[text(id="8:1", width=200), text(id="8:2", y=40, width=200)],
    )

    assert "off-scale-spacing" in {d.kind for d in critic.find_design_defects(tree)}
    assert "off-scale-spacing" not in kinds(tree)
    assert all(d.advisory for d in critic.find_design_defects(tree))


def test_on_scale_spacing_is_clean():
    tree = frame(
        name="Section", layoutMode="VERTICAL", itemSpacing=24, padding=[32, 80, 32, 80],
        children=[text(id="8:3", width=200), text(id="8:4", y=40, width=200)],
    )

    assert "off-scale-spacing" not in {d.kind for d in critic.find_design_defects(tree)}


def test_a_font_size_off_the_ramp_is_reported():
    tree = frame(children=[text(id="8:5", fontSize=17, width=200)])

    assert "off-ramp-type" in {d.kind for d in critic.find_design_defects(tree)}


def test_a_font_size_on_the_ramp_is_clean():
    tree = frame(children=[text(id="8:6", fontSize=16, width=200)])

    assert "off-ramp-type" not in {d.kind for d in critic.find_design_defects(tree)}


def test_a_hardcoded_fill_is_reported_and_a_token_backed_one_is_not():
    hardcoded = frame(id="8:7", name="Band", width=200, height=80, fill=fill("#123456"))
    tokenised = frame(id="8:8", name="Band", width=200, height=80,
                      fill=fill("#123456"), tokenBacked=True)

    assert "untokenised-fill" in {d.kind for d in critic.find_design_defects(hardcoded)}
    assert "untokenised-fill" not in {d.kind for d in critic.find_design_defects(tokenised)}


def test_tiny_nodes_are_exempt_from_the_token_audit():
    """An icon or a divider with a one-off colour is normal, not a violation."""
    tree = frame(id="8:9", name="Dot", width=8, height=8, fill=fill("#123456"))

    assert "untokenised-fill" not in {d.kind for d in critic.find_design_defects(tree)}


def test_the_layout_script_compiles_as_the_plugin_evaluates_it():
    """It grew fill/token/spacing readers; a syntax slip must fail here, not mid-run."""
    from tests.test_scaffold import compiles_as_async_body

    assert compiles_as_async_body(critic.layout_script("1:23"))


# ---- the harness must not report its own scaffolding as design problems ----
#
# A finished run listed six "design system notes" and every one was about the
# TODO frames the HARNESS had dropped in for steps it could not build: 14px
# text off the type ramp, two hardcoded greys, 4.2:1 contrast. The harness was
# failing its own checks and billing the user for it.


def test_a_todo_placeholder_is_not_judged_as_design():
    placeholder = {
        "id": "9:1", "name": "TODO — login form", "type": "FRAME", "x": 0, "y": 0,
        "width": 1440, "height": 160, "visible": True, "layoutMode": "VERTICAL",
        "itemSpacing": 17, "padding": [13, 13, 13, 13], "fill": fill("#F7F7F8"),
        "children": [text(id="9:2", name="Label", characters="TODO: login form",
                          fontSize=14, width=200, fill=fill("#95979E"))],
    }

    assert critic.find_design_defects(placeholder) == []


def test_a_real_section_is_still_judged():
    """The exemption is for TODO markers only, not for anything nearby."""
    section = {
        "id": "9:3", "name": "Hero", "type": "FRAME", "x": 0, "y": 0,
        "width": 1440, "height": 160, "visible": True, "layoutMode": "VERTICAL",
        "itemSpacing": 17, "padding": [13, 13, 13, 13], "fill": fill("#F7F7F8"),
        "children": [text(id="9:4", name="Label", characters="Hi", fontSize=14, width=200)],
    }

    kinds = {d.kind for d in critic.find_design_defects(section)}
    assert "off-scale-spacing" in kinds
    assert "off-ramp-type" in kinds


def test_a_placeholder_that_does_not_render_is_still_reported():
    """Excusing its colours must not excuse it being invisible."""
    placeholder = {
        "id": "9:5", "name": "TODO — hero", "type": "FRAME", "x": 0, "y": 0,
        "width": 1440, "height": 160, "visible": True, "layoutMode": "VERTICAL",
        "children": [text(id="9:6", name="Label", characters="TODO: hero", width=0, height=0)],
    }

    assert "collapsed" in {d.kind for d in critic.find_layout_defects(placeholder)}


# ---- duplicated regions -----------------------------------------------------


def test_a_section_that_rebuilds_a_sibling_is_caught():
    """A dashboard run built its sidebar twice: one step added the sidebar, the
    next rebuilt the whole shell around it. Both copies were well-formed, so
    only their identical text gives them away."""
    def sidebar(node_id, name):
        return {
            "id": node_id, "name": name, "type": "FRAME", "x": 0, "y": 0,
            "width": 240, "height": 400, "visible": True, "layoutMode": "VERTICAL",
            "children": [
                text(id=node_id + ":a", characters="Acme"),
                text(id=node_id + ":b", characters="Dashboard"),
                text(id=node_id + ":c", characters="Profile"),
                text(id=node_id + ":d", characters="Settings"),
            ],
        }

    tree = frame(id="root", name="Dashboard", width=1440, height=900, layoutMode="VERTICAL",
                 children=[sidebar("s1", "Sidebar"), sidebar("s2", "Shell")])

    defects = critic.find_duplicate_sections(tree)

    assert [d.kind for d in defects] == ["duplicate-section"]
    assert defects[0].node_id == "s2"          # the LATER copy is the duplicate
    assert "Sidebar" in defects[0].detail      # and it names what it repeats


def test_sections_that_merely_share_a_word_are_not_duplicates():
    def band(node_id, name, words):
        return {
            "id": node_id, "name": name, "type": "FRAME", "x": 0, "y": 0,
            "width": 1440, "height": 200, "visible": True, "layoutMode": "VERTICAL",
            "children": [text(id=f"{node_id}:{i}", characters=w) for i, w in enumerate(words)],
        }

    tree = frame(id="root", name="Landing", width=1440, height=900, layoutMode="VERTICAL",
                 children=[
                     band("b1", "Hero", ["Get started", "Plan your week", "Sign up"]),
                     band("b2", "Footer", ["Get started", "Careers", "Privacy"]),
                 ])

    assert critic.find_duplicate_sections(tree) == []


def test_a_step_is_only_blamed_for_its_own_duplicate():
    def nav(node_id):
        return {
            "id": node_id, "name": "Nav", "type": "FRAME", "x": 0, "y": 0,
            "width": 1440, "height": 80, "visible": True, "layoutMode": "HORIZONTAL",
            "children": [text(id=node_id + ":a", characters="Home"),
                         text(id=node_id + ":b", characters="Pricing"),
                         text(id=node_id + ":c", characters="Contact")],
        }

    tree = frame(id="root", name="Page", width=1440, height=900, layoutMode="VERTICAL",
                 children=[nav("n1"), nav("n2")])

    # Scoped to the FIRST copy: it duplicates nothing before it.
    assert critic.find_duplicate_sections(tree, scope_ids=["n1"]) == []
    assert len(critic.find_duplicate_sections(tree, scope_ids=["n2"])) == 1


# ---- a real trace: every form step became a TODO placeholder ---------------
#
# The tree read stopped at depth 4. A form field sits at depth 5:
#   screen -> section -> col -> Field -> Input -> placeholder text
# so `Input` always came back childless and was reported as an empty frame --
# a defect no attempt could fix. Three steps in a row failed their visual gate
# three times each and were demoted to TODO placeholders.


def test_the_layout_read_reaches_a_form_field_placeholder():
    script = critic.layout_script("1:2")
    assert critic.MAX_TREE_DEPTH >= 6, "a form field's text sits at depth 5"
    assert f"const MAX_TREE_DEPTH = {critic.MAX_TREE_DEPTH};" in script
    assert "depth < MAX_TREE_DEPTH" in script


def _nest(depth: int) -> dict:
    """A screen -> section -> col -> Field -> Input -> text chain."""
    text = {"id": "n:5", "name": "Placeholder", "type": "TEXT", "x": 0, "y": 0,
            "width": 120, "height": 20, "visible": True, "characters": "you@company.com",
            "fontSize": 15, "fill": {"r": 0.4, "g": 0.4, "b": 0.4}, "children": []}
    node = text
    for level in reversed(range(depth)):
        node = {"id": f"n:{level}", "name": ["Screen", "Section", "Col", "Field", "Input"][level],
                "type": "FRAME", "x": 0, "y": 0, "width": 372, "height": 56,
                "visible": True, "layoutMode": "VERTICAL", "fill": {"r": 1, "g": 1, "b": 1},
                "children": [node]}
    return node


def test_a_populated_input_is_not_reported_as_an_empty_frame():
    assert critic.find_layout_defects(_nest(5)) == []


def test_a_filled_decorative_block_is_not_a_blank_region():
    """The renderer's own `box` -- a chart area, an image placeholder, a glowing
    shape -- is a deliberately childless FILLED frame. Flagging it meant the
    harness failed its own output."""
    tree = {"id": "1:1", "name": "Hero", "type": "FRAME", "x": 0, "y": 0,
            "width": 800, "height": 600, "visible": True, "fill": {"r": 1, "g": 1, "b": 1},
            "children": [
                {"id": "1:2", "name": "Glow", "type": "FRAME", "x": 0, "y": 0,
                 "width": 240, "height": 120, "visible": True,
                 "fill": {"r": 0.49, "g": 0.23, "b": 0.93}, "children": []}]}

    assert [d.kind for d in critic.find_layout_defects(tree)] == []


def test_an_empty_frame_the_colour_of_its_background_IS_still_a_blank_region():
    """The exemption must not swallow the real defect: a white section on a
    white page is the 1440x900 void a failed run actually produced."""
    tree = {"id": "1:1", "name": "Screen", "type": "FRAME", "x": 0, "y": 0,
            "width": 1440, "height": 900, "visible": True, "fill": {"r": 1, "g": 1, "b": 1},
            "children": [
                {"id": "1:2", "name": "Left Visual", "type": "FRAME", "x": 0, "y": 0,
                 "width": 792, "height": 900, "visible": True,
                 "fill": {"r": 1, "g": 1, "b": 1}, "children": []}]}

    assert any(d.kind == "empty-frame" for d in critic.find_layout_defects(tree))
