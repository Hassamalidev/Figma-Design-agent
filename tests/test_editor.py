"""The edit compiler: what it produces, and above all what it refuses.

Weighted towards refusals on purpose. A bad create leaves an ugly section next
to the good ones; a bad edit damages work the user already has, so the checks
that stop an edit reaching Figma matter more than the ones that shape it.

The generated JavaScript is compiled exactly as the plugin evaluates it.
"""
from __future__ import annotations

import functools
import json
import re
import subprocess

import pytest

from agent import editor, inventory
from agent.renderer import SpecError


def compiles_as_async_body(code: str) -> bool:
    result = subprocess.run(
        ["node", "-e", "new (Object.getPrototypeOf(async function(){}).constructor)(process.argv[1])", "--", code],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


PAGE = {
    "roots": [{
        "id": "1:2", "name": "Login", "type": "FRAME", "width": 1440, "height": 900,
        "layoutMode": "VERTICAL", "children": [
            {"id": "1:3", "name": "Auth Card", "type": "FRAME", "width": 440, "height": 520,
             "layoutMode": "VERTICAL", "children": [
                 {"id": "1:4", "name": "Heading", "type": "TEXT", "width": 300, "height": 40,
                  "characters": "Welcome back", "fontSize": 32, "children": []},
                 {"id": "1:9", "name": "Button / Log in", "type": "FRAME", "width": 372,
                  "height": 48, "children": []}]}]}]
}

ROLES = {
    "accent": "color/primary",
    "text": "color/main-text",
    "surface": "color/card",
    "background": "color/background",
}
TOKENS = ["color/primary", "color/main-text", "color/card", "color/background"]


@pytest.fixture
def resolve():
    return functools.partial(inventory.resolve, inventory.build(PAGE))


def compile_one(resolve, *edits, protected=frozenset({"1:2"})):
    return editor.compile_edits(list(edits), resolve, ROLES, TOKENS, set(protected))


# ---- what it produces -----------------------------------------------------


def test_a_batch_of_edits_compiles_to_one_valid_script(resolve):
    code, touched = compile_one(
        resolve,
        {"op": "set_fill", "target": "1:9", "color": "accent"},
        {"op": "set_text", "target": "1:4", "value": "Sign in to Nexora"},
        {"op": "set_text_style", "target": "1:4", "style": "Display"},
        {"op": "set_size", "target": "1:9", "height": 56},
        {"op": "set_spacing", "target": "1:3", "gap": "lg", "padding": "xl"},
        {"op": "set_radius", "target": "1:3", "radius": "xl"},
        {"op": "set_name", "target": "1:9", "name": "Primary Button"},
        {"op": "reorder", "target": "1:9", "index": 0},
        {"op": "set_visible", "target": "1:4", "visible": True},
    )

    assert compiles_as_async_body(code)
    assert touched == ["1:9", "1:4", "1:3"]
    assert "appliedEdits" in code and "failedEdits" in code


def test_structural_edits_reuse_the_renderer(resolve):
    code, _ = compile_one(
        resolve,
        {"op": "insert", "parent": "1:3", "index": 0,
         "spec": {"kind": "text", "style": "Caption", "value": "Beta"}},
        {"op": "replace", "target": "1:9",
         "spec": {"kind": "button", "label": "Continue", "variant": "primary"}},
    )

    assert compiles_as_async_body(code)
    assert '"Beta"' in code and '"Continue"' in code


def test_one_bad_target_does_not_discard_the_whole_batch(resolve):
    """Atomicity is right for building one section and wrong for a batch of
    independent edits -- a single stale node would cost nine good changes."""
    code, _ = compile_one(resolve, {"op": "set_fill", "target": "1:9", "color": "accent"})

    assert code.count("try {") >= 1 and "catch (e)" in code
    assert "failed.push(" in code


def test_set_text_loads_the_font_the_node_already_uses(resolve):
    """Guessing a style string killed a live run ("Inter SemiBold" -- the real
    style has a space). Editing never has to guess: the node knows."""
    code, _ = compile_one(resolve, {"op": "set_text", "target": "1:4", "value": "Hi"})

    assert "loadFontAsync(n0.fontName)" in code
    assert "figma.mixed" in code  # ...and a mixed-font node is handled, not crashed on


def test_resizing_a_child_of_auto_layout_switches_it_to_a_fixed_size_first(resolve):
    """Otherwise the layout silently undoes the resize and the edit reports
    success while nothing visibly changed."""
    code, _ = compile_one(resolve, {"op": "set_size", "target": "1:9", "width": 400})

    assert "layoutSizingHorizontal = 'FIXED'" in code


def test_a_selector_target_expands_to_every_match(resolve):
    code, touched = compile_one(
        resolve, {"op": "set_fill", "target": {"type": "TEXT"}, "color": "text"}
    )

    assert touched == ["1:4"]
    assert compiles_as_async_body(code)


# ---- what it refuses ------------------------------------------------------


def test_an_id_that_is_not_on_the_canvas_never_reaches_figma(resolve):
    with pytest.raises(SpecError) as caught:
        compile_one(resolve, {"op": "set_fill", "target": "9:99", "color": "accent"})

    assert "9:99" in str(caught.value)


def test_an_unknown_op_is_named_alongside_the_real_ones(resolve):
    with pytest.raises(SpecError) as caught:
        compile_one(resolve, {"op": "rotate", "target": "1:9"})

    assert "rotate" in str(caught.value) and "set_fill" in str(caught.value)


def test_a_hex_colour_is_refused_so_edits_stay_token_backed(resolve):
    """Golden rule 5. An edit that hardcodes a colour is a colour that no longer
    changes when the palette does."""
    with pytest.raises(SpecError):
        compile_one(resolve, {"op": "set_fill", "target": "1:9", "color": "#FF0000"})


def test_a_whole_screen_frame_cannot_be_deleted(resolve):
    """Far too much damage for a single mis-parsed word."""
    with pytest.raises(SpecError) as caught:
        compile_one(resolve, {"op": "delete", "target": "1:2"})

    assert "whole screen frame" in str(caught.value)


def test_a_section_inside_a_screen_CAN_be_deleted(resolve):
    code, touched = compile_one(resolve, {"op": "delete", "target": "1:9"})

    assert touched == ["1:9"] and compiles_as_async_body(code)


def test_set_text_will_not_blank_a_node(resolve):
    with pytest.raises(SpecError) as caught:
        compile_one(resolve, {"op": "set_text", "target": "1:4", "value": "   "})

    assert "delete" in str(caught.value)


def test_spacing_stays_on_the_8px_scale(resolve):
    with pytest.raises(SpecError):
        compile_one(resolve, {"op": "set_spacing", "target": "1:3", "gap": 17})


def test_a_text_style_outside_the_ramp_is_refused(resolve):
    with pytest.raises(SpecError) as caught:
        compile_one(resolve, {"op": "set_text_style", "target": "1:4", "style": "Enormous"})

    assert "Heading" in str(caught.value)


def test_an_empty_or_oversized_batch_is_refused(resolve):
    with pytest.raises(SpecError):
        editor.compile_edits([], resolve, ROLES, TOKENS)
    with pytest.raises(SpecError) as caught:
        editor.compile_edits(
            [{"op": "set_visible", "target": "1:9", "visible": True}] * (editor.MAX_EDITS + 1),
            resolve, ROLES, TOKENS,
        )
    assert "split this across steps" in str(caught.value)


def test_a_real_token_name_works_where_no_role_does(resolve):
    """Same resolution as the renderer: a colour that works when building has
    to work when editing."""
    code, _ = compile_one(resolve, {"op": "set_fill", "target": "1:9", "color": "color/card"})

    assert '"color/card"' in code


# ---- a real trace: an edit run emptied the user's page ---------------------
#
# `delete` was guarded against removing a whole screen frame and `replace` was
# not -- and `replace` removes its target just as surely, via the renderer's
# replace_ids. So one edit took a whole screen away, and the same op with a
# `{"type":"FRAME"}` selector fanned out across every frame on the page.
#
# These tests are the boundary of what an edit may destroy. Loosening any of
# them is how a user loses work, so each one says what it is protecting.

WIDE_PAGE = {
    "roots": [
        {"id": "1:2", "name": "Login", "type": "FRAME", "width": 1440, "height": 900,
         "children": [
             {"id": "1:3", "name": "Card", "type": "FRAME", "width": 440, "height": 520,
              "children": []},
             {"id": "1:4", "name": "Old banner", "type": "FRAME", "width": 440, "height": 80,
              "children": []}]},
        {"id": "2:2", "name": "Sign Up", "type": "FRAME", "width": 1440, "height": 900,
         "children": [
             {"id": "2:3", "name": "Card", "type": "FRAME", "width": 440, "height": 520,
              "children": []}]},
    ]
}
SCREENS = frozenset({"1:2", "2:2"})


@pytest.fixture
def wide():
    return functools.partial(inventory.resolve, inventory.build(WIDE_PAGE))


def removed_by(code: str) -> list[str]:
    """Every node id the script would actually take off the canvas."""
    gone = re.findall(r"for \(const _oldId of (\[[^\]]*\])\)", code)
    ids = [i for group in gone for i in json.loads(group)]
    ids += re.findall(r'getNodeByIdAsync\("([^"]+)"\);\s*\n\s*if \(!n\d+ \|\| n\d+\.removed\)', code)
    return ids


@pytest.mark.parametrize("op, extra", [("delete", {}), ("replace", {"spec": {"kind": "text", "value": "x"}})])
def test_no_destructive_op_may_target_a_whole_screen(wide, op, extra):
    """The exact bug: `replace` on a top-level frame emptied the page."""
    with pytest.raises(SpecError) as caught:
        editor.compile_edits([{"op": op, "target": "1:2", **extra}], wide, ROLES, TOKENS, set(SCREENS))

    assert "whole screen frame" in str(caught.value)


@pytest.mark.parametrize("op, extra", [("delete", {}), ("replace", {"spec": {"kind": "text", "value": "x"}})])
def test_a_wide_selector_may_not_remove_what_it_matches(wide, op, extra):
    """`{"type":"FRAME"}` matches everything. Bulk RECOLOURING everything is a
    fine edit; bulk REMOVING it is how a page disappears -- and unlike a list of
    ids, the model never sees how far a selector reaches."""
    with pytest.raises(SpecError) as caught:
        editor.compile_edits(
            [{"op": op, "target": {"type": "FRAME"}, **extra}], wide, ROLES, TOKENS, set(SCREENS)
        )

    assert "removes what it matches" in str(caught.value)


def test_a_wide_selector_is_still_fine_for_a_NON_destructive_edit(wide):
    """The cap must not cost the legitimate bulk edit it was never about."""
    code, touched = editor.compile_edits(
        [{"op": "set_fill", "target": {"type": "FRAME"}, "color": "accent"}],
        wide, ROLES, TOKENS, set(SCREENS),
    )

    assert len(touched) == 5 and compiles_as_async_body(code)


def test_a_batch_may_not_remove_more_than_a_handful(wide):
    """Removing more than a few nodes is a redesign, not an edit."""
    with pytest.raises(SpecError) as caught:
        editor.compile_edits(
            [{"op": "delete", "target": "1:3"}] * (editor.MAX_REMOVALS_PER_BATCH + 1),
            wide, ROLES, TOKENS, set(SCREENS),
        )

    assert "redesign, not an edit" in str(caught.value)


def test_removal_fails_CLOSED_when_the_screens_are_unknown(wide):
    """If the inventory read failed, or a caller forgot to pass the screens, the
    guard has nothing to compare against. That must refuse, never allow."""
    with pytest.raises(SpecError) as caught:
        editor.compile_edits([{"op": "delete", "target": "1:3"}], wide, ROLES, TOKENS, set())

    assert "could not work out which frames are whole screens" in str(caught.value)


def test_the_generated_script_refuses_a_screen_at_runtime_too(wide):
    """Belt and braces: the inventory is a snapshot, and the canvas can move
    under it. The script re-checks against Figma itself."""
    code, _ = editor.compile_edits(
        [{"op": "replace", "target": "1:4", "spec": {"kind": "text", "value": "x"}}],
        wide, ROLES, TOKENS, set(SCREENS),
    )

    assert "target.type === 'PAGE'" in code
    assert "would empty the page" in code


def test_two_replaces_in_one_batch_do_not_collide(wide):
    """`const old` was emitted at the top level, so a second replace was a
    redeclaration and the whole script died before anything ran."""
    spec = {"kind": "text", "value": "x"}
    code, _ = editor.compile_edits(
        [{"op": "replace", "target": "1:3", "spec": spec},
         {"op": "replace", "target": "1:4", "spec": spec}],
        wide, ROLES, TOKENS, set(SCREENS),
    )

    assert compiles_as_async_body(code)


def test_a_legitimate_replace_and_delete_still_work(wide):
    """The guard has to leave the real job intact."""
    code, touched = editor.compile_edits(
        [{"op": "replace", "target": "1:4", "spec": {"kind": "text", "value": "x"}},
         {"op": "delete", "target": "1:3"}],
        wide, ROLES, TOKENS, set(SCREENS),
    )

    assert touched == ["1:4", "1:3"] and compiles_as_async_body(code)


def test_every_destructive_op_is_covered_by_the_guard():
    """A new op that removes something must not be addable without one."""
    import inspect

    source = inspect.getsource(editor._EditCompiler)
    for op in editor.DESTRUCTIVE_OPS:
        body = source.split(f"def {op}(self")[1].split("\n    def ")[0]
        assert "guard_removal" in body, f"{op} removes nodes but is not guarded"


# ---- pictures and prototype links ------------------------------------------
#
# The two things a real Figma design has that a static mockup does not. Both
# were added to edit mode for the same reason as everything else here: the user
# can ask for them, and rebuilding a screen to change one is not an edit.

from agent.assets import ImageAsset  # noqa: E402

HERO = ImageAsset(name="hero.png", key="hero", image_hash="hash-1", width=1600, height=900)
SCREEN_FRAMES = {"Login": "1:2", "Dashboard": "8:1"}


def compile_rich(resolve, *edits, assets=(HERO,), screens=None):
    return editor.compile_edits(
        list(edits), resolve, ROLES, TOKENS, {"1:2"},
        assets=list(assets), screens=dict(screens if screens is not None else SCREEN_FRAMES),
    )


def test_an_attached_picture_can_be_dropped_onto_an_existing_node(resolve):
    js, touched = compile_rich(resolve, {"op": "set_image", "target": "1:9", "asset": "hero.png"})

    assert "hash-1" in js and "'IMAGE'" in js
    assert touched == ["1:9"]
    assert compiles_as_async_body(js)


def test_set_image_with_no_attachment_says_what_to_do(resolve):
    with pytest.raises(SpecError, match="Attach an image"):
        compile_rich(resolve, {"op": "set_image", "target": "1:9"}, assets=())


def test_set_image_with_an_unknown_name_lists_the_real_ones(resolve):
    with pytest.raises(SpecError, match="hero.png"):
        compile_rich(resolve, {"op": "set_image", "target": "1:9", "asset": "banner.jpg"})


def test_a_button_can_be_made_to_navigate(resolve):
    js, _ = compile_rich(resolve, {"op": "set_interaction", "target": "1:9", "to": "Dashboard"})

    assert "setReactionsAsync(" in js
    assert '"destinationId": "8:1"' in js
    assert compiles_as_async_body(js)


def test_an_interaction_to_a_screen_that_does_not_exist_is_refused(resolve):
    """A dead prototype link looks wired and does nothing, which is worse than
    not wiring it: the user finds out by clicking."""
    with pytest.raises(SpecError, match="no screen"):
        compile_rich(resolve, {"op": "set_interaction", "target": "1:9", "to": "Checkout"})


def test_back_needs_no_screen_at_all(resolve):
    js, _ = compile_rich(
        resolve, {"op": "set_interaction", "target": "1:9", "action": "back"}, screens={}
    )

    assert '"BACK"' in js


def test_neither_new_op_can_remove_anything(resolve):
    """`DESTRUCTIVE_OPS` is the list the removal guard hangs off. Adding an op
    that removes something without adding it there is the hole that emptied a
    real user's file."""
    assert "set_image" not in editor.DESTRUCTIVE_OPS
    assert "set_interaction" not in editor.DESTRUCTIVE_OPS

    js, _ = compile_rich(
        resolve,
        {"op": "set_image", "target": "1:9", "asset": "hero.png"},
        {"op": "set_interaction", "target": "1:9", "to": "Dashboard"},
    )
    assert ".remove()" not in js


def test_an_inserted_section_does_not_end_the_batch_early(resolve):
    """The renderer's own `return` is rewritten when its body is inlined. That
    rewrite matched a fixed string and the return line has since grown a field
    -- so a missed match would silently discard every edit after an insert."""
    js, _ = compile_rich(
        resolve,
        {"op": "insert", "parent": "1:3", "spec": {"kind": "text", "value": "Hi"}},
        {"op": "set_name", "target": "1:9", "name": "Primary"},
    )

    assert "createdNodeIds: created" not in js  # the inlined return is gone
    assert js.rstrip().endswith(
        "return { createdNodeIds: madeIds, appliedEdits: applied, failedEdits: failed };"
    )
    assert compiles_as_async_body(js)
