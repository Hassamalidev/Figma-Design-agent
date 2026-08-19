"""The edit compiler: what it produces, and above all what it refuses.

Weighted towards refusals on purpose. A bad create leaves an ugly section next
to the good ones; a bad edit damages work the user already has, so the checks
that stop an edit reaching Figma matter more than the ones that shape it.

The generated JavaScript is compiled exactly as the plugin evaluates it.
"""
from __future__ import annotations

import functools
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

    assert "whole screen" in str(caught.value)


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
