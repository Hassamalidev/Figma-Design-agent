"""The declarative UI compiler.

Two things must hold, and the second is why this module exists at all:

1. The generated JavaScript must COMPILE, exactly as the plugin evals it.
2. It must be correct by construction -- the API mistakes the model made every
   single run (FILL before append, collapsed text, off-scale spacing) must be
   impossible to express, not merely discouraged.
"""
from __future__ import annotations

import re
import subprocess

import pytest

from agent import renderer

ROLES = {
    "background": "color/bg", "surface": "color/surface", "text": "color/ink",
    "text-muted": "color/slate", "accent": "color/indigo", "on-accent": "color/bg",
    "border": "color/line", "success": "color/green", "success-bg": "color/green-soft",
}


def compile_ok(spec, parent="1:2"):
    js, created = renderer.compile_spec(spec, parent, ROLES)
    return js, created


def node_js(spec):
    return compile_ok(spec)[0]


# ---- the generated JavaScript actually runs --------------------------------


def _node_available() -> bool:
    try:
        subprocess.run(["node", "--version"], capture_output=True, timeout=10)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _node_available(), reason="node not installed")
def test_generated_javascript_compiles_as_an_async_function_body():
    """The plugin evals this as `new AsyncFunction(code)`. A syntax slip must
    fail here, not halfway through a live run."""
    js, _ = compile_ok(
        {"kind": "section", "name": "KPI", "direction": "row", "children": [
            {"kind": "card", "children": [
                {"kind": "text", "style": "Caption", "value": "Total Revenue", "color": "text-muted"},
                {"kind": "text", "style": "Display", "value": "$128,430"},
                {"kind": "badge", "tone": "success", "label": "+12.5%"},
            ]},
            {"kind": "button", "label": "Export", "variant": "secondary"},
            {"kind": "input", "label": "Email", "placeholder": "you@co.com"},
            {"kind": "avatar"}, {"kind": "divider"}, {"kind": "box", "height": 220},
        ]}
    )
    probe = (
        "const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;\n"
        f"new AsyncFunction({js!r});\n"
    ).replace("\\n", "\\n")
    result = subprocess.run(
        ["node", "-e", "const AsyncFunction=Object.getPrototypeOf(async function(){}).constructor;"
         "const code=process.argv[1];new AsyncFunction(code);", "--", js],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


# ---- correct by construction ------------------------------------------------


def test_text_hugs_by_default_so_it_can_never_collapse():
    """A hugging text node sizes to its content, so the vertical-column-of-
    letters bug is not expressible."""
    js = node_js({"kind": "text", "value": "Recent Transactions", "style": "Subheading"})
    assert "textAutoResize = 'WIDTH_AND_HEIGHT'" in js


def test_wrapping_text_fills_first_then_grows_downward():
    js = node_js({"kind": "text", "value": "long copy", "wrap": True})
    fill_at = js.index("setFill(")
    height_at = js.index("textAutoResize = 'HEIGHT'")
    assert fill_at < height_at, "FILL must be applied before switching to HEIGHT autoresize"


def test_fill_is_only_ever_applied_after_appendchild():
    """`FILL can only be set on children of auto-layout frames` was the single
    most repeated error in live runs."""
    js = node_js({"kind": "section", "children": [{"kind": "text", "value": "hi"}]})
    for match in re.finditer(r"setFill\((n\d+)\)", js):
        var = match.group(1)
        append = js.index(f"appendChild({var})")
        assert append < match.start(), f"{var}: FILL applied before it was appended"


def test_fill_checks_the_parent_is_auto_layout_at_runtime():
    js = node_js({"kind": "text", "value": "hi"})
    assert "layoutMode' in p" in js and "!== 'NONE'" in js


def test_frames_hug_their_content_so_they_cannot_clip_it():
    js = node_js({"kind": "card", "children": [{"kind": "text", "value": "x"}]})
    assert "primaryAxisSizingMode = 'AUTO'" in js
    assert "counterAxisSizingMode = 'AUTO'" in js


def test_every_font_used_is_loaded_before_any_text_is_created():
    js = node_js({"kind": "col", "children": [
        {"kind": "text", "style": "Display", "value": "a"},
        {"kind": "text", "style": "Caption", "value": "b"},
    ]})
    assert "loadFontAsync({ family: \"Inter\", style: \"Bold\" })" in js
    assert "loadFontAsync({ family: \"Inter\", style: \"Regular\" })" in js
    assert js.index("loadFontAsync") < js.index("createText()")


def test_spacing_is_a_name_so_off_scale_values_cannot_be_expressed():
    js = node_js({"kind": "section", "gap": "md", "padding": "lg", "children": []})
    assert "itemSpacing = 16" in js
    assert "paddingTop = 24" in js


def test_an_unknown_spacing_name_falls_back_to_the_scale():
    js = node_js({"kind": "section", "gap": "37px", "children": []})
    assert "itemSpacing = 16" in js  # the md default, never 37


def test_colour_is_applied_as_a_style_never_a_hex():
    js = node_js({"kind": "text", "value": "x", "color": "accent"})
    assert 'applyFill(n1, "color/indigo")' in js
    assert "#" not in js  # no hardcoded colour anywhere


def test_an_unknown_colour_role_is_skipped_rather_than_guessed():
    js = node_js({"kind": "text", "value": "x", "color": "chartreuse"})
    # The helper is always defined in the preamble; what must be absent is a CALL.
    assert not re.search(r"await applyFill\(n\d+", js)


def test_text_styles_are_applied_by_name():
    js = node_js({"kind": "text", "style": "Heading", "value": "x"})
    assert 'applyTextStyle(n1, "Heading")' in js


def test_an_unknown_text_style_falls_back_to_body():
    js = node_js({"kind": "text", "style": "Enormous", "value": "x"})
    assert 'applyTextStyle(n1, "Body")' in js


def test_the_script_returns_every_node_it_created():
    js, created = compile_ok({"kind": "section", "children": [
        {"kind": "text", "value": "a"}, {"kind": "text", "value": "b"},
    ]})
    assert len(created) == 3  # section + 2 texts
    for var in created:
        assert f"created.push({var}.id)" in js
    assert "return { createdNodeIds: created };" in js


def test_a_primary_button_puts_readable_text_on_the_accent():
    js = node_js({"kind": "button", "label": "Create Report", "variant": "primary"})
    assert 'applyFill(n1, "color/indigo")' in js      # the button surface
    assert 'applyFill(n2, "color/bg")' in js          # on-accent label


def test_composite_kinds_expand_into_real_structure():
    _, created = compile_ok({"kind": "input", "label": "Email", "placeholder": "you@co.com"})
    assert len(created) == 4  # wrapper + label + box + placeholder


# ---- errors are the model's to fix -----------------------------------------


def test_an_unknown_kind_names_the_valid_ones():
    with pytest.raises(renderer.SpecError, match="Valid kinds"):
        compile_ok({"kind": "carousel"})


def test_a_non_object_node_is_rejected_clearly():
    with pytest.raises(renderer.SpecError, match="must be an object"):
        compile_ok({"kind": "col", "children": ["just a string"]})


def test_a_runaway_spec_is_refused_before_it_reaches_figma():
    huge = {"kind": "col", "children": [{"kind": "text", "value": str(i)} for i in range(600)]}
    with pytest.raises(renderer.SpecError, match="more than"):
        compile_ok(huge)


# ---- role mapping -----------------------------------------------------------


def test_roles_map_to_the_tokens_the_harness_actually_created():
    palette = [
        ("color/warm-sand", "#E8DCC8", "background (page/section fill)"),
        ("color/deep-navy", "#0B1F3A", "text (body copy on a light background)"),
        ("color/coral", "#FF6B4A", "accent (buttons, links, emphasis)"),
    ]
    roles = renderer.role_map(palette)

    assert roles["background"] == "color/warm-sand"
    assert roles["text"] == "color/deep-navy"
    assert roles["accent"] == "color/coral"
    # on-accent must be the light colour, so accent buttons stay readable.
    assert roles["on-accent"] == "color/warm-sand"


def test_role_map_never_yields_an_empty_style_name():
    assert all(v for v in renderer.role_map([]).values())


def test_a_section_stretches_to_the_page_width():
    """A section that hugs leaves a ragged column down the left of the page."""
    js, created = compile_ok({"kind": "section", "name": "Hero", "children": []})
    assert f"setFill({created[0]})" in js


def test_buttons_and_badges_hug_their_label_instead_of_stretching():
    js, created = compile_ok({"kind": "row", "children": [
        {"kind": "button", "label": "Export"},
        {"kind": "badge", "tone": "success", "label": "Completed"},
    ]})
    row, button, badge = created[0], created[1], created[3]
    assert f"setFill({row})" in js
    assert f"setFill({button})" not in js
    assert f"setFill({badge})" not in js


# ---- a real trace: 25 calls rejected for the wrong envelope ---------------
#
# The schema wants {"spec": {...}}, the prompt taught the bare tree, and a 20B
# model sent the tree as the arguments object. Every one of those calls was a
# correct UI tree rejected as "Missing required argument 'spec'" -- roughly
# half of every step's tool-call budget, in a 5-step run.

import json as _json

from tools.registry import coerce_spec

_TREE = {"kind": "section", "name": "Sign in", "children": [{"kind": "text", "value": "hi"}]}


@pytest.mark.parametrize(
    "arguments",
    [
        {"spec": _TREE},                       # the documented shape
        _TREE,                                 # the tree AS the arguments
        {"spec": _json.dumps(_TREE)},          # spec handed over as JSON text
        {"spec": {"spec": _TREE}},             # wrapped twice
        {"ui": _TREE},                         # a plausible other key
        {"spec": [_TREE]},                     # a bare list of sections
        _json.dumps({"spec": _TREE}),          # the whole argument blob as text
    ],
)
def test_every_envelope_a_model_actually_sends_finds_the_tree(arguments):
    assert coerce_spec(arguments) is not None


def test_something_that_is_not_a_ui_tree_is_still_rejected():
    """Tolerance must not become guessing -- an unreadable call has to come
    back as a readable error, not a silently invented section."""
    assert coerce_spec({"code": "figma.createFrame()"}) is None
    assert coerce_spec({"query": "how do I set a fill"}) is None
    assert coerce_spec("not json at all") is None
    assert coerce_spec(None) is None


def test_the_kinds_the_model_kept_asking_for_now_render():
    """`unknown kind 'ellipse'` / `'frame'` / `'checkbox'` / `'image'` cost a
    turn each. They are all things the renderer can already draw."""
    spec = {"kind": "frame", "name": "Auth", "children": [
        {"kind": "ellipse", "size": 32},
        {"kind": "heading", "style": "Heading", "value": "Welcome back"},
        {"kind": "image", "name": "Glow", "height": 120},
        {"kind": "checkbox", "label": "Remember me"},
        {"kind": "separator"},
        {"kind": "cta", "label": "Log in"}]}

    code, created = renderer.compile_spec(spec, "1:2", {"accent": "color/a"})

    assert len(created) >= 6
    assert "Remember me" in code


def test_a_spec_may_name_a_real_token_when_no_role_expresses_it():
    """`background` is by definition the LIGHTEST colour, so a dark panel was
    unbuildable: the model asked for it and got white."""
    spec = {"kind": "section", "name": "Hero", "background": "color/deep-background",
            "children": [{"kind": "text", "value": "AI at work"}]}

    code, _ = renderer.compile_spec(
        spec, "1:2", {"text": "color/ink"}, token_names=["color/deep-background"]
    )

    assert '"color/deep-background"' in code


def test_a_role_is_never_aliased_onto_something_it_vanishes_against():
    """border -> surface -> background painted every divider white on white."""
    info = [("color/paper", "#FFFFFF", "background (the page fill)"),
            ("color/ink", "#111111", "text (body copy)")]

    roles = renderer.role_map(info)

    assert roles.get("border") != "color/paper"


def test_a_label_is_not_written_twice_above_its_own_input():
    """`input` renders its own label, so a spec that also writes the label out
    produces "Email" stacked on "Email". The vision critic caught it correctly
    -- and then a whole repair budget went on a judgement call that cannot fail
    a step, so nothing was ever fixed."""
    spec = {"kind": "col", "name": "Form", "children": [
        {"kind": "text", "style": "Caption", "value": "Email"},
        {"kind": "input", "label": "Email", "placeholder": "you@company.com"},
        {"kind": "text", "style": "Caption", "value": "Password"},
        {"kind": "input", "label": "Password", "placeholder": "........"},
        {"kind": "text", "style": "Caption", "value": "Forgot password?"}]}

    code, _ = renderer.compile_spec(spec, "1:2", {})

    assert code.count('"Email"') == 1
    assert code.count('"Password"') == 1
    # A link that merely sits near a field is not a duplicate label.
    assert '"Forgot password?"' in code


def test_a_label_before_a_DIFFERENT_input_is_kept():
    spec = {"kind": "col", "children": [
        {"kind": "text", "value": "Account details"},
        {"kind": "input", "label": "Email"}]}

    code, _ = renderer.compile_spec(spec, "1:2", {})

    assert '"Account details"' in code and '"Email"' in code
