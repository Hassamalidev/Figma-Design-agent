"""Rules every script this project generates must obey, checked in one place.

The suite already compiles generated JavaScript, which catches syntax. It
cannot catch a call that PARSES and then throws inside Figma -- and one did:
`figma.getNodeById` is the sync getter, which this plugin's
`documentAccess: "dynamic-page"` makes throw. It threw after a `replace` had
already removed the old node, so the edit reported failure with the original
gone.

The gotchas in knowledge/gotchas.md are rules for the MODEL. These are the same
rules applied to the harness's own output, which is the code that actually runs
most of the time.
"""
from __future__ import annotations

import functools

import pytest

from agent import assets, critic, editor, interactions, inventory, renderer, scaffold

PAGE = {
    "roots": [{
        "id": "1:2", "name": "Login", "type": "FRAME", "width": 1440, "height": 900,
        "layoutMode": "VERTICAL", "children": [
            {"id": "1:3", "name": "Card", "type": "FRAME", "width": 440, "height": 520,
             "layoutMode": "VERTICAL", "children": [
                 {"id": "1:4", "name": "Heading", "type": "TEXT", "width": 300, "height": 40,
                  "characters": "Welcome", "fontSize": 32, "children": []}]}]}]
}
SPEC = {"kind": "card", "name": "Card", "children": [
    {"kind": "text", "style": "Heading", "value": "Welcome back"},
    {"kind": "input", "label": "Email", "placeholder": "you@co.com"},
    {"kind": "button", "label": "Sign in", "variant": "primary"}]}


def every_generated_script() -> dict[str, str]:
    """One of everything this project sends to Figma."""
    resolve = functools.partial(inventory.resolve, inventory.build(PAGE))
    roles = {"accent": "color/a", "surface": "color/s", "text": "color/t", "background": "color/b"}
    edits = [
        {"op": "set_fill", "target": "1:3", "color": "accent"},
        {"op": "set_text", "target": "1:4", "value": "Hello"},
        {"op": "delete", "target": "1:4"},
        {"op": "replace", "target": "1:3", "spec": SPEC},
        {"op": "insert", "parent": "1:2", "index": 0, "spec": SPEC},
    ]
    return {
        "renderer": renderer.compile_spec(SPEC, "1:2", roles, replace_ids=["1:9"])[0],
        "editor": editor.compile_edits(edits, resolve, roles, list(roles.values()), {"1:2"})[0],
        "inventory": inventory.inventory_script(),
        "layout": critic.layout_script("1:2"),
        "tokens": scaffold.build_token_script([("accent", "#6C5CE7")]),
        "text_styles": scaffold.build_text_style_script(),
        "screens": scaffold.build_screen_frames_script([{"name": "S", "x": 0, "y": 0, "width": 1440}]),
        "placeholder": scaffold.build_placeholder_section_script("1:2", "hero"),
        "remove": scaffold.build_remove_nodes_script(["1:9"]),
        "hug_fix": scaffold.build_hug_fix_script("1:2"),
        "bind": scaffold.build_bind_fills_script(["1:3"], [("accent", "#6C5CE7")]),
        "clear_page": scaffold.build_clear_page_script(),
        "screen_content": scaffold.build_screen_content_script(["1:2", "1:3"]),
        "fit_screens": scaffold.build_fit_screens_script(
            [{"id": "1:2", "viewport": 1024}, {"id": "1:3", "viewport": 1024}]
        ),
        "asset_upload": assets.build_upload_script(b"\x89PNG\r\n\x1a\n"),
        "prototype_read": interactions.build_candidates_script([{"id": "1:2", "name": "Login"}]),
        "prototype_apply": interactions.build_apply_script(
            [interactions.Link(source_id="1:9", label="Sign in", destination_id="1:20")]
        ),
        "prototype_flows": interactions.build_flow_script(
            [{"id": "1:2", "name": "Login", "start": True, "scrolls": True}]
        ),
    }


SCRIPTS = every_generated_script()


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_no_generated_script_uses_a_banned_sync_api(name):
    """`documentAccess: "dynamic-page"` makes each of these throw at runtime --
    which parses fine and so slips past a compile check."""
    banned = {
        "figma.getNodeById(": "use await figma.getNodeByIdAsync(id)",
        "figma.currentPage =": "use await figma.setCurrentPageAsync(page)",
        "figma.notify(": "it throws in this sandbox; return instead",
        "figma.createPage(": "screens are frames, never pages",
        "createVariableSet": "does not exist",
        "figma.createStyle(": "does not exist",
    }
    script = SCRIPTS[name]
    for call, fix in banned.items():
        assert call not in script, f"{name} uses {call} -- {fix}"


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_no_generated_script_uses_syntax_the_sandbox_rejects(name):
    """Optional chaining and nullish coalescing fail with "unexpected token"."""
    script = SCRIPTS[name]
    assert "?." not in script
    assert "??" not in script


@pytest.mark.parametrize(
    "name", sorted(n for n, s in SCRIPTS.items() if ".remove()" in s and n != "clear_page")
)
def test_every_script_that_removes_nodes_refuses_a_whole_screen(name):
    """Emptying the canvas is something the user asks for behind a confirmation
    dialog. Nothing else -- no cleanup pass, no repair, no edit -- may do it,
    and an edit run emptied a real file by doing exactly that.

    The guard is in the generated JAVASCRIPT, not only in the Python that calls
    it, so a future caller cannot reintroduce the hole by forgetting a check.
    """
    assert "parent.type === 'PAGE'" in SCRIPTS[name], (
        f"{name} removes nodes without checking it is not taking a whole screen"
    )


def test_the_clear_page_script_is_the_one_deliberate_exception():
    """It exists to empty a canvas, on purpose, behind a confirmation."""
    assert ".remove()" in SCRIPTS["clear_page"]
