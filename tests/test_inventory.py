"""The canvas index edit mode targets: reading it, listing it, finding in it.

No Figma and no model. The reader's JavaScript is compiled the same way the
plugin evaluates it, so a syntax slip fails here rather than mid-run.
"""
from __future__ import annotations

import subprocess

import pytest

from agent import critic, inventory


def compiles_as_async_body(code: str) -> bool:
    result = subprocess.run(
        ["node", "-e", "new (Object.getPrototypeOf(async function(){}).constructor)(process.argv[1])", "--", code],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


PAGE = {
    "pageId": "0:1",
    "pageName": "Nexora",
    "selection": ["1:9"],
    "roots": [
        {
            "id": "1:2", "name": "Login", "type": "FRAME", "width": 1440, "height": 900,
            "layoutMode": "VERTICAL", "fill": {"r": 1, "g": 1, "b": 1},
            "children": [
                {
                    "id": "1:3", "name": "Auth Card", "type": "FRAME", "width": 440,
                    "height": 520, "layoutMode": "VERTICAL", "fill": {"r": 1, "g": 1, "b": 1},
                    "children": [
                        {"id": "1:4", "name": "Heading", "type": "TEXT", "width": 300,
                         "height": 40, "characters": "Welcome back", "fontSize": 32,
                         "children": []},
                        {"id": "1:9", "name": "Button / Log in", "type": "FRAME",
                         "width": 372, "height": 48,
                         "fill": {"r": 0.42, "g": 0.36, "b": 0.9}, "children": [
                             {"id": "1:10", "name": "Text", "type": "TEXT", "width": 60,
                              "height": 20, "characters": "Log in", "fontSize": 15,
                              "children": []}]},
                    ],
                }
            ],
        },
        {
            "id": "2:2", "name": "Sign Up", "type": "FRAME", "width": 1440, "height": 900,
            "children": [
                {"id": "2:9", "name": "Button / Create account", "type": "FRAME",
                 "width": 372, "height": 48, "children": []}
            ],
        },
    ],
}


@pytest.fixture
def canvas():
    return inventory.build(PAGE)


def test_the_reader_script_compiles_as_the_plugin_runs_it():
    assert compiles_as_async_body(inventory.inventory_script())


def test_edit_mode_reads_the_canvas_with_the_SAME_reader_as_the_gate():
    """A second reader would drift, and "what the agent sees" would stop
    matching "what is checked"."""
    assert critic.NODE_READER_JS in inventory.inventory_script()
    assert "function describe" in critic.NODE_READER_JS


def test_every_node_is_addressable_by_its_real_id(canvas):
    assert canvas.by_id("1:10").text == "Log in"
    assert canvas.by_id("2:9").screen == "Sign Up"
    assert canvas.by_id("1:4").parent_id == "1:3"


def test_the_listing_leads_with_the_id(canvas):
    """The id is the only part the model must copy exactly. Burying it after the
    name made it likelier to be paraphrased."""
    lines = [line.strip() for line in inventory.format_listing(canvas).splitlines()]

    assert lines[0].startswith("PAGE")
    assert lines[1].startswith("1:2  FRAME")
    assert any(line.startswith("1:4  TEXT") and "Welcome back" in line for line in lines)


def test_scoping_the_listing_to_a_selection_keeps_what_is_inside_it(canvas):
    """Selecting a card means the card AND its contents -- a listing of one bare
    frame tells the model nothing it can act on."""
    listing = inventory.format_listing(canvas, ["1:9"])

    assert "1:9" in listing and "1:10" in listing   # the button and its label
    assert "1:4" not in listing                      # the heading is out of scope


def test_find_matches_by_name_text_and_type(canvas):
    assert inventory.find(canvas, {"name": "Button"}) == ["1:9", "2:9"]
    assert inventory.find(canvas, {"text": "Log in"}) == ["1:10"]
    assert inventory.find(canvas, {"type": "TEXT"}) == ["1:4", "1:10"]


def test_find_can_be_narrowed_to_one_screen(canvas):
    assert inventory.find(canvas, {"name": "Button", "screen": "Sign Up"}) == ["2:9"]


def test_an_exact_name_outranks_a_partial_one():
    page = {"roots": [{"id": "1:1", "name": "Root", "type": "FRAME", "children": [
        {"id": "1:2", "name": "Primary Button Wrapper", "type": "FRAME", "children": []},
        {"id": "1:3", "name": "Button", "type": "FRAME", "children": []}]}]}

    assert inventory.find(inventory.build(page), {"name": "button"})[0] == "1:3"


def test_a_hallucinated_id_is_refused_with_something_actionable(canvas):
    """Figma's own error for a missing node names neither the bad id nor the
    real ones, so a model that gets it just invents another."""
    ids, error = inventory.resolve(canvas, ["1:9", "9:99"])

    assert ids == []
    assert "9:99" in error and "canvas listing" in error


def test_resolve_accepts_an_id_a_list_or_a_selector(canvas):
    assert inventory.resolve(canvas, "1:9") == (["1:9"], "")
    assert inventory.resolve(canvas, ["1:9", "2:9"]) == (["1:9", "2:9"], "")
    assert inventory.resolve(canvas, {"name": "Button"})[0] == ["1:9", "2:9"]


def test_a_selector_matching_nothing_is_an_error_not_an_empty_edit(canvas):
    """Silently applying zero edits reports success and changes nothing, which
    is the most confusing possible outcome."""
    ids, error = inventory.resolve(canvas, {"name": "Carousel"})

    assert ids == [] and "nothing on the canvas matches" in error


def test_a_huge_page_is_capped_and_says_so():
    """A model that cannot see a node will confidently conclude it does not
    exist, so the truncation has to be visible."""
    kids = [{"id": f"1:{i}", "name": f"N{i}", "type": "FRAME", "width": 50, "height": 50,
             "children": []} for i in range(inventory.MAX_LISTED_NODES + 50)]
    canvas = inventory.build({"roots": [{"id": "1:0", "name": "Root", "type": "FRAME",
                                         "width": 100, "height": 100, "children": kids}]})

    assert canvas.truncated
    assert len(canvas.nodes) == inventory.MAX_LISTED_NODES
    assert "only the first" in inventory.format_listing(canvas)


def test_an_empty_page_says_so_rather_than_returning_nothing():
    canvas = inventory.build({"pageName": "Blank", "roots": []})

    assert canvas.is_empty()
    assert inventory.format_listing(canvas) == "(the page is empty)"
