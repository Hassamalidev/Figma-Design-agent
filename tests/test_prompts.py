"""The step prompt must carry the design context, not just the step sentence.

The planner is deliberately told to keep each step under 20 words and to strip
out colours, fonts and pixel values -- so if the brief and the plan outline do
not travel with the step, the component making every visual decision is the
only one in the pipeline that never sees the design.
"""
from __future__ import annotations

from agent.prompts import (
    repair_note,
    MAX_BRIEF_CHARS,
    design_context_note,
    plan_outline_note,
    step_user_message,
)

BRIEF = "Palette: Deep Navy (#0B1F3A), Warm Sand (#E8DCC8). Sections: nav, hero, footer."
PLAN = [
    "Add the nav bar into the root frame.",
    "Add the hero section into the root frame.",
    "Add the footer into the root frame.",
]


def test_step_prompt_carries_instruction_and_brief():
    prompt = step_user_message(
        "Add the hero section into the root frame.",
        docs="",
        state_summary="(none)",
        instruction="a landing page for a sailing school",
        brief=BRIEF,
    )
    assert "a landing page for a sailing school" in prompt
    assert "#0B1F3A" in prompt
    assert "Warm Sand" in prompt


def test_step_prompt_marks_the_current_step_in_the_plan():
    prompt = step_user_message(
        PLAN[1], docs="", state_summary="(none)", plan=PLAN, step_index=2
    )
    assert ">>> THIS STEP: Add the hero section" in prompt
    assert "[already built] Add the nav bar" in prompt
    assert "[comes later] Add the footer" in prompt


def test_outline_is_skipped_for_a_single_step_plan():
    """One step has no neighbours to sit between -- the outline is pure noise."""
    assert plan_outline_note(["only step"], 1) == ""
    assert plan_outline_note([], 1) == ""
    assert plan_outline_note(None, 1) == ""


def test_long_brief_is_truncated_so_it_cannot_crowd_out_the_docs():
    note = design_context_note("x", "line of brief text\n" * 400)
    assert len(note) < MAX_BRIEF_CHARS + 600
    assert "(brief truncated)" in note


def test_missing_context_degrades_quietly():
    """A failed enhancement must not produce a prompt full of empty headings."""
    assert design_context_note("", "") == ""
    prompt = step_user_message("do the thing", docs="", state_summary="(none)")
    assert "WHAT YOU ARE BUILDING" not in prompt
    assert "do the thing" in prompt


def test_step_prompt_still_carries_the_existing_harness_facts():
    """The new block must not have displaced root id, tokens or font styles."""
    prompt = step_user_message(
        "Add the hero section into the root frame.",
        docs="some docs",
        state_summary="(none)",
        root_frame_id="1:5",
        token_names=["color/accent"],
        text_style_names=["Heading"],
        font_styles=["Semi Bold"],
        instruction="a landing page",
        brief=BRIEF,
        plan=PLAN,
        step_index=2,
    )
    assert "1:5" in prompt
    assert "color/accent" in prompt
    assert "Heading" in prompt
    assert "Semi Bold" in prompt
    assert "some docs" in prompt


# ---- repair mode ----------------------------------------------------------

def test_repair_prompt_leads_with_the_fix_not_the_build():
    prompt = step_user_message(
        "Add the hero section into the root frame.",
        docs="", state_summary="(none)", root_frame_id="1:2",
        prior_node_ids=["1:5", "1:6"],
        prior_defects=["[collapsed] Headline: has no area (0x0)"],
    )
    assert prompt.startswith("CORRECTING THE PREVIOUS ATTEMPT")
    assert "1:5" in prompt and "1:6" in prompt
    assert "has no area" in prompt
    assert "MODIFIES the existing nodes" in prompt


def test_repair_prompt_does_not_tell_the_model_to_append_a_section():
    """The standard root note says 'append each section into it' -- following
    that during a repair is exactly how a page ends up with two heroes.
    """
    prompt = step_user_message(
        "Add the hero section into the root frame.",
        docs="", state_summary="(none)", root_frame_id="1:2",
        prior_node_ids=["1:5"],
        prior_defects=["[collapsed] Headline: has no area (0x0)"],
    )
    assert "Append each section into it" not in prompt
    assert "do not append anything to it" in prompt
    assert "1:2" in prompt  # the root id is still available if it's needed


def test_a_thrown_script_with_nothing_landed_is_not_a_repair():
    """Figma scripts are atomic: if nothing was created, rebuilding is right."""
    assert repair_note([], prior_error="boom") == ""
    assert repair_note(None, ["some defect"]) == ""


def test_repair_after_a_thrown_script_explains_what_survived():
    note = repair_note(["1:1"], prior_error="Property lineHeight failed validation")
    assert "1:1" in note
    assert "lineHeight" in note
    assert "continue from them rather than rebuilding" in note
