"""The benchmark scorer. Pure logic -- no Figma, no model, no network.

The scorer is the only thing that will tell us whether a change to the agent
helped, so its own failure modes matter: it must not reward a run for a
dimension nobody measured, and it must actually notice a broken page.
"""
from __future__ import annotations

import json

import pytest

from agent.state import RunResult, StepResult
from bench import score as scoring
from bench.capture import walk
from bench.spec import TASKS_DIR, Criterion, load_all_tasks, load_task


def text(node_id, name, characters, size=16, width=200, height=24, styled=True):
    return {
        "id": node_id, "name": name, "type": "TEXT", "x": 0, "y": 0,
        "width": width, "height": height, "visible": True, "layoutMode": None,
        "characters": characters, "fontSize": size, "textStyled": styled,
        "hasSolidFill": True, "fillBound": True, "children": [],
    }


def frame(node_id, name, children=(), spacing=16, padding=(24, 24, 24, 24), bound=True):
    return {
        "id": node_id, "name": name, "type": "FRAME", "x": 0, "y": 0,
        "width": 1440, "height": 400, "visible": True, "layoutMode": "VERTICAL",
        "itemSpacing": spacing, "padding": list(padding),
        "hasSolidFill": True, "fillBound": bound, "children": list(children),
    }


def good_login_tree():
    """A design that satisfies the login task."""
    return frame("0:root", "Beacon — Page", [
        frame("1:1", "Header", [text("t:1", "Wordmark", "Beacon", size=32)]),
        frame("1:2", "Sign in card", [
            text("t:2", "Heading", "Sign in to Beacon", size=32),
            text("t:3", "Email label", "Email"),
            text("t:4", "Password label", "Password"),
            text("t:5", "Forgot link", "Forgot your password?", size=13),
            frame("1:3", "Primary button", [text("t:6", "Button label", "Sign in", size=15)]),
            frame("1:4", "Google button", [text("t:7", "Google label", "Continue with Google", size=15)]),
        ]),
    ])


def clean_result(steps=4):
    return RunResult(
        instruction="x", success=True, created_node_ids=["0:root"],
        failed_steps=[], warnings=[],
        step_results=[
            StepResult(f"step {i}", ok=True, section_name=f"Section {i}")
            for i in range(steps)
        ],
    )


# ---- task definitions -----------------------------------------------------


def test_every_task_file_loads_and_has_criteria():
    tasks = load_all_tasks()
    assert len(tasks) >= 6
    for task in tasks:
        assert task.instruction.strip()
        assert task.criteria, f"{task.task_id} has no acceptance criteria"
        for criterion in task.criteria:
            assert criterion.kind() != "unknown", f"{task.task_id}: {criterion.label}"


def test_every_criterion_pattern_compiles():
    """A bad regex (or a JSON escape that ate its backslashes) would silently
    fail every criterion using it and quietly deflate the score."""
    import re as _re

    for task in load_all_tasks():
        for criterion in task.criteria:
            for pattern in (criterion.any_text, criterion.any_name):
                if pattern:
                    _re.compile(pattern)  # raises on a malformed pattern


def test_task_ids_match_their_filenames():
    """Otherwise `--rescore` loads a different task than the run used."""
    for path in TASKS_DIR.glob("*.json"):
        assert json.loads(path.read_text(encoding="utf-8"))["task_id"] == path.stem


def test_loading_an_unknown_task_says_what_is_available():
    with pytest.raises(FileNotFoundError, match="login"):
        load_task("does-not-exist")


# ---- criteria -------------------------------------------------------------


def test_text_criterion_searches_text_nodes_only():
    tree = good_login_tree()
    nodes = walk(tree)
    assert scoring.criterion_met(Criterion("x", any_text="(?i)password"), tree, nodes)
    # "Sign in card" is a frame NAME, so a text search must not find it.
    assert not scoring.criterion_met(Criterion("x", any_text="Sign in card"), tree, nodes)


def test_name_criterion_searches_every_node_name():
    tree = good_login_tree()
    assert scoring.criterion_met(Criterion("x", any_name="(?i)button"), tree, walk(tree))


def test_min_sections_counts_direct_children_of_the_root():
    tree = good_login_tree()
    assert scoring.criterion_met(Criterion("x", min_sections=2), tree, walk(tree))
    assert not scoring.criterion_met(Criterion("x", min_sections=3), tree, walk(tree))


def test_min_nodes_counts_by_type():
    tree = good_login_tree()
    nodes = walk(tree)
    assert scoring.criterion_met(Criterion("x", min_nodes={"type": "TEXT", "n": 7}), tree, nodes)
    assert not scoring.criterion_met(Criterion("x", min_nodes={"type": "TEXT", "n": 99}), tree, nodes)


# ---- dimensions -----------------------------------------------------------


def test_a_good_login_design_scores_well():
    score = scoring.score_task(load_task("login"), good_login_tree(), clean_result())
    assert score.total > 80
    assert score.unmet == []


def test_a_design_missing_a_requirement_loses_requirement_points_only():
    tree = good_login_tree()
    # Strip the Google option out of the card.
    tree["children"][1]["children"] = [
        c for c in tree["children"][1]["children"] if "Google" not in c["name"]
    ]
    score = scoring.score_task(load_task("login"), tree, clean_result())

    assert "Google sign-in is offered" in score.unmet
    dims = {d.name: d.score for d in score.dimensions}
    assert dims["requirements"] < 1.0
    assert dims["design_system"] == 1.0  # still fully token-backed


def test_unmeasured_visual_quality_is_excluded_not_scored_zero():
    """Counting an unmeasured dimension as zero would make simply switching on
    a vision judge look like a large improvement."""
    score = scoring.score_task(load_task("login"), good_login_tree(), clean_result())
    visual = next(d for d in score.dimensions if d.name == "visual")
    assert visual.score is None
    assert score.measured_weight == pytest.approx(0.85)

    judged = scoring.score_task(load_task("login"), good_login_tree(), clean_result(), visual=1.0)
    assert judged.measured_weight == pytest.approx(1.0)
    assert judged.total >= score.total


def test_duplicate_sections_are_penalised():
    """The exact failure repair-mode retries were built to prevent."""
    duped = clean_result()
    duped.step_results = [
        StepResult("a", ok=True, section_name="Hero"),
        StepResult("b", ok=True, section_name="Hero"),
        StepResult("c", ok=True, section_name="Hero"),
        StepResult("d", ok=True, section_name="Footer"),
    ]
    clean = scoring.score_figma_correctness(clean_result())
    dirty = scoring.score_figma_correctness(duped)
    assert dirty.score < clean.score
    assert "2 duplicate section(s)" in dirty.detail


def test_failed_steps_and_placeholders_are_penalised():
    limping = RunResult(
        instruction="x", success=False, created_node_ids=[],
        failed_steps=["a", "b"], warnings=[],
        step_results=[
            StepResult("a", ok=False, summary="exhausted retries (placeholder added)"),
            StepResult("b", ok=False, summary="exhausted retries (placeholder added)"),
            StepResult("c", ok=True, section_name="Footer"),
            StepResult("d", ok=True, section_name="Hero"),
        ],
    )
    assert scoring.score_figma_correctness(limping).score < 0.5


def test_off_scale_spacing_lowers_the_layout_score():
    tidy = frame("0:root", "Root", [text("t:1", "T", "Hello")], spacing=16, padding=(24, 24, 24, 24))
    messy = frame("0:root", "Root", [text("t:1", "T", "Hello")], spacing=13, padding=(37, 19, 7, 23))
    assert scoring.score_layout(messy, walk(messy)).score < scoring.score_layout(tidy, walk(tidy)).score


def test_hardcoded_fills_lower_the_design_system_score():
    bound = frame("0:root", "Root", [text("t:1", "T", "Hello")], bound=True)
    loose = frame("0:root", "Root", [dict(text("t:1", "T", "Hello"), fillBound=False)], bound=False)
    assert scoring.score_design_system(walk(loose)).score == 0.0
    assert scoring.score_design_system(walk(bound)).score == 1.0


def test_off_ramp_font_sizes_lower_the_typography_score():
    on_ramp = frame("0:root", "Root", [text("t:1", "T", "Hi", size=32)])
    off_ramp = frame("0:root", "Root", [text("t:1", "T", "Hi", size=27)])
    assert scoring.score_typography(walk(off_ramp)).score < scoring.score_typography(walk(on_ramp)).score


def test_collapsed_text_is_caught_as_unreadable():
    broken = frame("0:root", "Root", [text("t:1", "T", "Invisible", width=0)])
    dim = scoring.score_typography(walk(broken))
    assert dim.score < 0.6
    assert "1/1 unreadable" in dim.detail


def test_a_dimension_with_nothing_to_measure_reports_none():
    empty = frame("0:root", "Root", [])
    assert scoring.score_typography(walk(empty)).score is None


def test_the_report_renders_without_a_visual_score():
    report = scoring.format_score(scoring.score_task(load_task("login"), good_login_tree(), clean_result()))
    assert "login:" in report
    assert "n/a" in report  # visual, not measured


# ---- the live-file inspector ----------------------------------------------

def test_mostly_empty_sections_are_flagged():
    """Vertical emptiness is what makes a generated page look unfinished, and
    every other check is blind to it: a 400px section holding one 40px label
    passes geometry cleanly."""
    from bench import inspect as inspect_mod

    tree = frame("0:root", "Root", [
        dict(frame("1:1", "Chart Area", [text("t:1", "Title", "Revenue", height=40)]), height=400),
        dict(frame("1:2", "Blank Band", []), height=300),
        dict(frame("1:3", "Tight", [dict(text("t:2", "Row", "Data"), height=380)]), height=400),
    ])
    lines = "\n".join(inspect_mod.report_sections(tree))

    assert "Chart Area" in lines and "mostly empty" in lines
    assert "EMPTY, no children" in lines
    # The well-filled section must NOT be flagged.
    tight = [ln for ln in inspect_mod.report_sections(tree) if "Tight" in ln][0]
    assert "empty" not in tight.lower()


def test_loose_nodes_outside_the_root_frame_are_listed():
    """Stray squares beside a design are nodes a script created and never
    parented, so they land on the page instead of inside a section."""
    from bench import inspect as inspect_mod

    page = [
        {"id": "0:root", "name": "Root", "type": "FRAME", "x": 0, "y": 0, "width": 1440, "height": 900},
        {"id": "9:1", "name": "Rectangle 12", "type": "RECTANGLE", "x": -40, "y": 10, "width": 16, "height": 16},
    ]
    orphans = inspect_mod.report_orphans(page, "0:root")

    assert len(orphans) == 1
    assert "Rectangle 12" in orphans[0]


def test_inspecting_a_file_we_did_not_build_excludes_correctness():
    """`bench.inspect` has no run history. Reporting a perfect figma_correctness
    from no evidence is exactly the flattery measured_weight exists to prevent.
    """
    score = scoring.score_task(load_task("login"), good_login_tree(), result=None)

    correctness = next(d for d in score.dimensions if d.name == "figma_correctness")
    assert correctness.score is None
    assert "no run history" in correctness.detail
    # requirements + layout + design_system + typography = 0.70
    assert score.measured_weight == pytest.approx(0.70)


def test_an_empty_run_history_is_also_treated_as_unmeasured():
    empty = RunResult(instruction="x", success=True, created_node_ids=[],
                      failed_steps=[], warnings=[], step_results=[])
    assert scoring.score_figma_correctness(empty).score is None
