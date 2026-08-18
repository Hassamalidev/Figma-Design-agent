"""Requirement coverage: did the design contain what was ASKED for?

The rule that matters most here is the negative one -- a false "missing" tells
the user their working design is broken, which is worse than not checking at
all. So most of these tests are about what must NOT be reported.
"""
from __future__ import annotations

from agent import requirements


def node(name="Frame", characters="", children=None):
    return {"name": name, "characters": characters, "children": children or []}


def labels(instruction):
    return [r.label for r in requirements.expected_requirements(instruction)]


# ---- what gets asserted ---------------------------------------------------


def test_requirements_come_from_the_instruction_not_from_thin_air():
    found = labels("a mobile sign-in screen with email + password and a Google button")

    assert "password field" in found
    assert "email field" in found
    assert "Google sign-in" in found
    # Never assumed: a sign-in screen might legitimately have no search or table.
    assert "search field" not in found
    assert "data table" not in found


def test_nothing_is_asserted_for_an_instruction_that_names_nothing_checkable():
    """No requirements is a real answer. A made-up score is not."""
    coverage = requirements.check_coverage("make it look nicer", node())

    assert coverage.expected == 0
    assert coverage.ratio is None
    assert not coverage.satisfied_nothing


def test_quoted_copy_is_a_literal_requirement():
    found = labels("""a screen with a 'Welcome back' heading""")

    assert 'copy "Welcome back"' in found


# ---- checking against the finished tree -----------------------------------


def test_a_built_requirement_is_met_from_either_the_node_name_or_its_text():
    tree = node("Root", children=[
        node("Password Field"),                       # found by NAME
        node("Label", characters="Email address"),    # found by TEXT
    ])

    coverage = requirements.check_coverage("sign-in with email and password", tree)

    assert set(coverage.missing) == set()
    assert coverage.ratio == 1.0


def test_a_silently_dropped_requirement_is_reported():
    """The failure this whole module exists for: a sign-in screen with no
    password field that every other gate passes."""
    tree = node("Root", children=[
        node("Email Field"),
        node("Button", characters="Sign in"),
    ])

    coverage = requirements.check_coverage(
        "a sign-in screen with email and password fields", tree
    )

    assert "password field" in coverage.missing
    assert "email field" in coverage.met


def test_matching_is_case_insensitive_and_reads_nested_nodes():
    tree = node("Root", children=[node("Card", children=[node("PASSWORD input")])])

    coverage = requirements.check_coverage("password", tree)

    assert coverage.missing == []


# ---- the one result confident enough to fail a run ------------------------


def test_a_design_matching_none_of_the_instruction_is_a_failure():
    coverage = requirements.check_coverage(
        "a dashboard with a sidebar, a chart and a data table", node("Root")
    )

    assert coverage.satisfied_nothing
    assert coverage.expected >= requirements.MIN_REQUIREMENTS_TO_JUDGE


def test_one_missing_item_is_a_flaw_not_a_failure():
    tree = node("Root", children=[
        node("Sidebar"), node("Revenue Chart"), node("Users table"),
    ])

    coverage = requirements.check_coverage(
        "a dashboard with a sidebar, a chart, a data table and pagination", tree
    )

    assert coverage.missing == ["pagination"]
    assert not coverage.satisfied_nothing


def test_too_few_requirements_to_judge_never_fails_a_run():
    """One unmatched keyword is not evidence the whole design is wrong."""
    coverage = requirements.check_coverage("a screen with a chart", node("Root"))

    assert coverage.missing == ["chart"]
    assert not coverage.satisfied_nothing


def test_every_pattern_compiles_and_its_own_evidence_matches_its_trigger():
    """A trigger whose evidence can never match would report every design as
    broken. Cheap to guarantee, expensive to discover in a live run."""
    import re

    for label, trigger, evidence in requirements._ELEMENTS:
        re.compile(trigger)
        re.compile(evidence)
        assert re.search(evidence, label, re.IGNORECASE) or re.search(
            evidence, trigger.replace("\\b", "").replace("?", ""), re.IGNORECASE
        ), f"{label}: evidence cannot match anything its own trigger describes"
