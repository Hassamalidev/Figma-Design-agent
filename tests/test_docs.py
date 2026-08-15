"""query_docs must never answer with silence.

Live runs showed the model rephrasing the same search up to 6 times in a row
when a query returned an empty string, burning the step's entire turn budget
without touching the canvas.
"""
from __future__ import annotations

from tools.docs import NO_MATCH, query_docs


def test_a_matching_query_returns_real_docs():
    result = query_docs("load font before setting characters")

    assert result != NO_MATCH
    assert "loadFontAsync" in result


def test_a_query_with_no_match_says_so_and_says_stop():
    result = query_docs("zzzz nonexistent quantum blockchain topic")

    assert result == NO_MATCH
    assert "execute_figma_js" in result  # points at the way forward


def test_hallucinated_variable_apis_are_documented():
    """The exact APIs a live run invented should now be retrievable."""
    result = query_docs("createVariableSet getVariableByName variables")

    assert "createVariableCollection" in result


# ---- the gotchas are carried, not searched for ----------------------------

def test_query_docs_is_not_offered_as_a_tool():
    """It cost two or three round trips per step to retrieve a 4k-token corpus
    that now simply lives in the system prompt."""
    from tools.registry import TOOL_SCHEMAS

    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert "query_docs" not in names
    assert "execute_figma_js" in names


def test_dispatch_still_answers_a_hallucinated_doc_call():
    """A small model may call it from memory; documentation beats an error."""
    from tools.registry import dispatch

    result = dispatch("query_docs", {"query": "load font style"}, bridge=None)
    assert result["ok"] is True
    assert result["result"]


def test_the_system_prompt_carries_the_whole_gotchas_reference():
    from agent.prompts import system_prompt
    from knowledge.index import gotchas_text

    prompt = system_prompt(gotchas_text())
    assert "REFERENCE: FIGMA PLUGIN API" in prompt
    # Traps the model cannot infer, now always in context.
    assert "createComponentSet" in prompt
    assert "Semi Bold" in prompt
    assert "0-1" in prompt


def test_system_prompt_without_gotchas_is_unchanged():
    from agent.prompts import SYSTEM_PROMPT, system_prompt

    assert system_prompt("") == SYSTEM_PROMPT


def test_per_step_retrieval_pulls_typings_not_the_inlined_gotchas():
    """Retrieving gotchas again per step would spend the same context twice."""
    from knowledge.index import retrieve

    typings_only = retrieve("text node auto resize", sources=("api_types.d.ts",))
    assert "gotchas.md" not in typings_only
