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


# ---- retrieval quality ----------------------------------------------------
#
# These are regressions, not aspirations: every query below returned the WRONG
# chunk under the previous raw-overlap scorer, because counting matched words
# rewards long chunks. The worst case was "line height on a text node", which
# retrieved the typings file's PREAMBLE -- the longest chunk in the corpus and
# the one carrying the least information.


def top_heading(query: str) -> str:
    from knowledge.index import retrieve

    result = retrieve(query, sources=("api_types.d.ts",), max_chars=400)
    return result.splitlines()[0] if result else ""


def test_plain_english_reaches_the_api_that_answers_it():
    """Plan steps are plain English by design (under 20 words, no API names),
    so retrieval has to bridge the gap to compound identifiers on its own."""
    assert "TextNode" in top_heading("set the line height on a text node")
    assert "VariablesAPI" in top_heading("bind a paint to a colour variable")
    assert "AutoLayoutMixin" in top_heading("make a child fill its auto layout parent")


def test_a_long_low_signal_chunk_no_longer_wins_on_length_alone():
    """The typings preamble is the longest chunk and answers nothing."""
    assert "header" not in top_heading("set the line height on a text node")


def test_identifiers_are_indexed_whole_and_in_parts():
    from knowledge.index import _tokenize

    tokens = _tokenize("setBoundVariableForPaint")

    assert "setboundvariableforpaint" in tokens  # the exact name still matches
    assert {"bound", "variable", "paint"} <= set(tokens)  # and so does English


def test_repeated_identical_queries_are_served_from_cache():
    """Every retry of a step asks the identical question, and the corpus cannot
    change during a run -- so the second answer can only be the first one
    recomputed."""
    from knowledge.index import retrieve

    retrieve.cache_clear()
    first = retrieve("append a section into the root frame", sources=("api_types.d.ts",))
    second = retrieve("append a section into the root frame", sources=("api_types.d.ts",))

    assert first == second
    assert retrieve.cache_info().hits == 1


def test_the_backend_is_swappable_and_an_unknown_name_is_survivable():
    """CLAUDE.md promises embeddings are a one-file change. An unknown backend
    must degrade to keyword search, never take a run down."""
    from knowledge import index

    assert index.set_backend("keyword") == "keyword"
    assert index.set_backend("nonsense-backend") == "keyword"
    assert index.retrieve("text node", sources=("api_types.d.ts",))


def test_the_embedding_backend_falls_back_when_the_dependency_is_absent():
    """sentence-transformers pulls in PyTorch and is deliberately not required.
    Asking for it without installing it must still return documentation."""
    from knowledge.index import EmbeddingBackend, _chunks

    backend = EmbeddingBackend(model_name="definitely-not-a-real-model")
    backend._ensure_loaded = lambda: False  # simulate the missing dependency

    ranked = backend.rank("text node line height", _chunks())

    assert ranked, "must fall back to keyword scoring rather than returning nothing"
