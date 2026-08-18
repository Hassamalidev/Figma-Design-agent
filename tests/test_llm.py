"""Pure logic in agent/llm.py's local-model fallback -- no network, no model."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.llm import _recover_tool_call_from_content, _strip_code_fence

TOOLS = [{"type": "function", "function": {"name": "execute_figma_js"}}]


def message(content):
    return SimpleNamespace(content=content, tool_calls=None)


def test_recovers_a_tool_call_emitted_as_plain_json():
    raw = json.dumps({"name": "execute_figma_js", "arguments": {"code": "return {createdNodeIds: []}"}})
    result = _recover_tool_call_from_content(message(raw), TOOLS)

    assert result.content is None
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.function.name == "execute_figma_js"
    assert json.loads(call.function.arguments) == {"code": "return {createdNodeIds: []}"}


def test_recovers_a_tool_call_wrapped_in_a_code_fence():
    raw = "```json\n" + json.dumps({"name": "execute_figma_js", "arguments": {"code": "x"}}) + "\n```"
    result = _recover_tool_call_from_content(message(raw), TOOLS)

    assert result.tool_calls[0].function.name == "execute_figma_js"


def test_leaves_ordinary_prose_untouched():
    original = message("The rectangle has been created successfully.")
    result = _recover_tool_call_from_content(original, TOOLS)

    assert result is original


def test_leaves_unrelated_json_untouched():
    original = message(json.dumps({"status": "done"}))
    result = _recover_tool_call_from_content(original, TOOLS)

    assert result is original


def test_leaves_unknown_tool_name_untouched():
    original = message(json.dumps({"name": "not_a_real_tool", "arguments": {}}))
    result = _recover_tool_call_from_content(original, TOOLS)

    assert result is original


def test_strip_code_fence_variants():
    assert _strip_code_fence("```json\n{}\n```") == "{}"
    assert _strip_code_fence("```\n{}\n```") == "{}"
    assert _strip_code_fence("{}") == "{}"


def test_recovers_the_invented_calls_wrapper():
    """Real shape seen in a live run: the model batched calls under {"calls": [...]}."""
    raw = json.dumps(
        {
            "calls": [
                {"name": "execute_figma_js", "arguments": {"code": "a"}},
                {"name": "execute_figma_js", "arguments": {"code": "b"}},
            ]
        }
    )
    result = _recover_tool_call_from_content(message(raw), TOOLS)

    assert len(result.tool_calls) == 2
    assert [json.loads(c.function.arguments)["code"] for c in result.tool_calls] == ["a", "b"]


def test_recovers_calls_buried_in_prose_with_multiple_fenced_blocks():
    """Also real: the model narrated, then emitted several ```json blocks."""
    raw = (
        "Below are three separate calls:\n\n"
        "```json\n" + json.dumps({"name": "execute_figma_js", "arguments": {"code": "first"}}) + "\n```\n\n"
        "And the second:\n\n"
        "```json\n" + json.dumps({"name": "execute_figma_js", "arguments": {"code": "second"}}) + "\n```\n"
    )
    result = _recover_tool_call_from_content(message(raw), TOOLS)

    assert [json.loads(c.function.arguments)["code"] for c in result.tool_calls] == ["first", "second"]


# ---- transient endpoint failures ------------------------------------------

class _Boom:
    """A client whose first N calls raise, then succeeds."""

    def __init__(self, error, failures: int):
        self._error, self._left = error, failures
        self.calls = 0

        class _Completions:
            def create(inner, **kwargs):
                self.calls += 1
                if self._left > 0:
                    self._left -= 1
                    raise self._error
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))]
                )

        self.chat = SimpleNamespace(completions=_Completions())


def _client_with(monkeypatch, error, failures):
    from agent import llm as llm_mod

    monkeypatch.setattr(llm_mod.time, "sleep", lambda _s: None)  # no real backoff in tests
    client = llm_mod.ModelClient("http://x/v1", "k", "m")
    fake = _Boom(error, failures)
    client._client = fake
    return client, fake


def _response(status: int):
    """A response real enough for the SDK's exception constructors."""
    import httpx2 as httpx  # the http library the openai SDK actually vendors

    request = httpx.Request("POST", "http://x/v1/chat/completions")
    return httpx.Response(status_code=status, request=request)


def _server_error():
    from openai import InternalServerError

    return InternalServerError("boom", response=_response(500), body=None)


def test_a_transient_500_is_retried_not_fatal(monkeypatch):
    """A single 500 from Ollama Cloud used to kill a whole benchmark sweep."""
    client, fake = _client_with(monkeypatch, _server_error(), failures=2)
    assert client.complete([{"role": "user", "content": "hi"}]).content == "ok"
    assert fake.calls == 3


def test_an_endpoint_that_never_recovers_still_raises(monkeypatch):
    """Absorbing failures forever would hide a genuinely dead endpoint."""
    from agent import llm as llm_mod
    from openai import InternalServerError

    client, fake = _client_with(monkeypatch, _server_error(), failures=99)
    with pytest.raises(InternalServerError):
        client.complete([{"role": "user", "content": "hi"}])
    assert fake.calls == llm_mod.MAX_TRANSIENT_RETRIES + 1


def test_a_bad_request_is_not_retried(monkeypatch):
    """A 400 is a real answer -- a text-only model rejecting an image. Retrying
    it four times just wastes four round trips before the same result."""
    from openai import BadRequestError

    error = BadRequestError("no images", response=_response(400), body=None)
    client, fake = _client_with(monkeypatch, error, failures=99)
    with pytest.raises(BadRequestError):
        client.complete([{"role": "user", "content": "hi"}])
    assert fake.calls == 1


# ---- measurement at the swap point ----------------------------------------
#
# Model calls are the most expensive thing a run does (~50 per design). They are
# instrumented HERE rather than at each call site because CLAUDE.md's first
# golden rule is that all model access goes through this file -- so measuring it
# here measures the loop, the planner and the critic by construction.


class _StubCompletions:
    """A transport that hands back canned outcomes. No SDK call, no network."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _client_returning(*outcomes):
    from agent.llm import ModelClient

    client = ModelClient.__new__(ModelClient)  # skip OpenAI() construction
    completions = _StubCompletions(outcomes)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client._model = "stub"
    return client, completions


def _completion(content="ok", prompt_tokens=0, completion_tokens=0):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def test_every_model_call_is_timed_and_counted():
    from agent import metrics

    client, _ = _client_returning(_completion(), _completion())
    with metrics.recording() as measured:
        client.complete([{"role": "user", "content": "hi"}])
        client.complete([{"role": "user", "content": "hi"}])

    assert measured.model.count == 2
    assert measured.model.errors == 0


def test_token_usage_is_recorded_when_the_endpoint_reports_it():
    from agent import metrics

    client, _ = _client_returning(_completion(prompt_tokens=1200, completion_tokens=90))
    with metrics.recording() as measured:
        client.complete([{"role": "user", "content": "hi"}])

    assert measured.prompt_tokens == 1200
    assert measured.completion_tokens == 90


def test_a_missing_usage_field_is_not_an_error():
    """Not every OpenAI-compatible server fills usage in, and a statistic must
    never take a run down."""
    from agent import metrics

    reply = _completion()
    reply.usage = None
    client, _ = _client_returning(reply)
    with metrics.recording() as measured:
        client.complete([{"role": "user", "content": "hi"}])

    assert measured.model.count == 1
    assert measured.prompt_tokens == 0


def test_a_failed_call_is_recorded_as_an_error_and_still_raises():
    from agent import metrics

    client, _ = _client_returning(ValueError("bad request"))
    with metrics.recording() as measured:
        with pytest.raises(ValueError):
            client.complete([{"role": "user", "content": "hi"}])

    assert measured.model.count == 1
    assert measured.model.errors == 1


def test_absorbed_endpoint_blips_are_counted_separately_from_calls(monkeypatch):
    """A run that 'felt slow' is often this number, not the model itself -- and
    a retry inside the client is still ONE logical call to the loop."""
    from agent import llm as llm_mod
    from agent import metrics

    monkeypatch.setattr(llm_mod.time, "sleep", lambda _s: None)  # no real backoff
    client, completions = _client_returning(_server_error(), _completion())

    with metrics.recording() as measured:
        client.complete([{"role": "user", "content": "hi"}])

    assert completions.calls == 2                   # retried once at the transport
    assert measured.model.count == 1                # one logical call
    assert measured.model_transient_retries == 1    # with the blip made visible
