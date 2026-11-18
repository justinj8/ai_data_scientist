import json

import httpx2
import pytest

from adsci.llm.base import AccessDenied, AuthError, ProviderUnavailable, RateLimitError
from adsci.llm.openai_adapter import OpenAICompatAdapter, openai_family, to_responses_input, xai_family
from adsci.llm.types import (
    Message,
    TextDelta,
    TextPart,
    ToolCallEvent,
    ToolCallPart,
    ToolResultPart,
    ToolSpec,
)

from .sse import openai_incomplete_turn, openai_tool_turn

TOOLS = [ToolSpec(name="run_sql", description="d", parameters={"type": "object", "properties": {}})]


class Recorder:
    def __init__(self, response):
        self.response, self.requests = response, []

    def __call__(self, request):
        self.requests.append(request)
        return self.response


def make(response, provider="openai", billed_to="api_key"):
    rec = Recorder(response)

    async def token():
        return "tok-123"

    a = OpenAICompatAdapter(provider, token, billed_to=billed_to,
                            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(rec)), max_retries=0)
    return a, rec


def sse_response(body):
    return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})


def _history():
    return [
        Message(role="user", parts=[TextPart(text="Which region?")]),
        Message(role="assistant", parts=[TextPart(text="Plan"), ToolCallPart(id="c1", name="run_sql", arguments={"q": 1})]),
        Message(role="user", parts=[ToolResultPart(tool_call_id="c1", content="West")]),
    ]


def test_history_to_responses_items():
    items = to_responses_input(_history(), "openai")
    assert items[0] == {"role": "user", "content": "Which region?"}
    assert items[2] == {"type": "function_call", "call_id": "c1", "name": "run_sql", "arguments": json.dumps({"q": 1})}
    assert items[3] == {"type": "function_call_output", "call_id": "c1", "output": "West"}


def test_provider_state_replay_and_xai_drops_reasoning():
    out = [{"type": "reasoning", "id": "rs", "encrypted_content": "e", "summary": []},
           {"type": "function_call", "call_id": "c1", "name": "run_sql", "arguments": "{}"}]
    m = Message(role="assistant", parts=[], provider_state={"provider": "openai", "output": out})
    assert to_responses_input([m], "openai") == out
    mx = Message(role="assistant", parts=[], provider_state={"provider": "xai", "output": out})
    assert to_responses_input([mx], "xai") == out[1:]


async def test_stream_text_and_function_call():
    a, rec = make(sse_response(openai_tool_turn()))
    events = [e async for e in a.stream(model="gpt-6.2-astra", system="sys", messages=_history(), tools=TOOLS)]
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Plan: query"
    [call] = [e for e in events if isinstance(e, ToolCallEvent)]
    assert call.call.id == "call_1" and call.call.arguments == {"source": "files", "query": "select 1"}
    end = events[-1]
    assert end.stop_reason == "tool_use" and end.usage.input_tokens == 800
    assert end.usage.cost_usd == pytest.approx(800 * 8 / 1e6 + 300 * 40 / 1e6)
    assert end.provider_state["output"][0]["type"] == "reasoning"
    req = rec.requests[-1]
    body = json.loads(req.content)
    assert req.url.host == "api.openai.com" and req.headers["authorization"] == "Bearer tok-123"
    assert body["instructions"] == "sys" and body["store"] is False and body["stream"] is True
    assert "reasoning.encrypted_content" in body["include"] and body["tools"][0]["type"] == "function"


async def test_plan_usage_has_no_dollar_cost():
    a, _ = make(sse_response(openai_tool_turn()), billed_to="chatgpt_plan")
    events = [e async for e in a.stream(model="gpt-6.2-astra", system="s", messages=_history(), tools=TOOLS)]
    assert events[-1].usage.cost_usd is None and events[-1].usage.billed_to == "chatgpt_plan"


async def test_incomplete_maps_to_max_tokens():
    a, _ = make(sse_response(openai_incomplete_turn()))
    events = [e async for e in a.stream(model="gpt-6.2-astra", system="s", messages=_history(), tools=[])]
    assert events[-1].stop_reason == "max_tokens"


async def test_xai_uses_xai_host_and_no_encrypted_reasoning():
    a, rec = make(sse_response(openai_tool_turn()), provider="xai")
    [e async for e in a.stream(model="grok-4.7", system="s", messages=_history(), tools=TOOLS)]
    body = json.loads(rec.requests[-1].content)
    assert rec.requests[-1].url.host == "api.x.ai" and "include" not in body and "reasoning" not in body


@pytest.mark.parametrize("status,provider,billed,exc", [
    (401, "openai", "api_key", AuthError),
    (429, "openai", "chatgpt_plan", RateLimitError),
    (500, "openai", "api_key", ProviderUnavailable),
    (403, "xai", "supergrok_plan", AccessDenied),
])
async def test_errors_mapped(status, provider, billed, exc):
    a, _ = make(httpx2.Response(status, json={"error": {"message": "nope", "type": "x"}}), provider, billed)
    with pytest.raises(exc) as e:
        [x async for x in a.stream(model="m", system="s", messages=_history(), tools=[])]
    if exc is AccessDenied:
        assert "API key" in e.value.message
    if billed == "chatgpt_plan":
        assert "ChatGPT plan" in e.value.message


@pytest.mark.parametrize("mid,fam", [("gpt-6.2-astra", "gpt"), ("gpt-6-luna", "gpt"), ("gpt-image-2.6-flare", "other"),
                                     ("gpt-realtime-2.1-mini", "other"), ("gpt-6-cyber", "other"), ("o3", "other")])
def test_openai_family(mid, fam):
    assert openai_family(mid) == fam


@pytest.mark.parametrize("mid,fam", [("grok-4.7", "grok"), ("grok-imagine-image-2.0", "other"), ("grok-code-fast-1", "other"),
                                     ("grok-4.20-0309-reasoning", "grok"), ("grok-build-0.1", "other")])
def test_xai_family(mid, fam):
    assert xai_family(mid) == fam


async def test_list_models():
    a, _ = make(httpx2.Response(200, json={"object": "list", "data": [
        {"id": "gpt-6.2-astra", "object": "model", "created": 1760000000, "owned_by": "openai"},
        {"id": "gpt-image-2.6-flare", "object": "model", "created": 1761000000, "owned_by": "openai"}]}))
    models = {m.id: m for m in await a.list_models()}
    assert models["gpt-6.2-astra"].family == "gpt" and models["gpt-6.2-astra"].created.year == 2025
    assert models["gpt-image-2.6-flare"].supports_tools is False
