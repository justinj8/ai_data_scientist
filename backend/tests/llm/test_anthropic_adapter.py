import json

import anthropic
import httpx2
import pytest

from adsci.llm.anthropic_adapter import AnthropicAdapter, to_anthropic_messages
from adsci.llm.base import AuthError, RateLimitError
from adsci.llm.types import (
    Message,
    TextDelta,
    TextPart,
    ToolCallEvent,
    ToolCallPart,
    ToolResultPart,
    ToolSpec,
    TurnEnd,
)

from .sse import anthropic_refusal_turn, anthropic_tool_turn


class Recorder:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return self.response


def adapter_with(response):
    rec = Recorder(response)
    client = anthropic.AsyncAnthropic(
        api_key="sk-test", base_url="https://api.anthropic.com", max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(rec)),
    )
    return AnthropicAdapter(api_key="unused", client=client), rec


def sse_response(body):
    return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})
TOOLS = [ToolSpec(name="run_sql", description="d", parameters={"type": "object", "properties": {}})]


def _history():
    return [
        Message(role="user", parts=[TextPart(text="Which region?")]),
        Message(role="assistant", parts=[TextPart(text="Plan"), ToolCallPart(id="t1", name="run_sql", arguments={"q": 1})]),
        Message(role="user", parts=[ToolResultPart(tool_call_id="t1", content="West", is_error=False)]),
    ]


def test_neutral_history_to_anthropic_blocks():
    out = to_anthropic_messages(_history())
    assert out[1]["content"][1] == {"type": "tool_use", "id": "t1", "name": "run_sql", "input": {"q": 1}}
    assert out[2]["content"][0] == {"type": "tool_result", "tool_use_id": "t1", "content": "West", "is_error": False}


def test_provider_state_replayed_verbatim():
    raw = [{"type": "thinking", "thinking": "", "signature": "s"}, {"type": "text", "text": "Plan"}]
    m = Message(role="assistant", parts=[TextPart(text="Plan")], provider_state={"provider": "anthropic", "content": raw})
    assert to_anthropic_messages([m])[0]["content"] == raw
    other = Message(role="assistant", parts=[TextPart(text="Plan")], provider_state={"provider": "openai", "content": []})
    assert to_anthropic_messages([other])[0]["content"] == [{"type": "text", "text": "Plan"}]


async def test_stream_text_then_tool_call():
    adapter, rec = adapter_with(sse_response(anthropic_tool_turn()))
    events = [e async for e in adapter.stream(model="claude-opus-5-5", system="sys", messages=_history(), tools=TOOLS)]
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Plan: 1. query"
    [call] = [e for e in events if isinstance(e, ToolCallEvent)]
    assert call.call.id == "toolu_1" and call.call.arguments == {"source": "files", "query": "select 1"}
    end = events[-1]
    assert isinstance(end, TurnEnd) and end.stop_reason == "tool_use"
    assert end.usage.input_tokens == 1000 and end.usage.output_tokens == 500
    assert end.usage.cost_usd == pytest.approx(1000 * 4 / 1e6 + 500 * 20 / 1e6)
    assert end.provider_state["provider"] == "anthropic"
    assert end.provider_state["content"][0]["type"] == "thinking"  # thinking block kept for replay
    body = json.loads(rec.requests[-1].content)
    assert body["model"] == "claude-opus-5-5" and body["system"] == "sys"
    assert body["fallbacks"] == "default" and body["output_config"]["effort"] == "high"
    assert "thinking" not in body  # Opus 5.5 is always adaptive


async def test_refusal_maps_to_refusal_stop_reason():
    adapter, _ = adapter_with(sse_response(anthropic_refusal_turn()))
    events = [e async for e in adapter.stream(model="claude-opus-5-5", system="s", messages=_history(), tools=[])]
    assert events[-1].stop_reason == "refusal"


@pytest.mark.parametrize("status,exc,action", [(401, AuthError, "update_credentials"), (429, RateLimitError, "switch_model")])
async def test_http_errors_mapped(status, exc, action):
    adapter, _ = adapter_with(httpx2.Response(status, json={"type": "error", "error": {"type": "x", "message": "nope"}}))
    with pytest.raises(exc) as e:
        [e async for e in adapter.stream(model="claude-opus-5-5", system="s", messages=_history(), tools=[])]
    assert e.value.user_action == action and e.value.provider == "anthropic"


async def test_list_models_families():
    adapter, rec = adapter_with(httpx2.Response(200, json={
        "data": [
            {"type": "model", "id": "claude-opus-5-5", "display_name": "Claude Opus 5.5", "created_at": "2026-09-01T00:00:00Z"},
            {"type": "model", "id": "claude-sonnet-5-5", "display_name": "Claude Sonnet 5.5", "created_at": "2026-10-01T00:00:00Z"},
            {"type": "model", "id": "claude-haiku-5-5", "display_name": "Claude Haiku 5.5", "created_at": "2026-10-01T00:00:00Z"},
        ],
        "has_more": False, "first_id": "claude-opus-5-5", "last_id": "claude-haiku-5-5"}))
    models = {m.id: m for m in await adapter.list_models()}
    assert rec.requests[0].url.path == "/v1/models"
    assert models["claude-opus-5-5"].family == "opus" and models["claude-sonnet-5-5"].family == "sonnet"
    assert models["claude-haiku-5-5"].family == "other" and models["claude-opus-5-5"].created.year == 2026


def test_default_client_ignores_ambient_base_url(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example.com")
    assert str(AnthropicAdapter(api_key="k").client.base_url).startswith("https://api.anthropic.com")
