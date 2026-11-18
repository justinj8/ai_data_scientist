import json
from pathlib import Path

from adsci.llm.base import AuthError
from adsci.llm.fake import FakeAdapter
from adsci.llm.types import ModelInfo, TextDelta, ToolCallEvent, ToolCallPart, TurnEnd, Usage

SAMPLE = Path(__file__).resolve().parents[3] / "sample_data" / "out"
MODEL = ModelInfo(id="fake-model", provider="fake", family="other", display_name="Fake")
U = Usage(input_tokens=1, output_tokens=1)


def parse_sse(text):
    events = []
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        data = [line[5:].strip() for line in frame.split("\n") if line.startswith("data:")]
        if data:
            events.append(json.loads("\n".join(data)))
    return events


def _session_with_data(client):
    sid = client.post("/api/sessions").json()["session_id"]
    with open(SAMPLE / "customers.csv", "rb") as f:
        client.post(f"/api/sessions/{sid}/files", files={"file": ("customers.csv", f, "text/csv")})
    return sid


def test_ask_streams_agent_events(client):
    sid = _session_with_data(client)
    fake = FakeAdapter([
        [TextDelta(text="Plan: count."), ToolCallEvent(call=ToolCallPart(id="c", name="run_sql", arguments={
            "source": "files", "query": "select count(*) as n from customers", "purpose": "count"})),
         TurnEnd(stop_reason="tool_use", usage=U)],
        [TextDelta(text="There are 5000 customers."), TurnEnd(stop_reason="end_turn", usage=U)],
    ])
    seen = {}

    async def resolve(key):
        seen["key"] = key
        return fake, MODEL

    client.app.state.models.resolve = resolve
    r = client.post(f"/api/sessions/{sid}/ask", json={"question": "How many customers?", "model": "sonnet"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert events[0]["type"] == "model" and events[-1]["type"] == "done"
    assert events[-1]["answer"] == "There are 5000 customers." and seen["key"] == "sonnet"
    table = next(e for e in events if e["type"] == "step_output")["blocks"][0]
    assert table["rows"] == [[5000]]


def test_ask_without_credentials_streams_error(client):
    sid = _session_with_data(client)

    async def resolve(key):
        raise AuthError("No Anthropic credentials. Add an API key in Settings.", "anthropic")

    client.app.state.models.resolve = resolve
    events = parse_sse(client.post(f"/api/sessions/{sid}/ask", json={"question": "q"}).text)
    assert events == [{"type": "error", "message": "No Anthropic credentials. Add an API key in Settings.",
                       "user_action": "update_credentials"}]


def test_ask_rejects_empty_question(client):
    sid = client.post("/api/sessions").json()["session_id"]
    assert client.post(f"/api/sessions/{sid}/ask", json={"question": "  "}).status_code == 422


def test_unexpected_exception_becomes_error_event(client):
    sid = _session_with_data(client)

    class Broken:
        provider = "fake"

        async def stream(self, **kw):
            raise RuntimeError("boom")
            yield  # pragma: no cover

    async def resolve(key):
        return Broken(), MODEL

    client.app.state.models.resolve = resolve
    events = parse_sse(client.post(f"/api/sessions/{sid}/ask", json={"question": "q"}).text)
    assert events[-1]["type"] == "error" and events[-1]["user_action"] == "retry"
