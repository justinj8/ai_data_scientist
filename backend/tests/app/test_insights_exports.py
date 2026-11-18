import json

import nbformat

from adsci.llm.fake import FakeAdapter
from adsci.llm.types import ModelInfo, TextDelta, ToolCallEvent, ToolCallPart, TurnEnd, Usage

from .test_chat_route import SAMPLE, parse_sse

MODEL = ModelInfo(id="fake-model", provider="fake", family="other", display_name="Fake")
U = Usage(input_tokens=1, output_tokens=1)


def _session(client):
    sid = client.post("/api/sessions").json()["session_id"]
    with open(SAMPLE / "orders.csv", "rb") as f:
        client.post(f"/api/sessions/{sid}/files", files={"file": ("orders.csv", f, "text/csv")})
    return sid


def _ask(client, sid):
    fake = FakeAdapter([
        [TextDelta(text="Plan: count"), ToolCallEvent(call=ToolCallPart(id="c", name="run_sql", arguments={
            "source": "files", "query": "select region, count(*) as n from orders group by 1 order by 1",
            "purpose": "count by region"})), TurnEnd(stop_reason="tool_use", usage=U)],
        [TextDelta(text="Orders are spread across 4 regions."), TurnEnd(stop_reason="end_turn", usage=U)],
    ], json_responses=[{"issues": []}])

    async def resolve(key):
        return fake, MODEL

    client.app.state.models.resolve = resolve
    return parse_sse(client.post(f"/api/sessions/{sid}/ask", json={"question": "How are orders spread by region?"}).text)


def test_overview(client):
    sid = _session(client)
    o = client.get(f"/api/sessions/{sid}/overview", params={"source": "files", "table": "orders"}).json()
    labels = {m["label"]: m["value"] for m in o["metrics"]}
    assert labels["Rows"] == "71,630" and "Columns" in labels
    assert len(o["distributions"]) >= 1 and o["distributions"][0]["type"] == "chart"
    assert o["correlation"]["type"] == "chart"
    assert client.get(f"/api/sessions/{sid}/overview", params={"source": "files", "table": "nope"}).status_code == 404


def test_insights_pin_patch_unpin(client):
    sid = _session(client)
    block = {"type": "text", "text": "West leads", "format": "plain"}
    created = client.post(f"/api/sessions/{sid}/insights", json={"title": "Finding", "block": block}).json()
    second = client.post(f"/api/sessions/{sid}/insights", json={"title": "Two", "block": block}).json()
    assert [i["id"] for i in client.get(f"/api/sessions/{sid}/insights").json()["insights"]] == [created["id"], second["id"]]
    client.patch(f"/api/sessions/{sid}/insights/{second['id']}", json={"order": 0, "note": "check Q4"})
    items = client.get(f"/api/sessions/{sid}/insights").json()["insights"]
    assert [i["id"] for i in items] == [second["id"], created["id"]] and items[0]["note"] == "check Q4"
    assert client.delete(f"/api/sessions/{sid}/insights/{created['id']}").status_code == 204
    assert len(client.get(f"/api/sessions/{sid}/insights").json()["insights"]) == 1


def test_exports_after_a_question(client):
    sid = _session(client)
    events = _ask(client, sid)
    assert events[-1]["type"] == "done"
    nb_resp = client.get(f"/api/sessions/{sid}/export/notebook")
    assert "attachment" in nb_resp.headers["content-disposition"]
    nb = nbformat.reads(nb_resp.text, as_version=4)
    nbformat.validate(nb)
    assert any("group by 1" in c.source for c in nb.cells)
    report = client.get(f"/api/sessions/{sid}/export/report").text
    assert "How are orders spread by region?" in report and "Verified" in report and "```sql" in report
    client.post(f"/api/sessions/{sid}/insights", json={"title": "Pinned table", "block": {
        "type": "table", "columns": ["a"], "rows": [[1]], "total_rows": 1}})
    assert "Pinned table" in client.get(f"/api/sessions/{sid}/export/report").text
    assert json.loads(client.get(f"/api/sessions/{sid}/questions").text)["questions"][0]["answer"].startswith("Orders")
