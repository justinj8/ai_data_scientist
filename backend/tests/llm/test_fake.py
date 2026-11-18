import pytest

from adsci.llm.base import RateLimitError
from adsci.llm.fake import FakeAdapter
from adsci.llm.types import (
    Message,
    TextDelta,
    TextPart,
    ToolCallEvent,
    ToolCallPart,
    ToolResultPart,
    TurnEnd,
    Usage,
)


def test_message_roundtrip_json():
    m = Message(role="assistant", parts=[TextPart(text="hi"), ToolCallPart(id="c1", name="run_sql", arguments={"q": 1})])
    again = Message.model_validate_json(m.model_dump_json())
    assert again == m and isinstance(again.parts[1], ToolCallPart)
    r = Message(role="user", parts=[ToolResultPart(tool_call_id="c1", content="ok", is_error=False)])
    assert Message.model_validate_json(r.model_dump_json()).parts[0].tool_call_id == "c1"


def test_message_text_helper():
    m = Message(role="assistant", parts=[TextPart(text="a"), ToolCallPart(id="x", name="t", arguments={}), TextPart(text="b")])
    assert m.text == "ab" and [c.id for c in m.tool_calls] == ["x"]


async def test_fake_adapter_replays_turns_and_records():
    usage = Usage(input_tokens=1, output_tokens=2, cost_usd=None, billed_to="api_key")
    fake = FakeAdapter([
        [TextDelta(text="Plan: 1."), ToolCallEvent(call=ToolCallPart(id="1", name="run_sql", arguments={})),
         TurnEnd(stop_reason="tool_use", usage=usage)],
        [TextDelta(text="Done"), TurnEnd(stop_reason="end_turn", usage=usage)],
    ])
    msgs = [Message(role="user", parts=[TextPart(text="q")])]
    first = [e async for e in fake.stream(model="m", system="s", messages=msgs, tools=[])]
    assert isinstance(first[1], ToolCallEvent) and first[-1].stop_reason == "tool_use"
    second = [e async for e in fake.stream(model="m", system="s", messages=msgs, tools=[])]
    assert second[0].text == "Done"
    assert len(fake.calls) == 2 and fake.calls[0]["messages"][0].parts[0].text == "q"


async def test_fake_adapter_raises_scripted_error():
    fake = FakeAdapter([RateLimitError("slow down", provider="fake")])
    with pytest.raises(RateLimitError) as e:
        [e async for e in fake.stream(model="m", system="s", messages=[], tools=[])]
    assert e.value.user_action == "switch_model" and e.value.retryable


async def test_fake_list_models_and_json():
    fake = FakeAdapter([], json_responses=[{"issues": []}])
    assert (await fake.list_models())[0].provider == "fake"
    assert await fake.complete_json(model="m", system="s", prompt="p", schema={}) == {"issues": []}
