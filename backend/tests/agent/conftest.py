from pathlib import Path

import pytest

from adsci.app.sessions import SessionManager
from adsci.llm.types import TextDelta, ToolCallEvent, ToolCallPart, TurnEnd, Usage

SAMPLE = Path(__file__).resolve().parents[3] / "sample_data" / "out"
USAGE = Usage(input_tokens=100, output_tokens=50, cost_usd=0.001, billed_to="api_key")


def text(t):
    return TextDelta(text=t)


def call(cid, name, args):
    return ToolCallEvent(call=ToolCallPart(id=cid, name=name, arguments=args))


def end(stop="end_turn"):
    return TurnEnd(stop_reason=stop, usage=USAGE, provider_state={"provider": "fake", "content": []})


@pytest.fixture
def session(settings):
    mgr = SessionManager(settings)
    s = mgr.create()
    s.files.add_file(SAMPLE / "orders.csv")
    yield s
    mgr.close(s.id)
