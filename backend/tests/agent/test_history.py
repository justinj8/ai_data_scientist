from adsci.agent.history import compact_history
from adsci.llm.types import Message, TextPart, ToolCallPart, ToolResultPart


def convo(n_questions, result_size):
    msgs = []
    for i in range(n_questions):
        msgs += [
            Message(role="user", parts=[TextPart(text=f"q{i}")]),
            Message(role="assistant", parts=[ToolCallPart(id=f"c{i}", name="run_sql", arguments={})],
                    provider_state={"provider": "anthropic", "content": [{"type": "thinking"}]}),
            Message(role="user", parts=[ToolResultPart(tool_call_id=f"c{i}", content="x" * result_size)]),
            Message(role="assistant", parts=[TextPart(text=f"a{i}")]),
        ]
    return msgs


def test_under_budget_is_append_only_identity():
    msgs = convo(3, 1000)
    assert compact_history(msgs, char_budget=100_000) == msgs


def test_over_budget_clips_old_results_and_strips_their_provider_state():
    msgs = convo(3, 50_000)
    out = compact_history(msgs, char_budget=100_000, keep_last=1)
    results = [p.content for m in out for p in m.parts if isinstance(p, ToolResultPart)]
    assert len(results[0]) <= 600 and len(results[1]) <= 600 and len(results[2]) == 50_000
    assert out[1].provider_state is None and out[9].provider_state is not None
