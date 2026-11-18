import pytest

from adsci.agent.orchestrator import Orchestrator
from adsci.agent.verifier import Verifier
from adsci.llm.base import RateLimitError
from adsci.llm.fake import FakeAdapter
from adsci.llm.types import ModelInfo, TextPart, ToolResultPart
from adsci.tools.base import default_registry

from .conftest import call, end, text

MODEL = ModelInfo(id="fake-model", provider="fake", family="other", display_name="Fake")
Q_SQL = {"source": "files", "query": "select region, count(*) as n from orders group by 1", "purpose": "count"}


@pytest.fixture
def orch(settings):
    return Orchestrator(default_registry(), Verifier(settings), settings)


async def collect(orch, session, question, adapter):
    return [e async for e in orch.run(session, question, adapter, MODEL)]


def types(events):
    return [e.type for e in events]


async def test_happy_path_event_order(orch, session):
    fake = FakeAdapter([
        [text("Plan:\n1. Count orders by region."), call("c1", "run_sql", Q_SQL), end("tool_use")],
        [text("West has the most orders."), end()],
    ])
    events = await collect(orch, session, "Which region has most orders?", fake)
    t = [x for x in types(events) if x != "text_delta"]
    assert t == ["model", "plan", "step_start", "step_output", "answer", "verification", "usage", "done"]
    start = next(e for e in events if e.type == "step_start")
    assert start.tool == "run_sql" and start.language == "sql" and "group by" in start.code
    out = next(e for e in events if e.type == "step_output")
    assert not out.is_error and out.blocks[0].type == "table"
    done = events[-1]
    assert done.answer == "West has the most orders." and done.steps == 1
    usage = next(e for e in events if e.type == "usage")
    assert usage.usage.input_tokens == 200 and usage.usage.cost_usd == pytest.approx(0.002)
    assert len(session.questions) == 1 and session.questions[0].answer == done.answer


async def test_tool_error_is_fed_back_for_self_correction(orch, session):
    def script(messages):
        last = messages[-1]
        results = [p for p in last.parts if isinstance(p, ToolResultPart)]
        if not results:
            return [call("c1", "run_sql", {**Q_SQL, "query": "drop table orders"}), end("tool_use")]
        if results[0].is_error:
            assert "read-only" in results[0].content
            return [call("c2", "run_sql", Q_SQL), end("tool_use")]
        return [text("Fixed it."), end()]

    events = await collect(orch, session, "count", FakeAdapter(script))
    outs = [e for e in events if e.type == "step_output"]
    assert [o.is_error for o in outs] == [True, False] and events[-1].type == "done"


async def test_step_cap_forces_an_answer(settings, session):
    settings.max_agent_steps = 3
    orch = Orchestrator(default_registry(), Verifier(settings), settings)

    def script(messages):
        last = messages[-1]
        if any(isinstance(p, TextPart) and "step limit" in p.text.lower() for p in last.parts):
            return [text("Best answer with what I have."), end()]
        n = sum(1 for m in messages for p in m.parts if isinstance(p, ToolResultPart))
        return [call(f"c{n}", "run_sql", Q_SQL), end("tool_use")]

    events = await collect(orch, session, "loop forever", FakeAdapter(script))
    assert types(events).count("step_start") == 3
    assert next(e for e in events if e.type == "answer").text == "Best answer with what I have."


async def test_model_that_ignores_cap_is_stopped(settings, session):
    settings.max_agent_steps = 2
    orch = Orchestrator(default_registry(), Verifier(settings), settings)
    events = await collect(orch, session, "q", FakeAdapter(lambda m: [call("x", "run_sql", Q_SQL), end("tool_use")]))
    assert types(events).count("step_start") == 2
    assert "step limit" in next(e for e in events if e.type == "answer").text.lower()


async def test_history_persists_and_switching_models_reuses_it(orch, session):
    first = FakeAdapter([[text("Plan"), call("c1", "run_sql", Q_SQL), end("tool_use")], [text("Answer one."), end()]])
    await collect(orch, session, "Question one?", first)
    second = FakeAdapter([[text("Answer two."), end()]])
    await collect(orch, session, "Follow-up?", second)
    msgs = second.calls[0]["messages"]
    assert msgs[0].parts[0].text == "Question one?" and msgs[-1].parts[0].text == "Follow-up?"
    assert any(isinstance(p, ToolResultPart) for m in msgs for p in m.parts)
    assert "Answer one." in [m.text for m in msgs if m.role == "assistant"]


async def test_provider_error_emits_single_error_and_keeps_history_clean(orch, session):
    fake = FakeAdapter([RateLimitError("Anthropic rate limit reached", provider="anthropic")])
    events = await collect(orch, session, "q", fake)
    assert types(events) == ["model", "error"]
    assert events[-1].user_action == "switch_model" and "rate limit" in events[-1].message
    assert session.history == [] and session.questions == []


async def test_refusal_is_reported(orch, session):
    events = await collect(orch, session, "q", FakeAdapter([[end("refusal")]]))
    assert events[-1].type == "error" and events[-1].user_action == "switch_model"


async def test_system_prompt_lists_sources_and_rules(orch, session):
    fake = FakeAdapter([[text("ok"), end()]])
    await collect(orch, session, "q", fake)
    system = fake.calls[0]["system"]
    assert "files" in system and "orders" in system and "revenue" in system
    assert "effect size" in system.lower() and "plan:" in system.lower()
    assert {t.name for t in fake.calls[0]["tools"]} >= {"run_sql", "run_python", "statistical_test", "train_model"}
