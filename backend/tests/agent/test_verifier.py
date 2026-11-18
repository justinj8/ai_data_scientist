import pytest

from adsci.agent.orchestrator import Orchestrator
from adsci.agent.verifier import Verifier, extract_numbers, ground
from adsci.llm.fake import FakeAdapter
from adsci.llm.types import ModelInfo
from adsci.tools.base import ToolOutput, default_registry

from .conftest import call, end, text

MODEL = ModelInfo(id="fake-model", provider="fake", family="other", display_name="Fake")


@pytest.mark.parametrize("raw,value,pct", [
    ("Revenue was 1,234.5 dollars", 1234.5, False),
    ("a change of -3.2 points", -3.2, False),
    ("churn is 12% overall", 12.0, True),
    ("sales hit $3.4M", 3_400_000.0, False),
    ("about 1.2k orders", 1200.0, False),
    ("2.5 billion rows", 2_500_000_000.0, False),
    ("p = 0.031", 0.031, False),
])
def test_extract_numbers(raw, value, pct):
    [m] = extract_numbers(raw)
    assert m.value == pytest.approx(value) and m.is_percent is pct


def test_extract_ignores_dates_years_and_list_markers():
    text = "1. On 2025-02-10 revenue spiked.\n2. In 2024 it grew.\n- Q4 was strong."
    assert extract_numbers(text) == []  # "Q4" is a label, not a statistic


@pytest.mark.parametrize("answer,evidence,grounded", [
    ("Revenue grew 8.3%.", ["growth 0.0831"], True),
    ("West AOV is 12.4% higher.", ["ratio=1.1238"], True),
    ("Churn is 31%.", ["churn_rate 0.2"], False),
    ("The top 3 regions", ["nothing"], True),        # small integers are exempt
    ("Total was $4.66M.", ["total 4658795.4"], True),
    ("Mean spend is 61.7", ["mean=61.73"], True),
    ("Mean spend is 61.9", ["mean=61.73"], False),
    ("It fell 3.2%.", ["change -0.032"], True),
])
def test_ground(answer, evidence, grounded):
    assert (ground(answer, evidence, "question?") == []) is grounded


def test_numbers_in_question_are_exempt():
    assert ground("Over 30 days revenue rises.", [], "Forecast the next 30 days") == []


async def test_verifier_combines_grounding_and_critic(settings):
    v = Verifier(settings)
    outs = [ToolOutput(summary="churn_rate 0.2", data={"rate": 0.2})]
    fake = FakeAdapter([], json_responses=[{"issues": ["Claims causation from observational data."]}])
    r = await v.verify("q", "Churn is 20%. Tickets cause churn; churn is 31%.", outs, fake, MODEL)
    assert r.status == "warnings" and r.ungrounded == ["31%"] and r.issues == ["Claims causation from observational data."]
    assert "churn_rate 0.2" in fake.calls[-1]["prompt"]


async def test_critic_failure_is_a_note_not_a_warning(settings):
    class Failing(FakeAdapter):
        async def complete_json(self, **kw):
            raise RuntimeError("down")

    r = await Verifier(settings).verify("q", "Churn is 20%.", [ToolOutput(summary="0.2")], Failing([]), MODEL)
    assert r.status == "verified" and r.notes == ["The reviewer model was unavailable; only numbers were checked."]


async def test_orchestrator_revises_ungrounded_answer(settings, session):
    orch = Orchestrator(default_registry(), Verifier(settings), settings)
    q = {"source": "files", "query": "select count(*) as n from orders", "purpose": "count"}
    fake = FakeAdapter([
        [text("Plan: count"), call("c1", "run_sql", q), end("tool_use")],
        [text("There are 99,999 orders."), end()],
        [text("There are 71,630 orders."), end()],
    ], json_responses=[{"issues": []}, {"issues": []}])
    events = [e async for e in orch.run(session, "How many orders?", fake, MODEL)]
    answers = [e for e in events if e.type == "answer"]
    assert [a.revised for a in answers] == [False, True] and answers[-1].text == "There are 71,630 orders."
    streams = [c for c in fake.calls if "messages" in c]  # calls also records critic requests
    revision_prompt = streams[2]["messages"][-1].parts[0].text
    assert "99,999" in revision_prompt
    v = next(e for e in events if e.type == "verification")
    assert v.status == "verified" and session.questions[-1].verification.status == "verified"
