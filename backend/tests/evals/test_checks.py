import pytest

from adsci.evals.checks import load_questions, score

TRUTH = {"yoy_revenue_growth_2025": 0.084965, "west_aov_ratio": 1.108315, "n_orders": 71630,
         "anomaly_spike_dates": ["2024-03-14", "2024-07-04"], "top_category_2025": "Electronics"}


@pytest.mark.parametrize("answer,check,passed", [
    ("Revenue grew 8.5% year over year.", {"type": "numeric", "key": "yoy_revenue_growth_2025", "tolerance": 0.02}, True),
    ("Revenue grew 12% year over year.", {"type": "numeric", "key": "yoy_revenue_growth_2025", "tolerance": 0.02}, False),
    ("West orders are 10.8% larger on average.", {"type": "numeric", "key": "west_aov_ratio", "tolerance": 0.01}, True),
    ("There are 71,630 orders.", {"type": "numeric", "key": "n_orders", "tolerance": 0}, True),
    ("Spikes on 2024-03-14 and 2024-07-04.", {"type": "set", "key": "anomaly_spike_dates"}, True),
    ("Spike on 2024-03-14 only.", {"type": "set", "key": "anomaly_spike_dates"}, False),
    ("ELECTRONICS led sales.", {"type": "contains_all", "values": ["electronics"]}, True),
    ("Region shows no significant association.", {"type": "contains_any", "values": ["not statistically significant", "no significant"]}, True),
    ("tickets and login gaps", {"type": "contains_any", "values": ["last_login", "login"]}, True),
])
async def test_score(answer, check, passed):
    assert (await score(answer, check, TRUTH)).passed is passed


async def test_rubric_uses_judge():
    async def judge(question, answer, criteria):
        return {"pass": "interval" in answer, "reason": "checked"}

    r = await score("Forecast with a 95% interval.", {"type": "rubric", "criteria": "gives an interval"}, TRUTH,
                    question="q", judge=judge)
    assert r.passed and r.detail == "checked"


def test_question_file_is_valid():
    qs = load_questions()
    assert len(qs) >= 20 and len({q["id"] for q in qs}) == len(qs)
    assert all(q["check"]["type"] in {"numeric", "set", "contains_all", "contains_any", "rubric"} for q in qs)


def test_results_table():
    from adsci.evals.run import results_markdown

    rows = [
        {"model": "claude-opus-5-5", "id": "a", "passed": True, "steps": 2, "tool_errors": 0, "verified": True,
         "latency_s": 10.0, "cost_usd": 0.05, "error": None},
        {"model": "claude-opus-5-5", "id": "b", "passed": False, "steps": 4, "tool_errors": 1, "verified": False,
         "latency_s": 20.0, "cost_usd": 0.07, "error": None},
    ]
    md = results_markdown(rows)
    assert "| claude-opus-5-5 | 50% (1/2) | 3.0 | 17% | 50% | 15.0s | $0.060 |" in md
    assert "| a | ✅ |" in md and "| b | ❌ |" in md
