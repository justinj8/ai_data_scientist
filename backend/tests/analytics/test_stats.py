import numpy as np
import pandas as pd
import pytest

from adsci.analytics.stats import association_test, compare_groups

CAUSAL = ("causes", "caused", "drives", "driven by", "leads to", "results in", "because of")


def test_west_vs_rest_aov_is_significant(orders):
    df = orders.assign(segment=np.where(orders["region"] == "West", "West", "Other"))
    r = compare_groups(df, "revenue", "segment")
    assert r.test in {"Welch t-test", "Mann-Whitney U"}
    assert r.p_value < 0.001
    assert r.groups["West"].mean > r.groups["Other"].mean
    assert r.effect_size is not None and r.effect_size_name
    assert r.ci is not None


def test_region_vs_churn_matches_ground_truth(customers, truth):
    r = association_test(customers, "region", "churned")
    assert r.test == "Chi-square test of independence"
    assert r.p_value == pytest.approx(truth["region_churn_p_value"], abs=1e-9)
    assert r.p_value > 0.05 and r.effect_size_name == "Cramér's V"
    assert "not statistically significant" in r.interpretation


def test_normal_equal_variance_uses_student():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"g": ["a"] * 200 + ["b"] * 200, "v": np.r_[rng.normal(0, 1, 200), rng.normal(0.5, 1, 200)]})
    r = compare_groups(df, "v", "g")
    assert r.test == "Student t-test" and r.effect_size_name == "Cohen's d"
    assert any(a.name == "normality" and a.passed for a in r.assumptions)


def test_skewed_data_uses_mann_whitney():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"g": ["a"] * 200 + ["b"] * 200, "v": np.r_[rng.lognormal(0, 1.5, 200), rng.lognormal(0.3, 1.5, 200)]})
    r = compare_groups(df, "v", "g")
    assert r.test == "Mann-Whitney U" and r.effect_size_name == "rank-biserial correlation"


def test_three_groups_with_posthoc(customers):
    r = compare_groups(customers, "monthly_spend", "plan")
    assert r.test in {"One-way ANOVA", "Welch ANOVA", "Kruskal-Wallis H"}
    assert r.p_value < 0.001 and len(r.posthoc) == 3
    assert all(0 <= p.p_adjusted <= 1 for p in r.posthoc)


def test_small_sample_warning():
    df = pd.DataFrame({"g": ["a"] * 8 + ["b"] * 30, "v": list(range(8)) + list(range(30))})
    r = compare_groups(df, "v", "g")
    assert any("small sample" in w.lower() for w in r.warnings)


def test_numeric_association_spearman_for_skewed(customers):
    r = association_test(customers, "last_login_days", "support_tickets")
    assert r.test in {"Pearson correlation", "Spearman correlation"}
    assert r.ci is not None and -1 <= r.ci[0] <= r.ci[1] <= 1


def test_numeric_vs_categorical_delegates_to_group_comparison(customers):
    r = association_test(customers, "last_login_days", "churned")
    assert r.test in {"Welch t-test", "Mann-Whitney U", "Student t-test"} and r.p_value < 0.001


@pytest.mark.parametrize("fn,args", [(compare_groups, ("v", "g")), (association_test, ("v", "g"))])
def test_interpretation_never_causal(fn, args):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"g": rng.choice(["a", "b"], 300), "v": rng.normal(size=300)})
    text = fn(df, *args).interpretation.lower()
    assert not any(c in text for c in CAUSAL)
