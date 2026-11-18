import pytest

from adsci.analytics.modeling import train_and_evaluate


@pytest.fixture(scope="module")
def churn_report(customers):
    df = customers.assign(leak=customers["churned"].astype(float))
    return train_and_evaluate(df, "churned", seed=0)


def test_churn_beats_baseline(churn_report):
    r = churn_report
    assert r.task == "classification" and r.primary_metric == "roc_auc"
    best = next(m for m in r.models if m.name == r.best_model)
    assert best.test["roc_auc"] > r.baseline["roc_auc"] + 0.15
    assert {m.name for m in r.models} == {"baseline", "linear", "gradient_boosting"}
    assert set(best.cv_mean) == set(best.cv_std) and "f1" in best.cv_mean


def test_churn_importances_match_planted_drivers(churn_report, truth):
    top2 = {i.feature for i in churn_report.importances[:2]}
    assert top2 & set(truth["churn_top_drivers"][:2])


def test_ids_and_leakage_dropped(churn_report):
    dropped = dict(churn_report.dropped_features)
    assert "customer_id" in dropped and "leak" in dropped
    assert "leak" not in churn_report.features_used


def test_regression_revenue(orders):
    df = orders.sample(20000, random_state=0)
    r = train_and_evaluate(df, "revenue", features=["units", "unit_price", "discount"], seed=0)
    assert r.task == "regression" and r.primary_metric == "r2"
    best = next(m for m in r.models if m.name == r.best_model)
    assert best.test["r2"] > 0.9 and {"rmse", "mae"} <= set(best.test)


def test_deterministic(customers):
    df = customers.sample(1500, random_state=1)
    a = train_and_evaluate(df, "churned", seed=3)
    b = train_and_evaluate(df, "churned", seed=3)
    assert a.model_dump() == b.model_dump()


def test_class_imbalance_warning(customers):
    df = customers.assign(rare=(customers.index % 25 == 0))
    r = train_and_evaluate(df.drop(columns=["churned"]), "rare", seed=0)
    assert any("imbalance" in w.lower() for w in r.warnings)
