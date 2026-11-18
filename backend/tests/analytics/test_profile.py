import numpy as np

from adsci.analytics.profile import profile_frame


def _col(p, name):
    return next(c for c in p.columns if c.name == name)


def _kinds(p, column=None):
    return {w.kind for w in p.warnings if column is None or w.column == column}


def test_customers_profile_basics(customers):
    p = profile_frame(customers)
    assert p.n_rows == 5000 and p.n_cols == customers.shape[1]
    assert _col(p, "customer_id").semantic == "id"
    assert "id_like" in _kinds(p, "customer_id")
    assert _col(p, "region").semantic == "categorical"
    assert _col(p, "churned").semantic == "boolean"
    assert _col(p, "monthly_spend").semantic == "numeric"
    assert {"mean", "median", "std"} <= set(_col(p, "monthly_spend").stats)


def test_missing_constant_leakage_and_duplicates(customers):
    df = customers.copy()
    df["mostly_missing"] = np.where(np.arange(len(df)) % 10 < 6, np.nan, 1.0)
    df["constant"] = 7
    df["leak"] = df["churned"].astype(float)
    df = __import__("pandas").concat([df, df.head(3)])
    p = profile_frame(df, target="churned")
    assert "high_missing" in _kinds(p, "mostly_missing")
    assert abs(_col(p, "mostly_missing").missing_pct - 60.0) < 0.5
    assert "constant" in _kinds(p, "constant")
    assert "possible_leakage" in _kinds(p, "leak")
    assert p.duplicate_rows == 3 and "duplicates" in _kinds(p)


def test_correlations_symmetric_with_unit_diagonal(customers):
    p = profile_frame(customers)
    c = p.correlations
    assert c["tenure_months"]["tenure_months"] == 1.0
    assert c["nps"]["support_tickets"] == c["support_tickets"]["nps"]
    assert "customer_id" not in c  # identifiers are excluded


def test_outliers_use_iqr_rule():
    import pandas as pd

    df = pd.DataFrame({"x": list(range(100)) + [10_000] * 10})
    p = profile_frame(df)
    assert abs(_col(p, "x").outlier_pct - 100 * 10 / 110) < 0.01
    assert "outliers" in _kinds(p, "x") and "high_skew" in _kinds(p, "x")


def test_top_values_for_categoricals(orders):
    p = profile_frame(orders.head(5000))
    top = _col(p, "region").top_values
    assert {t["value"] for t in top} == {"North", "South", "East", "West"}
    assert _col(p, "order_date").semantic == "datetime"
