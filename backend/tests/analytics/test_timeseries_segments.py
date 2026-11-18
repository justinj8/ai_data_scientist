import pandas as pd
import pytest

from adsci.analytics.anomalies import detect_anomalies
from adsci.analytics.forecasting import forecast_series
from adsci.analytics.segmentation import segment

CHURN_FEATURES = ["tenure_months", "monthly_spend", "support_tickets", "last_login_days", "nps"]


@pytest.fixture(scope="module")
def daily(orders):
    return orders.groupby("order_date", as_index=False)["revenue"].sum()


@pytest.fixture(scope="module")
def fc(daily):
    return forecast_series(daily, "order_date", "revenue", horizon=30)


def test_forecast_beats_seasonal_naive(fc):
    assert fc.freq == "D" and fc.seasonal_period == 7
    assert fc.backtest.mape < fc.backtest.naive_mape
    assert len(fc.forecast) == 30
    assert all(p.lower <= p.value <= p.upper for p in fc.forecast)
    assert fc.forecast[0].date == "2026-01-01"


def test_forecast_needs_enough_history(daily):
    with pytest.raises(ValueError, match="history"):
        forecast_series(daily.head(40), "order_date", "revenue", horizon=30)


def test_time_series_anomalies_include_planted_days(daily, truth):
    r = detect_anomalies(daily, date_col="order_date", value_col="revenue")
    found = {a.index_or_date for a in r.anomalies}
    assert set(truth["anomaly_spike_dates"]) <= found
    assert set(truth["outage_dates"]) <= found
    assert r.method == "robust_zscore"
    assert len(found) < 40  # not flagging everything


def test_tabular_anomalies(customers):
    r = detect_anomalies(customers, columns=CHURN_FEATURES, contamination=0.01)
    assert r.method == "isolation_forest" and len(r.anomalies) == 50
    assert all(a.top_features for a in r.anomalies)


def test_segmentation_recovers_planted_clusters():
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=1500, centers=3, n_features=4, cluster_std=1.0, random_state=0)
    df = pd.DataFrame(X, columns=list("abcd"))
    r = segment(df, list("abcd"), seed=0)
    assert r.k == 3 and r.silhouette > 0.5 and not r.warnings
    assert sum(p.size for p in r.profiles) == 1500 and all(p.distinguishing for p in r.profiles)
    assert segment(df, list("abcd"), seed=0).model_dump() == r.model_dump()


def test_segmentation_warns_on_weak_structure(customers):
    r = segment(customers, CHURN_FEATURES, seed=0)
    assert 2 <= r.k <= 8 and sum(p.size for p in r.profiles) == len(customers)
    assert r.silhouette < 0.25 and any("weak" in w.lower() for w in r.warnings)
    assert "monthly_spend" in r.log_transformed


def test_segmentation_fixed_k(customers):
    assert segment(customers, CHURN_FEATURES, k=3).k == 3


def test_monthly_frequency_inferred():
    s = pd.DataFrame({"m": pd.date_range("2018-01-01", periods=72, freq="MS"),
                      "v": [100 + (i % 12) * 5 + i for i in range(72)]})
    r = forecast_series(s, "m", "v", horizon=6)
    assert r.freq in {"MS", "M"} and r.seasonal_period == 12


def test_partial_final_period_is_dropped_with_warning(orders):
    cut = orders[orders["order_date"] <= "2025-11-05"]
    daily = cut.groupby("order_date", as_index=False)["revenue"].sum()
    r = forecast_series(daily, "order_date", "revenue", horizon=3, freq="MS")
    assert r.history[-1].date == "2025-10-01"
    assert r.forecast[0].date == "2025-11-01"
    assert any("incomplete" in w.lower() for w in r.warnings)
    assert min(p.value for p in r.forecast) > 250_000  # not dragged down by a 5-day "month"
