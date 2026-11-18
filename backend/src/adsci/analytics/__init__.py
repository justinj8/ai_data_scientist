"""Pure, deterministic data-science functions used by the agent tools and by exported notebooks."""

from adsci.analytics.anomalies import detect_anomalies  # noqa: E402
from adsci.analytics.forecasting import forecast_series  # noqa: E402
from adsci.analytics.modeling import train_and_evaluate  # noqa: E402
from adsci.analytics.profile import profile_frame  # noqa: E402
from adsci.analytics.segmentation import segment  # noqa: E402
from adsci.analytics.stats import association_test, compare_groups  # noqa: E402

__all__ = [
    "association_test",
    "compare_groups",
    "detect_anomalies",
    "forecast_series",
    "profile_frame",
    "segment",
    "train_and_evaluate",
]
