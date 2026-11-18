"""Anomaly detection: STL residual robust z-scores for time series, Isolation Forest for tables."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL

Z_THRESHOLD = 3.5
PERIODS = {"D": 7, "B": 5, "W": 52, "MS": 12, "M": 12, "h": 24, "H": 24}


class Anomaly(BaseModel):
    index_or_date: str
    score: float
    value: float | None = None
    expected: float | None = None
    top_features: list[str] = []


class AnomalyReport(BaseModel):
    method: Literal["isolation_forest", "robust_zscore"]
    contamination: float | None
    threshold: float | None
    n_checked: int
    anomalies: list[Anomaly]
    warnings: list[str] = []


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return (x - med) / mad if mad > 0 else np.zeros_like(x)


def detect_anomalies(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    date_col: str | None = None,
    value_col: str | None = None,
    contamination: float = 0.01,
    seed: int = 0,
) -> AnomalyReport:
    if date_col and value_col:
        s = df[[date_col, value_col]].dropna()
        series = pd.Series(s[value_col].astype(float).to_numpy(), index=pd.to_datetime(s[date_col])).sort_index()
        freq = pd.infer_freq(series.index[:200]) or "D"
        series = series.groupby(level=0).sum()
        series = series.asfreq(freq, fill_value=0.0)  # a missing period means nothing was recorded
        period = PERIODS.get(freq.split("-")[0], 7)
        warns = []
        if len(series) < 2 * period + 1:
            raise ValueError(f"Need at least {2 * period + 1} points to detect anomalies in a series.")
        # Non-robust STL: the robust variant hugs the data so tightly that residual spread is understated
        # and ordinary noise gets flagged.
        res = STL(series, period=period, robust=False).fit()
        z = _robust_z(res.resid.to_numpy())
        expected = (res.trend + res.seasonal).to_numpy()
        idx = np.where(np.abs(z) > Z_THRESHOLD)[0]
        fmt = "%Y-%m-%d %H:%M" if freq in ("h", "H") else "%Y-%m-%d"
        anomalies = sorted(
            (
                Anomaly(index_or_date=series.index[i].strftime(fmt), score=float(z[i]),
                        value=float(series.iloc[i]), expected=float(expected[i]))
                for i in idx
            ),
            key=lambda a: -abs(a.score),
        )
        return AnomalyReport(method="robust_zscore", contamination=None, threshold=Z_THRESHOLD,
                             n_checked=len(series), anomalies=anomalies, warnings=warns)

    cols = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    if not cols:
        raise ValueError("No numeric columns to check for anomalies.")
    X = df[cols].astype(float)
    X = X.fillna(X.median())
    forest = IsolationForest(contamination=contamination, random_state=seed).fit(X)
    flags = forest.predict(X) == -1
    scores = -forest.score_samples(X)
    zs = pd.DataFrame({c: _robust_z(X[c].to_numpy()) for c in cols}, index=X.index)
    anomalies = []
    for i in np.where(flags)[0]:
        row = zs.iloc[i].abs().sort_values(ascending=False)
        anomalies.append(Anomaly(index_or_date=str(X.index[i]), score=float(scores[i]),
                                 top_features=[f"{c} (z={zs.iloc[i][c]:.1f})" for c in row.index[:3]]))
    anomalies.sort(key=lambda a: -a.score)
    return AnomalyReport(
        method="isolation_forest", contamination=contamination, threshold=None, n_checked=len(X),
        anomalies=anomalies,
        warnings=[f"Isolation Forest flags about {contamination:.0%} of rows by construction (the contamination setting)."],
    )
