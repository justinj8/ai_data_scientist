"""Time-series forecasting with model selection by rolling-origin backtest against a seasonal-naive baseline."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pydantic import BaseModel
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX

CANDIDATE_PERIODS = {"D": 7, "B": 5, "W": 52, "MS": 12, "M": 12, "ME": 12, "QS": 4, "Q": 4, "QE": 4, "h": 24, "H": 24}
FOLDS = 3


class Point(BaseModel):
    date: str
    value: float
    lower: float | None = None
    upper: float | None = None


class BacktestResult(BaseModel):
    folds: int
    horizon: int
    mape: float
    rmse: float
    naive_mape: float
    naive_rmse: float


class ForecastReport(BaseModel):
    freq: str
    model: str
    seasonal_period: int | None
    history: list[Point]
    forecast: list[Point]
    backtest: BacktestResult
    candidates: dict[str, float]  # model -> backtest MAPE
    warnings: list[str] = []


def _fmt(ts: pd.Timestamp, freq: str) -> str:
    return ts.strftime("%Y-%m-%d %H:%M") if freq in ("h", "H") else ts.strftime("%Y-%m-%d")


_PERIOD_FREQ = {"MS": "M", "ME": "M", "M": "M", "QS": "Q", "QE": "Q", "Q": "Q", "W": "W", "D": "D", "B": "B",
                "h": "h", "H": "h", "YS": "Y", "YE": "Y", "Y": "Y", "A": "Y"}


def _last_bin_incomplete(raw_index: pd.DatetimeIndex, series: pd.Series, freq: str) -> bool:
    """True when the data stops before the last period ends (e.g. a month with only five days of data)."""
    base = freq.split("-")[0]
    pfreq = _PERIOD_FREQ.get(base)
    if pfreq is None or len(raw_index.unique()) < 3:
        return False
    step = pd.Series(raw_index.unique().sort_values()).diff().median()
    if pfreq == "W" and "-" in freq:
        pfreq = freq  # keep the week anchor
    period_end = pd.Period(series.index[-1], freq=pfreq).end_time
    return raw_index.max() + step <= period_end


def _prepare(df: pd.DataFrame, date_col: str, value_col: str, freq: str | None, agg: str) -> tuple[pd.Series, str]:
    s = df[[date_col, value_col]].dropna()
    idx = pd.to_datetime(s[date_col])
    series = pd.Series(s[value_col].astype(float).to_numpy(), index=idx).sort_index()
    if freq is None:
        uniq = series.index.unique()
        freq = pd.infer_freq(uniq[: min(len(uniq), 200)]) if len(uniq) >= 3 else None
        if freq is None:
            step = pd.Series(uniq).diff().median()
            freq = "D" if step <= pd.Timedelta(days=1) else "W" if step <= pd.Timedelta(days=7) else "MS"
        freq = {"W-SUN": "W"}.get(freq, freq)
    raw_index = series.index
    series = series.resample(freq).agg(agg)
    if agg == "sum":
        series = series.fillna(0.0)
    series = series.interpolate(limit_direction="both")
    series.attrs["dropped_partial"] = False
    if len(series) > 3 and _last_bin_incomplete(raw_index, series, freq):
        label = series.index[-1]
        series = series.iloc[:-1]
        series.attrs["dropped_partial"] = str(label.date())
    return series, freq


def _seasonal_period(y: pd.Series, freq: str) -> int | None:
    base = freq.split("-")[0]
    p = CANDIDATE_PERIODS.get(base)
    if p is None or len(y) < 3 * p:
        return None
    # Seasonal strength (Hyndman & Athanasopoulos): 1 - Var(R) / Var(S + R) from an STL decomposition.
    res = STL(y, period=p).fit()
    strength = 1 - np.var(res.resid) / np.var(res.seasonal + res.resid)
    return p if strength >= 0.1 else None


def _mape(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.abs(actual) > 1e-9
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask]))) if mask.any() else float("nan")


def _rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def _fit_predict(name: str, train: pd.Series, horizon: int, period: int | None, alpha: float = 0.05):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "ets":
            model = ETSModel(
                train, error="add", trend="add", damped_trend=True,
                seasonal="add" if period else None, seasonal_periods=period,
            ).fit(disp=False)
            pred = model.get_prediction(start=len(train), end=len(train) + horizon - 1).summary_frame(alpha=alpha)
            return pred["mean"].to_numpy(), pred["pi_lower"].to_numpy(), pred["pi_upper"].to_numpy()
        seasonal_order = (1, 0, 1, period) if period else (0, 0, 0, 0)
        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=seasonal_order).fit(disp=False)
        fc = model.get_forecast(horizon)
        ci = fc.conf_int(alpha=alpha)
        return fc.predicted_mean.to_numpy(), ci.iloc[:, 0].to_numpy(), ci.iloc[:, 1].to_numpy()


def _naive(train: pd.Series, horizon: int, period: int | None) -> np.ndarray:
    p = period or 1
    last = train.to_numpy()[-p:]
    return np.resize(last, horizon)


def forecast_series(
    df: pd.DataFrame, date_col: str, value_col: str, horizon: int, freq: str | None = None, agg: str = "sum"
) -> ForecastReport:
    y, freq = _prepare(df, date_col, value_col, freq, agg)
    min_len = max(3 * horizon, 2 * horizon * FOLDS // 2 + 10, 24)
    if len(y) < min_len + FOLDS * horizon // 2:
        raise ValueError(
            f"Not enough history to forecast {horizon} periods with a backtest: have {len(y)} {freq} points, "
            f"need at least {min_len + FOLDS * horizon // 2}."
        )
    period = _seasonal_period(y, freq)
    warns: list[str] = []
    if y.attrs.get("dropped_partial"):
        warns.append(
            f"The last period ({y.attrs['dropped_partial']}) is incomplete in the data, so it was left out of "
            "training and the forecast starts with it."
        )
    if len(y) < 2 * horizon * FOLDS:
        warns.append("Short history: backtest folds overlap little with the forecast horizon.")

    scores: dict[str, list[tuple[float, float]]] = {"ets": [], "sarima": []}
    naive_scores: list[tuple[float, float]] = []
    for k in range(FOLDS, 0, -1):
        cut = len(y) - k * horizon
        train, test = y.iloc[:cut], y.iloc[cut : cut + horizon].to_numpy()
        nv = _naive(train, horizon, period)
        naive_scores.append((_mape(test, nv), _rmse(test, nv)))
        for name in scores:
            try:
                pred, _, _ = _fit_predict(name, train, horizon, period)
                scores[name].append((_mape(test, pred), _rmse(test, pred)))
            except Exception:  # noqa: BLE001 - a model that fails to fit is simply not selected
                scores[name].append((float("inf"), float("inf")))
    candidates = {n: float(np.mean([s[0] for s in v])) for n, v in scores.items()}
    best = min(candidates, key=candidates.get)
    pred, lo, hi = _fit_predict(best, y, horizon, period)
    future = pd.date_range(y.index[-1], periods=horizon + 1, freq=freq)[1:]  # starts at the dropped period, if any
    nm = float(np.mean([s[0] for s in naive_scores]))
    if candidates[best] >= nm:
        warns.append("The model does not beat a seasonal-naive forecast in backtests; treat the forecast with caution.")
    return ForecastReport(
        freq=freq,
        model={"ets": "ETS (additive, damped trend)", "sarima": "SARIMA(1,1,1)"}[best]
        + (f" with period-{period} seasonality" if period else ""),
        seasonal_period=period,
        history=[Point(date=_fmt(t, freq), value=float(v)) for t, v in y.iloc[-max(4 * horizon, 60):].items()],
        forecast=[
            Point(date=_fmt(t, freq), value=float(v), lower=float(min(a, v)), upper=float(max(b, v)))
            for t, v, a, b in zip(future, pred, lo, hi, strict=True)
        ],
        backtest=BacktestResult(
            folds=FOLDS, horizon=horizon,
            mape=candidates[best], rmse=float(np.mean([s[1] for s in scores[best]])),
            naive_mape=nm, naive_rmse=float(np.mean([s[1] for s in naive_scores])),
        ),
        candidates=candidates,
        warnings=warns,
    )
