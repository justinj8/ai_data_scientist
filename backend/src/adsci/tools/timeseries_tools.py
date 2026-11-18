"""Tools for forecasting, anomaly detection and segmentation."""

from __future__ import annotations

import asyncio

import plotly.express as px
import plotly.graph_objects as go
from pydantic import BaseModel, Field

from adsci.analytics.anomalies import detect_anomalies
from adsci.analytics.forecasting import forecast_series
from adsci.analytics.segmentation import segment
from adsci.data.base import DataRef
from adsci.kernel.plotly_json import figure_to_plain
from adsci.tools.analytic_base import record
from adsci.tools.base import (
    ChartBlock,
    Metric,
    MetricsBlock,
    TableBlock,
    TextBlock,
    Tool,
    ToolContext,
    ToolOutput,
)


class ForecastInput(BaseModel):
    data: DataRef
    date_col: str
    value_col: str
    horizon: int = Field(ge=1, le=365, description="Number of future periods to forecast.")
    freq: str | None = Field(None, description="Pandas frequency (D, W, MS...); inferred if omitted.")
    agg: str = Field("sum", description="How to aggregate values within a period: sum or mean.")


class ForecastTool(Tool):
    name = "forecast"
    description = (
        "Forecast a time series. Detects seasonality, compares ETS and SARIMA in a rolling-origin backtest "
        "against a seasonal-naive baseline, and returns the forecast with 95% prediction intervals."
    )
    Input = ForecastInput

    async def run(self, args: ForecastInput, ctx: ToolContext) -> ToolOutput:
        df, warns = await ctx.load(args.data)
        r = await asyncio.to_thread(forecast_series, df, args.date_col, args.value_col, args.horizon, args.freq, args.agg)
        fig = go.Figure()
        fig.add_scatter(x=[p.date for p in r.history], y=[p.value for p in r.history], name="history", mode="lines")
        fig.add_scatter(x=[p.date for p in r.forecast], y=[p.upper for p in r.forecast], mode="lines",
                        line={"width": 0}, showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=[p.date for p in r.forecast], y=[p.lower for p in r.forecast], mode="lines",
                        line={"width": 0}, fill="tonexty", name="95% interval")
        fig.add_scatter(x=[p.date for p in r.forecast], y=[p.value for p in r.forecast], name="forecast",
                        mode="lines", line={"dash": "dash"})
        fig.update_layout(title=f"{args.value_col}: {args.horizon}-period forecast")
        bt = r.backtest
        ui = [
            MetricsBlock(items=[
                Metric(label="Model", value=r.model),
                Metric(label="Backtest MAPE", value=f"{bt.mape:.1%}", detail=f"seasonal naive {bt.naive_mape:.1%}"),
                Metric(label="Forecast total", value=f"{sum(p.value for p in r.forecast):,.2f}"),
                Metric(label="Frequency", value=r.freq),
            ]),
            ChartBlock(figure=figure_to_plain(fig), title=f"{args.value_col} forecast"),
            TableBlock(columns=["date", "forecast", "lower 95%", "upper 95%"],
                       rows=[[p.date, round(p.value, 4), round(p.lower, 4), round(p.upper, 4)] for p in r.forecast],
                       total_rows=len(r.forecast)),
        ]
        if r.warnings or warns:
            ui.append(TextBlock(format="markdown", text="\n".join(f"- ⚠ {w}" for w in r.warnings + warns)))
        total = sum(p.value for p in r.forecast)
        lines = [
            f"Model: {r.model} (freq {r.freq}, seasonal period {r.seasonal_period}).",
            f"Backtest ({bt.folds} folds x {bt.horizon}): MAPE {bt.mape:.4f} vs seasonal-naive MAPE {bt.naive_mape:.4f}; "
            f"RMSE {bt.rmse:.6g} vs {bt.naive_rmse:.6g}. Candidate MAPEs: {r.candidates}.",
            f"Forecast total over {args.horizon} periods: {total:.6g} "
            f"(95% interval per period). Last observed value: {r.history[-1].value:.6g} on {r.history[-1].date}.",
            "Forecast (date, value, lower, upper):",
        ] + [f"{p.date}, {p.value:.6g}, {p.lower:.6g}, {p.upper:.6g}" for p in r.forecast[:60]]
        lines += [f"WARNING: {w}" for w in r.warnings + warns]
        call = (f"forecast_series(df, date_col={args.date_col!r}, value_col={args.value_col!r}, "
                f"horizon={args.horizon}, freq={args.freq!r}, agg={args.agg!r})")
        data = r.model_dump(exclude={"history"})
        data["forecast_total"] = total
        return ToolOutput(summary="\n".join(lines)[: ctx.settings.model_summary_chars], ui=ui,
                          code=record(ctx, args.data, call), language="python", data=data)


class AnomalyInput(BaseModel):
    data: DataRef
    date_col: str | None = Field(None, description="For time series: the date column (with value_col).")
    value_col: str | None = None
    columns: list[str] | None = Field(None, description="For tabular data: numeric columns to check.")
    contamination: float = Field(0.01, gt=0, lt=0.5, description="Expected outlier share (tabular only).")


class AnomalyTool(Tool):
    name = "detect_anomalies"
    description = (
        "Find anomalies. Time series (date_col + value_col): STL decomposition with robust z-scores of the "
        "residuals (|z| > 3.5). Tabular (columns): Isolation Forest with the contributing features."
    )
    Input = AnomalyInput

    async def run(self, args: AnomalyInput, ctx: ToolContext) -> ToolOutput:
        df, warns = await ctx.load(args.data)
        r = await asyncio.to_thread(detect_anomalies, df, args.columns, args.date_col, args.value_col, args.contamination)
        ui: list = [MetricsBlock(items=[
            Metric(label="Method", value=r.method.replace("_", " ")),
            Metric(label="Checked", value=f"{r.n_checked:,}"),
            Metric(label="Anomalies", value=str(len(r.anomalies))),
        ])]
        if r.method == "robust_zscore":
            s = df[[args.date_col, args.value_col]].groupby(args.date_col, as_index=False).sum().sort_values(args.date_col)
            fig = px.line(s, x=args.date_col, y=args.value_col, title=f"{args.value_col} with anomalies")
            fig.add_scatter(x=[a.index_or_date for a in r.anomalies], y=[a.value for a in r.anomalies],
                            mode="markers", marker={"size": 10, "color": "#e4572e"}, name="anomaly")
            ui.append(ChartBlock(figure=figure_to_plain(fig), title=f"{args.value_col} anomalies"))
            rows = [[a.index_or_date, round(a.value, 4), round(a.expected, 4), round(a.score, 2)] for a in r.anomalies]
            cols = ["date", "value", "expected", "z"]
        else:
            rows = [[a.index_or_date, round(a.score, 4), "; ".join(a.top_features)] for a in r.anomalies]
            cols = ["row", "anomaly score", "top features"]
        ui.append(TableBlock(columns=cols, rows=rows[:200], total_rows=len(rows)))
        lines = [f"{r.method}: {len(r.anomalies)} anomalies among {r.n_checked} points."]
        lines += [", ".join(str(v) for v in row) for row in rows[:40]]
        lines += [f"NOTE: {w}" for w in r.warnings + warns]
        call = (f"detect_anomalies(df, columns={args.columns!r}, date_col={args.date_col!r}, "
                f"value_col={args.value_col!r}, contamination={args.contamination})")
        return ToolOutput(summary="\n".join(lines)[: ctx.settings.model_summary_chars], ui=ui,
                          code=record(ctx, args.data, call), language="python", data=r.model_dump())


class SegmentInput(BaseModel):
    data: DataRef
    features: list[str] = Field(min_length=1, description="Numeric columns to cluster on.")
    k: int | None = Field(None, ge=2, le=12, description="Number of segments; chosen by silhouette if omitted.")


class SegmentTool(Tool):
    name = "segment"
    description = (
        "Segment rows (e.g. customers) with k-means on scaled numeric features (skewed ones log-transformed); "
        "k chosen by silhouette score. Returns segment sizes, means and what distinguishes each segment."
    )
    Input = SegmentInput

    async def run(self, args: SegmentInput, ctx: ToolContext) -> ToolOutput:
        df, warns = await ctx.load(args.data)
        r = await asyncio.to_thread(segment, df, args.features, args.k)
        ui: list = [
            MetricsBlock(items=[
                Metric(label="Segments", value=str(r.k)),
                Metric(label="Silhouette", value=f"{r.silhouette:.3f}"),
                Metric(label="Rows", value=f"{sum(p.size for p in r.profiles):,}"),
            ]),
            TableBlock(
                columns=["segment", "size", "share"] + r.features + ["distinguishing"],
                rows=[[p.cluster, p.size, round(p.share, 4)] + [round(p.means[f], 4) for f in r.features]
                      + ["; ".join(p.distinguishing)] for p in r.profiles],
                total_rows=len(r.profiles),
            ),
        ]
        sizes = px.bar(x=[f"Segment {p.cluster}" for p in r.profiles], y=[p.size for p in r.profiles],
                       title="Segment sizes", labels={"x": "segment", "y": "rows"})
        ui.append(ChartBlock(figure=figure_to_plain(sizes), title="Segment sizes"))
        if r.warnings or warns:
            ui.append(TextBlock(format="markdown", text="\n".join(f"- ⚠ {w}" for w in r.warnings + warns)))
        lines = [f"k={r.k} segments, silhouette {r.silhouette:.4f} (by k: {r.silhouette_by_k}). "
                 f"Log-transformed: {r.log_transformed or 'none'}."]
        for p in r.profiles:
            means = ", ".join(f"{f}={v:.4g}" for f, v in p.means.items())
            lines.append(f"Segment {p.cluster}: n={p.size} ({p.share:.1%}); {means}; distinguishing: {'; '.join(p.distinguishing)}")
        lines += [f"WARNING: {w}" for w in r.warnings + warns]
        call = f"segment(df, features={args.features!r}, k={args.k!r})"
        return ToolOutput(summary="\n".join(lines)[: ctx.settings.model_summary_chars], ui=ui,
                          code=record(ctx, args.data, call), language="python", data=r.model_dump(exclude={"labels"}))
