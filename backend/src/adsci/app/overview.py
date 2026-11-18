"""The automatic dataset overview shown on the dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px

from adsci.analytics.profile import profile_frame
from adsci.kernel.plotly_json import figure_to_plain
from adsci.tools.base import ChartBlock, Metric

MAX_DISTRIBUTIONS = 8
SAMPLE_FOR_PLOTS = 50_000


def build_overview(df: pd.DataFrame) -> dict:
    p = profile_frame(df)
    missing = float(df.isna().mean().mean() * 100) if len(df) else 0.0
    metrics = [
        Metric(label="Rows", value=f"{p.n_rows:,}"),
        Metric(label="Columns", value=str(p.n_cols)),
        Metric(label="Missing cells", value=f"{missing:.1f}%"),
        Metric(label="Duplicate rows", value=f"{p.duplicate_rows:,}"),
    ]
    dates = [c.name for c in p.columns if c.semantic == "datetime"]
    if dates:
        d = pd.to_datetime(df[dates[0]], errors="coerce").dropna()
        if len(d):
            metrics.append(Metric(label="Date range", value=f"{d.min():%Y-%m-%d} → {d.max():%Y-%m-%d}", detail=dates[0]))
    plot_df = df.sample(SAMPLE_FOR_PLOTS, random_state=0) if len(df) > SAMPLE_FOR_PLOTS else df
    charts: list[ChartBlock] = []
    for c in p.columns:
        if len(charts) >= MAX_DISTRIBUTIONS:
            break
        if c.semantic == "numeric":
            fig = px.histogram(plot_df, x=c.name, nbins=40, title=c.name)
        elif c.semantic in ("categorical", "boolean") and c.top_values:
            top = pd.DataFrame(c.top_values)
            top["value"] = top["value"].astype(str)
            fig = px.bar(top, x="value", y="count", title=c.name)
        elif c.semantic == "datetime":
            s = pd.to_datetime(df[c.name], errors="coerce").dropna().dt.to_period("M").dt.to_timestamp()
            counts = s.value_counts().sort_index().rename_axis("month").reset_index(name="rows")
            fig = px.line(counts, x="month", y="rows", title=f"{c.name} (rows per month)")
        else:
            continue
        fig.update_layout(showlegend=False, margin={"l": 30, "r": 10, "t": 40, "b": 30}, height=260)
        charts.append(ChartBlock(figure=figure_to_plain(fig), title=c.name))
    correlation = None
    if p.correlations:
        names = list(p.correlations)
        z = [[p.correlations[a][b] for b in names] for a in names]
        fig = px.imshow(z, x=names, y=names, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f",
                        title="Correlations")
        correlation = ChartBlock(figure=figure_to_plain(fig), title="Correlations")
    return {
        "metrics": [m.model_dump() for m in metrics],
        "warnings": [w.model_dump() for w in p.warnings],
        "distributions": [c.model_dump() for c in charts],
        "correlation": correlation.model_dump() if correlation else None,
        "columns": [c.model_dump(include={"name", "semantic", "missing_pct", "n_unique"}) for c in p.columns],
    }
