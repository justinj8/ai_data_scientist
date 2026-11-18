from __future__ import annotations

import json
from typing import Literal

import plotly.express as px
from pydantic import BaseModel, Field

from adsci.data.base import DataRef
from adsci.kernel.manager import PlotlyOutput
from adsci.kernel.plotly_json import figure_to_plain
from adsci.tools.base import ChartBlock, Tool, ToolContext, ToolOutput, summarize_frame

Kind = Literal["bar", "line", "scatter", "histogram", "box", "heatmap", "pie"]
Agg = Literal["sum", "mean", "count", "median"]


class CreateChartInput(BaseModel):
    data: DataRef = Field(description="Where the chart's data comes from.")
    kind: Kind
    x: str | None = None
    y: str | list[str] | None = None
    color: str | None = None
    title: str
    agg: Agg | None = Field(None, description="Aggregate y by x (and color) before plotting.")


def _prepare(df, args: CreateChartInput):
    ys = [args.y] if isinstance(args.y, str) else (args.y or [])
    if args.agg and args.x and args.kind not in ("histogram", "box", "heatmap"):
        keys = [args.x] + ([args.color] if args.color else [])
        if args.agg == "count":
            df = df.groupby(keys, dropna=False).size().reset_index(name="count")
            ys = ["count"]
        else:
            df = df.groupby(keys, dropna=False)[ys].agg(args.agg).reset_index()
    return df, ys


def build_figure(df, args: CreateChartInput):
    df, ys = _prepare(df, args)
    y = ys[0] if len(ys) == 1 else (ys or None)
    common = {"title": args.title}
    if args.kind == "bar":
        fig = px.bar(df, x=args.x, y=y, color=args.color, **common)
    elif args.kind == "line":
        fig = px.line(df.sort_values(args.x) if args.x else df, x=args.x, y=y, color=args.color, **common)
    elif args.kind == "scatter":
        fig = px.scatter(df, x=args.x, y=y, color=args.color, **common)
    elif args.kind == "histogram":
        fig = px.histogram(df, x=args.x, color=args.color, **common)
    elif args.kind == "box":
        fig = px.box(df, x=args.x, y=y, color=args.color, **common)
    elif args.kind == "heatmap":
        fig = px.density_heatmap(df, x=args.x, y=y, z=args.color, histfunc=args.agg or "count", **common)
    else:
        fig = px.pie(df, names=args.x, values=y, **common)
    return fig, df


def chart_code(args: CreateChartInput) -> str:
    """A notebook cell that rebuilds the chart; every value is a Python literal (repr), so quotes are safe."""
    return (
        f"df = sql({args.data.sql!r}, source={args.data.source!r})\n"
        "from adsci.tools.chart_tool import CreateChartInput, build_figure\n"
        f"fig, _ = build_figure(df, CreateChartInput.model_validate_json({args.model_dump_json()!r}))\n"
        "show(fig)"
    )


class CreateChartTool(Tool):
    name = "create_chart"
    description = (
        "Create a chart (bar, line, scatter, histogram, box, heatmap, pie) from a SQL query. Use agg to "
        "aggregate y by x. For complex visuals use run_python with px/go and show(fig)."
    )
    Input = CreateChartInput

    async def run(self, args: CreateChartInput, ctx: ToolContext) -> ToolOutput:
        df, warns = await ctx.load(args.data)
        fig, plotted = build_figure(df, args)
        figure = figure_to_plain(fig)
        code = chart_code(args)
        ctx.notebook.add_python(code, [PlotlyOutput(figure=figure)])
        summary = f"Created {args.kind} chart '{args.title}' from {len(df)} rows. Plotted data:\n"
        summary += summarize_frame(plotted)
        if warns:
            summary += "\n" + "\n".join(warns)
        return ToolOutput(
            summary=summary,
            ui=[ChartBlock(figure=figure, title=args.title)],
            code=code,
            language="python",
            data={"plotted": json.loads(plotted.head(100).to_json(orient="split", date_format="iso", index=False))},
        )
