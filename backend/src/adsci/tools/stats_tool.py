from __future__ import annotations

import asyncio
from typing import Literal

import plotly.express as px
from pydantic import BaseModel, Field, model_validator

from adsci.analytics.stats import association_test, compare_groups
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


class StatsTestInput(BaseModel):
    data: DataRef
    kind: Literal["compare_groups", "association"] = Field(
        description="compare_groups: numeric `value` across levels of `group`. association: any two columns x, y."
    )
    value: str | None = None
    group: str | None = None
    x: str | None = None
    y: str | None = None

    @model_validator(mode="after")
    def _needs_columns(self):
        if self.kind == "compare_groups" and not (self.value and self.group):
            raise ValueError("compare_groups needs both 'value' and 'group'")
        if self.kind == "association" and not (self.x and self.y):
            raise ValueError("association needs both 'x' and 'y'")
        return self


class StatsTestTool(Tool):
    name = "statistical_test"
    description = (
        "Hypothesis test that checks its own assumptions and reports an effect size and CI. compare_groups picks "
        "Student/Welch t, Mann-Whitney, ANOVA, Welch ANOVA or Kruskal-Wallis (with Holm post-hoc); association "
        "picks chi-square/Fisher or Pearson/Spearman. Use it before claiming any difference is real."
    )
    Input = StatsTestInput

    async def run(self, args: StatsTestInput, ctx: ToolContext) -> ToolOutput:
        df, warns = await ctx.load(args.data)
        if args.kind == "compare_groups":
            r = await asyncio.to_thread(compare_groups, df, args.value, args.group)
            call = f"compare_groups(df, value={args.value!r}, group={args.group!r})"
        else:
            r = await asyncio.to_thread(association_test, df, args.x, args.y)
            call = f"association_test(df, x={args.x!r}, y={args.y!r})"
        ui: list = [MetricsBlock(items=[
            Metric(label="Test", value=r.test),
            Metric(label="p-value", value="< 0.001" if r.p_value < 0.001 else f"{r.p_value:.4f}"),
            Metric(label=r.effect_size_name or "Effect size",
                   value="n/a" if r.effect_size is None else f"{r.effect_size:.3f}", detail=r.effect_magnitude),
            Metric(label="n", value=f"{r.n:,}"),
        ])]
        if r.groups:
            ui.append(TableBlock(
                columns=["group", "n", "mean", "median", "std"],
                rows=[[k, g.n, round(g.mean, 4), round(g.median, 4), round(g.std, 4)] for k, g in r.groups.items()],
                total_rows=len(r.groups),
            ))
            if args.kind == "compare_groups" or r.groups:
                num, cat = (args.value, args.group) if args.kind == "compare_groups" else (None, None)
                if num and cat:
                    plot_df = df[[num, cat]].dropna()
                    if len(plot_df) > 20_000:
                        plot_df = plot_df.sample(20_000, random_state=0)
                    fig = px.box(plot_df, x=cat, y=num, points=False, title=f"{num} by {cat}")
                    ui.append(ChartBlock(figure=figure_to_plain(fig), title=f"{num} by {cat}"))
        notes = [f"- {'✓' if a.passed else '✗'} {a.name}: {a.detail}" for a in r.assumptions]
        notes += [f"- ⚠ {w}" for w in r.warnings + warns]
        ui.append(TextBlock(format="markdown", text=r.interpretation + ("\n\n" + "\n".join(notes) if notes else "")))

        lines = [r.interpretation, f"statistic={r.statistic:.6g}, p={r.p_value:.6g}, n={r.n}"]
        if r.effect_size is not None:
            lines.append(f"effect: {r.effect_size_name}={r.effect_size:.6g} ({r.effect_magnitude})")
        if r.ci:
            lines.append(f"{r.ci_label}: [{r.ci[0]:.6g}, {r.ci[1]:.6g}]")
        if r.groups:
            lines.append("groups: " + "; ".join(
                f"{k}: n={g.n} mean={g.mean:.6g} median={g.median:.6g}" for k, g in r.groups.items()))
        for ph in r.posthoc:
            lines.append(f"posthoc {ph.a} vs {ph.b}: diff={ph.mean_diff:.6g} p_holm={ph.p_adjusted:.4g}")
        lines += [f"assumption {a.name}: {'passed' if a.passed else 'FAILED'} ({a.detail})" for a in r.assumptions]
        lines += [f"WARNING: {w}" for w in r.warnings + warns]
        return ToolOutput(
            summary="\n".join(lines)[: ctx.settings.model_summary_chars],
            ui=ui,
            code=record(ctx, args.data, call),
            language="python",
            data=r.model_dump(),
        )
