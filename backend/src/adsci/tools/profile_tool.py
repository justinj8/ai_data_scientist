from __future__ import annotations

import plotly.express as px
from pydantic import BaseModel, Field

from adsci.analytics.profile import profile_frame
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


class ProfileInput(BaseModel):
    data: DataRef
    target: str | None = Field(None, description="Optional target column, to check features for leakage.")


class ProfileTool(Tool):
    name = "profile_dataset"
    description = (
        "Profile a dataset: column types, missingness, distributions, outliers, correlations and data-quality "
        "warnings (ID-like columns, constants, skew, duplicates, target leakage). Run this before modeling."
    )
    Input = ProfileInput

    async def run(self, args: ProfileInput, ctx: ToolContext) -> ToolOutput:
        df, warns = await ctx.load(args.data)
        p = profile_frame(df, target=args.target)
        missing_total = float(df.isna().mean().mean() * 100) if len(df) else 0.0
        ui: list = [
            MetricsBlock(items=[
                Metric(label="Rows", value=f"{p.n_rows:,}"),
                Metric(label="Columns", value=str(p.n_cols)),
                Metric(label="Missing cells", value=f"{missing_total:.1f}%"),
                Metric(label="Duplicate rows", value=f"{p.duplicate_rows:,}"),
            ]),
            TableBlock(
                columns=["column", "type", "missing %", "unique", "mean", "median", "outliers %"],
                rows=[
                    [c.name, c.semantic, round(c.missing_pct, 2), c.n_unique,
                     None if not c.stats else round(c.stats["mean"], 4),
                     None if not c.stats else round(c.stats["median"], 4),
                     None if c.outlier_pct is None else round(c.outlier_pct, 2)]
                    for c in p.columns
                ],
                total_rows=len(p.columns),
            ),
        ]
        if p.correlations:
            names = list(p.correlations)
            z = [[p.correlations[a][b] for b in names] for a in names]
            fig = px.imshow(z, x=names, y=names, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                            title="Correlation matrix", text_auto=".2f")
            ui.append(ChartBlock(figure=figure_to_plain(fig), title="Correlation matrix"))
        if p.warnings:
            ui.append(TextBlock(format="markdown", text="\n".join(f"- **{w.kind}**: {w.message}" for w in p.warnings)))

        lines = [f"{p.n_rows} rows x {p.n_cols} columns; {p.duplicate_rows} duplicate rows."]
        for c in p.columns:
            extra = ""
            if c.stats:
                extra = f" mean={c.stats['mean']:.4g} median={c.stats['median']:.4g} std={c.stats['std']:.4g}"
            elif c.top_values:
                extra = " top=" + ", ".join(f"{t['value']}({t['pct']}%)" for t in c.top_values[:5])
            lines.append(f"- {c.name}: {c.semantic}, missing {c.missing_pct:.1f}%, unique {c.n_unique}{extra}")
        strong = sorted(
            ((a, b, r) for a, row in p.correlations.items() for b, r in row.items() if a < b and abs(r) >= 0.3),
            key=lambda t: -abs(t[2]),
        )[:10]
        if strong:
            lines.append("Notable correlations: " + "; ".join(f"{a}~{b} r={r:.2f}" for a, b, r in strong))
        lines += [f"WARNING {w.kind}: {w.message}" for w in p.warnings] + warns
        code = record(ctx, args.data, f"profile_frame(df, target={args.target!r})")
        return ToolOutput(
            summary="\n".join(lines)[: ctx.settings.model_summary_chars],
            ui=ui,
            code=code,
            language="python",
            data=p.model_dump(exclude={"correlations"}),
        )
