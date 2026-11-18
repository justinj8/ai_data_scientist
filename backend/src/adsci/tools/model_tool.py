from __future__ import annotations

import asyncio
from typing import Literal

import plotly.express as px
from pydantic import BaseModel, Field

from adsci.analytics.modeling import train_and_evaluate
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


class TrainModelInput(BaseModel):
    data: DataRef
    target: str
    features: list[str] | None = Field(None, description="Feature columns; default: all usable columns.")
    task: Literal["classification", "regression"] | None = None


class TrainModelTool(Tool):
    name = "train_model"
    description = (
        "Train and evaluate predictive models (baseline vs linear vs gradient boosting) with 5-fold CV on 80% of "
        "the data and a held-out 20% test set. Drops identifiers, constants and leaky features, and reports "
        "permutation feature importance. Use it to find what predicts a target."
    )
    Input = TrainModelInput

    async def run(self, args: TrainModelInput, ctx: ToolContext) -> ToolOutput:
        df, warns = await ctx.load(args.data)
        r = await asyncio.to_thread(train_and_evaluate, df, args.target, args.features, args.task)
        pm = r.primary_metric
        best = next(m for m in r.models if m.name == r.best_model)
        ui: list = [MetricsBlock(items=[
            Metric(label="Task", value=r.task),
            Metric(label="Best model", value=r.best_model.replace("_", " ")),
            Metric(label=f"Test {pm}", value=f"{best.test[pm]:.3f}", detail=f"baseline {r.baseline[pm]:.3f}"),
            Metric(label="Train / test rows", value=f"{r.n_train:,} / {r.n_test:,}"),
        ])]
        metrics = list(best.test)
        ui.append(TableBlock(
            columns=["model"] + [f"cv {m}" for m in metrics] + [f"test {m}" for m in metrics],
            rows=[[m.name] + [f"{m.cv_mean[k]:.4f} ± {m.cv_std[k]:.4f}" for k in metrics]
                  + [round(m.test[k], 4) for k in metrics] for m in r.models],
            total_rows=len(r.models),
        ))
        top = r.importances[:15]
        fig = px.bar(x=[i.importance for i in top][::-1], y=[i.feature for i in top][::-1], orientation="h",
                     error_x=[i.std for i in top][::-1], title=f"Permutation importance ({pm} drop)",
                     labels={"x": f"decrease in {pm}", "y": "feature"})
        ui.append(ChartBlock(figure=figure_to_plain(fig), title="Feature importance"))
        notes = [f"- dropped `{f}`: {why}" for f, why in r.dropped_features] + [f"- ⚠ {w}" for w in r.warnings + warns]
        if notes:
            ui.append(TextBlock(format="markdown", text="\n".join(notes)))

        lines = [f"{r.task} on target '{r.target}' ({r.n_train} train / {r.n_test} test rows). Primary metric: {pm}."]
        for m in r.models:
            cv = ", ".join(f"{k}={m.cv_mean[k]:.4f}±{m.cv_std[k]:.4f}" for k in m.cv_mean)
            te = ", ".join(f"{k}={v:.4f}" for k, v in m.test.items())
            lines.append(f"- {m.name}: CV [{cv}]; test [{te}]")
        lines.append(f"Best model: {r.best_model}.")
        lines.append("Permutation importance (test set): " + "; ".join(
            f"{i.feature}={i.importance:.4f}±{i.std:.4f}" for i in r.importances[:10]))
        if r.class_balance:
            lines.append("Class balance: " + ", ".join(f"{k}={v:.3f}" for k, v in r.class_balance.items()))
        lines += [f"Dropped {f}: {why}" for f, why in r.dropped_features]
        lines += [f"WARNING: {w}" for w in r.warnings + warns]
        lines.append("Importance shows predictive value, not causal effect.")
        call = (f"train_and_evaluate(df, target={args.target!r}, features={args.features!r}, "
                f"task={args.task!r})")
        return ToolOutput(
            summary="\n".join(lines)[: ctx.settings.model_summary_chars], ui=ui,
            code=record(ctx, args.data, call), language="python", data=r.model_dump(),
        )
