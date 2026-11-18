from __future__ import annotations

from pydantic import BaseModel, Field

from adsci.kernel.manager import ImageOutput, PlotlyOutput, TableOutput, TextOutput
from adsci.tools.base import (
    ChartBlock,
    ErrorBlock,
    ImageBlock,
    TableBlock,
    TextBlock,
    Tool,
    ToolContext,
    ToolOutput,
    clip,
)


class RunPythonInput(BaseModel):
    code: str = Field(description="Python code to run in the persistent session kernel.")
    purpose: str = Field(description="One short sentence: what this code is for.")


def _table_csv(o: TableOutput, n: int = 20) -> str:
    lines = [",".join(o.columns)] + [",".join("" if v is None else str(v) for v in r) for r in o.rows[:n]]
    more = f"\n... ({o.total_rows} rows total)" if o.total_rows > n else ""
    return "\n".join(lines) + more


class RunPythonTool(Tool):
    name = "run_python"
    description = (
        "Run Python in a persistent Jupyter kernel (variables persist between calls). Preloaded: pd, np, px, go, "
        "stats (scipy.stats), sm (statsmodels.api), sql(query, source='files') -> DataFrame, show(fig) to display "
        "a Plotly figure. The last expression's value is displayed. Use for custom analysis, cross-source joins "
        "and anything the specialised tools don't cover."
    )
    Input = RunPythonInput

    async def run(self, args: RunPythonInput, ctx: ToolContext) -> ToolOutput:
        kernel = await ctx.kernel()
        r = await kernel.execute(args.code, ctx.settings.kernel_timeout_s)
        ui: list = []
        parts: list[str] = []
        tables = []
        if r.stdout:
            ui.append(TextBlock(text=r.stdout))
            parts.append("stdout:\n" + clip(r.stdout, 2000))
        for o in r.outputs:
            if isinstance(o, TableOutput):
                ui.append(TableBlock(columns=o.columns, rows=o.rows, total_rows=o.total_rows))
                parts.append("table:\n" + _table_csv(o))
                tables.append({"columns": o.columns, "rows": o.rows[:50]})
            elif isinstance(o, PlotlyOutput):
                title = ((o.figure.get("layout") or {}).get("title") or {}).get("text")
                ui.append(ChartBlock(figure=o.figure, title=title))
                parts.append(f"chart displayed: {title or 'untitled'}")
            elif isinstance(o, ImageOutput):
                ui.append(ImageBlock(png_b64=o.png_b64))
                parts.append("image displayed")
            elif isinstance(o, TextOutput):
                ui.append(TextBlock(text=o.text))
                parts.append("value: " + clip(o.text, 1500))
        if r.error:
            tb = "\n".join(r.error.traceback.splitlines()[-15:])
            ui.append(ErrorBlock(message=f"{r.error.ename}: {r.error.evalue}", detail=r.error.traceback or None))
            parts.append(f"ERROR {r.error.ename}: {r.error.evalue}\n{tb}")
        if r.restarted:
            parts.append("NOTE: the kernel was restarted; earlier variables are gone.")
        ctx.notebook.add_python(args.code, r.outputs)
        summary = clip("\n".join(parts) or "(no output)", ctx.settings.model_summary_chars)
        return ToolOutput(
            summary=("ERROR: " if r.error else "") + summary,
            ui=ui,
            code=args.code,
            language="python",
            is_error=r.error is not None,
            data={"stdout": clip(r.stdout, 5000), "tables": tables},
        )
