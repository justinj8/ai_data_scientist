from __future__ import annotations

from pydantic import BaseModel, Field

from adsci.kernel.manager import TableOutput
from adsci.tools.base import Tool, ToolContext, ToolOutput, frame_to_table, summarize_frame


class RunSQLInput(BaseModel):
    source: str = Field(description="Name of the data source to query (see the schema in the system prompt).")
    query: str = Field(description="A single read-only SELECT statement in the source's SQL dialect.")
    purpose: str = Field(description="One short sentence: what this query is for.")


class RunSQLTool(Tool):
    name = "run_sql"
    description = (
        "Run a read-only SELECT against one data source and get the result table. Prefer aggregating in SQL. "
        "Results shown to you are truncated; full results go to the user."
    )
    Input = RunSQLInput

    async def run(self, args: RunSQLInput, ctx: ToolContext) -> ToolOutput:
        df = await ctx.query(args.source, args.query)
        table = frame_to_table(df, ctx.settings.preview_rows)
        note = ""
        if len(df) >= ctx.settings.sql_row_limit:
            note = f"\nNOTE: result truncated at {ctx.settings.sql_row_limit:,} rows; aggregate in SQL instead."
        ctx.notebook.add_sql(
            args.query, args.source, [TableOutput(columns=table.columns, rows=table.rows, total_rows=table.total_rows)]
        )
        return ToolOutput(
            summary=f"Query on '{args.source}' returned {len(df)} rows.{note}\n{summarize_frame(df)}",
            ui=[table],
            code=args.query,
            language="sql",
            data={"columns": table.columns, "rows": table.rows[:100], "row_count": len(df)},
        )
