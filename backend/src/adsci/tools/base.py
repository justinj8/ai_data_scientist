"""Tool framework: typed inputs, dual outputs (compact summary for the model, rich blocks for the UI)."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from adsci.config import Settings
from adsci.data.base import DataRef
from adsci.data.sql_guard import UnsafeSQLError
from adsci.llm.types import ToolSpec

# ---------------------------------------------------------------- UI blocks


class TableBlock(BaseModel):
    type: Literal["table"] = "table"
    columns: list[str]
    rows: list[list]
    total_rows: int


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str
    format: Literal["plain", "markdown"] = "plain"


class ChartBlock(BaseModel):
    type: Literal["chart"] = "chart"
    figure: dict
    title: str | None = None


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    png_b64: str


class Metric(BaseModel):
    label: str
    value: str
    detail: str | None = None


class MetricsBlock(BaseModel):
    type: Literal["metrics"] = "metrics"
    items: list[Metric]


class ErrorBlock(BaseModel):
    type: Literal["error"] = "error"
    message: str
    detail: str | None = None


UIBlock = Annotated[
    TableBlock | TextBlock | ChartBlock | ImageBlock | MetricsBlock | ErrorBlock, Field(discriminator="type")
]


class ToolOutput(BaseModel):
    summary: str  # what the model sees
    ui: list[UIBlock] = []
    code: str | None = None
    language: Literal["sql", "python"] | None = None
    is_error: bool = False
    data: dict[str, Any] | None = None  # structured results, used for numeric grounding


# ---------------------------------------------------------------- helpers


def clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 40] + f"\n... [truncated {len(text) - n + 40} chars]"


def _jsonable_rows(df: pd.DataFrame) -> tuple[list[str], list[list]]:
    payload = json.loads(df.to_json(orient="split", date_format="iso", default_handler=str, index=False))
    return [str(c) for c in payload["columns"]], payload["data"]


def frame_to_table(df: pd.DataFrame, max_rows: int = 50) -> TableBlock:
    cols, rows = _jsonable_rows(df.head(max_rows))
    return TableBlock(columns=cols, rows=rows, total_rows=int(len(df)))


def summarize_frame(df: pd.DataFrame, max_rows: int = 20, max_cols: int = 25) -> str:
    """A compact text view of a DataFrame that stays small however big the frame is."""
    n_rows, n_cols = df.shape
    view = df.iloc[:, :max_cols]
    lines = [f"Shape: {n_rows} rows x {n_cols} columns"]
    lines.append("Columns: " + ", ".join(f"{c} ({t})" for c, t in zip(view.columns, view.dtypes.astype(str), strict=True)))
    if n_cols > max_cols:
        lines.append(f"(showing first {max_cols} of {n_cols} columns)")
    shown = view.head(max_rows)
    lines.append(f"First {len(shown)} rows (CSV):")
    lines.append(shown.to_csv(index=False, float_format="%.6g").strip())
    if n_rows > max_rows:
        numeric = view.select_dtypes("number")
        if not numeric.empty:
            desc = numeric.iloc[:, :10].describe().T[["mean", "std", "min", "max"]]
            lines.append("Numeric summary (first 10 numeric columns):")
            lines.append(desc.to_csv(float_format="%.6g").strip())
    return clip("\n".join(lines), 3500)


# ---------------------------------------------------------------- context and tools


@dataclass
class ToolContext:
    session: Any  # adsci.app.sessions.Session
    settings: Settings

    @property
    def notebook(self):
        from adsci.kernel.notebook import NotebookRecorder

        s = self.session
        if s.notebook is None:
            s.notebook = NotebookRecorder(s.id, self.settings.internal_base_url, s.internal_token, list(s.sources))
        s.notebook.sources = list(s.sources)
        return s.notebook

    async def kernel(self):
        from adsci.kernel.manager import KernelSession

        s = self.session
        if s.kernel is None:
            k = KernelSession(
                s.id,
                s.dir / "work",
                internal_url=self.settings.internal_base_url,
                token=s.internal_token,
                preview_rows=self.settings.preview_rows,
            )
            await k.start()
            s.kernel = k
        return s.kernel

    async def query(self, source: str, sql: str, limit: int | None = None) -> pd.DataFrame:
        src = self.session.get_source(source)
        return await asyncio.to_thread(src.query, sql, limit)

    async def load(self, data: DataRef) -> tuple[pd.DataFrame, list[str]]:
        """Load data for analysis (up to ``analysis_row_limit`` rows); returns the frame and warnings."""
        limit = self.settings.analysis_row_limit
        df = await self.query(data.source, data.sql, limit)
        warns = []
        if len(df) >= limit:
            warns.append(f"Data was truncated at {limit:,} rows; aggregate or filter in SQL for complete results.")
        return df, warns


class Tool(ABC):
    name: str
    description: str
    Input: type[BaseModel]

    @abstractmethod
    async def run(self, args: BaseModel, ctx: ToolContext) -> ToolOutput: ...

    def spec(self) -> ToolSpec:
        schema = self.Input.model_json_schema()
        return ToolSpec(name=self.name, description=self.description, parameters=schema)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    async def execute(self, name: str, raw_args: dict, ctx: ToolContext) -> ToolOutput:
        tool = self._tools.get(name)
        if tool is None:
            return _error(f"Unknown tool '{name}'. Available tools: {', '.join(self._tools)}.")
        try:
            args = tool.Input.model_validate(raw_args)
        except ValidationError as e:
            problems = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'arguments'}: {err['msg']}" for err in e.errors())
            return _error(f"Invalid arguments for {name}: {problems}")
        try:
            return await tool.run(args, ctx)
        except UnsafeSQLError as e:
            return _error(str(e))
        except KeyError as e:
            return _error(str(e.args[0]) if e.args else str(e))
        except Exception as e:  # noqa: BLE001 - every failure goes back to the model to self-correct
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            return _error(f"{name} failed: {type(e).__name__}: {msg}")


def _error(message: str) -> ToolOutput:
    return ToolOutput(summary=f"ERROR: {message}", ui=[ErrorBlock(message=message)], is_error=True)


def default_registry() -> ToolRegistry:
    from adsci.tools import registry_tools

    return ToolRegistry(registry_tools())
