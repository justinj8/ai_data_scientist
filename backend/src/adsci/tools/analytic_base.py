"""Shared plumbing for tools that wrap a pure analytics function."""

from __future__ import annotations

from adsci.data.base import DataRef
from adsci.kernel.manager import Output


def notebook_code(data: DataRef, call: str) -> str:
    """Notebook cell that reproduces a tool call: load the data with sql(), then call the analytics function."""
    return f'df = sql("""{data.sql}""", source="{data.source}")\n{call}'


def record(ctx, data: DataRef, call: str, outputs: list[Output] | None = None) -> str:
    code = notebook_code(data, call)
    ctx.notebook.add_python(code, outputs or [])
    return code
