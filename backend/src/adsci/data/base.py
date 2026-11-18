"""Core data-source types shared by file and database sources."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import pandas as pd
from pydantic import BaseModel


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    nullable: bool = True
    original_name: str | None = None


class TableInfo(BaseModel):
    name: str
    columns: list[ColumnInfo]
    row_count: int | None = None
    original_name: str | None = None


class LoadReport(BaseModel):
    table: str
    rows: int
    warnings: list[str] = []


class DataRef(BaseModel):
    """A pointer to data: a read-only SQL query against a named source."""

    source: str
    sql: str


@runtime_checkable
class DataSource(Protocol):
    name: str
    kind: Literal["file", "database"]
    dialect: str  # sqlglot dialect name

    def schema(self) -> list[TableInfo]: ...

    def query(self, sql: str, limit: int | None = None) -> pd.DataFrame: ...

    def test_connection(self) -> None: ...

    def close(self) -> None: ...
