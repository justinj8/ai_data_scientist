"""Load CSV and XLSX files into a per-session DuckDB database."""

from __future__ import annotations

import csv
import re
import threading
import warnings as _warnings
from pathlib import Path
from typing import Literal

import duckdb
import openpyxl
import pandas as pd

from adsci.data.base import ColumnInfo, LoadReport, TableInfo
from adsci.data.identifiers import dedupe, normalize_identifier
from adsci.data.sql_guard import guard_sql

_NUMBER = re.compile(r"^[-+]?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?$")
_HAS_DIGIT = re.compile(r"\d")
_META_COLUMNS = "_adsci_columns"
_META_TABLES = "_adsci_tables"


_DATEISH = re.compile(
    r"^(\d{4}-\d{1,2}-\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?)?"
    r"|\d{4}/\d{1,2}/\d{1,2}"
    r"|\d{1,2}[/.-]\d{1,2}[/.-](\d{4}|\d{2})"
    r"|[A-Za-z]{3,9}\.? \d{1,2},? \d{4}"
    r"|\d{1,2} [A-Za-z]{3,9}\.? \d{4})$"
)
_ISO = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}")
_DATE_FORMATS = [
    "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y", "%d-%m-%Y", "%m-%d-%Y", "%d/%m/%y", "%m/%d/%y",
    "%b %d %Y", "%b %d, %Y", "%B %d %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y",
]


def _parse_dates(values: pd.Series) -> tuple[str | None, pd.Series | None, list[str]]:
    """Pick ONE format for the whole column (no per-value guessing); returns (format, parsed, notes)."""
    if values.str.match(_ISO).mean() >= 0.9:
        return "ISO8601", pd.to_datetime(values, format="ISO8601", errors="coerce"), []
    results = {f: pd.to_datetime(values, format=f, errors="coerce") for f in _DATE_FORMATS}
    best = max(results, key=lambda f: results[f].notna().sum())
    notes: list[str] = []
    for day_first, month_first in (("%d/%m/%Y", "%m/%d/%Y"), ("%d-%m-%Y", "%m-%d-%Y"), ("%d/%m/%y", "%m/%d/%y")):
        if best in (day_first, month_first):
            d, m = results[day_first].notna().sum(), results[month_first].notna().sum()
            if d == m:  # every value fits both orders: genuinely ambiguous
                best = month_first
                notes.append("dates could be day/month or month/day; read as month/day")
    return best, results[best], notes


def _coerce_types(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Convert text columns that are really numbers (``1,234.50``) or dates; report anything lost."""
    warns: list[str] = []
    for col in df.columns:
        s = df[col]
        if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
            continue
        values = s.dropna().astype(str).str.strip()
        values = values[values != ""]
        if values.empty:
            continue
        if values.str.match(_NUMBER).all():
            df[col] = pd.to_numeric(s.astype(str).str.strip().str.replace(",", "", regex=False), errors="coerce")
            continue
        if values.str.match(_DATEISH).mean() < 0.9:
            continue
        fmt, parsed, notes = _parse_dates(values)
        if parsed is None or parsed.notna().mean() < 0.9:
            continue
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            df[col] = pd.to_datetime(s.astype("string").str.strip(), format=fmt, errors="coerce")
        lost = int(parsed.isna().sum())
        if lost:
            plural = lost != 1
            warns.append(f"Column {col}: {lost} value{'s' if plural else ''} couldn't be read as a date and "
                         f"{'were' if plural else 'was'} left empty.")
        warns += [f"Column {col}: {n}." for n in notes]
    return df, warns


def _detect_encoding(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def load_csv(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read a CSV with sniffed encoding and delimiter; returns the frame (original headers) and warnings."""
    path = Path(path)
    warns: list[str] = []
    encoding = _detect_encoding(path)
    with path.open(encoding=encoding, newline="") as f:
        sample = f.read(65536)
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","
    if encoding not in ("utf-8-sig",):
        warns.append(f"Read with {encoding} encoding.")
    df = pd.read_csv(path, sep=delimiter, encoding=encoding, low_memory=False)
    df.columns = dedupe([str(c).strip() for c in df.columns])
    df, notes = _coerce_types(df)
    return df, warns + notes


def _find_header(rows: list[tuple]) -> int:
    """The header is the first "wide" row: title and note rows above it fill only a cell or two.
    Header cells may be numbers (e.g. years), so only fill matters."""
    width = max((sum(v is not None and str(v).strip() != "" for v in r) for r in rows), default=0)
    for i, row in enumerate(rows[:10]):
        filled = sum(v is not None and str(v).strip() != "" for v in row)
        if width and filled >= 0.6 * width:
            return i
    return 0


def _header_name(v, i: int) -> str:
    if v is None or str(v).strip() == "":
        return f"column_{i + 1}"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def load_xlsx(path: Path, skipped: list[str] | None = None) -> dict[str, tuple[pd.DataFrame, list[str]]]:
    """Read every non-empty sheet. Empty sheets are skipped and reported via ``skipped``."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    try:
        for ws in wb.worksheets:
            rows = [r for r in ws.iter_rows(values_only=True) if any(v is not None and str(v).strip() for v in r)]
            if not rows:
                if skipped is not None:
                    skipped.append(f"Sheet '{ws.title}' is empty and was skipped.")
                continue
            warns: list[str] = []
            h = _find_header(rows)
            if h:
                warns.append(f"Sheet '{ws.title}': skipped {h} title row(s) above the header.")
            header = list(rows[h])
            while header and header[-1] is None:
                header.pop()
            names = dedupe([_header_name(v, i) for i, v in enumerate(header)])
            data = [list(r[: len(names)]) + [None] * (len(names) - len(r)) for r in rows[h + 1 :]]
            df = pd.DataFrame(data, columns=names)
            df, notes = _coerce_types(df)
            out[ws.title] = (df, warns + notes)
    finally:
        wb.close()
    return out


class FileDataSource:
    """All uploaded files of a session, stored as tables in one DuckDB database."""

    kind: Literal["file"] = "file"
    dialect = "duckdb"

    def __init__(self, name: str, db_path: Path, row_limit: int = 10_000, max_limit: int = 1_000_000) -> None:
        self.name = name
        self.row_limit = row_limit
        self.max_limit = max_limit
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # No file or network access from SQL: uploads are loaded through pandas, so queries never need it.
        self._con = duckdb.connect(str(self.db_path), config={"enable_external_access": False})
        self._lock = threading.Lock()
        self._con.execute(
            f"CREATE TABLE IF NOT EXISTS {_META_COLUMNS} (table_name TEXT, column_name TEXT, original_name TEXT)"
        )
        self._con.execute(f"CREATE TABLE IF NOT EXISTS {_META_TABLES} (table_name TEXT, original_name TEXT)")

    def add_file(self, path: Path) -> list[LoadReport]:
        path = Path(path)
        stem = normalize_identifier(path.stem)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df, warns = load_csv(path)
            return [self._store(stem, path.name, df, warns)]
        if suffix == ".xlsx":
            skipped: list[str] = []
            sheets = load_xlsx(path, skipped)
            reports = [
                self._store(f"{stem}__{normalize_identifier(sheet)}", f"{path.name} [{sheet}]", df, warns)
                for sheet, (df, warns) in sheets.items()
            ]
            if reports:
                reports[0].warnings.extend(skipped)
            return reports
        raise ValueError(f"Unsupported file type '{suffix}'. Upload a .csv or .xlsx file.")

    def _store(self, table: str, original: str, df: pd.DataFrame, warns: list[str]) -> LoadReport:
        originals = [str(c) for c in df.columns]
        df = df.copy()
        df.columns = dedupe([normalize_identifier(c) for c in originals])
        with self._lock:
            self._con.register("_adsci_upload", df)
            try:
                self._con.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM _adsci_upload')
            finally:
                self._con.unregister("_adsci_upload")
            self._con.execute(f"DELETE FROM {_META_COLUMNS} WHERE table_name = ?", [table])
            self._con.execute(f"DELETE FROM {_META_TABLES} WHERE table_name = ?", [table])
            self._con.executemany(
                f"INSERT INTO {_META_COLUMNS} VALUES (?, ?, ?)",
                [[table, n, o] for n, o in zip(df.columns, originals, strict=True)],
            )
            self._con.execute(f"INSERT INTO {_META_TABLES} VALUES (?, ?)", [table, original])
        return LoadReport(table=table, rows=len(df), warnings=list(warns))

    def schema(self) -> list[TableInfo]:
        with self._lock:
            cols = self._con.execute(
                "SELECT table_name, column_name, data_type, is_nullable FROM information_schema.columns "
                "WHERE table_schema = 'main' AND table_name NOT LIKE '\\_adsci%' ESCAPE '\\' "
                "ORDER BY table_name, ordinal_position"
            ).fetchall()
            originals = {
                (t, c): o for t, c, o in self._con.execute(f"SELECT * FROM {_META_COLUMNS}").fetchall()
            }
            table_originals = dict(self._con.execute(f"SELECT * FROM {_META_TABLES}").fetchall())
            tables: dict[str, TableInfo] = {}
            for t, c, dtype, nullable in cols:
                info = tables.setdefault(t, TableInfo(name=t, columns=[], original_name=table_originals.get(t)))
                info.columns.append(
                    ColumnInfo(name=c, dtype=dtype, nullable=nullable == "YES", original_name=originals.get((t, c)))
                )
            for t, info in tables.items():
                info.row_count = self._con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
        return list(tables.values())

    def query(self, sql: str, limit: int | None = None) -> pd.DataFrame:
        """Run a guarded, read-only query returning ``limit`` rows (default ``row_limit``, capped at ``max_limit``)."""
        sql = guard_sql(sql, self.dialect, min(limit or self.row_limit, self.max_limit))
        with self._lock:
            return self._con.execute(sql).df()

    def test_connection(self) -> None:
        with self._lock:
            self._con.execute("SELECT 1")

    def close(self) -> None:
        with self._lock:
            self._con.close()
