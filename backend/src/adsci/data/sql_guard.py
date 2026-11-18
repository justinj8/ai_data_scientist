"""Read-only SQL guard: parse with sqlglot, allow a single SELECT, cap the row count."""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, SqlglotError


class UnsafeSQLError(ValueError):
    """Raised for SQL that is not a single read-only SELECT. The message is safe to show to users."""


_FORBIDDEN_NODE_NAMES = [
    "Insert", "Update", "Delete", "Drop", "Create", "Alter", "Command", "Copy", "Merge", "Into",
    "Pragma", "Attach", "Detach", "Use", "Set", "Grant", "Revoke", "TruncateTable", "LoadData", "Transaction",
    "Commit", "Rollback", "Analyze", "Kill", "Refresh", "Cache", "Uncache", "Describe",
]
_FORBIDDEN_NODES = tuple(t for n in _FORBIDDEN_NODE_NAMES if (t := getattr(exp, n, None)) is not None)
_FORBIDDEN_FUNC_EXACT = {"glob", "load_file", "pg_sleep", "sleep", "benchmark", "query_table", "query"}
_FORBIDDEN_FUNC_PREFIX = ("read_", "pg_read", "pg_ls", "pg_stat_file", "lo_", "dblink", "xp_", "sp_")
_FORBIDDEN_FUNC_SUFFIX = ("_scan",)
_ALLOWED_ROOTS = tuple(t for n in ("Select", "Union", "Intersect", "Except", "SetOperation") if (t := getattr(exp, n, None)))


def _function_name(node: exp.Expression) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    try:
        return node.sql_name().lower()
    except Exception:  # pragma: no cover - defensive
        return type(node).__name__.lower()


def guard_sql(sql: str, dialect: str, limit: int) -> str:
    """Return ``sql`` rewritten with a row limit, or raise :class:`UnsafeSQLError`."""
    if not sql or not sql.strip():
        raise UnsafeSQLError("The query is empty.")
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except (ParseError, SqlglotError) as e:
        raise UnsafeSQLError(f"Could not parse the SQL ({dialect}): {str(e).splitlines()[0]}") from e
    if len(statements) != 1:
        raise UnsafeSQLError("Only a single SELECT statement is allowed (this connection is read-only).")
    root = statements[0]
    if not isinstance(root, _ALLOWED_ROOTS):
        raise UnsafeSQLError(
            f"Only SELECT queries are allowed (this connection is read-only); got {root.key.upper()}."
        )
    for node in root.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise UnsafeSQLError(f"{node.key.upper()} is not allowed: this connection is read-only.")
        if isinstance(node, exp.Func):
            name = _function_name(node)
            if (
                name in _FORBIDDEN_FUNC_EXACT
                or name.startswith(_FORBIDDEN_FUNC_PREFIX)
                or name.endswith(_FORBIDDEN_FUNC_SUFFIX)
            ):
                raise UnsafeSQLError(f"The function {name}() is not allowed: queries may only read tables.")

    existing = root.args.get("limit")
    current: int | None = None
    if isinstance(existing, exp.Limit):
        lit = existing.expression
        if isinstance(lit, exp.Literal) and lit.is_int:
            current = int(lit.this)
    elif isinstance(existing, exp.Fetch):
        count = existing.args.get("count")
        if isinstance(count, exp.Literal) and count.is_int:
            current = int(count.this)
    if current is None or current > limit:
        root = root.limit(limit, copy=False) if hasattr(root, "limit") else root
    return root.sql(dialect=dialect)
