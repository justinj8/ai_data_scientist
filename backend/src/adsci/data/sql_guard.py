"""Read-only SQL guard: parse with sqlglot, allow a single SELECT, cap the row count."""

from __future__ import annotations

import re

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
# Functions that read files, run SQL on other servers (whose SQL hides in a string), or have side effects
# (killing sessions, changing settings, locks, notifications). Matched case-insensitively.
_FORBIDDEN_FUNC_EXACT = {
    "glob", "load_file", "pg_sleep", "sleep", "benchmark", "query_table", "query",
    # federated / nested-SQL carriers
    "openquery", "openrowset", "opendatasource", "external_query", "query_to_xml", "query_to_xmlschema",
    "query_to_xml_and_xmlschema", "cursor_to_xml", "cursor_to_xmlschema",
    # settings and session control
    "set_config", "pg_notify", "pg_reload_conf", "pg_rotate_logfile", "pg_promote", "pg_switch_wal",
    "pg_log_backend_memory_contexts",
    # MySQL locks
    "get_lock", "release_lock", "release_all_locks", "master_pos_wait", "source_pos_wait",
}
_FORBIDDEN_FUNC_PREFIX = (
    "read_", "pg_read", "pg_ls", "pg_stat_file", "pg_file", "lo_", "dblink", "xp_", "sp_",
    "pg_terminate", "pg_cancel", "pg_advisory", "pg_try_advisory", "pg_create_", "pg_drop_", "pg_replication",
    "pg_logical", "pg_backup", "pg_start_backup", "pg_stop_backup", "pg_import_",
    "system$",  # Snowflake system functions (abort sessions, cancel queries, ...)
)
_FORBIDDEN_FUNC_SUFFIX = ("_scan",)
# A string literal that is itself a SQL statement, e.g. a UDF or linked server asked to run it.
_EMBEDDED_STATEMENT = re.compile(
    r"^\s*(insert|update|delete|merge|upsert|replace|drop|create|alter|truncate|grant|revoke|exec|execute|call"
    r"|copy|attach|detach|set|use|begin|commit|rollback|kill)\b",
    re.IGNORECASE,
)
# DuckDB treats a quoted table name like '/data/x.csv' as "read this file" (a replacement scan).
_FILE_LIKE = re.compile(r"[/\\*?]|\.(csv|tsv|txt|json|jsonl|ndjson|parquet|xlsx?|gz|zst|db|duckdb|sqlite3?)$", re.I)
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
        if isinstance(node, exp.Table) and _FILE_LIKE.search(node.name or ""):
            raise UnsafeSQLError(
                f"'{node.name}' looks like a file path; queries may only read loaded tables, not files."
            )
        if isinstance(node, exp.Func):
            name = _function_name(node)
            if (
                name in _FORBIDDEN_FUNC_EXACT
                or name.startswith(_FORBIDDEN_FUNC_PREFIX)
                or name.endswith(_FORBIDDEN_FUNC_SUFFIX)
            ):
                raise UnsafeSQLError(
                    f"The function {name}() is not allowed: queries may only read loaded tables, without side "
                    "effects or access to other servers."
                )
            for arg in node.find_all(exp.Literal):
                if arg.is_string and _EMBEDDED_STATEMENT.match(arg.this or ""):
                    raise UnsafeSQLError(
                        "A SQL statement inside a function argument is not allowed: this connection is read-only."
                    )

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
