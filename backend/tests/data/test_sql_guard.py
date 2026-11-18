import pytest

from adsci.data.sql_guard import UnsafeSQLError, guard_sql

DIALECTS = ["duckdb", "postgres", "mysql", "sqlite", "tsql", "snowflake", "bigquery"]


@pytest.mark.parametrize("dialect", DIALECTS)
def test_select_gets_limit(dialect):
    out = guard_sql("select a from t", dialect, 100).lower()
    assert "limit 100" in out or "top 100" in out or "fetch first 100" in out


@pytest.mark.parametrize(
    "bad",
    [
        "delete from t",
        "drop table t",
        "insert into t values (1)",
        "update t set a=1",
        "select 1; drop table t",
        "create table x as select 1",
        "copy t to '/tmp/x'",
        "attach 'x.db'",
        "pragma table_info(t)",
        "",
        "   ",
        "select * into newtable from t",
    ],
)
def test_rejects_non_select(bad):
    with pytest.raises(UnsafeSQLError):
        guard_sql(bad, "duckdb", 100)


def test_keeps_smaller_existing_limit():
    assert "limit 5" in guard_sql("select * from t limit 5", "duckdb", 100).lower()


def test_caps_larger_limit():
    out = guard_sql("select * from t limit 99999", "duckdb", 100).lower()
    assert "limit 100" in out and "99999" not in out


def test_cte_allowed():
    out = guard_sql("with x as (select 1 as a) select a from x", "postgres", 10)
    assert "limit 10" in out.lower()


def test_union_allowed():
    guard_sql("select 1 as a union all select 2", "duckdb", 10)


def test_trailing_semicolon_ok():
    guard_sql("select 1;", "duckdb", 10)


@pytest.mark.parametrize("fn", ["read_csv('/etc/passwd')", "read_parquet('x')", "glob('*')", "sqlite_scan('a','b')"])
def test_duckdb_file_functions_rejected(fn):
    with pytest.raises(UnsafeSQLError):
        guard_sql(f"select * from {fn}", "duckdb", 10)


def test_unparseable_sql_rejected_with_message():
    with pytest.raises(UnsafeSQLError, match="parse"):
        guard_sql("selec * fro t (", "duckdb", 10)


def test_quoted_identifiers_preserved():
    out = guard_sql('select "Order Date" from "my table"', "duckdb", 10)
    assert '"Order Date"' in out and '"my table"' in out


@pytest.mark.parametrize("q", [
    "select * from '/etc/hosts'",
    "select * from 'data/*.csv'",
    'select * from "/tmp/x.parquet"',
    "select * from 'C:\\\\data\\\\x.csv'",
    "select * from orders join 'other.json' on true",
])
def test_file_path_tables_rejected(q):
    with pytest.raises(UnsafeSQLError, match="file"):
        guard_sql(q, "duckdb", 10)


@pytest.mark.parametrize("dialect,q", [
    ("tsql", "SELECT * FROM OPENQUERY(LinkedSrv, 'SELECT 1')"),
    ("tsql", "SELECT * FROM OPENROWSET('SQLNCLI', 'Server=x;', 'SELECT 1')"),
    ("tsql", "SELECT * FROM OPENDATASOURCE('SQLNCLI', 'Data Source=x').db.dbo.t"),
    ("postgres", "SELECT query_to_xml('select 1', true, false, '')"),
    ("postgres", "SELECT pg_terminate_backend(123)"),
    ("postgres", "SELECT pg_cancel_backend(123)"),
    ("postgres", "SELECT set_config('default_transaction_read_only','off',false)"),
    ("postgres", "SELECT pg_notify('c','x')"),
    ("postgres", "SELECT pg_advisory_lock(1)"),
    ("postgres", "SELECT pg_reload_conf()"),
    ("bigquery", "SELECT * FROM EXTERNAL_QUERY('conn', 'SELECT 1')"),
    ("snowflake", "SELECT SYSTEM$ABORT_SESSION(1)"),
    ("snowflake", "SELECT SYSTEM$CANCEL_QUERY('x')"),
    ("mysql", "SELECT GET_LOCK('x', 10)"),
])
def test_federated_and_side_effect_functions_rejected(dialect, q):
    with pytest.raises(UnsafeSQLError, match="not allowed"):
        guard_sql(q, dialect, 10)


@pytest.mark.parametrize("q", [
    "SELECT some_udf('INSERT INTO t VALUES (1)')",
    "SELECT f('  delete from t where 1=1')",
    "SELECT f('UPDATE t SET a = 1')",
    "SELECT f('drop table t')",
    "SELECT f('EXEC sp_who')",
])
def test_sql_statements_hidden_in_string_arguments_rejected(q):
    with pytest.raises(UnsafeSQLError, match="SQL statement"):
        guard_sql(q, "postgres", 10)


@pytest.mark.parametrize("q", [
    "SELECT * FROM notes WHERE body LIKE '%update%'",
    "SELECT * FROM t WHERE status = 'deleted'",
    "SELECT pg_size_pretty(pg_total_relation_size('orders'))",
    "SELECT pg_typeof(1)",
])
def test_ordinary_text_and_info_functions_still_allowed(q):
    guard_sql(q, "postgres", 10)
