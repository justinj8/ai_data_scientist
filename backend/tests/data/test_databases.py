import os
import shutil

import pytest
from sqlalchemy import exc as sa_exc
from sqlalchemy import text

from adsci.data.databases import ConnectionConfig, ConnectionFailed, SQLAlchemyDataSource, build_url
from adsci.data.sql_guard import UnsafeSQLError

from .conftest import SAMPLE


@pytest.fixture
def retail_sqlite(tmp_path):
    p = tmp_path / "retail.db"
    shutil.copy(SAMPLE / "retail.db", p)
    return p


def test_sqlite_source_schema_and_query(retail_sqlite, settings):
    src = SQLAlchemyDataSource("shop", ConnectionConfig(engine="sqlite", database=str(retail_sqlite)), settings)
    tables = {t.name: t for t in src.schema()}
    assert {"orders", "customers"} <= set(tables)
    assert tables["orders"].row_count > 70000
    assert "revenue" in [c.name for c in tables["orders"].columns]
    assert len(src.query("select * from orders")) == settings.sql_row_limit
    assert len(src.query("select * from orders", limit=3)) == 3
    with pytest.raises(UnsafeSQLError):
        src.query("delete from orders")
    assert src.dialect == "sqlite" and src.kind == "database"


def test_sqlite_is_read_only_even_if_guard_bypassed(retail_sqlite, settings):
    src = SQLAlchemyDataSource("shop", ConnectionConfig(engine="sqlite", database=str(retail_sqlite)), settings)
    with src.engine.connect() as c, pytest.raises(sa_exc.DBAPIError):
        c.execute(text("delete from orders"))


@pytest.mark.parametrize(
    "cfg,prefix",
    [
        (dict(engine="postgresql", host="h", port=5432, database="d", username="u", password="p"), "postgresql+psycopg"),
        (dict(engine="mysql", host="h", database="d", username="u", password="p"), "mysql+pymysql"),
        (dict(engine="sqlite", database="/tmp/x.db"), "sqlite"),
        (dict(engine="mssql", host="h", database="d", username="u", password="p"), "mssql+pymssql"),
        (dict(engine="snowflake", account="acc", username="u", password="p", database="DB", db_schema="PUBLIC", warehouse="WH"), "snowflake"),
        (dict(engine="bigquery", project="proj", dataset="ds"), "bigquery"),
    ],
)
def test_url_building(cfg, prefix):
    assert build_url(ConnectionConfig(**cfg)).drivername == prefix


def test_raw_url_overrides_fields():
    cfg = ConnectionConfig(engine="postgresql", url="postgresql+psycopg://a:b@host:1/db")
    assert build_url(cfg).host == "host"


def test_secrets_not_in_repr_or_errors(settings):
    cfg = ConnectionConfig(engine="postgresql", host="127.0.0.1", port=1, database="x", username="u", password="hunter2")
    assert "hunter2" not in repr(cfg)
    with pytest.raises(ConnectionFailed) as e:
        SQLAlchemyDataSource("pg", cfg, settings).test_connection()
    assert "hunter2" not in str(e.value)


def test_unknown_engine_rejected():
    with pytest.raises(ValueError):
        ConnectionConfig(engine="oracle")


@pytest.mark.skipif(not os.getenv("ADSCI_TEST_PG_URL"), reason="needs Postgres (ADSCI_TEST_PG_URL)")
def test_postgres_live(settings):
    cfg = ConnectionConfig(engine="postgresql", url=os.environ["ADSCI_TEST_PG_URL"])
    src = SQLAlchemyDataSource("pg", cfg, settings)
    src.test_connection()
    assert {"orders", "customers"} <= {t.name for t in src.schema()}
    assert src.query("select count(*) as n from orders")["n"][0] > 0
    with src.engine.connect() as c, pytest.raises(sa_exc.DBAPIError):
        c.execute(text("create table z(a int)"))


@pytest.mark.parametrize(
    "cfg",
    [
        dict(engine="snowflake", account="acc", username="u", password="p", database="DB", warehouse="WH"),
        dict(engine="mssql", host="h", database="d", username="u", password="p"),
        dict(engine="mysql", host="h", database="d", username="u", password="p"),
    ],
)
def test_engine_creation_is_lazy(cfg, settings):
    # Creating the source must not open a network connection.
    SQLAlchemyDataSource("x", ConnectionConfig(**cfg), settings)


def test_bigquery_without_credentials_fails_cleanly(settings, monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    with pytest.raises(ConnectionFailed, match="service-account"):
        SQLAlchemyDataSource("bq", ConnectionConfig(engine="bigquery", project="p", dataset="d"), settings)


def test_query_limit_can_exceed_default_up_to_analysis_limit(retail_sqlite, settings):
    settings.analysis_row_limit = 20_000
    src = SQLAlchemyDataSource("shop", ConnectionConfig(engine="sqlite", database=str(retail_sqlite)), settings)
    assert len(src.query("select * from orders", limit=1_000_000)) == 20_000


def test_postgres_ssl_verifies_certificate_and_hostname(settings):
    from adsci.data.databases import build_engine

    cfg = ConnectionConfig(engine="postgresql", host="db.example.com", database="d", username="u", password="p", ssl=True)
    q = build_engine(cfg, settings).url.query
    assert q["sslmode"] == "verify-full" and q["sslrootcert"] == "system"


def test_mysql_ssl_verifies_certificate_and_hostname(settings):
    from adsci.data.databases import _connect_args

    args = _connect_args(ConnectionConfig(engine="mysql", host="h", database="d", username="u", password="p", ssl=True),
                         settings)
    assert args["ssl"]["ca"] and args["ssl_verify_cert"] is True and args["ssl_verify_identity"] is True
