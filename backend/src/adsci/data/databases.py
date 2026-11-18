"""Live, read-only database sources: Postgres, MySQL/MariaDB, SQLite, SQL Server, Snowflake, BigQuery."""

from __future__ import annotations

import json
import threading
from typing import Literal

import pandas as pd
import sqlalchemy as sa
from pydantic import BaseModel, ConfigDict, SecretStr
from sqlalchemy.engine import URL, make_url

from adsci.config import Settings
from adsci.data.base import ColumnInfo, TableInfo
from adsci.data.sql_guard import guard_sql

Engine = Literal["postgresql", "mysql", "sqlite", "mssql", "snowflake", "bigquery"]

DIALECTS: dict[str, str] = {
    "postgresql": "postgres",
    "mysql": "mysql",
    "sqlite": "sqlite",
    "mssql": "tsql",
    "snowflake": "snowflake",
    "bigquery": "bigquery",
}
_DRIVERS = {
    "postgresql": "postgresql+psycopg",
    "mysql": "mysql+pymysql",
    "mssql": "mssql+pymssql",
}
_DEFAULT_PORTS = {"postgresql": 5432, "mysql": 3306, "mssql": 1433}
MAX_TABLES = 200


class ConnectionFailed(Exception):
    """A connection could not be opened. The message never contains credentials."""


class ConnectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: Engine
    host: str | None = None
    port: int | None = None
    database: str | None = None
    username: str | None = None
    password: SecretStr | None = None
    ssl: bool = False
    account: str | None = None  # Snowflake account identifier
    warehouse: str | None = None
    role: str | None = None
    db_schema: str | None = None
    project: str | None = None  # BigQuery
    dataset: str | None = None
    credentials_json: SecretStr | None = None  # BigQuery service-account JSON
    url: SecretStr | None = None  # raw SQLAlchemy URL; overrides the fields above

    def secrets(self) -> list[str]:
        out = [s.get_secret_value() for s in (self.password, self.credentials_json, self.url) if s]
        if self.url:
            pw = make_url(self.url.get_secret_value()).password
            if pw:
                out.append(str(pw))
        return [s for s in out if s]


def build_url(cfg: ConnectionConfig) -> URL:
    if cfg.url is not None:
        return make_url(cfg.url.get_secret_value())
    password = cfg.password.get_secret_value() if cfg.password else None
    if cfg.engine == "sqlite":
        return make_url(f"sqlite:///file:{cfg.database}?mode=ro&uri=true")
    if cfg.engine == "bigquery":
        return URL.create("bigquery", host=cfg.project, database=cfg.dataset)
    if cfg.engine == "snowflake":
        database = cfg.database if not cfg.db_schema else f"{cfg.database}/{cfg.db_schema}"
        query = {k: v for k, v in {"warehouse": cfg.warehouse, "role": cfg.role}.items() if v}
        return URL.create(
            "snowflake", username=cfg.username, password=password, host=cfg.account, database=database, query=query
        )
    query = {}
    if cfg.ssl and cfg.engine == "postgresql":
        query["sslmode"] = "require"
    return URL.create(
        _DRIVERS[cfg.engine],
        username=cfg.username,
        password=password,
        host=cfg.host,
        port=cfg.port or _DEFAULT_PORTS.get(cfg.engine),
        database=cfg.database,
        query=query,
    )


def build_engine(cfg: ConnectionConfig, settings: Settings) -> sa.Engine:
    """Create an engine whose sessions are read-only and time-limited where the database supports it."""
    url = build_url(cfg)
    timeout_s = settings.sql_timeout_s
    kwargs: dict = {"pool_pre_ping": True}
    backend = url.get_backend_name()
    if backend == "postgresql":
        kwargs["connect_args"] = {
            "options": f"-c default_transaction_read_only=on -c statement_timeout={timeout_s * 1000}",
            "connect_timeout": 10,
        }
    elif backend == "mysql":
        kwargs["connect_args"] = {"connect_timeout": 10, "read_timeout": timeout_s + 5}
        if cfg.ssl:
            kwargs["connect_args"]["ssl"] = {}
    elif backend == "mssql":
        kwargs["connect_args"] = {"timeout": timeout_s, "login_timeout": 10}
    elif backend == "snowflake":
        kwargs["connect_args"] = {"session_parameters": {"STATEMENT_TIMEOUT_IN_SECONDS": timeout_s}}
    elif backend == "bigquery":
        url = url.update_query_dict({"maximum_bytes_billed": str(settings.bigquery_max_bytes_billed)})
        if cfg.credentials_json:
            kwargs["credentials_info"] = json.loads(cfg.credentials_json.get_secret_value())
    elif backend == "sqlite":
        kwargs = {"connect_args": {"check_same_thread": False}}

    engine = sa.create_engine(url, **kwargs)

    if backend == "mysql":

        @sa.event.listens_for(engine, "connect")
        def _mysql_session(dbapi_conn, _record):  # pragma: no cover - needs a live MySQL
            cur = dbapi_conn.cursor()
            cur.execute("SET SESSION TRANSACTION READ ONLY")
            cur.execute(f"SET SESSION MAX_EXECUTION_TIME = {timeout_s * 1000}")
            cur.close()

    return engine


def _scrub(message: str, secrets: list[str]) -> str:
    for s in secrets:
        message = message.replace(s, "***")
    return message


class SQLAlchemyDataSource:
    kind: Literal["database"] = "database"

    def __init__(self, name: str, cfg: ConnectionConfig, settings: Settings) -> None:
        self.name = name
        self.cfg = cfg
        self.settings = settings
        self.dialect = DIALECTS[cfg.engine]
        try:
            self.engine = build_engine(cfg, settings)
        except Exception as e:  # noqa: BLE001 - driver-specific (e.g. missing Google credentials)
            hint = (
                " Provide a service-account JSON key for BigQuery."
                if cfg.engine == "bigquery"
                else ""
            )
            first = str(e).splitlines()[0] if str(e) else type(e).__name__
            raise ConnectionFailed(_scrub(f"Could not set up {cfg.engine}: {first}.{hint}", cfg.secrets())) from None
        self._schema_cache: list[TableInfo] | None = None
        self._lock = threading.Lock()

    def test_connection(self) -> None:
        try:
            with self.engine.connect() as c:
                c.execute(sa.text("SELECT 1"))
        except Exception as e:  # noqa: BLE001 - every driver raises its own types
            first = str(e).splitlines()[0] if str(e) else type(e).__name__
            raise ConnectionFailed(_scrub(f"Could not connect to {self.cfg.engine}: {first}", self.cfg.secrets())) from None

    def schema(self, refresh: bool = False) -> list[TableInfo]:
        with self._lock:
            if self._schema_cache is not None and not refresh:
                return self._schema_cache
            insp = sa.inspect(self.engine)
            schema = self.cfg.db_schema if self.cfg.engine != "snowflake" else None
            names = list(insp.get_table_names(schema=schema))
            try:
                names += list(insp.get_view_names(schema=schema))
            except NotImplementedError:  # pragma: no cover
                pass
            tables: list[TableInfo] = []
            for t in sorted(set(names))[:MAX_TABLES]:
                cols = [
                    ColumnInfo(name=c["name"], dtype=str(c["type"]), nullable=bool(c.get("nullable", True)))
                    for c in insp.get_columns(t, schema=schema)
                ]
                qualified = f"{schema}.{t}" if schema else t
                tables.append(TableInfo(name=qualified, columns=cols, row_count=self._row_count(qualified)))
            self._schema_cache = tables
            return tables

    def _row_count(self, table: str) -> int | None:
        try:
            with self.engine.connect() as c:
                if self.cfg.engine == "sqlite":
                    return int(c.execute(sa.text(f'SELECT count(*) FROM "{table}"')).scalar_one())
                if self.cfg.engine == "postgresql":
                    v = c.execute(
                        sa.text("SELECT reltuples::bigint FROM pg_class WHERE oid = to_regclass(:t)"), {"t": table}
                    ).scalar()
                    return int(v) if v is not None and v >= 0 else None
        except Exception:  # noqa: BLE001 - row counts are best-effort
            return None
        return None

    def query(self, sql: str, limit: int | None = None) -> pd.DataFrame:
        """Run a guarded, read-only query; at most ``limit`` (default ``sql_row_limit``) rows."""
        cap = self.settings.sql_row_limit
        safe = guard_sql(sql, self.dialect, min(limit or cap, cap))
        with self.engine.connect() as c:
            return pd.read_sql(sa.text(safe), c)

    def close(self) -> None:
        self.engine.dispose()
