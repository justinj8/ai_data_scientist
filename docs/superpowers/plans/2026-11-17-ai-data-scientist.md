# AI Data Scientist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, multi-provider AI data-scientist web app: upload CSV/XLSX or connect live databases, ask questions in plain English, and get verified answers backed by visible SQL, Python, statistics/ML and charts on a dashboard.

**Architecture:** A FastAPI backend (Python package `adsci`) owns data sources (DuckDB for files, SQLAlchemy for live DBs), a per-session Jupyter kernel, a registry of deterministic data-science tools, a provider-neutral LLM layer (Anthropic / OpenAI / xAI adapters, API-key or subscription auth), and a hand-rolled agent loop that streams events over SSE and verifies numbers before answering. A React + Vite + TS frontend renders the data panel, chat step-timeline and dashboard. One Docker image serves both; a seeded Postgres container backs the live-DB demo.

**Tech Stack:** Python 3.12+, uv, FastAPI, sse-starlette, pydantic v2 / pydantic-settings, DuckDB, pandas, pyarrow, openpyxl, SQLAlchemy 2, sqlglot, psycopg 3, PyMySQL, pymssql, snowflake-sqlalchemy, sqlalchemy-bigquery, jupyter_client, ipykernel, nbformat, scipy, statsmodels, scikit-learn, plotly, anthropic, openai, httpx, cryptography, keyring, pytest, respx. Frontend: React 18, Vite, TypeScript, Tailwind, @tanstack/react-query, react-plotly.js + plotly.js-dist-min, lucide-react, react-markdown, prism-react-renderer, vitest.

**Spec:** `docs/superpowers/specs/2026-11-17-ai-data-scientist-design.md`

## Global Constraints

- Inputs: **CSV and XLSX** files; **live connections to multiple databases at once**: PostgreSQL, MySQL/MariaDB, SQLite, Microsoft SQL Server, Snowflake, Google BigQuery — all in the default install.
- Models: default **latest Claude Opus**; options latest Opus, latest Sonnet, latest OpenAI GPT, latest Grok. "Latest" is resolved at runtime from each provider's model list; fallback table as of 2026-11-17: `claude-opus-5-5`, `claude-sonnet-5-5`, OpenAI GPT‑6.2 flagship (exact ID verified in Task 15), `grok-4.7`.
- Credentials: API key for every provider. Subscription sign-in: **ChatGPT plan — supported**; **SuperGrok — experimental** with API-key fallback on 403; **Claude subscription — not offered** (Anthropic terms), Settings explains why.
- Credentials are encrypted at rest, never logged, never in prompts, notebooks, exports or git; masked in the UI after entry.
- SQL is read-only: single `SELECT`/`WITH` only, limit injected, statement timeout.
- Agent caps: default **12** tool steps per question; one verification revision round.
- Every number in a final answer must be grounded in tool outputs or flagged.
- One-command run: `docker compose up`. Native dev: `uv run` + `npm run dev`.
- Python package root: `backend/src/adsci/`; tests in `backend/tests/` mirroring package paths.
- Permutation importance only (no SHAP) — settles the spec's open builder call.
- SQL Server driver: `pymssql` (`mssql+pymssql://`), to avoid the ODBC system dependency.
- Frontend is built into static files served by FastAPI in Docker (single origin `http://127.0.0.1:8000`).

## Review Focus

1. **Messy CSVs** (semicolon delimiter, latin-1 encoding, UTF-8 BOM, `1,234.50` thousands separators, mixed date formats) → load with sensible types instead of failing or loading one giant text column. Test in Task 3.
2. **Messy XLSX** (title rows above the header, empty sheets, duplicate column names) → header detected, empty sheets skipped with a warning, duplicates suffixed `_2`. Test in Task 3.
3. **Awkward identifiers** (spaces, reserved words like `order`, unicode, leading digits in table/column names) → normalized names are valid unquoted SQL identifiers and the original name is kept. Test in Task 3.
4. **Kernel death** (user code calls `os._exit`, or exhausts memory) → the tool returns an error result, the kernel restarts automatically, the next step works. Test in Task 7.
5. **Provider failure mid-stream** (401 revoked key, 429 rate/usage limit, network drop) → the SSE stream emits a single `error` event with a user-facing message and suggested action ("update key" / "switch model") and closes; no hung request. Test in Task 18.

---

## File Structure

```
backend/
  pyproject.toml
  src/adsci/
    config.py                 # Settings
    app/main.py               # FastAPI factory, static files, routers
    app/sessions.py           # Session, SessionManager
    app/routes/{data,chat,insights,exports,models,credentials,auth,internal}.py
    data/base.py              # ColumnInfo, TableInfo, DataSource protocol, DataRef
    data/identifiers.py       # normalize_identifier, dedupe
    data/files.py             # load_csv, load_xlsx, FileDataSource (DuckDB)
    data/databases.py         # ConnectionConfig, build_engine, SQLAlchemyDataSource
    data/sql_guard.py         # guard_sql, UnsafeSQLError
    kernel/manager.py         # KernelSession, ExecutionResult, Output
    kernel/bootstrap.py       # code injected at kernel start (sql(), show())
    kernel/notebook.py        # NotebookRecorder
    analytics/{profile,stats,modeling,forecasting,segmentation,anomalies}.py  # pure functions
    tools/base.py             # Tool, ToolOutput, UIBlock, ToolRegistry, summarize_frame
    tools/{sql_tool,python_tool,chart_tool,profile_tool,stats_tool,model_tool,forecast_tool,segment_tool,anomaly_tool}.py
    llm/types.py              # neutral Message/parts/ToolSpec/StreamEvent/ModelInfo
    llm/base.py               # LLMAdapter protocol, ProviderError hierarchy
    llm/fake.py               # FakeAdapter (scripted)
    llm/anthropic_adapter.py
    llm/openai_adapter.py     # OpenAI + xAI (base_url/auth parametrized)
    llm/registry.py           # ModelRegistry, FALLBACK_MODELS, family rules
    auth/store.py             # CredentialStore
    auth/chatgpt.py           # Sign in with ChatGPT (PKCE loopback)
    auth/grok.py              # SuperGrok device-code (experimental)
    agent/prompts.py
    agent/events.py           # AgentEvent models
    agent/history.py          # compact_history
    agent/orchestrator.py
    agent/verifier.py
    exports/report.py         # markdown report
  tests/...
frontend/
  src/{api,hooks,components,features/{data,chat,dashboard,settings},styles}
sample_data/generate.py, sample_data/out/*, sample_data/postgres/init.sql
evals/{questions.yaml,run.py,checks.py,results/}
docker-compose.yml, Dockerfile, .env.example, .github/workflows/ci.yml, README.md
```

---

## Phase 0 — Foundation

### Task 1: Repository and backend skeleton

**Files:**
- Create: `.gitignore`, `backend/pyproject.toml`, `backend/src/adsci/__init__.py`, `backend/src/adsci/config.py`, `backend/src/adsci/app/main.py`, `backend/tests/test_health.py`

**Interfaces:**
- Produces: `Settings` (pydantic-settings, env prefix `ADSCI_`) with fields `data_dir: Path = Path("data")`, `max_agent_steps: int = 12`, `kernel_timeout_s: int = 60`, `sql_row_limit: int = 10_000`, `sql_timeout_s: int = 30`, `preview_rows: int = 50`, `model_summary_chars: int = 4000`, `session_idle_minutes: int = 60`, `bigquery_max_bytes_billed: int = 1_000_000_000`, `public_base_url: str = "http://127.0.0.1:8000"`; `get_settings() -> Settings` (cached). `create_app(settings: Settings | None = None) -> FastAPI`.

- [ ] **Step 1:** `git init`; `.gitignore` covering `data/`, `.env`, `*.key`, `__pycache__/`, `.venv/`, `node_modules/`, `frontend/dist/`, `sample_data/out/*.db`, `evals/results/raw/`.
- [ ] **Step 2:** `uv init --package` in `backend/` with name `adsci`, `requires-python = ">=3.12"`; add all Tech Stack runtime deps and dev deps (`pytest`, `pytest-asyncio`, `respx`, `ruff`). Run `uv sync`. Expected: resolves; if `pymssql` or `snowflake-connector-python` lacks a wheel for the local Python, pin `.python-version` to 3.12 and re-sync.
- [ ] **Step 3: Write the failing test**

```python
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
```
(`client` fixture in `tests/conftest.py`: `TestClient(create_app(Settings(data_dir=tmp_path)))`.)

- [ ] **Step 4:** Run `uv run pytest tests/test_health.py -v` → FAIL (import error).
- [ ] **Step 5:** Implement `config.py` and `create_app` with `GET /api/health`; `pytest-asyncio` mode `auto` in pyproject.
- [ ] **Step 6:** Run tests → PASS. `uv run ruff check .` → clean.
- [ ] **Step 7:** Commit `chore: backend skeleton with settings and health endpoint` (also commit the spec and this plan).

### Task 2: Synthetic sample data with planted ground truth

**Files:**
- Create: `sample_data/generate.py`, `sample_data/postgres/init.sql`, `backend/tests/test_sample_data.py`
- Output (committed): `sample_data/out/orders.csv`, `customers.csv`, `retail_workbook.xlsx`, `ground_truth.json`; (generated, ignored) `retail.db`

**Interfaces:**
- Produces: `generate(out_dir: Path, seed: int = 42) -> dict` returning the ground-truth dict also written to `ground_truth.json`. Datasets registered by name for Task 6: `"retail"` (orders + customers).

Data definitions (exact):
- `orders` 2023-01-01..2025-12-31, ~60k rows: `order_id, order_date, customer_id, region∈{North,South,East,West}, category∈{Electronics,Home,Apparel,Beauty,Sports}, channel∈{web,app,store}, units, unit_price, discount, revenue, promo`.
  Planted: weekly seasonality (weekend volume ×1.2); Q4 volume ×1.35; +8% YoY volume growth; promo orders units ×1.25 and discount 0.10–0.20; West average order value ×1.12; spike days ×4 revenue on 2024-03-14, 2024-07-04, 2024-09-21, 2025-02-10, 2025-06-18; outage days ≈0 orders on 2024-11-05, 2025-08-12.
- `customers` 5,000 rows: `customer_id, signup_date, region, plan∈{basic,pro,enterprise}, tenure_months, monthly_spend, support_tickets, last_login_days, nps, churned`. Churn via logistic model: +support_tickets, +last_login_days, −tenure_months, basic plan higher, −nps; **region has no effect** (null control). Base churn ≈ 20%.
- `retail_workbook.xlsx`: sheet `Orders_2025` (2025 orders), sheet `Targets` (monthly revenue target by region), sheet `Notes` with a 2-row title block above the header at row 3, and an empty sheet `Scratch`.
- `ground_truth.json` keys (computed from generated data, not hard-coded): `total_revenue_2025`, `yoy_revenue_growth_2025`, `promo_aov_uplift`, `west_aov_ratio`, `top_category_2025`, `q4_share_2024`, `anomaly_spike_dates`, `outage_dates`, `churn_rate`, `churn_top_drivers` (ordered), `region_churn_p_value`.
- `retail.db` (SQLite) and `postgres/init.sql` (`CREATE TABLE` + `COPY ... FROM '/seed/orders.csv' CSV HEADER`) contain the same two tables.

- [ ] **Step 1: Write the failing tests**

```python
def test_planted_truths(tmp_path):
    gt = generate(tmp_path, seed=42)
    assert 0.06 <= gt["yoy_revenue_growth_2025"] <= 0.12
    assert 1.08 <= gt["west_aov_ratio"] <= 1.16
    assert gt["promo_aov_uplift"] > 0.05
    assert set(gt["anomaly_spike_dates"]) == {"2024-03-14","2024-07-04","2024-09-21","2025-02-10","2025-06-18"}
    assert 0.15 <= gt["churn_rate"] <= 0.25
    assert gt["churn_top_drivers"][0] in {"last_login_days", "support_tickets"}
    assert gt["region_churn_p_value"] > 0.05

def test_outputs_exist_and_deterministic(tmp_path):
    a = generate(tmp_path / "a", seed=42); b = generate(tmp_path / "b", seed=42)
    assert a == b
    assert (tmp_path / "a" / "retail_workbook.xlsx").exists() and (tmp_path / "a" / "retail.db").exists()
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement with numpy `default_rng(seed)`; truths computed with pandas/scipy from the generated frames. **Step 4:** Run → PASS. Run `uv run python ../sample_data/generate.py` to refresh `sample_data/out/`.
- [ ] **Step 5:** Commit `feat: seeded synthetic retail datasets with ground truth`.

---

## Phase 1 — Data layer

### Task 3: File loading (CSV, XLSX) into DuckDB

**Files:**
- Create: `data/base.py`, `data/identifiers.py`, `data/files.py`, `tests/data/test_identifiers.py`, `tests/data/test_files.py`, `tests/fixtures/messy/*` (small hand-made files)

**Interfaces:**
- Produces:
  - `ColumnInfo(name: str, dtype: str, nullable: bool, original_name: str | None = None)`; `TableInfo(name: str, columns: list[ColumnInfo], row_count: int | None, original_name: str | None = None)` — pydantic models.
  - `class DataSource(Protocol)`: `name: str`, `kind: Literal["file","database"]`, `dialect: str` (sqlglot dialect), `schema() -> list[TableInfo]`, `query(sql: str, limit: int | None = None) -> pd.DataFrame`, `test_connection() -> None`, `close() -> None`.
  - `DataRef(source: str, sql: str)` pydantic model (used by every analytic tool).
  - `normalize_identifier(raw: str) -> str` (lowercase snake_case, ASCII-folded, leading digit → `_` prefix, reserved word → `_` suffix, empty → `col`); `dedupe(names: list[str]) -> list[str]` (`x`, `x_2`, `x_3`).
  - `FileDataSource(name: str, db_path: Path)` with `add_file(path: Path) -> list[LoadReport]`; `LoadReport(table: str, rows: int, warnings: list[str])`. CSV → one table `normalize(stem)`; XLSX → one table per non-empty sheet `normalize(stem)__normalize(sheet)`.
  - `load_csv(path) -> tuple[pd.DataFrame, list[str]]`, `load_xlsx(path) -> dict[str, tuple[pd.DataFrame, list[str]]]`.

- [ ] **Step 1: Write the failing tests** (covers Review Focus 1–3)

```python
@pytest.mark.parametrize("raw,expected", [("Order Date","order_date"),("order","order_"),("2024 Sales","_2024_sales"),("Café №","cafe_no"),("","col")])
def test_normalize_identifier(raw, expected): assert normalize_identifier(raw) == expected

def test_dedupe(): assert dedupe(["a","a","b","a"]) == ["a","a_2","b","a_3"]

def test_semicolon_latin1_bom_thousands(tmp_path):  # fixture: 'messy_semicolon_latin1.csv'
    df, warns = load_csv(FIX / "messy_semicolon_latin1.csv")
    assert list(df.columns) == ["region", "revenue", "order_date"]
    assert df["revenue"].dtype.kind == "f" and df["revenue"].iloc[0] == 1234.5
    assert str(df["order_date"].dtype).startswith("datetime64")

def test_xlsx_title_rows_empty_sheet_dupes():
    sheets = load_xlsx(FIX / "messy.xlsx")
    assert "scratch" not in sheets                       # empty sheet skipped
    df, warns = sheets["notes"]
    assert list(df.columns)[:2] == ["item", "item_2"]    # header found at row 3, dupes suffixed

def test_file_source_roundtrip(tmp_path):
    src = FileDataSource("files", tmp_path / "s.duckdb")
    reports = src.add_file(SAMPLE / "retail_workbook.xlsx")
    assert {r.table for r in reports} >= {"retail_workbook__orders_2025", "retail_workbook__targets", "retail_workbook__notes"}
    df = src.query("select count(*) n from retail_workbook__targets")
    assert df["n"][0] > 0
    t = next(t for t in src.schema() if t.name == "retail_workbook__targets")
    assert all(c.original_name for c in t.columns)
```
Empty-sheet skip must also produce a warning on the source-level `LoadReport` list: assert `any("Scratch" in w for r in reports for w in r.warnings)`.

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. CSV: sniff encoding (try utf-8-sig, then cp1252/latin-1), delimiter via `csv.Sniffer` on first 64 KB; coerce object columns matching `^-?[\d,]+(\.\d+)?$` to float and date-like columns via `pd.to_datetime(format="mixed")` when ≥90% parse. XLSX via openpyxl: header row = first row within the first 10 where ≥60% cells non-empty and strings. Tables written to DuckDB via `con.register` + `CREATE TABLE AS`. Original names stored in a `_adsci_columns` metadata table. **Step 4:** PASS. **Step 5:** Commit `feat: CSV/XLSX loaders into DuckDB with identifier normalization`.

### Task 4: SQL guard

**Files:**
- Create: `data/sql_guard.py`, `tests/data/test_sql_guard.py`

**Interfaces:**
- Produces: `guard_sql(sql: str, dialect: str, limit: int) -> str` (returns the rewritten SQL in the same dialect); `class UnsafeSQLError(ValueError)` with a user-facing message.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("dialect", ["duckdb","postgres","mysql","sqlite","tsql","snowflake","bigquery"])
def test_select_gets_limit(dialect):
    out = guard_sql("select a from t", dialect, 100)
    assert ("limit 100" in out.lower()) or ("top 100" in out.lower())

@pytest.mark.parametrize("bad", ["delete from t","drop table t","insert into t values (1)","update t set a=1",
    "select 1; drop table t","create table x as select 1","copy t to '/tmp/x'","attach 'x.db'","pragma table_info(t)"])
def test_rejects_non_select(bad):
    with pytest.raises(UnsafeSQLError): guard_sql(bad, "duckdb", 100)

def test_keeps_smaller_existing_limit(): assert "limit 5" in guard_sql("select * from t limit 5", "duckdb", 100).lower()
def test_caps_larger_limit(): assert "limit 100" in guard_sql("select * from t limit 99999", "duckdb", 100).lower()
def test_cte_allowed(): guard_sql("with x as (select 1 a) select a from x", "postgres", 10)
def test_duckdb_file_functions_rejected():
    with pytest.raises(UnsafeSQLError): guard_sql("select * from read_csv('/etc/passwd')", "duckdb", 10)
```

- [ ] **Step 2:** FAIL. **Step 3:** Implement with `sqlglot.parse(sql, read=dialect)`; exactly one statement; root must be `exp.Select`/`exp.Union`/`exp.With`-wrapped select; walk tree rejecting any DML/DDL/Command node and DuckDB table functions `read_*`, `glob`, `*_scan`; apply `.limit()`. **Step 4:** PASS. **Step 5:** Commit `feat: read-only SQL guard across dialects`.

### Task 5: Live database sources

**Files:**
- Create: `data/databases.py`, `tests/data/test_databases.py`

**Interfaces:**
- Consumes: `DataSource`, `TableInfo`, `guard_sql`, `Settings`.
- Produces:
  - `ConnectionConfig` (pydantic): `engine: Literal["postgresql","mysql","sqlite","mssql","snowflake","bigquery"]`, `host: str | None`, `port: int | None`, `database: str | None`, `username: str | None`, `password: SecretStr | None`, `ssl: bool = False`, `account: str | None` (Snowflake), `warehouse: str | None`, `role: str | None`, `db_schema: str | None`, `project: str | None` (BigQuery), `dataset: str | None`, `credentials_json: SecretStr | None`, `url: SecretStr | None` (raw SQLAlchemy URL overrides fields). `model_config` forbids extra; `__repr__` never shows secrets.
  - `build_engine(cfg: ConnectionConfig, settings: Settings) -> sqlalchemy.Engine` — drivers: `postgresql+psycopg`, `mysql+pymysql`, `sqlite` (`?mode=ro&uri=true`), `mssql+pymssql`, `snowflake://` (snowflake-sqlalchemy), `bigquery://project/dataset` with `credentials_info=json.loads(...)`.
  - `SQLAlchemyDataSource(name: str, cfg: ConnectionConfig, settings: Settings)` implementing `DataSource`; `dialect` mapping `{"postgresql":"postgres","mysql":"mysql","sqlite":"sqlite","mssql":"tsql","snowflake":"snowflake","bigquery":"bigquery"}`.
  - `class ConnectionFailed(Exception)` — message stripped of secrets.

- [ ] **Step 1: Write the failing tests**

```python
def test_sqlite_source_schema_and_query(retail_sqlite):
    src = SQLAlchemyDataSource("shop", ConnectionConfig(engine="sqlite", database=str(retail_sqlite)), settings)
    names = {t.name for t in src.schema()}
    assert {"orders","customers"} <= names
    assert len(src.query("select * from orders")) == settings.sql_row_limit  # limit injected
    with pytest.raises(UnsafeSQLError): src.query("delete from orders")

def test_sqlite_is_read_only_even_if_guard_bypassed(retail_sqlite):
    src = SQLAlchemyDataSource("shop", ConnectionConfig(engine="sqlite", database=str(retail_sqlite)), settings)
    with src.engine.connect() as c, pytest.raises(Exception): c.exec_driver_sql("delete from orders")

@pytest.mark.parametrize("cfg,prefix", [...one case per engine...])
def test_url_building(cfg, prefix): assert build_engine(cfg, settings).url.drivername.startswith(prefix)

def test_secrets_not_in_errors_or_repr():
    cfg = ConnectionConfig(engine="postgresql", host="127.0.0.1", port=1, database="x", username="u", password="hunter2")
    assert "hunter2" not in repr(cfg)
    with pytest.raises(ConnectionFailed) as e: SQLAlchemyDataSource("pg", cfg, settings).test_connection()
    assert "hunter2" not in str(e.value)

@pytest.mark.skipif(not os.getenv("ADSCI_TEST_PG_URL"), reason="needs Postgres")
def test_postgres_live(pg_cfg):  # read-only + statement timeout
    src = SQLAlchemyDataSource("pg", pg_cfg, settings)
    assert src.query("select count(*) n from orders")["n"][0] > 0
    with src.engine.connect() as c, pytest.raises(Exception): c.exec_driver_sql("create table z(a int)")
```

- [ ] **Step 2:** FAIL. **Step 3:** Implement. Per-connection session setup on `connect` event: Postgres `SET default_transaction_read_only = on; SET statement_timeout = <ms>`; MySQL `SET SESSION TRANSACTION READ ONLY; SET SESSION MAX_EXECUTION_TIME = <ms>`; Snowflake `ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = <s>`; BigQuery job config `maximum_bytes_billed = settings.bigquery_max_bytes_billed`. Schema via `sqlalchemy.inspect` (skip system schemas; cap 200 tables, row counts only where cheap). **Step 4:** PASS (Postgres test runs in CI via service container). **Step 5:** Commit `feat: live database sources for six engines, read-only`.

### Task 6: Sessions and data API

**Files:**
- Create: `app/sessions.py`, `app/routes/data.py`, `tests/app/test_data_routes.py`
- Modify: `app/main.py` (register router, `SessionManager` on `app.state`)

**Interfaces:**
- Consumes: `FileDataSource`, `SQLAlchemyDataSource`, `ConnectionConfig`, `TableInfo`.
- Produces:
  - `Session`: `id: str`, `dir: Path`, `sources: dict[str, DataSource]`, `files: FileDataSource` (the single file source named `"files"`, created lazily), `history: list[Message]` (Task 13 type), `insights: list[Insight]`, `model_id: str | None`, `notebook: NotebookRecorder` (Task 7), `kernel: KernelSession | None` (Task 7), `internal_token: str`, `last_active: datetime`. Method `get_source(name) -> DataSource` raising `KeyError` with available names.
  - Fields `kernel` and `notebook` are added in Task 7; `history` is `list[Any]` until Task 13 types it as `list[Message]`.
  - `SessionManager(settings)`: `create() -> Session`, `get(id) -> Session` (404 via `HTTPException`), `close(id)`, `reap_idle(now) -> int`. `create_app` lifespan runs `reap_idle` every 5 minutes and closes all sessions on shutdown.
  - Routes: `POST /api/sessions` → `{"session_id"}`; `GET /api/sessions/{id}/schema` → `{"sources":[{"name","kind","dialect","tables":[TableInfo]}]}`; `POST /api/sessions/{id}/files` (multipart `file`, `.csv`/`.xlsx` only, max 200 MB) → `{"tables":[LoadReport]}`; `POST /api/connections/test` (ConnectionConfig) → `{"ok": bool, "error": str|None}`; `POST /api/sessions/{id}/connections` (`{"name", "config"}`) → `{"name","tables"}`; `DELETE /api/sessions/{id}/sources/{name}`; `POST /api/sessions/{id}/samples/{dataset}` (`retail` → loads CSVs into `files` and attaches `retail.db` as source `retail_sqlite`).

- [ ] **Step 1: Write the failing tests**: `reap_idle` closes a session whose `last_active` is older than `session_idle_minutes`; create session → upload `orders.csv` → schema lists `files.orders`; upload `.txt` → 415; add SQLite connection named `shop` → schema has two sources; duplicate source name → 409; delete source → gone; sample `retail` → sources `files` and `retail_sqlite`; unknown session → 404.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS. **Step 5:** Commit `feat: session manager and data routes`.

---

## Phase 2 — Execution and tools

### Task 7: Kernel manager and notebook recorder

**Files:**
- Create: `kernel/manager.py`, `kernel/bootstrap.py`, `kernel/notebook.py`, `app/routes/internal.py`, `tests/kernel/test_manager.py`, `tests/kernel/test_notebook.py`

**Interfaces:**
- Consumes: `Session`, `DataSource.query`.
- Produces:
  - `Output` = `TextOutput(text)` | `TableOutput(columns: list[str], rows: list[list], total_rows: int)` | `PlotlyOutput(figure: dict)` | `ImageOutput(png_b64: str)` (pydantic, discriminated by `type`).
  - `ExecutionResult(stdout: str, outputs: list[Output], error: ExecError | None, duration_s: float, restarted: bool = False)`; `ExecError(ename, evalue, traceback: str)`.
  - `KernelSession(session_id: str, workdir: Path, internal_url: str, token: str)`: `async start()`, `async execute(code: str, timeout_s: int) -> ExecutionResult`, `async restart()`, `async shutdown()`.
  - Bootstrap defines in the kernel: `sql(query: str, source: str = "files") -> pd.DataFrame` (POSTs to `{internal_url}/internal/sessions/{id}/sql` with header `X-ADSCI-Token`, receives Parquet bytes), `show(fig)` (displays mimetype `application/vnd.plotly.v1+json`), imports `pd, np, px, go, stats (scipy.stats), sm (statsmodels.api)` and `from adsci.analytics import *`. Pandas DataFrame display results captured as `TableOutput` (first `preview_rows`).
  - `POST /internal/sessions/{id}/sql` body `{"query","source"}` → `application/vnd.apache.parquet`; 403 on bad token; only bound for requests from 127.0.0.1.
  - `NotebookRecorder`: `add_markdown(text)`, `add_python(code, outputs: list[Output])`, `add_sql(query, source, outputs)` (rendered as `df = sql("""...""", source="...")` cell), `to_ipynb() -> str` (nbformat v4, first cell = setup with `from adsci.kernel.bootstrap import *`-equivalent instructions and connection names only).

- [ ] **Step 1: Write the failing tests** (async, real kernel, Review Focus 4)

```python
async def test_state_persists(kernel):
    await kernel.execute("x = 41", 10)
    r = await kernel.execute("print(x + 1)", 10)
    assert r.stdout.strip() == "42" and r.error is None

async def test_error_returned(kernel):
    r = await kernel.execute("1/0", 10)
    assert r.error.ename == "ZeroDivisionError"

async def test_timeout_interrupts(kernel):
    r = await kernel.execute("import time; time.sleep(30)", 2)
    assert r.error and "timed out" in r.error.evalue.lower()
    assert (await kernel.execute("print(1)", 5)).stdout.strip() == "1"

async def test_kernel_death_restarts(kernel):
    r = await kernel.execute("import os; os._exit(1)", 10)
    assert r.error and r.restarted
    assert (await kernel.execute("print('alive')", 10)).stdout.strip() == "alive"

async def test_plotly_and_table_capture(kernel):
    r = await kernel.execute("import plotly.express as px; show(px.bar(x=[1,2], y=[3,4])); pd.DataFrame({'a':[1,2]})", 20)
    assert any(o.type == "plotly" for o in r.outputs) and any(o.type == "table" for o in r.outputs)

async def test_sql_helper_roundtrip(live_app_with_session):  # uvicorn on a free port + sample loaded
    r = await kernel.execute("print(len(sql('select * from orders limit 7')))", 20)
    assert r.stdout.strip() == "7"
```
Notebook test: recorder with one markdown, one sql, one python cell → `nbformat.validate(nbformat.reads(nb, 4))` passes and contains no string from a fake password.

- [ ] **Step 2:** FAIL. **Step 3:** Implement with `jupyter_client.AsyncKernelManager`; kernel cwd = `workdir`; memory cap via `resource.setrlimit(RLIMIT_AS, 2 GB)` in bootstrap (skipped on macOS where unsupported, noted in README); output cap 200 KB stdout. **Step 4:** PASS. **Step 5:** Commit `feat: stateful per-session Jupyter kernel with sql()/show() helpers and notebook recorder`.

### Task 8: Tool framework + `run_sql`, `run_python`, `create_chart`

**Files:**
- Create: `tools/base.py`, `tools/sql_tool.py`, `tools/python_tool.py`, `tools/chart_tool.py`, `tests/tools/test_base.py`, `tests/tools/test_core_tools.py`

**Interfaces:**
- Consumes: `Session`, `KernelSession`, `guard_sql`, `NotebookRecorder`, `Output`.
- Produces:
  - `UIBlock` = `TableBlock(columns, rows, total_rows)` | `TextBlock(text, format: Literal["plain","markdown"])` | `ChartBlock(figure: dict, title: str | None)` | `MetricsBlock(items: list[Metric(label, value, detail: str | None)])` | `ErrorBlock(message, detail: str | None)`.
  - `ToolOutput(summary: str, ui: list[UIBlock], code: str | None = None, language: Literal["sql","python"] | None = None, is_error: bool = False, data: dict | None = None)` — `summary` is what the model sees (≤ `settings.model_summary_chars`), `data` is structured JSON results (for verification grounding).
  - `class Tool(ABC)`: `name: str`, `description: str`, `Input: type[BaseModel]`, `async run(self, args: BaseModel, ctx: ToolContext) -> ToolOutput`. `ToolContext(session: Session, settings: Settings)`.
  - `ToolRegistry`: `register(tool)`, `specs() -> list[ToolSpec]` (JSON Schema from `Input.model_json_schema()`), `async execute(name: str, raw_args: dict, ctx) -> ToolOutput` — validation errors and exceptions become `is_error=True` outputs with a corrective message; never raises. `default_registry() -> ToolRegistry` registering all tools from Tasks 8–12.
  - `summarize_frame(df: pd.DataFrame, max_rows: int = 20) -> str` (shape, dtypes, head rows as CSV, numeric describe when > max_rows); `clip(text: str, n: int) -> str`.
  - Tool inputs: `RunSQLInput(source: str, query: str, purpose: str)`; `RunPythonInput(code: str, purpose: str)`; `CreateChartInput(data: DataRef, kind: Literal["bar","line","scatter","histogram","box","heatmap","pie"], x: str | None, y: str | list[str] | None, color: str | None, title: str, agg: Literal["sum","mean","count","median"] | None = None)`.
  - Every tool appends to `session.notebook`.

- [ ] **Step 1: Write the failing tests**: registry turns a pydantic validation error into `is_error` with field names in `summary`; unknown tool → `is_error`; `summarize_frame` of a 100k×300 frame stays under 4000 chars; `run_sql` on sample returns `TableBlock` with `total_rows` and `code == query`, `language == "sql"`, unsafe SQL → `is_error` mentioning "read-only"; `run_python` returning a figure yields `ChartBlock`; `create_chart` bar with `agg="sum"` returns a figure whose trace y-sum equals the SQL sum; each tool adds exactly one notebook cell.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS. **Step 5:** Commit `feat: tool framework with sql, python and chart tools`.

### Task 9: `profile_dataset`

**Files:**
- Create: `analytics/profile.py`, `tools/profile_tool.py`, `tests/analytics/test_profile.py`

**Interfaces:**
- Produces: `profile_frame(df: pd.DataFrame, max_categories: int = 10) -> DatasetProfile` with `n_rows, n_cols, duplicate_rows, columns: list[ColumnProfile(name, dtype, semantic: Literal["numeric","categorical","datetime","boolean","text","id"], missing_pct, n_unique, top_values, stats: dict | None, outlier_pct: float | None)], correlations: dict[str, dict[str, float]], warnings: list[DataQualityWarning(column, kind: Literal["high_missing","constant","id_like","high_skew","outliers","duplicates","possible_leakage"], message)]`. Tool input `ProfileInput(data: DataRef, target: str | None = None)`; `possible_leakage` raised when |corr(feature, target)| > 0.95. Tool UI: MetricsBlock + TableBlock of columns + ChartBlock correlation heatmap.

- [ ] **Step 1: Tests**: on `customers`: `customer_id` semantic `id`, warning `id_like`; a column injected 60% NaN → `high_missing`; constant column → `constant`; synthetic `leak = churned*1.0` with `target="churned"` → `possible_leakage`; correlations symmetric with 1.0 diagonal; outlier_pct uses 1.5×IQR.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat: dataset profiling with data-quality warnings`.

### Task 10: `statistical_test`

**Files:**
- Create: `analytics/stats.py`, `tools/stats_tool.py`, `tests/analytics/test_stats.py`

**Interfaces:**
- Produces: `TestResult(test: str, statistic: float, p_value: float, effect_size: float | None, effect_size_name: str | None, ci: tuple[float, float] | None, n: int, groups: dict[str, GroupSummary] | None, assumptions: list[AssumptionCheck(name, passed: bool, detail)], warnings: list[str], interpretation: str)`.
  - `compare_groups(df, value: str, group: str, alpha: float = 0.05) -> TestResult` — 2 groups: Shapiro (n≤5000, else skip with note) + Levene → Student t / Welch t / Mann-Whitney U; effect Cohen's d or rank-biserial; CI of mean difference. ≥3 groups: ANOVA / Welch ANOVA / Kruskal-Wallis; effect eta² / epsilon²; post-hoc pairwise with Holm correction in `warnings`/`groups`.
  - `test_association(df, x: str, y: str) -> TestResult` — cat×cat: chi-square (Fisher for 2×2 with expected < 5), Cramér's V; num×num: Pearson if both ~normal else Spearman, with CI via Fisher z.
  - Tool input `StatsTestInput(data: DataRef, kind: Literal["compare_groups","association"], value: str | None, group: str | None, x: str | None, y: str | None)`.
  - `interpretation` never uses causal verbs ("causes", "drives", "leads to").

- [ ] **Step 1: Tests** (with known answers): West vs other regions AOV → Welch or Mann-Whitney, p < 0.001, effect > 0; region×churned → chi-square p equals `ground_truth["region_churn_p_value"]` (±1e-9) and > 0.05; two normal samples with equal variance → "Student"; heavy-skew samples → "Mann-Whitney"; n < 20 group → warning containing "small sample"; interpretation contains none of the causal verbs.
- [ ] **Step 2–4:** FAIL → implement (scipy, statsmodels) → PASS. **Step 5:** Commit `feat: assumption-aware statistical testing with effect sizes`.

### Task 11: `train_model`

**Files:**
- Create: `analytics/modeling.py`, `tools/model_tool.py`, `tests/analytics/test_modeling.py`

**Interfaces:**
- Produces: `train_and_evaluate(df, target: str, features: list[str] | None = None, task: Literal["classification","regression"] | None = None, seed: int = 0) -> ModelReport` with `task, n_train, n_test, baseline: Metrics, models: list[ModelResult(name: Literal["baseline","linear","gradient_boosting"], cv_mean: dict[str,float], cv_std: dict[str,float], test: dict[str,float])], best_model, importances: list[FeatureImportance(feature, importance, std)], dropped_features: list[tuple[str, str]], warnings`.
  - Metrics: classification → ROC-AUC, F1, accuracy (+ class balance warning if minority < 10%); regression → RMSE, MAE, R².
  - Pipeline: `ColumnTransformer` (impute + scale numeric; impute + one-hot categorical, `handle_unknown="ignore"`, high-cardinality > 50 → dropped with reason); 5-fold (stratified for classification) CV on 80% train; held-out 20% test; `HistGradientBoosting*` and `LogisticRegression`/`Ridge`; `DummyClassifier/Regressor` baseline; permutation importance on the test set (10 repeats).
  - Auto-drops ID-like and leakage columns (reuses Task 9 detection) and reports them.
  - Tool input `TrainModelInput(data: DataRef, target: str, features: list[str] | None = None, task: ... | None = None)`; UI: metrics table vs baseline, importance bar chart.

- [ ] **Step 1: Tests**: churn model best test ROC-AUC > baseline + 0.15; top-2 importances ⊇ one of `ground_truth["churn_top_drivers"][:2]`; `customer_id` in `dropped_features`; leakage column dropped; regression on `revenue` from `units, unit_price, discount` R² > 0.9; same seed → identical report.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat: cross-validated modeling with baselines and permutation importance`.

### Task 12: `forecast`, `segment`, `detect_anomalies`

**Files:**
- Create: `analytics/forecasting.py`, `analytics/segmentation.py`, `analytics/anomalies.py`, `tools/forecast_tool.py`, `tools/segment_tool.py`, `tools/anomaly_tool.py`, tests for each under `tests/analytics/`

**Interfaces:**
- Produces:
  - `forecast_series(df, date_col: str, value_col: str, horizon: int, freq: str | None = None, agg: str = "sum") -> ForecastReport(freq, model: str, seasonal_period: int | None, history: list[Point], forecast: list[Point(date, value, lower, upper)], backtest: BacktestResult(folds, mape, rmse, naive_mape, naive_rmse), warnings)` — infer freq; resample; ETS (statsmodels `ExponentialSmoothing`) vs SARIMA chosen by backtest; rolling-origin backtest with 3 folds of length `horizon`; seasonal-naive baseline; 80% and 95% intervals (report 95%).
  - `segment(df, features: list[str], k: int | None = None, k_range: tuple[int,int] = (2, 8), seed: int = 0) -> SegmentReport(k, silhouette, sizes, profiles: list[ClusterProfile(cluster, size, means: dict, distinguishing: list[str])])`.
  - `detect_anomalies(df, columns: list[str] | None = None, date_col: str | None = None, value_col: str | None = None, contamination: float = 0.01) -> AnomalyReport(method: Literal["isolation_forest","robust_zscore"], contamination, anomalies: list[Anomaly(index_or_date, score, value, top_features)], warnings)` — time series (date_col+value_col) → STL residual robust z (|z| > 3.5); tabular → Isolation Forest.
  - Tool inputs mirror signatures with `data: DataRef`.

- [ ] **Step 1: Tests**: daily revenue forecast (horizon 30) backtest MAPE < naive MAPE and detects weekly period 7; anomalies on daily revenue include all 5 `anomaly_spike_dates` and both `outage_dates`; segmentation on churn features with k=None picks k∈[2,8] with silhouette > 0.2 and profiles' sizes sum to n; seeds deterministic.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat: forecasting with backtests, segmentation and anomaly detection`.

---

## Phase 3 — LLM layer

### Task 13: Neutral LLM types and fake adapter

**Files:**
- Create: `llm/types.py`, `llm/base.py`, `llm/fake.py`, `tests/llm/test_fake.py`

**Interfaces:**
- Produces:
  - Parts: `TextPart(text)`, `ToolCallPart(id, name, arguments: dict)`, `ToolResultPart(tool_call_id, content: str, is_error: bool = False)`. `Message(role: Literal["user","assistant"], parts: list[Part])` — tool results live in `user` messages.
  - `ToolSpec(name, description, parameters: dict)`.
  - Stream events: `TextDelta(text)`, `ToolCallEvent(call: ToolCallPart)`, `TurnEnd(stop_reason: Literal["end_turn","tool_use","max_tokens"], usage: Usage)`. `Usage(input_tokens: int, output_tokens: int, cost_usd: float | None, billed_to: Literal["api_key","chatgpt_plan","supergrok_plan"])`.
  - `ModelInfo(id, provider: Literal["anthropic","openai","xai","fake"], family: Literal["opus","sonnet","gpt","grok","other"], display_name, created: datetime | None, supports_tools: bool)`.
  - `class LLMAdapter(Protocol)`: `provider: str`; `def stream(self, *, model: str, system: str, messages: list[Message], tools: list[ToolSpec], max_tokens: int = 8192) -> AsyncIterator[StreamEvent]`; `async def list_models(self) -> list[ModelInfo]`; `async def complete_json(self, *, model, system, prompt, schema: dict) -> dict` (used by the critic).
  - Errors: `ProviderError(message, provider, retryable: bool, user_action: Literal["update_credentials","switch_model","retry","none"])` with subclasses `AuthError` (401/403), `RateLimitError` (429, incl. plan usage limit), `ProviderUnavailable` (5xx/network), `AccessDenied` (tier gating).
  - `FakeAdapter(script: list[list[StreamEvent]] | Callable[[list[Message]], list[StreamEvent]])` — replays one scripted turn per `stream` call; records received messages; can raise a scripted `ProviderError`.

- [ ] **Step 1–4:** Tests: `Message` round-trips through JSON; `FakeAdapter` yields scripted events in order and records calls. FAIL → implement → PASS. **Step 5:** Commit `feat: provider-neutral LLM types and fake adapter`.

### Task 14: Anthropic adapter

**Files:**
- Create: `llm/anthropic_adapter.py`, `tests/llm/test_anthropic_adapter.py`, `tests/llm/fixtures/anthropic_*.json`

**Interfaces:**
- Consumes: Task 13 types.
- Produces: `AnthropicAdapter(api_key: str, client: anthropic.AsyncAnthropic | None = None)` implementing `LLMAdapter`; `to_anthropic_messages(messages) -> list[dict]`; `PRICES: dict[str, tuple[float, float]]` (USD per 1M input/output; Opus 5.5 $4/$20 — confirm Sonnet in the skill reference).

- [ ] **Step 0:** Load the `claude-api` skill and follow it for streaming tool use, model IDs, and the models-list endpoint.
- [ ] **Step 1: Tests** (respx-mocked HTTP using recorded SSE fixtures): neutral history with text + tool call + tool result converts to Anthropic `tool_use`/`tool_result` blocks with matching ids; a streamed turn with text then a tool call yields `TextDelta`s, one `ToolCallEvent` with parsed JSON arguments, and `TurnEnd(stop_reason="tool_use")` with usage and cost; 401 → `AuthError(user_action="update_credentials")`; 429 → `RateLimitError`; `list_models` maps `claude-opus-*` → family `opus`, `claude-sonnet-*` → `sonnet`, with `created` from `created_at`.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat: Anthropic adapter with streaming tool use`.

### Task 15: OpenAI and xAI adapter

**Files:**
- Create: `llm/openai_adapter.py`, `tests/llm/test_openai_adapter.py`, fixtures

**Interfaces:**
- Produces: `OpenAICompatAdapter(provider: Literal["openai","xai"], token_provider: Callable[[], Awaitable[str]], billed_to: Literal["api_key","chatgpt_plan","supergrok_plan"], base_url: str | None = None)` using the Responses API (`client.responses.stream`) with function tools; factory helpers `openai_with_key(key)`, `openai_with_chatgpt(token_provider)`, `xai_with_key(key)` (base_url `https://api.x.ai/v1`), `xai_with_supergrok(token_provider)`. Token provider is awaited per request so OAuth refresh (Task 17) is transparent. When `billed_to != "api_key"`, `Usage.cost_usd = None`.

- [ ] **Step 0:** Verify current Responses API streaming event names and function-call output item shape (Context7 docs for `openai` Python SDK); verify xAI supports the Responses endpoint (docs.x.ai) — if not, implement xAI via Chat Completions behind the same class. Record the exact current OpenAI flagship model IDs from `GET /v1/models` docs for the fallback table.
- [ ] **Step 1: Tests**: neutral history ↔ Responses `input` items (`function_call` / `function_call_output` with `call_id`); streamed fixture yields deltas, a `ToolCallEvent`, and `TurnEnd`; 401/403/429/5xx mapped to Task 13 errors; xAI 403 with tier message → `AccessDenied(user_action="update_credentials")` whose message suggests an API key; `list_models` family rules: OpenAI `^gpt-\d` excluding `mini|nano|image|audio|realtime|transcribe|tts|search|codex|embedding` → `gpt`; xAI `^grok-\d` excluding `image|imagine|video|voice|code|build|fast|mini` → `grok`.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat: OpenAI and xAI adapter over the Responses API`.

### Task 16: Credential store and model registry

**Files:**
- Create: `auth/store.py`, `llm/registry.py`, `app/routes/credentials.py`, `app/routes/models.py`, `tests/auth/test_store.py`, `tests/llm/test_registry.py`

**Interfaces:**
- Produces:
  - `Credential` = `ApiKeyCredential(provider, api_key: SecretStr)` | `OAuthCredential(provider, kind: Literal["chatgpt_plan","supergrok_plan"], access_token: SecretStr, refresh_token: SecretStr | None, expires_at: datetime, client_id: str | None, account_label: str | None)`.
  - `CredentialStore(path: Path, key: bytes)`: `get(provider) -> Credential | None` (falls back to env `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`/`XAI_API_KEY`), `set(cred)`, `delete(provider, kind: str | None = None)`, `status() -> dict[str, ProviderStatus(configured: bool, kind, masked: str | None, account_label)]`. `load_store_key(settings) -> bytes` — keyring service `adsci`, else `ADSCI_SECRET_KEY_FILE` (created 0600). A provider may hold both an API key and an OAuth credential; `preferred(provider)` returns OAuth if valid else API key.
  - `adapter_for(provider: str, store: CredentialStore) -> LLMAdapter` (raises `AuthError` when unconfigured).
  - `ModelRegistry(store, adapter_factory=adapter_for, ttl=timedelta(hours=24))`: `async choices() -> list[ModelChoice(key: Literal["opus","sonnet","gpt","grok"], model: ModelInfo, available: bool, reason: str | None)]`, `async resolve(key_or_model_id: str) -> tuple[LLMAdapter, ModelInfo]`, `DEFAULT_KEY = "opus"`. `FALLBACK_MODELS: dict[str, str] = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5", "gpt": <verified in Task 15>, "grok": "grok-4.7"}`. Latest = max by `created`, else by parsed version tuple from the id.
  - Routes: `GET /api/models` → choices + `default`; `GET /api/credentials` → status (masked); `PUT /api/credentials/{provider}/api-key` `{"api_key"}` → validates with `list_models()` before saving (400 with provider message on failure); `DELETE /api/credentials/{provider}?kind=`. `GET /api/credentials` response includes for anthropic `{"subscription_supported": false, "subscription_note": "Anthropic's terms don't allow third-party apps to use Claude Pro/Max subscriptions. Use an API key from console.anthropic.com."}`.

- [ ] **Step 1: Tests**: stored file contains no plaintext key; wrong key can't decrypt; env fallback works; masked = `sk-…abcd` style (last 4); registry picks the newer of two fake opus models by `created`, falls back to `FALLBACK_MODELS` when listing raises, marks `gpt` unavailable with reason when OpenAI unconfigured; `PUT` with a key that fails validation is not saved.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat: encrypted credential store and latest-model registry`.

### Task 17: Subscription sign-in (ChatGPT plan; SuperGrok experimental)

**Files:**
- Create: `auth/chatgpt.py`, `auth/grok.py`, `app/routes/auth.py`, `tests/auth/test_chatgpt.py`, `tests/auth/test_grok.py`

**Interfaces:**
- Consumes: `CredentialStore`, `OAuthCredential`, `Settings.public_base_url`.
- Produces:
  - `ChatGPTAuth(store, settings, http: httpx.AsyncClient)`: `start() -> str` (authorize URL; generates and stores state, nonce, PKCE verifier; persists `ext_agent_host_id` UUID `urn:uuid:…` under `data_dir`), `async finish(code: str, state: str, client_id: str) -> OAuthCredential` (exchanges code, verifies ID token signature/iss/aud/nonce, requires scope `chatgpt.tokens.use.direct`), `async token() -> str` (refreshes when < 5 min left). Redirect URI `{public_base_url}/api/auth/chatgpt/callback`.
  - `GrokAuth(store, http)`: `async start() -> DeviceCode(verification_uri, user_code, expires_in, interval)`, `async poll() -> Literal["pending","approved","denied","expired"]`, `async token() -> str`. Labeled experimental everywhere.
  - Routes: `POST /api/auth/chatgpt/start` → `{"authorize_url"}`; `GET /api/auth/chatgpt/callback` → HTML page "Connected — you can close this tab" (or error); `POST /api/auth/grok/start` → device code; `GET /api/auth/grok/status`; `POST /api/auth/{provider}/logout`.

- [ ] **Step 0:** Read OpenAI's "Sign in with ChatGPT → ChatGPT plan usage" docs (registration and sign-in, profiles and sessions, models and inference, errors and recovery, UI/UX guidelines) and the devkit's `packages/local/src/oauth.ts`; copy the exact authorize/token/JWKS endpoints, `client_id=dynamic_agent_client`, `agent_name_hint="AI Data Scientist"`, scopes (identity + `offline_access` + `resource.invoke` + `chatgpt.tokens.use.direct`), resource `https://api.openai.com/v1`, and the loopback-redirect rules into constants in `auth/chatgpt.py`. Confirm the model list for plan usage comes from the devkit's model-discovery endpoint and use it in `list_models` when `billed_to="chatgpt_plan"`. For Grok, take the device-code endpoints and client id from xAI's docs if published; otherwise from an open-source implementation (Hermes Agent / OpenClaw) and cite it in a code comment. If the ChatGPT redirect cannot be a non-root loopback path, run a dedicated loopback listener on a fixed port and publish it in Docker.
- [ ] **Step 1: Tests** (mock authorization server with respx + a local RSA JWKS): `start()` URL contains state, `code_challenge_method=S256`, required scopes and host id; `finish()` with wrong state → error; missing `chatgpt.tokens.use.direct` → stored with sharing disabled and a clear message; expired access token triggers one refresh; Grok poll sequence pending → approved stores credential; Grok inference 403 surfaces `AccessDenied` advice to use an API key.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat: Sign in with ChatGPT plan usage and experimental SuperGrok sign-in`.

---

## Phase 4 — Agent

### Task 18: Orchestrator and chat SSE route

**Files:**
- Create: `agent/events.py`, `agent/prompts.py`, `agent/history.py`, `agent/orchestrator.py`, `app/routes/chat.py`, `tests/agent/test_orchestrator.py`, `tests/app/test_chat_route.py`

**Interfaces:**
- Consumes: `ToolRegistry`, `ToolOutput`, `LLMAdapter`, `ModelRegistry.resolve`, `Session`, `Verifier` (Task 19; until then a stub returning verified).
- Produces:
  - `AgentEvent` (pydantic, discriminated `type`): `plan(text)`, `step_start(step_id, tool, purpose, code, language, args)`, `step_output(step_id, blocks: list[UIBlock], is_error, duration_s)`, `answer_delta(text)`, `answer_reset()`, `verification(status: Literal["verified","warnings"], ungrounded: list[str], issues: list[str])`, `usage(Usage, model_id)`, `done(answer: str, steps: int, question_id)`, `error(message, user_action)`.
  - `Orchestrator(registry: ToolRegistry, verifier: Verifier, settings: Settings)`: `async run(session: Session, question: str, adapter: LLMAdapter, model: ModelInfo) -> AsyncIterator[AgentEvent]`.
  - `build_system_prompt(session: Session, tools: list[ToolSpec]) -> str` — includes every source's name/dialect/tables/columns (cap 12k chars, then table names only), the analysis rules from spec §6.3, "start your first reply with a numbered `Plan:` before calling tools", "cite only numbers that appear in tool results", "for cross-source questions, load each side with `sql()` in `run_python` and join in pandas".
  - `compact_history(messages: list[Message], keep_last: int = 1) -> list[Message]` — tool results older than the last `keep_last` questions clipped to 500 chars.
  - `POST /api/sessions/{id}/ask` `{"question", "model": "opus"|"sonnet"|"gpt"|"grok"|<model id> | null}` → `text/event-stream` of `AgentEvent` JSON (`event: <type>`). Default model `opus`. Session `model_id` updated.

Loop: stream turn → collect text + tool calls → first turn's leading text → `plan`; for each tool call emit `step_start`, `await registry.execute`, emit `step_output`, append `ToolResultPart(content=output.summary)`; repeat until `end_turn` or `max_agent_steps` (then one final turn with tools disabled asking for an answer with what's known); final turn text streamed as `answer_delta`; then verification (Task 19); `done`. Any `ProviderError` → single `error` event with its `user_action`, then stream ends.

- [ ] **Step 1: Tests** (FakeAdapter, sample data): scripted plan → `run_sql` → answer yields events in order `plan, step_start, step_output, answer_delta+, verification, usage, done`; a tool error result is fed back and the scripted model's corrected call succeeds; step cap 3 with a model that always calls tools → exactly 3 steps then a forced answer; history persists so a second question's adapter call contains the first Q&A, with old tool results clipped; switching model between questions sends the same neutral history to a different FakeAdapter; **Review Focus 5:** adapter raising `RateLimitError` mid-stream → exactly one `error` event with `user_action="switch_model"` and the HTTP stream closes; route test parses SSE frames end to end with `TestClient`.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat: agent orchestrator with streamed tool loop`.

### Task 19: Verifier (numeric grounding + critic)

**Files:**
- Create: `agent/verifier.py`, `tests/agent/test_verifier.py`
- Modify: `agent/orchestrator.py` (replace stub; one revision round)

**Interfaces:**
- Produces:
  - `extract_numbers(text: str) -> list[NumberMention(raw: str, value: float, is_percent: bool, span: tuple[int,int])]` — handles `1,234.5`, `-3.2`, `12%`, `$3.4M`, `1.2k`, `2.5 billion`; ignores dates (`2025-02-10`), years 1900–2100 when standalone, list markers.
  - `ground(answer: str, evidence: list[str], question: str) -> list[NumberMention]` (ungrounded) — a mention is grounded if any evidence number `e` satisfies: equal after rounding `e` to the mention's displayed decimals; or relative diff ≤ 0.5%; percent mentions also match `e*100` and `e`; integers ≤ 10 and numbers in the question are exempt. Evidence = every `ToolOutput.summary` plus flattened numbers from `ToolOutput.data` for this question.
  - `Verifier(settings)`: `async verify(question, answer, outputs: list[ToolOutput], adapter, model) -> VerificationResult(status, ungrounded: list[str], issues: list[str])`; critic uses `adapter.complete_json` with schema `{"issues": [string]}` checking unsupported claims, causal language, ignored failed assumptions, small samples; critic failure → status unaffected, issue "critic unavailable".
  - Orchestrator: if `status == "warnings"`, send one revision message listing ungrounded numbers/issues (≤ 3 more tool steps allowed), emit `answer_reset` then the revised `answer_delta`s, re-verify once, emit final `verification`.

- [ ] **Step 1: Tests**: table-driven `extract_numbers` cases above; `ground("Revenue grew 8.3%", ["growth 0.0831"], q)` → grounded; `"West AOV is 12.4% higher"` vs evidence ratio `1.1238` → grounded; `"churn is 31%"` vs evidence `0.2`  → ungrounded; `"top 3 regions"` exempt; orchestrator with a FakeAdapter whose first answer contains an ungrounded number emits `answer_reset` and a revised answer, and final `verification.status == "verified"` when the revision fixes it.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat: numeric grounding and critic verification with one revision round`.

### Task 20: Overview, insights and exports API

**Files:**
- Create: `app/routes/insights.py`, `app/routes/exports.py`, `exports/report.py`, `tests/app/test_insights_exports.py`

**Interfaces:**
- Consumes: `profile_frame`, `NotebookRecorder.to_ipynb`, `AgentEvent` history (store each question's final events on the session as `QuestionRecord(id, question, model_id, plan, steps: list[StepRecord], answer, verification)`).
- Produces:
  - `GET /api/sessions/{id}/overview?source=&table=` → `{"metrics":[Metric], "warnings":[DataQualityWarning], "distributions":[ChartBlock], "correlation": ChartBlock | null}` (table sampled to 200k rows).
  - `Insight(id, question_id | None, title, block: UIBlock, note: str | None, created_at)`; `GET/POST /api/sessions/{id}/insights`, `DELETE /api/sessions/{id}/insights/{iid}`, `PATCH` for title/note/order.
  - `GET /api/sessions/{id}/export/notebook` → `.ipynb` download; `GET /api/sessions/{id}/export/report` → Markdown with each question, answer, verification badge, code blocks, and pinned insights (charts as Plotly JSON links omitted; tables as Markdown).

- [ ] **Step 1–4:** Tests: overview on sample `orders` has `n_rows` metric and ≥ 1 distribution; pin/unpin round-trip; notebook export validates with nbformat and contains no credential strings; report contains question text and "Verified". FAIL → implement → PASS. **Step 5:** Commit `feat: overview dashboard data, insights board API and exports`.

---

## Phase 5 — Frontend

All frontend tasks: load the `frontend-design` skill first (Task 21). Verify each task by running `npm run dev` against the backend and checking in the browser; vitest for logic.

### Task 21: Frontend scaffold, API client and app shell

**Files:**
- Create: `frontend/` (Vite React TS), `src/api/client.ts`, `src/api/types.ts`, `src/api/sse.ts`, `src/hooks/useSession.ts`, `src/App.tsx`, `src/styles/theme.css`, `src/api/sse.test.ts`

**Interfaces:**
- Produces: TS types mirroring backend models (`AgentEvent`, `UIBlock`, `TableInfo`, `ModelChoice`, `ProviderStatus`, `Insight`, `ConnectionConfig`); `api.*` functions for every route in Tasks 6, 16, 17, 20; `streamAsk(sessionId, body, onEvent: (e: AgentEvent) => void, signal: AbortSignal): Promise<void>` (fetch POST + ReadableStream SSE parser handling multi-line `data:` and partial chunks); Vite proxy `/api → http://127.0.0.1:8000`.
- Layout shell: top bar (app name, model picker slot, settings button, theme toggle), three columns (Data 280px / Analysis flex / Dashboard 420px, collapsible; tabs on < 1024px). Design tokens as CSS variables with light/dark; distinctive type pairing and accent per the frontend-design skill.

- [ ] **Step 1:** vitest: SSE parser handles a frame split across chunks and two frames in one chunk. **Step 2:** FAIL. **Step 3:** Implement scaffold, client, shell. **Step 4:** `npm run test`, `npm run build`, `npx tsc --noEmit` pass; shell renders in browser. **Step 5:** Commit `feat(frontend): scaffold, typed API client, SSE streaming, app shell`.

### Task 22: Data panel

**Files:** `src/features/data/{DataPanel,UploadDropzone,ConnectionDialog,SchemaTree}.tsx`

- Upload dropzone (`.csv,.xlsx`, progress, per-table load warnings shown); "Try sample data" button; "Add database" dialog: engine picker with icons → engine-specific fields (BigQuery: project, dataset, service-account JSON upload; Snowflake: account, warehouse, role, database, schema; others: host/port/db/user/password/SSL) plus "Use connection URL" toggle; **Test connection** then **Connect**; source list with remove; schema tree source → table → columns (type icon, null %, original name tooltip). Clicking a table sets the Overview target.
- [ ] Verify in browser: upload `orders.csv`, connect sample SQLite and compose Postgres (`postgresql://adsci:adsci@localhost:5432/retail`), schema shows all three sources. vitest: connection form builds a valid `ConnectionConfig` per engine. Commit `feat(frontend): data panel with uploads, DB connections and schema explorer`.

### Task 23: Analysis chat with step timeline and model picker

**Files:** `src/features/chat/{ChatPanel,QuestionThread,StepCard,CodeBlock,AnswerView,VerificationBadge,ModelPicker,useAsk}.tsx`, `src/features/chat/reducer.ts`, `reducer.test.ts`

**Interfaces:**
- Produces: `chatReducer(state: ThreadState, event: AgentEvent): ThreadState` (pure; handles `answer_reset`).
- UI: per question — Plan card; StepCards (tool icon, purpose, collapsible SQL/Python with syntax highlighting + copy, output table (virtualized, first 50 rows), chart (react-plotly, responsive, theme-aware), error state); streaming answer (markdown); verification badge (✓ Verified / ⚠ n notes, expandable); usage chip ("$0.03 · Opus 5.5" or "Using ChatGPT plan"); pin button on every chart/table/answer; suggested follow-ups as chips; starter questions from schema when empty; Stop button (abort). Model picker in top bar: Opus (default) / Sonnet / GPT / Grok with resolved model id, unavailable ones disabled with reason and a link to Settings; "Advanced" lists all models. Error events show the message and an action button (Open settings / Switch model / Retry).
- [ ] vitest: reducer test feeding a recorded event sequence (including `answer_reset`) produces expected thread state. Verify in browser against a real provider: ask "Which region has the highest average order value, and is the difference significant?" → plan, SQL step, stats step, chart, verified answer. Commit `feat(frontend): streaming analysis timeline, verification badge, model picker`.

### Task 24: Dashboard (Overview + Insights board) and exports

**Files:** `src/features/dashboard/{DashboardPanel,OverviewView,InsightsBoard,InsightCard}.tsx`

- Overview: KPI cards (rows, columns, missing %, date range, duplicates), data-quality warning list with severity, distribution small multiples, correlation heatmap. Insights board: responsive grid of pinned cards (title editable, note, remove, drag to reorder via `PATCH` order), empty state explaining pinning. Export menu: Notebook (.ipynb), Report (.md), chart PNG (Plotly `downloadImage`).
- [ ] Verify in browser: load sample → overview populated; pin two items → appear on board, survive reload (served from API); exports download and the notebook opens in Jupyter and runs top-to-bottom against the running backend. Commit `feat(frontend): overview dashboard, insights board, exports`.

### Task 25: Settings — providers and credentials

**Files:** `src/features/settings/{SettingsDialog,ProviderCard,ApiKeyForm,SignInChatGPT,SignInGrok}.tsx`

- One card per provider (Anthropic, OpenAI, xAI): status, masked key, "Paste API key" (validated server-side, inline error), Remove. OpenAI: "Continue with ChatGPT" button (opens authorize URL in a new tab, polls `/api/credentials` until connected), shows "Using ChatGPT plan", **Manage usage** → `https://chatgpt.com/settings/usage`. xAI: "Sign in with Grok (experimental)" showing verification URL + user code, polling status, and on `AccessDenied` the message "Your Grok plan doesn't include API access here — add an xAI API key instead." Anthropic: API key form plus the note from `GET /api/credentials` explaining why subscription sign-in isn't available. Defaults section: default model, max steps.
- [ ] Verify in browser: add/remove keys; complete a ChatGPT sign-in with a Plus/Pro account if available (else verify the flow up to the authorize page and record it as unverified); model picker availability updates. Commit `feat(frontend): provider settings with API keys and subscription sign-in`.

---

## Phase 6 — Ship

### Task 26: Docker, compose, CI

**Files:** `Dockerfile`, `docker-compose.yml`, `.env.example`, `.github/workflows/ci.yml`, `backend/src/adsci/app/main.py` (serve `frontend/dist` at `/` with SPA fallback when present)

- Dockerfile multi-stage: `node:22-alpine` builds frontend → `python:3.12-slim` installs backend with uv (`--frozen`), copies `dist/`, generates sample data, runs `uvicorn adsci.app.main:app --host 0.0.0.0 --port 8000`. Compose services: `app` (ports `8000:8000`, volume `adsci-data:/app/data`, `env_file: .env` optional, `ADSCI_SECRET_KEY_FILE=/app/data/secret.key`, `ADSCI_PUBLIC_BASE_URL=http://127.0.0.1:8000`) and `postgres` (`postgres:16`, db/user/password `retail`/`adsci`/`adsci`, mounts `sample_data/out` at `/seed` and `sample_data/postgres/init.sql` into `/docker-entrypoint-initdb.d/`, port `5432:5432`). Inside compose, the app reaches it as host `postgres`.
- `.env.example`: `ANTHROPIC_API_KEY=`, `OPENAI_API_KEY=`, `XAI_API_KEY=` (all optional; keys can be added in the UI).
- CI: backend job (uv sync, ruff, pytest with a `postgres:16` service seeded via init.sql and `ADSCI_TEST_PG_URL`), frontend job (npm ci, tsc, vitest, build), docker build job.
- [ ] Verify: `docker compose up --build` from a clean clone → open `http://127.0.0.1:8000`, load sample data, connect `postgres` source, ask a question with a real key → verified answer. Commit `chore: docker compose with seeded Postgres, CI workflow`.

### Task 27: Evaluation harness

**Files:** `evals/questions.yaml`, `evals/checks.py`, `evals/run.py`, `evals/README.md`, `backend/tests/evals/test_checks.py`

**Interfaces:**
- Consumes: `Orchestrator`, `ModelRegistry`, `SessionManager`, `ground_truth.json`.
- Produces: question schema `{id, question, dataset: "retail", sources: ["files"]|["retail_sqlite"]|["files","retail_sqlite"], check: {type: "numeric", key, tolerance} | {type: "contains_all", values: [...]} | {type: "set", key} | {type: "rubric", criteria: str}}`; `score(answer: str, events: list[AgentEvent], check, truth) -> CheckResult(passed, detail)`; CLI `uv run python -m evals.run --models opus,sonnet,gpt,grok --repeats 1 --out evals/results/` writing `results.md` (per model: accuracy, avg steps, tool-error rate, verification-warning rate, median latency, avg cost) and raw JSONL. Rubric checks use a fixed judge (`opus`).
- 24 questions covering: aggregates (2025 revenue, top category), growth (YoY), group comparison + significance (West AOV, promo uplift), null result (region vs churn must say "no significant"), modeling (churn drivers, AUC above baseline), forecasting (next-30-day revenue with interval), anomalies (spike + outage dates), segmentation, data quality (duplicates/missing), cross-source (orders from CSV joined with SQLite customers by plan), XLSX targets vs actuals.
- [ ] Tests for `checks.py` only (numeric tolerance, percent parsing, set match). Run evals for each configured model; commit `evals/results/results.md`. Commit `feat: evaluation harness with cross-model comparison`.

### Task 28: README, docs and final verification

**Files:** `README.md`, `docs/architecture.svg` (or Mermaid in README), `docs/screenshots/*`, `LICENSE` (MIT)

- README: hero GIF placeholder + screenshots, one-paragraph pitch, feature list, architecture diagram, "How the agent works" (loop, tools, verification), model/provider matrix including the subscription table and the Claude-terms explanation, quick start (`docker compose up`, add a key in Settings), dev setup, eval results table, safety scope (honest sandbox limits + upgrade path), project structure, suggested resume bullets.
- [ ] Final verification (use superpowers:verification-before-completion): clean clone → `docker compose up --build` works; backend `pytest` all green; frontend `tsc`, `vitest`, `build` green; end-to-end ask on each configured provider; notebook export runs. Record a 60–90 s demo screen recording checklist in `docs/demo-script.md` (the user records the video). Commit `docs: README, architecture and demo script`.
