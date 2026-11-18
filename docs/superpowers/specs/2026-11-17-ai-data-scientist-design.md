# AI Data Scientist — Design

**Date:** 2026-11-17
**Status:** Approved 2026-11-17 (v2 + Snowflake/BigQuery in default install)

## 1. Intent

A portfolio project for a data-science / ML-engineering resume. It upgrades the common "AI Data Analyst" demo (CSV → SQL → chart) into an **AI Data Scientist**: an agent that plans an analysis, calls real tools (SQL, a stateful Python kernel, statistical and ML routines), checks its own numbers, and presents results on a dashboard.

It has to work for two audiences:

- **Recruiters / hiring managers** skimming for ~2 minutes: README hero GIF, a demo video, a clear architecture diagram, crisp bullet claims backed by an eval score.
- **Technical interviewers** reading the code: a clean, tested, explainable codebase where the agent loop, provider abstraction, tool design, sandboxing and statistical rigor can each be defended line by line.

Delivery is **a GitHub repo + demo video**, runnable locally with one command (`docker compose up`). It is built so it *could* be deployed later, but deployment is out of scope now.

## 2. Goals

1. **Data input** (user requirement):
   - Upload **CSV** and **XLSX** files (every sheet of a workbook becomes its own table).
   - **Live connections to multiple databases at once** — several named connections can be active in one session; queries run against the live source at question time.
2. **Model choice** (user requirement): default **latest Claude Opus**; the user can switch to **latest Claude Sonnet, latest OpenAI GPT (ChatGPT) model, or latest xAI Grok model** from the UI at any time, including mid-conversation.
3. **Bring-your-own credentials** (user requirement): for every provider the user can paste an **API key**; where the provider permits it, the user can instead **sign in with their subscription** (see §6.7 for what each provider allows).
4. Ask questions in plain English; get an answer backed by visible SQL, visible Python, charts, and an interpretation.
5. Go beyond descriptive analytics: profiling & data quality, hypothesis testing with assumption checks and effect sizes, predictive modeling with cross-validation and feature importance, forecasting with backtests, segmentation, anomaly detection.
6. A **dashboard**: an auto-generated overview on load, plus an "Insights board" where answers (charts, KPIs, findings) are pinned.
7. **Transparency and reproducibility:** every step's code is shown; the session exports to a runnable `.ipynb` notebook and a Markdown report.
8. **Self-verification:** the final answer is checked against tool outputs before it is shown (see §6.4).
9. A small **evaluation harness** with ground-truth questions, run across providers, producing a model-comparison table for the README.

## 3. Non-goals

- App-level user accounts / multi-user auth, billing.
- Hosting / public deployment (designed to allow it later).
- Write operations on connected databases.
- Big-data scale (target: up to ~1 GB / a few million rows on a laptop).
- Production-grade sandbox security (see §8 for what *is* done).
- Other file formats (Parquet, JSON) in v1 — easy to add later via the same loader interface.
- Providers beyond Anthropic, OpenAI and xAI in v1 (the adapter interface makes more addable).

## 4. Anti-goals (would be a failure even if it "works")

- **Hallucinated numbers:** any figure in an answer that wasn't computed by a tool.
- **Black-box answers:** a conclusion without the code that produced it.
- **Statistically naive output:** p-values without effect sizes, causal language from correlations, models without a baseline or held-out evaluation.
- **Terms-of-service violations:** using a subscription in a way its provider forbids (see §6.7).
- **Leaked credentials:** API keys, OAuth tokens or DB passwords in logs, prompts, exports or git.
- **Generic template look:** the UI should look like a deliberate product, not a default component-library page.
- **Demo that needs paid infrastructure** beyond the user's own model access.

## 5. Approaches considered

### 5.1 Agent orchestration

| Option | Trade-off | Decision |
|---|---|---|
| **Hand-rolled tool-use loop over a provider-neutral interface** | Fully transparent and explainable; we own streaming, step limits, error recovery. More code to write. | **Chosen** |
| LangGraph | Recognizable keyword; graph abstraction for planner/executor/critic. Extra abstraction to learn, debug, and defend. | Rejected for v1 |
| Claude Agent SDK | Shaped around a coding agent; Claude-only, which conflicts with multi-provider. | Rejected |

### 5.2 Multi-provider LLM layer

| Option | Trade-off | Decision |
|---|---|---|
| **Own adapter layer** (one adapter per provider over a common message/tool schema) | Shows real understanding of each provider's tool-calling format; supports subscription OAuth tokens, which generic libraries don't. ~3 adapters to maintain. | **Chosen** |
| LiteLLM | One call for 100+ providers, less code. Doesn't handle ChatGPT-plan or SuperGrok sign-in; hides the interesting engineering. | Rejected (could back an "other providers" adapter later) |
| LangChain chat models | Common abstraction, heavy dependency, same subscription gap. | Rejected |

### 5.3 Python execution

| Option | Trade-off | Decision |
|---|---|---|
| **Stateful Jupyter kernel per session** | Variables persist across steps; rich output capture; enables `.ipynb` export. Process-level isolation only. | **Chosen** |
| Fresh subprocess per call | Simple but stateless. | Rejected |
| Docker-per-execution / E2B | Strongest isolation; extra infra/cost. | Upgrade path for a public deployment |

### 5.4 Delivery

Chosen by the user: **B — repo + demo video**, local one-command run.

## 6. Architecture

```
┌──────────────────────────── React dashboard (Vite + TS) ─────────────────────────────┐
│ Data panel (files + DB connections, schema) │ Chat + live step timeline │ Dashboard  │
│ Settings: model picker · API keys · "Sign in with ChatGPT / Grok"                    │
└──────────────▲──────────────────────────────┴──────────────▲────────────┴─────▲──────┘
               │ REST                                        │ SSE              │ REST
┌──────────────┴─────────────────────────────────────────────┴──────────────────┴──────┐
│ FastAPI                                                                              │
│  Session manager ─ Data layer (DuckDB for files · SQLAlchemy for live DBs)           │
│                  ─ Kernel manager (Jupyter) ─ Credential store (encrypted, local)    │
│  Agent orchestrator: plan → tool loop (≤ N steps) → draft → verify → answer          │
│  Tool registry: profile · sql · python · stats_test · train_model · forecast ·       │
│                 segment · detect_anomalies · chart                                   │
│  LLM layer: ModelRegistry ─ AnthropicAdapter │ OpenAIAdapter │ XAIAdapter            │
│             auth: API key │ ChatGPT plan (OAuth) │ SuperGrok (OAuth, experimental)   │
└──────────────────────────────────────┬───────────────────────────────────────────────┘
                                       │
                 Anthropic API · OpenAI Responses API · xAI API
```

### 6.1 Backend (Python, FastAPI, managed with `uv`)

- **Session manager:** one session = a set of data sources + one kernel + conversation history + pinned insights + selected model. In-memory registry, files on disk under `data/sessions/<id>/`. Idle kernels are reaped.
- **Data layer** — a uniform `DataSource` interface: `name`, `dialect`, `schema()`, `query(sql, limit) -> DataFrame`, `test_connection()`.
  - **Files (CSV, XLSX):** loaded into a per-session **DuckDB** database. CSV: delimiter/encoding/type sniffing. XLSX: each sheet → table `<file>__<sheet>`, header-row detection, merged/empty rows trimmed. Column names normalized; originals kept as metadata.
  - **Live databases:** one `SQLAlchemyDataSource` per connection. Supported in v1:
    - **PostgreSQL** (incl. Redshift/Supabase/Neon-compatible), **MySQL/MariaDB**, **SQLite**, **Microsoft SQL Server**, **Snowflake**, **Google BigQuery** — all included in the default install and Docker image (user-confirmed).
    Connections are added from the UI via a form per engine (host/port/db/user/password/SSL, or service-account JSON for BigQuery) or a raw SQLAlchemy URL. Schema introspection is cached and refreshable.
  - **Multiple sources in one session:** each source has a name; `run_sql` takes a `source` argument and the agent's system prompt lists every source's schema. **Cross-source analysis** (e.g. join Postgres orders with an uploaded XLSX budget) is done by pulling each side into the Python kernel and joining in pandas — the agent is instructed to do this and the UI shows both queries.
- **SQL guard:** every statement parsed with **sqlglot** in the source's dialect; only a single `SELECT`/`WITH` allowed; `LIMIT`/`TOP` injected if missing; statement timeout set; read-only transaction where the driver supports it.
- **Kernel manager:** a Jupyter kernel per session with a bootstrap that preloads pandas/numpy/scipy/statsmodels/sklearn/plotly and helpers: `sql(query, source=...) -> DataFrame` (same guard), `show(fig)` to emit a chart, and small file tables preloaded as DataFrames. Wall-clock timeout and output-size cap per execution.
- **Notebook exporter:** every executed step appended to an `nbformat` notebook (SQL as `sql("...", source="...")` cells; narration as Markdown). DB passwords never appear — the notebook references connections by name.

### 6.2 Tools exposed to the agent

High-level tools are deterministic, unit-tested functions returning structured JSON (plus optional charts). `run_python` is the escape hatch.

| Tool | What it does | Rigor built in |
|---|---|---|
| `profile_dataset` | Types, missingness, cardinality, distributions, outliers, correlations, duplicates, data-quality warnings | Flags ID-like/constant columns, leakage suspects, skew |
| `run_sql` | Guarded read-only query on a named source; preview + row count | Row limit, timeout, schema-aware errors |
| `run_python` | Executes code in the session kernel | Timeout, output caps, traceback returned for self-correction |
| `statistical_test` | Picks the test from data types and assumptions (Welch t / Mann-Whitney, ANOVA / Kruskal-Wallis, chi-square / Fisher, Pearson / Spearman) | Normality & variance checks, effect size + CI, sample-size warnings, multiple-comparison correction |
| `train_model` | Classification/regression: baseline vs. linear vs. gradient boosting | K-fold CV, held-out test, leakage checks, permutation importance (SHAP if available) |
| `forecast` | ETS / SARIMA with seasonality detection | Rolling-origin backtest vs. naive baseline, prediction intervals |
| `segment` | Scaled k-means, k by silhouette, per-cluster profiles | Reports silhouette; describes clusters in feature terms |
| `detect_anomalies` | Isolation Forest / robust z-score | States contamination assumption; top contributing features |
| `create_chart` | Declarative spec → Plotly figure | Labelled axes, sensible defaults |

Tool outputs sent to the model are truncated/summarized to control tokens; full outputs go to the UI.

### 6.3 Agent loop

1. **System prompt:** schemas of all sources, profile summaries, tool list, analysis rules (check assumptions, report effect sizes, no causal claims from observational data, only cite tool-computed numbers).
2. **Plan:** a short numbered plan, streamed as the "Plan" step.
3. **Act:** provider-neutral tool-use loop; each call/result streamed. Errors go back to the model for self-correction. Caps: steps (default 12) and tokens per question.
4. **Draft answer:** findings, caveats, suggested follow-ups.
5. **Verify** (§6.4), then emit.
6. History persists within a session; older tool outputs compacted to summaries. **Switching model mid-session** works because history is stored in the neutral format and re-rendered for the new provider.

### 6.4 Verification ("checks numbers")

- **Numeric grounding (deterministic):** every number in the draft must match (with rounding tolerance) a number in this question's tool outputs.
- **Critic pass (LLM, short):** checks for unsupported claims, causal over-reach, ignored assumption failures, small samples.
- One revision round if issues are found; UI badge ✓ verified / ⚠ with notes.

### 6.5 Streaming protocol

SSE events per question: `plan`, `step_start`, `step_output`, `chart`, `answer_delta`, `verification`, `usage` (tokens + estimated cost, or "ChatGPT plan"), `done`, `error`.

### 6.6 LLM layer: providers and models

- **Neutral schema:** `Message`, `ToolCall`, `ToolResult`, `ToolSpec` (JSON Schema). Each adapter converts to/from its provider's format and streams normalized events.
  - `AnthropicAdapter` — Messages API with tool use and streaming.
  - `OpenAIAdapter` — Responses API with function tools and streaming; same adapter serves API-key and ChatGPT-plan tokens (only the bearer token and model list differ).
  - `XAIAdapter` — xAI's OpenAI-compatible API (`https://api.x.ai/v1`); reuses the OpenAI wire format with xAI base URL and auth.
- **ModelRegistry — "latest" resolved at runtime:** on credential setup and daily thereafter, list each provider's models with the user's credentials and pick the newest of each family by release date / version: Claude **Opus**, Claude **Sonnet**, OpenAI flagship **GPT**, xAI **Grok**. A pinned fallback table (as of 2026-11-17: `claude-opus-5-5`, `claude-sonnet-5-5`, OpenAI GPT‑6.2 flagship, `grok-4.7`; exact OpenAI IDs verified during implementation) is used if listing fails. The UI shows the resolved model ID, and an "Advanced" toggle allows picking any listed model.
- **Default:** latest Claude Opus. If no Anthropic credential is configured, the UI prompts the user to add one or pick another provider they've connected.
- Capability check: only models that support tool calling are offered.

### 6.7 Credentials and subscriptions

What each provider permits (researched 2026-11-17; re-verify before release):

| Provider | API key | Subscription sign-in | Status in this app |
|---|---|---|---|
| **OpenAI** | ✅ | ✅ **Sign in with ChatGPT** with *ChatGPT plan usage* — officially offered by OpenAI for open-source and locally run apps (Plus/Pro). OAuth 2.0 + PKCE via loopback redirect; dynamic client registration; scopes include `offline_access` and `chatgpt.tokens.use.direct`; inference via the public Responses API. | **Supported** |
| **xAI (Grok)** | ✅ | ⚠ SuperGrok / X Premium+ OAuth device-code flow (`accounts.x.ai`), used by other open-source agents; xAI gates it by tier (reports of 403 for non-Heavy tiers). | **Experimental**, with automatic "use an API key instead" fallback on 403 |
| **Anthropic (Claude)** | ✅ | ❌ Anthropic's terms forbid third-party apps from offering Claude.ai login or routing Pro/Max plan usage (enforced since 2026-04-04). | **API key only**; Settings explains why |

- **Settings screen:** per-provider card — "Paste API key" (validated with a test call) and, where supported, "Sign in with ChatGPT" / "Sign in with Grok (experimental)". Shows connection status, which credential is active, and "Manage usage" (for ChatGPT: link to ChatGPT Settings → Usage). Users may configure several providers at once.
- **Storage:** credentials are stored only on the user's machine, encrypted at rest (Fernet; key from the OS keychain via `keyring` when running natively, or from a local secret file in a Docker volume). `.env` remains supported for keys. Nothing is sent anywhere except to its own provider.
- **Docker note:** the ChatGPT loopback redirect needs a fixed callback port published from the container; Grok's device-code flow needs no callback.
- **Plan-usage labeling:** when a request runs on a subscription, the UI shows "Using ChatGPT plan" / "Using SuperGrok plan" instead of a dollar estimate; usage-limit errors surface a clear message with "Manage usage" / "Switch model".

### 6.8 Frontend (React + Vite + TypeScript + Tailwind; Plotly via `react-plotly.js`)

- **Left — Data:** upload CSV/XLSX, add database connections (engine picker → form → "Test connection"), list of active sources, schema explorer (source → tables → columns with type icons, null %, samples), sample datasets.
- **Center — Analysis:** chat with a live **step timeline** (Plan → each tool call with collapsible SQL/Python, output table, chart → Verification → Answer). Model picker in the chat header (Opus / Sonnet / GPT / Grok, plus "Advanced"). Starter questions generated from the schema.
- **Right / tab — Dashboard:** *Overview* (KPI cards, data-quality issues, distributions, correlation heatmap) and *Insights board* (pinned charts/tables/findings in a grid).
- **Settings:** providers & credentials (§6.7), defaults (model, step cap).
- **Exports:** `.ipynb`, Markdown report, chart PNGs.
- Light/dark theme; distinctive, deliberate visual style.

### 6.9 Sample data and evaluation

- **Synthetic, seeded datasets with planted ground truth:** e-commerce orders (seasonality, promotion uplift, regional difference, anomalous days) and customer churn (known drivers). Shipped as **CSV, XLSX (multi-sheet)** and as a **SQLite** DB; `docker compose` also starts an optional **Postgres** container seeded with the same data so live-DB and multi-source demos work out of the box.
- **Eval harness** (`evals/`): ~20–30 ground-truth questions (numeric tolerance or rubric). Runs per model; reports accuracy, steps, tool-error rate, verification-flag rate, latency and cost → a **model-comparison table** in the README (Opus vs Sonnet vs GPT vs Grok).

### 6.10 Testing

- `pytest` for every deterministic tool, file loaders (CSV/XLSX edge cases), SQL guard per dialect, data-source layer (SQLite + Postgres via testcontainers or the compose service), kernel manager, notebook exporter, numeric-grounding checker, credential store.
- **Adapter contract tests:** recorded provider responses (fixtures) verify each adapter's conversion of messages/tool calls/streams; OAuth flows tested against a mock authorization server.
- Agent loop tested with a **fake LLM adapter** replaying scripted tool calls (no network in CI).
- Frontend: typecheck + lint; a few component tests.
- GitHub Actions: backend tests + frontend build.

### 6.11 Repo layout

```
ai_data_scientist/
  backend/
    app/            # FastAPI app, routes, sessions, SSE
    agent/          # orchestrator, prompts, verifier
    llm/            # neutral schema, adapters (anthropic, openai, xai), model registry
    auth/           # credential store, ChatGPT OAuth, Grok device flow
    tools/          # one module per tool + registry
    data/           # DataSource impls, file loaders, SQL guard
    kernel/         # kernel manager, bootstrap, notebook exporter
    tests/
    pyproject.toml
  frontend/
  sample_data/      # generator + generated CSV/XLSX/SQLite + Postgres seed
  evals/
  docs/
  docker-compose.yml
  .env.example
  README.md
```

## 7. Constraints

- The user brings their own model access (any of: Anthropic key, OpenAI key or ChatGPT Plus/Pro, xAI key or eligible SuperGrok tier). Nothing else paid.
- Python 3.13 / Node 24 locally; Docker pins Python 3.12-slim and Node LTS.
- One-command run (`docker compose up`) plus native dev mode (`uv run` + `npm run dev`).
- ChatGPT plan usage is permitted for open-source/local apps only; a hosted or paid version would need OpenAI's approval (waitlist). Noted in README.

## 8. Safety posture (honest scope)

- SQL is read-only by parsing + read-only transactions + limits.
- Python runs in a per-session kernel inside the backend container: timeouts, output caps, memory limit, confined working directory. **Not** a hardened sandbox; README states this and names the upgrade path.
- Credentials (API keys, OAuth tokens, DB passwords) are encrypted at rest, never logged, never placed in prompts, notebooks or exports, and masked in the UI after entry.

## 9. Decided by the user

- Delivery: repo + demo video, local one-command run (option B).
- Data science / advanced-analysis focus; plain-English questions → SQL, Python, charts; dashboard.
- Inputs: **CSV, XLSX, and live connections to multiple databases**.
- Models: **default latest Opus**; options latest Opus, latest Sonnet, latest OpenAI GPT, latest Grok.
- Credentials: users can use **API keys for any provider** or **their subscriptions** — implemented where each provider's terms allow (§6.7).

## 10. Builder's calls (made in this design; open to correction)

- Hand-rolled agent loop + own provider adapters (not LiteLLM/LangChain); stateful Jupyter kernel.
- Claude subscription sign-in **excluded** (provider terms); Grok subscription marked **experimental**.
- "Latest" resolved dynamically from each provider's model list, with a pinned fallback.
- Database list: Postgres, MySQL/MariaDB, SQLite, SQL Server, Snowflake, BigQuery — all in the default install (Snowflake/BigQuery confirmed by the user).
- Cross-source joins happen in the Python kernel, not via federated SQL.
- FastAPI + React/Vite/TS/Tailwind + Plotly; DuckDB for files; SQLAlchemy for databases; sqlglot for SQL guarding.
- Synthetic seeded datasets with planted truths; a seeded Postgres container for demos; eval harness with model comparison.
- No app-level auth; credentials stored locally, encrypted.

## 11. Left to the builder

- Exact prompt wording, visual styling, chart defaults.
- Thresholds (step cap, timeouts, preview sizes, model-list refresh interval).
- Exact model-family matching rules per provider (verified against live model lists during implementation).
- Whether SHAP is included or permutation importance only.
