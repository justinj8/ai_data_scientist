"""The system prompt: role, analysis standards, tools and the session's data sources."""

from __future__ import annotations

from adsci.llm.types import ToolSpec

SCHEMA_BUDGET = 12_000

ROLE = """You are an expert data scientist working with a user on their data. You answer questions by \
running analyses with your tools, not from memory. Work like a careful senior data scientist: \
understand the data first, choose methods that fit the question, check assumptions, and report \
uncertainty honestly."""

RULES = """How to work:
- Start your first reply with a short numbered plan, beginning with "Plan:", then call tools.
- Prefer SQL for filtering, joining and aggregating; use run_python for anything custom.
- Before modeling or testing, understand the data (profile_dataset, or a quick SQL look).
- To compare groups or test a relationship, use statistical_test: report the test, p-value, effect size \
and confidence interval, and mention failed assumptions or small samples.
- To find what predicts an outcome, use train_model and compare against the baseline it reports.
- For questions about the future, use forecast; for unusual values, detect_anomalies; for groups of \
similar records, segment.
- Questions that span several data sources: load each side with sql(query, source=...) inside \
run_python and join them in pandas.
- If a tool returns an error, read it, fix the query or code, and try again.
- Charts help: create one when it makes the answer clearer.

Standards for the final answer:
- Only state numbers that appear in your tool results. Round sensibly and keep units.
- Observational data shows association, not causation: never say one variable causes, drives or \
leads to another unless the data comes from an experiment.
- Say when a difference is not statistically significant, and when results are uncertain.
- Lead with the direct answer, then the key evidence, then caveats, then 1-3 suggested follow-up \
questions. Use short Markdown sections and bullet points. Do not repeat tables the user already sees."""


def schema_section(session) -> str:
    lines: list[str] = []
    for name, src in session.sources.items():
        try:
            tables = src.schema()
        except Exception as e:  # noqa: BLE001 - one broken source shouldn't hide the others
            lines.append(f"\nSource `{name}` ({src.dialect}): schema unavailable ({e}).")
            continue
        lines.append(f"\nSource `{name}` — {src.kind}, SQL dialect: {src.dialect}")
        for t in tables:
            rows = f", ~{t.row_count:,} rows" if t.row_count is not None else ""
            orig = f" (from {t.original_name})" if t.original_name else ""
            cols = ", ".join(f"{c.name} {c.dtype}" for c in t.columns)
            lines.append(f"- {t.name}{orig}{rows}: {cols}")
    text = "\n".join(lines)
    if len(text) > SCHEMA_BUDGET:
        compact = []
        for name, src in session.sources.items():
            compact.append(f"\nSource `{name}` ({src.dialect}) tables: " + ", ".join(t.name for t in src.schema()))
        text = "\n".join(compact) + "\n(Schemas are large; query information_schema or run a LIMIT 5 query for columns.)"
    return text or "\n(No data sources are connected yet. Ask the user to upload a file or connect a database.)"


def build_system_prompt(session, tools: list[ToolSpec]) -> str:
    tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    return f"{ROLE}\n\n{RULES}\n\nTools:\n{tool_lines}\n\nData sources:{schema_section(session)}"
