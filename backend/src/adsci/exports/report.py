"""A Markdown report of a session: questions, answers, verification, code and pinned insights."""

from __future__ import annotations

from datetime import UTC, datetime


def _table_md(columns: list, rows: list, total: int, limit: int = 20) -> str:
    head = "| " + " | ".join(str(c) for c in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in rows[:limit]]
    more = [f"\n_{total - limit} more rows not shown._"] if total > limit else []
    return "\n".join([head, sep, *body, *more])


def block_md(block: dict) -> str:
    t = block.get("type")
    if t == "table":
        return _table_md(block["columns"], block["rows"], block.get("total_rows", len(block["rows"])))
    if t == "text":
        return block["text"] if block.get("format") == "markdown" else f"```\n{block['text']}\n```"
    if t == "metrics":
        return "\n".join(f"- **{m['label']}:** {m['value']}" + (f" ({m['detail']})" if m.get("detail") else "")
                         for m in block["items"])
    if t == "chart":
        return f"_Chart: {block.get('title') or 'untitled'} (interactive in the app)._"
    if t == "error":
        return f"> Error: {block['message']}"
    return ""


def build_report(session) -> str:
    out = ["# AI Data Scientist report", f"_Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC._", ""]
    out.append("**Data sources:** " + (", ".join(f"`{n}` ({s.dialect})" for n, s in session.sources.items()) or "none"))
    for q in session.questions:
        out += ["", f"## {q.question}", f"_Model: {q.model_id}_", ""]
        v = q.verification
        if v is not None and v.status == "verified":
            out.append("✅ **Verified**: every number in the answer was found in the analysis results.")
        elif v is not None:
            problems = [f"number {n} not found in results" for n in v.ungrounded] + v.issues
            out.append("⚠️ **Needs review**: " + "; ".join(problems))
        out += ["", q.answer, "", "<details><summary>Analysis steps</summary>", ""]
        if q.plan:
            out += [q.plan, ""]
        for i, s in enumerate(q.steps, start=1):
            out.append(f"**Step {i}: {s.tool}**" + (f" — {s.purpose}" if s.purpose else ""))
            if s.code:
                out += [f"```{s.language or ''}", s.code, "```"]
            for b in s.blocks:
                md = block_md(b)
                if md:
                    out += [md, ""]
        out += ["</details>"]
    if session.insights:
        out += ["", "## Pinned insights"]
        for ins in sorted(session.insights, key=lambda i: i.order):
            out += ["", f"### {ins.title}", block_md(ins.block.model_dump())]
            if ins.note:
                out += ["", f"_Note: {ins.note}_"]
    return "\n".join(out) + "\n"
