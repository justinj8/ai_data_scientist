"""Run the evaluation questions against one or more models and write a comparison table.

    uv run --project backend python -m adsci.evals.run --models opus,sonnet,gpt,grok

Uses the credentials saved in the app (or ANTHROPIC_API_KEY / OPENAI_API_KEY / XAI_API_KEY). Every run calls
the real model APIs and costs money.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import statistics
import threading
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import httpx
import uvicorn

from adsci.app.main import create_app
from adsci.config import Settings
from adsci.evals.checks import load_questions, score
from adsci.llm.base import ProviderError
from adsci.samples import ensure_samples

ROOT = Path(__file__).resolve().parents[4]
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"pass": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["pass", "reason"],
    "additionalProperties": False,
}
JUDGE_SYSTEM = "You grade answers from a data-analysis assistant against a rubric. Be strict but fair."


def results_markdown(rows: list[dict]) -> str:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    out = [
        "| Model | Accuracy | Avg steps | Tool-error rate | Verified answers | Median latency | Avg cost |",
        "|---|---|---|---|---|---|---|",
    ]
    for model, rs in by_model.items():
        n = len(rs)
        passed = sum(r["passed"] for r in rs)
        steps = sum(r["steps"] for r in rs)
        errs = sum(r["tool_errors"] for r in rs)
        costs = [r["cost_usd"] for r in rs if r["cost_usd"] is not None]
        out.append(
            f"| {model} | {passed / n:.0%} ({passed}/{n}) | {steps / n:.1f} | {errs / max(steps, 1):.0%} | "
            f"{sum(r['verified'] for r in rs) / n:.0%} | {statistics.median(r['latency_s'] for r in rs):.1f}s | "
            + (f"${sum(costs) / len(costs):.3f}" if costs else "plan") + " |"
        )
    models = list(by_model)
    out += ["", "| Question | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    ids = list(dict.fromkeys(r["id"] for r in rows))
    for qid in ids:
        cells = []
        for m in models:
            r = next((x for x in by_model[m] if x["id"] == qid), None)
            cells.append("—" if r is None else ("✅" if r["passed"] else "❌"))
        out.append(f"| {qid} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _run(args) -> None:
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    settings = Settings(data_dir=Path(args.data_dir), internal_base_url=base, port=port)
    app = create_app(settings)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", proxy_headers=False))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        await asyncio.sleep(0.05)

    truth = json.loads((ensure_samples(settings) / "ground_truth.json").read_text())
    questions = load_questions()[: args.limit or None]
    try:
        judge_adapter, judge_model = await app.state.models.resolve(args.judge)
    except ProviderError as e:
        judge_adapter = judge_model = None
        print(f"Judge model unavailable ({e.message}); rubric questions will fail.")

    async def judge(question: str, answer: str, criteria: str) -> dict:
        prompt = f"Question:\n{question}\n\nAnswer:\n{answer}\n\nRubric:\n{criteria}\n\nDoes the answer meet the rubric?"
        return await judge_adapter.complete_json(model=judge_model.id, system=JUDGE_SYSTEM, prompt=prompt,
                                                 schema=JUDGE_SCHEMA)

    out_dir = Path(args.out)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    async with httpx.AsyncClient(base_url=base, timeout=120) as http:
        for key in args.models.split(","):
            try:
                adapter, model = await app.state.models.resolve(key.strip())
            except ProviderError as e:
                print(f"Skipping {key}: {e.message}")
                continue
            raw = (out_dir / "raw" / f"{model.id}.jsonl").open("w")
            for q in questions:
                sid = (await http.post("/api/sessions")).json()["session_id"]
                await http.post(f"/api/sessions/{sid}/samples/retail")
                session = app.state.sessions.get(sid)
                started = time.monotonic()
                events = [e async for e in app.state.orchestrator.run(session, q["question"], adapter, model)]
                latency = time.monotonic() - started
                done = next((e for e in events if e.type == "done"), None)
                err = next((e for e in events if e.type == "error"), None)
                answer = done.answer if done else ""
                result = await score(answer, q["check"], truth, q["question"], judge if judge_adapter else None)
                usage = next((e for e in events if e.type == "usage"), None)
                verification = next((e for e in reversed(events) if e.type == "verification"), None)
                row = {
                    "model": model.id, "id": q["id"], "passed": result.passed and done is not None,
                    "detail": result.detail, "steps": done.steps if done else 0,
                    "tool_errors": sum(1 for e in events if e.type == "step_output" and e.is_error),
                    "verified": bool(verification and verification.status == "verified"),
                    "latency_s": round(latency, 2), "cost_usd": usage.usage.cost_usd if usage else None,
                    "error": err.message if err else None,
                }
                rows.append(row)
                raw.write(json.dumps({**row, "question": q["question"], "answer": answer}) + "\n")
                raw.flush()
                print(f"{model.id:24} {q['id']:28} {'PASS' if row['passed'] else 'FAIL'}  {row['detail'][:70]}")
                app.state.sessions.close(sid)
            raw.close()
    server.should_exit = True
    if rows:
        stamp = datetime.now(UTC).strftime("%Y-%m-%d")
        md = f"# Evaluation results ({stamp})\n\n{len(questions)} questions on the seeded retail data. " \
             f"Rubric questions graded by {judge_model.id if judge_model else 'n/a'}.\n\n" + results_markdown(rows)
        (out_dir / "results.md").write_text(md)
        print(md)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="opus,sonnet,gpt,grok")
    p.add_argument("--judge", default="opus")
    p.add_argument("--limit", type=int, default=0, help="Only the first N questions (0 = all).")
    p.add_argument("--data-dir", default=str(ROOT / "data"), help="App data dir holding saved credentials.")
    p.add_argument("--out", default=str(ROOT / "evals" / "results"))
    asyncio.run(_run(p.parse_args()))


if __name__ == "__main__":
    main()
