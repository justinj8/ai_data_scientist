"""Scoring rules for the evaluation questions in evals/questions.yaml."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

import yaml
from pydantic import BaseModel

from adsci.agent.verifier import extract_numbers

QUESTIONS = Path(__file__).resolve().parents[4] / "evals" / "questions.yaml"
Judge = Callable[[str, str, str], Awaitable[dict]]


class CheckResult(BaseModel):
    passed: bool
    detail: str


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return yaml.safe_load(path.read_text())


def _numeric(answer: str, target: float, tol: float) -> tuple[bool, str]:
    found = []
    for m in extract_numbers(answer):
        candidates = [m.value]
        if m.is_percent:
            candidates += [m.value / 100, 1 + m.value / 100]
        found.append(m.raw)
        for c in candidates:
            if (tol == 0 and abs(c - target) < 1e-9) or (tol > 0 and abs(c - target) <= tol * max(abs(target), 1e-9)):
                return True, f"found {m.raw} ≈ {target}"
    return False, f"expected ≈ {target}; numbers in answer: {', '.join(found) or 'none'}"


async def score(answer: str, check: dict, truth: dict, question: str = "", judge: Judge | None = None) -> CheckResult:
    text = answer.lower()
    kind = check["type"]
    if kind == "numeric":
        ok, detail = _numeric(answer, float(truth[check["key"]]), float(check.get("tolerance", 0.01)))
        return CheckResult(passed=ok, detail=detail)
    if kind == "set":
        missing = [v for v in truth[check["key"]] if str(v).lower() not in text]
        return CheckResult(passed=not missing, detail=f"missing: {missing}" if missing else "all present")
    if kind == "contains_all":
        missing = [v for v in check["values"] if v.lower() not in text]
        return CheckResult(passed=not missing, detail=f"missing: {missing}" if missing else "all present")
    if kind == "contains_any":
        hit = next((v for v in check["values"] if v.lower() in text), None)
        return CheckResult(passed=hit is not None, detail=f"found '{hit}'" if hit else "none of the expected phrases")
    if kind == "rubric":
        if judge is None:
            return CheckResult(passed=False, detail="no judge configured")
        verdict = await judge(question, answer, check["criteria"])
        return CheckResult(passed=bool(verdict.get("pass")), detail=str(verdict.get("reason", "")))
    raise ValueError(f"Unknown check type {kind}")
