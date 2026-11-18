"""Answer verification: every number must come from a tool result, and a critic checks the reasoning.

1. Numeric grounding (deterministic): each number in the answer must match a number in this question's
   tool outputs at the precision the answer displays (12.4% matches 0.1238, 1.1238 or 12.38; $4.66M matches
   4,658,795.4). Small integers (<= 10) and numbers from the question are exempt.
2. Critic (LLM): a short review for unsupported claims, causal over-reach, ignored assumption failures and
   small samples.
"""

from __future__ import annotations

import json
import math
import re
from typing import Literal

from pydantic import BaseModel

from adsci.config import Settings

_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b")
_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s")
_NUMBER = re.compile(
    r"(?<![\w.])(?P<sign>-)?(?P<cur>\$)?(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<dec>\d+))?"
    r"(?:\s?(?P<suffix>%|k|K|M|B|bn|million|billion|thousand)\b|(?P<pct>%))?(?![\w])"
)
_SCALES = {"k": 1e3, "K": 1e3, "thousand": 1e3, "M": 1e6, "million": 1e6, "B": 1e9, "bn": 1e9, "billion": 1e9}
_EVIDENCE_NUMBER = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")
EVIDENCE_PROMPT_CHARS = 12_000

CRITIC_SYSTEM = (
    "You review answers written by a data-science assistant. Compare the answer with the tool results it was "
    "based on and flag only material problems: claims the results don't support; causal language (causes, "
    "drives, leads to) drawn from observational data; ignoring failed assumptions or warnings shown in the "
    "results; presenting a non-significant result as a real difference; conclusions from very small samples. "
    'Return JSON {"issues": [...]} with at most 5 short, specific items, or an empty list if the answer is sound.'
)
CRITIC_SCHEMA = {
    "type": "object",
    "properties": {"issues": {"type": "array", "items": {"type": "string"}}},
    "required": ["issues"],
    "additionalProperties": False,
}


class NumberMention(BaseModel):
    raw: str
    value: float
    is_percent: bool
    decimals: int
    scale: float
    span: tuple[int, int]


class VerificationResult(BaseModel):
    status: Literal["verified", "warnings"] = "verified"
    ungrounded: list[str] = []
    issues: list[str] = []
    notes: list[str] = []


def extract_numbers(text: str) -> list[NumberMention]:
    clean = _DATE.sub(lambda m: " " * len(m.group()), text)
    clean = _LIST_MARKER.sub(lambda m: " " * len(m.group()), clean)
    out = []
    for m in _NUMBER.finditer(clean):
        whole = m.group("int").replace(",", "")
        dec = m.group("dec") or ""
        suffix = m.group("suffix") or m.group("pct")
        number = float(f"{whole}.{dec}" if dec else whole)
        if not dec and not suffix and not m.group("cur") and len(whole) == 4 and 1900 <= number <= 2100:
            continue  # a year
        scale = _SCALES.get(suffix or "", 1.0)
        value = -number if m.group("sign") else number
        out.append(NumberMention(raw=text[m.start():m.end()].strip(), value=value * scale,
                                 is_percent=suffix == "%", decimals=len(dec), scale=scale,
                                 span=(m.start(), m.end())))
    return out


def _flatten(obj, acc: list[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, int | float):
        if math.isfinite(obj):
            acc.append(float(obj))
    elif isinstance(obj, str):
        acc.extend(float(n.replace(",", "")) for n in _EVIDENCE_NUMBER.findall(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten(v, acc)
    elif isinstance(obj, list | tuple):
        for v in obj:
            _flatten(v, acc)


def evidence_numbers(evidence: list) -> list[float]:
    acc: list[float] = []
    _flatten(evidence, acc)
    return acc


def _matches(m: NumberMention, e: float) -> bool:
    shown = abs(m.value / m.scale)
    candidates = [e]
    if m.is_percent:
        candidates += [e * 100, (e - 1) * 100]
    for c in candidates:
        c = abs(c / m.scale)
        if round(c, m.decimals) == round(shown, m.decimals):
            return True
        if m.decimals == 0 and shown >= 100:  # "about 4,700,000" for 4,658,795
            zeros = len(str(int(shown))) - len(str(int(shown)).rstrip("0"))
            if zeros and round(c, -zeros) == shown:
                return True
    return False


def ground(answer: str, evidence: list, question: str) -> list[NumberMention]:
    nums = evidence_numbers(evidence)
    exempt = {abs(m.value) for m in extract_numbers(question)}
    bad = []
    for m in extract_numbers(answer):
        if m.decimals == 0 and m.scale == 1 and not m.is_percent and abs(m.value) <= 10:
            continue
        if abs(m.value) in exempt:
            continue
        if not any(_matches(m, e) for e in nums):
            bad.append(m)
    return bad


class Verifier:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def verify(self, question: str, answer: str, outputs: list, adapter, model) -> VerificationResult:
        evidence: list = []
        for o in outputs:
            evidence.append(o.summary)
            if o.data is not None:
                evidence.append(o.data)
        ungrounded = ground(answer, evidence, question)
        result = VerificationResult(ungrounded=list(dict.fromkeys(m.raw for m in ungrounded)))
        if self.settings.verifier_critic and outputs:
            results_text = "\n\n".join(f"[tool result {i + 1}]\n{o.summary}" for i, o in enumerate(outputs))
            prompt = (
                f"Question:\n{question}\n\nTool results:\n{results_text[:EVIDENCE_PROMPT_CHARS]}\n\n"
                f"Answer to review:\n{answer}"
            )
            try:
                review = await adapter.complete_json(model=model.id, system=CRITIC_SYSTEM, prompt=prompt,
                                                     schema=CRITIC_SCHEMA)
                result.issues = [str(i) for i in (review or {}).get("issues", [])][:5]
            except Exception:  # noqa: BLE001 - verification must never break the answer
                result.notes.append("The reviewer model was unavailable; only numbers were checked.")
        if result.ungrounded or result.issues:
            result.status = "warnings"
        return result


def to_json(result: VerificationResult) -> str:
    return json.dumps(result.model_dump())
