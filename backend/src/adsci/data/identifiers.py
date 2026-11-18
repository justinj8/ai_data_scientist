"""Turn arbitrary file/sheet/column names into safe, unquoted SQL identifiers."""

from __future__ import annotations

import re
import unicodedata

RESERVED = {
    "all", "analyse", "analyze", "and", "any", "array", "as", "asc", "between", "both", "by", "case",
    "cast", "check", "collate", "column", "constraint", "create", "current_date", "current_time",
    "current_user", "default", "delete", "desc", "distinct", "do", "drop", "else", "end", "except",
    "false", "fetch", "for", "foreign", "from", "grant", "group", "having", "in", "insert", "intersect",
    "into", "is", "join", "lateral", "leading", "like", "limit", "not", "null", "offset", "on", "only",
    "or", "order", "primary", "references", "returning", "select", "some", "table", "then", "to",
    "trailing", "true", "union", "unique", "update", "user", "using", "values", "when", "where",
    "window", "with",
}


def normalize_identifier(raw: str) -> str:
    text = unicodedata.normalize("NFKD", str(raw)).encode("ascii", "ignore").decode()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text.strip())
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    text = re.sub(r"_+", "_", text)
    if not text:
        return "col"
    if text[0].isdigit():
        text = "_" + text
    if text in RESERVED:
        text += "_"
    return text


def dedupe(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name in seen:
            seen[name] += 1
            candidate = f"{name}_{seen[name]}"
            while candidate in seen:
                seen[name] += 1
                candidate = f"{name}_{seen[name]}"
            seen[candidate] = 1
            out.append(candidate)
        else:
            seen[name] = 1
            out.append(name)
    return out
