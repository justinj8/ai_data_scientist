"""Conversation history management.

History is append-only by default: Claude Opus 5.5 binds each thinking block to the exact history before it,
so editing earlier turns invalidates later thinking. Only when a session's history exceeds the character
budget are older tool results clipped, and those turns' raw provider state (thinking) is stripped.
"""

from __future__ import annotations

from adsci.llm.types import Message, TextPart, ToolResultPart

CLIP_CHARS = 500


def _size(messages: list[Message]) -> int:
    return sum(len(p.model_dump_json()) for m in messages for p in m.parts)


def _question_starts(messages: list[Message]) -> list[int]:
    return [
        i for i, m in enumerate(messages)
        if m.role == "user" and any(isinstance(p, TextPart) for p in m.parts)
        and not any(isinstance(p, ToolResultPart) for p in m.parts)
    ]


def compact_history(messages: list[Message], char_budget: int = 600_000, keep_last: int = 1) -> list[Message]:
    if _size(messages) <= char_budget:
        return messages
    starts = _question_starts(messages)
    cutoff = starts[-keep_last] if len(starts) >= keep_last else 0
    out = []
    for i, m in enumerate(messages):
        if i >= cutoff:
            out.append(m)
            continue
        parts = [
            ToolResultPart(tool_call_id=p.tool_call_id, is_error=p.is_error,
                           content=p.content if len(p.content) <= CLIP_CHARS else p.content[:CLIP_CHARS] + " …[clipped]")
            if isinstance(p, ToolResultPart) else p
            for p in m.parts
        ]
        out.append(Message(role=m.role, parts=parts, provider_state=None))
    return out
