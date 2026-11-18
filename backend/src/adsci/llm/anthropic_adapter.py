"""Anthropic (Claude) adapter: Messages API streaming with tool use.

Notes for current models (Opus 5.5 / Sonnet 5.5):
- Thinking is adaptive and always on; we omit the ``thinking`` parameter and control depth with ``effort``.
- Thinking blocks must be replayed unchanged, so each assistant turn's raw content is kept in
  ``provider_state`` and sent back verbatim.
- Server-side refusal fallbacks (``fallbacks="default"``) are enabled so a safety-classifier refusal
  is retried on a fallback model instead of ending the analysis.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import datetime

import anthropic

from adsci.llm.base import AccessDenied, AuthError, ProviderError, ProviderUnavailable, RateLimitError
from adsci.llm.types import (
    Message,
    ModelInfo,
    StreamEvent,
    TextDelta,
    TextPart,
    ToolCallEvent,
    ToolCallPart,
    ToolResultPart,
    ToolSpec,
    TurnEnd,
    Usage,
)

PROVIDER = "anthropic"
# USD per million tokens (input, output).
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5-2": (8.0, 40.0),
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
DEFAULT_BASE_URL = "https://api.anthropic.com"


def _family(model_id: str) -> str:
    if model_id.startswith("claude-opus"):
        return "opus"
    if model_id.startswith("claude-sonnet"):
        return "sonnet"
    return "other"


def to_anthropic_messages(messages: list[Message]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        state = m.provider_state
        if m.role == "assistant" and state and state.get("provider") == PROVIDER:
            out.append({"role": "assistant", "content": state["content"]})
            continue
        results = [
            {"type": "tool_result", "tool_use_id": p.tool_call_id, "content": p.content, "is_error": p.is_error}
            for p in m.parts
            if isinstance(p, ToolResultPart)
        ]
        others: list[dict] = []
        for p in m.parts:
            if isinstance(p, TextPart) and p.text:
                others.append({"type": "text", "text": p.text})
            elif isinstance(p, ToolCallPart):
                others.append({"type": "tool_use", "id": p.id, "name": p.name, "input": p.arguments})
        # Tool results must come first in a user turn.
        out.append({"role": m.role, "content": results + others})
    return out


def _cost(model: str, usage) -> float | None:
    price = PRICES.get(model)
    if price is None:
        return None
    return usage.input_tokens * price[0] / 1e6 + usage.output_tokens * price[1] / 1e6


def _map_error(e: Exception) -> ProviderError:
    msg = getattr(e, "message", None) or str(e)
    if isinstance(e, anthropic.AuthenticationError):
        return AuthError(f"Anthropic rejected the API key: {msg}", PROVIDER)
    if isinstance(e, anthropic.PermissionDeniedError):
        return AccessDenied(f"This Anthropic key can't use that model: {msg}", PROVIDER)
    if isinstance(e, anthropic.RateLimitError):
        return RateLimitError(f"Anthropic rate limit reached: {msg}", PROVIDER)
    if isinstance(e, anthropic.NotFoundError):
        return ProviderError(f"Model not found on Anthropic: {msg}", PROVIDER, user_action="switch_model")
    if isinstance(e, anthropic.InternalServerError | anthropic.APIConnectionError):
        return ProviderUnavailable(f"Anthropic is unavailable: {msg}", PROVIDER)
    if isinstance(e, anthropic.APIStatusError):
        return ProviderError(f"Anthropic API error ({e.status_code}): {msg}", PROVIDER)
    return ProviderUnavailable(f"Anthropic request failed: {msg}", PROVIDER)


class AnthropicAdapter:
    provider = PROVIDER

    def __init__(
        self,
        api_key: str,
        client: anthropic.AsyncAnthropic | None = None,
        effort: str = "high",
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        # base_url is explicit so an ambient ANTHROPIC_BASE_URL (e.g. a developer proxy) is never used silently.
        self.client = client or anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url, max_retries=2)
        self.effort = effort

    async def stream(
        self, *, model: str, system: str, messages: list[Message], tools: list[ToolSpec], max_tokens: int = 32000
    ) -> AsyncIterator[StreamEvent]:
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": to_anthropic_messages(messages),
            "output_config": {"effort": self.effort},
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }
        if tools:
            kwargs["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
        try:
            async with self.client.beta.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "text" and event.text:
                        yield TextDelta(text=event.text)
                final = await stream.get_final_message()
        except anthropic.APIError as e:
            raise _map_error(e) from None

        for block in final.content:
            if block.type == "tool_use":
                args = block.input if isinstance(block.input, dict) else json.loads(block.input or "{}")
                yield ToolCallEvent(call=ToolCallPart(id=block.id, name=block.name, arguments=args))
        stop = final.stop_reason
        stop_reason = (
            "tool_use" if stop == "tool_use" else "max_tokens" if stop == "max_tokens"
            else "refusal" if stop == "refusal" else "end_turn"
        )
        yield TurnEnd(
            stop_reason=stop_reason,
            usage=Usage(
                input_tokens=final.usage.input_tokens,
                output_tokens=final.usage.output_tokens,
                cost_usd=_cost(final.model or model, final.usage),
                billed_to="api_key",
            ),
            provider_state={
                "provider": PROVIDER,
                "model": final.model,
                "content": [b.model_dump(mode="json", exclude_none=True) for b in final.content],
            },
        )

    async def list_models(self) -> list[ModelInfo]:
        try:
            page = await self.client.models.list(limit=100)
            models = [m async for m in page]
        except anthropic.APIError as e:
            raise _map_error(e) from None
        return [
            ModelInfo(
                id=m.id,
                provider=PROVIDER,
                family=_family(m.id),
                display_name=m.display_name,
                created=m.created_at if isinstance(m.created_at, datetime) else None,
            )
            for m in models
        ]

    async def complete_json(self, *, model: str, system: str, prompt: str, schema: dict) -> dict:
        try:
            msg = await self.client.messages.create(
                model=model,
                max_tokens=8000,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            )
        except anthropic.APIError as e:
            raise _map_error(e) from None
        text = "".join(b.text for b in msg.content if b.type == "text")
        return json.loads(text)
