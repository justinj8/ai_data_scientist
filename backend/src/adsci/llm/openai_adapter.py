"""OpenAI and xAI adapter over the Responses API (xAI serves an OpenAI-compatible Responses endpoint).

Requests are stateless (``store=False``). For OpenAI we request encrypted reasoning items and replay them
verbatim through ``provider_state`` so the model keeps its reasoning across tool calls.
The bearer token comes from an async ``token_provider`` on every request, so API keys and refreshed
subscription OAuth tokens (ChatGPT plan, SuperGrok) share one code path.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from typing import Literal

import httpx2
import openai

from adsci.llm.base import AccessDenied, AuthError, ProviderError, ProviderUnavailable, RateLimitError
from adsci.llm.types import (
    BilledTo,
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

ProviderName = Literal["openai", "xai"]
BASE_URLS = {"openai": "https://api.openai.com/v1", "xai": "https://api.x.ai/v1"}
# USD per million tokens (input, output), as of 2026-11-17.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-6.2-astra": (8.0, 40.0),
    "gpt-6.2-sol": (2.0, 10.0),
    "gpt-6.1-terra": (1.0, 5.0),
    "gpt-6-luna": (0.10, 0.50),
    "grok-4.7": (2.0, 6.0),
    "grok-4.6": (2.0, 6.0),
}
_OPENAI_EXCLUDE = re.compile(
    r"mini|nano|image|audio|realtime|transcribe|tts|search|codex|embedding|cyber|rosalind|live|law|whisper|translate"
)
_XAI_EXCLUDE = re.compile(r"image|imagine|video|voice|code|build|fast|mini|vision")
_PLAN_NAMES = {"chatgpt_plan": "ChatGPT plan", "supergrok_plan": "SuperGrok plan"}


def openai_family(model_id: str) -> str:
    return "gpt" if re.match(r"^gpt-\d", model_id) and not _OPENAI_EXCLUDE.search(model_id) else "other"


def xai_family(model_id: str) -> str:
    return "grok" if re.match(r"^grok-\d", model_id) and not _XAI_EXCLUDE.search(model_id) else "other"


def to_responses_input(messages: list[Message], provider: str) -> list[dict]:
    items: list[dict] = []
    for m in messages:
        state = m.provider_state
        if m.role == "assistant" and state and state.get("provider") == provider:
            output = state["output"]
            if provider == "xai":  # xAI reasoning items are not replayable without encrypted content
                output = [o for o in output if o.get("type") != "reasoning"]
            items.extend(output)
            continue
        for p in m.parts:
            if isinstance(p, ToolResultPart):
                items.append({"type": "function_call_output", "call_id": p.tool_call_id, "output": p.content})
        for p in m.parts:
            if isinstance(p, TextPart) and p.text:
                items.append({"role": m.role, "content": p.text})
            elif isinstance(p, ToolCallPart):
                items.append({"type": "function_call", "call_id": p.id, "name": p.name,
                              "arguments": json.dumps(p.arguments)})
    return items


class OpenAICompatAdapter:
    def __init__(
        self,
        provider: ProviderName,
        token_provider: Callable[[], Awaitable[str]],
        billed_to: BilledTo = "api_key",
        base_url: str | None = None,
        http_client: httpx2.AsyncClient | None = None,
        max_retries: int = 2,
        effort: str = "high",
    ) -> None:
        self.provider = provider
        self.token_provider = token_provider
        self.billed_to = billed_to
        self.base_url = base_url or BASE_URLS[provider]  # explicit: never pick up OPENAI_BASE_URL silently
        self.http_client = http_client
        self.max_retries = max_retries
        self.effort = effort

    async def _client(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            api_key=await self.token_provider(),
            base_url=self.base_url,
            http_client=self.http_client,
            max_retries=self.max_retries,
        )

    def _map_error(self, e: Exception) -> ProviderError:
        msg = getattr(e, "message", None) or str(e)
        who = "OpenAI" if self.provider == "openai" else "xAI"
        plan = _PLAN_NAMES.get(self.billed_to)
        if isinstance(e, openai.AuthenticationError):
            what = f"Your {plan} sign-in has expired" if plan else f"{who} rejected the API key"
            return AuthError(f"{what}: {msg}", self.provider)
        if isinstance(e, openai.PermissionDeniedError):
            if plan:
                return AccessDenied(
                    f"Your {plan} doesn't include API access for this app; add an {who} API key instead. ({msg})",
                    self.provider,
                )
            return AccessDenied(f"This {who} API key can't use that model: {msg}", self.provider)
        if isinstance(e, openai.RateLimitError):
            what = f"{plan} usage limit reached" if plan else f"{who} rate limit reached"
            return RateLimitError(f"{what}: {msg}", self.provider)
        if isinstance(e, openai.NotFoundError):
            return ProviderError(f"Model not found on {who}: {msg}", self.provider, user_action="switch_model")
        if isinstance(e, openai.InternalServerError | openai.APIConnectionError):
            return ProviderUnavailable(f"{who} is unavailable: {msg}", self.provider)
        if isinstance(e, openai.APIStatusError):
            return ProviderError(f"{who} API error ({e.status_code}): {msg}", self.provider)
        return ProviderUnavailable(f"{who} request failed: {msg}", self.provider)

    def _cost(self, model: str, inp: int, out: int) -> float | None:
        if self.billed_to != "api_key" or model not in PRICES:
            return None
        pin, pout = PRICES[model]
        return inp * pin / 1e6 + out * pout / 1e6

    async def stream(
        self, *, model: str, system: str, messages: list[Message], tools: list[ToolSpec], max_tokens: int = 32000
    ) -> AsyncIterator[StreamEvent]:
        kwargs: dict = {
            "model": model,
            "instructions": system,
            "input": to_responses_input(messages, self.provider),
            "max_output_tokens": max_tokens,
            "store": False,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = [
                {"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters,
                 "strict": False}
                for t in tools
            ]
        if self.provider == "openai":
            kwargs["reasoning"] = {"effort": self.effort}
            kwargs["include"] = ["reasoning.encrypted_content"]
        final = None
        try:
            client = await self._client()
            stream = await client.responses.create(**kwargs)
            async for event in stream:
                if event.type == "response.output_text.delta" and event.delta:
                    yield TextDelta(text=event.delta)
                elif event.type in ("response.completed", "response.incomplete", "response.failed"):
                    final = event.response
        except openai.APIError as e:
            raise self._map_error(e) from None
        if final is None:
            raise ProviderUnavailable("The response stream ended before completing.", self.provider)
        if final.status == "failed":
            err = getattr(final, "error", None)
            raise ProviderError(f"The model failed: {getattr(err, 'message', 'unknown error')}", self.provider)

        output = [o.model_dump(mode="json", exclude_none=True) for o in final.output]
        calls = [o for o in output if o.get("type") == "function_call"]
        for c in calls:
            yield ToolCallEvent(call=ToolCallPart(id=c["call_id"], name=c["name"],
                                                  arguments=json.loads(c.get("arguments") or "{}")))
        reason = getattr(getattr(final, "incomplete_details", None), "reason", None)
        if calls:
            stop = "tool_use"
        elif final.status == "incomplete" and reason == "max_output_tokens":
            stop = "max_tokens"
        elif final.status == "incomplete" and reason == "content_filter":
            stop = "refusal"
        else:
            stop = "end_turn"
        usage = final.usage
        inp, out = (usage.input_tokens, usage.output_tokens) if usage else (0, 0)
        yield TurnEnd(
            stop_reason=stop,
            usage=Usage(input_tokens=inp, output_tokens=out, cost_usd=self._cost(model, inp, out),
                        billed_to=self.billed_to),
            provider_state={"provider": self.provider, "model": final.model, "output": output},
        )

    async def _plan_models(self) -> list[ModelInfo]:
        """Plan tokens get a different model-list shape: {"models": [{"slug", "display_name", "visibility"}]}."""
        http = self.http_client or httpx2.AsyncClient(timeout=30)
        try:
            r = await http.get(f"{self.base_url}/models",
                               headers={"Authorization": f"Bearer {await self.token_provider()}"})
        except httpx2.HTTPError as e:
            raise ProviderUnavailable(f"OpenAI is unavailable: {e}", self.provider) from None
        if r.status_code >= 400:
            raise self._map_error(openai.APIStatusError(r.text, response=r, body=None))
        out = []
        for m in r.json().get("models", []):
            if m.get("visibility", "list") != "list":
                continue
            family = openai_family(m["slug"])
            out.append(ModelInfo(id=m["slug"], provider=self.provider, family=family,
                                 display_name=m.get("display_name") or m["slug"], supports_tools=family != "other"))
        return out

    async def list_models(self) -> list[ModelInfo]:
        if self.billed_to == "chatgpt_plan":
            return await self._plan_models()
        try:
            client = await self._client()
            models = [m async for m in client.models.list()]
        except openai.APIError as e:
            raise self._map_error(e) from None
        fam = openai_family if self.provider == "openai" else xai_family
        out = []
        for m in models:
            family = fam(m.id)
            out.append(ModelInfo(
                id=m.id, provider=self.provider, family=family, display_name=m.id,
                created=datetime.fromtimestamp(m.created, tz=UTC) if getattr(m, "created", None) else None,
                supports_tools=family != "other",
            ))
        return out

    async def complete_json(self, *, model: str, system: str, prompt: str, schema: dict) -> dict:
        try:
            client = await self._client()
            r = await client.responses.create(
                model=model, instructions=system, input=prompt, store=False,
                text={"format": {"type": "json_schema", "name": "result", "schema": schema, "strict": False}},
            )
        except openai.APIError as e:
            raise self._map_error(e) from None
        return json.loads(r.output_text)


def _static(token: str) -> Callable[[], Awaitable[str]]:
    async def provide() -> str:
        return token

    return provide


def openai_with_key(key: str) -> OpenAICompatAdapter:
    return OpenAICompatAdapter("openai", _static(key), billed_to="api_key")


def openai_with_chatgpt(token_provider: Callable[[], Awaitable[str]]) -> OpenAICompatAdapter:
    return OpenAICompatAdapter("openai", token_provider, billed_to="chatgpt_plan")


def xai_with_key(key: str) -> OpenAICompatAdapter:
    return OpenAICompatAdapter("xai", _static(key), billed_to="api_key")


def xai_with_supergrok(token_provider: Callable[[], Awaitable[str]]) -> OpenAICompatAdapter:
    return OpenAICompatAdapter("xai", token_provider, billed_to="supergrok_plan")
