"""A scripted adapter for tests and offline demos: replays one scripted turn per ``stream`` call."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

from adsci.llm.base import ProviderError
from adsci.llm.types import Message, ModelInfo, StreamEvent, ToolSpec

Turn = list[StreamEvent] | ProviderError


class FakeAdapter:
    provider = "fake"

    def __init__(
        self,
        script: list[Turn] | Callable[[list[Message]], Turn],
        json_responses: list[dict] | None = None,
    ) -> None:
        self.script = script
        self.json_responses = list(json_responses or [])
        self.calls: list[dict] = []
        self._i = 0

    async def stream(
        self, *, model: str, system: str, messages: list[Message], tools: list[ToolSpec], max_tokens: int = 8192
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append({"model": model, "system": system, "messages": list(messages), "tools": list(tools)})
        if callable(self.script):
            turn = self.script(messages)
        else:
            if self._i >= len(self.script):
                raise AssertionError("FakeAdapter script exhausted")
            turn = self.script[self._i]
            self._i += 1
        if isinstance(turn, ProviderError):
            raise turn
        for event in turn:
            yield event

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id="fake-model", provider="fake", family="other", display_name="Fake model")]

    async def complete_json(self, *, model: str, system: str, prompt: str, schema: dict) -> dict:
        self.calls.append({"model": model, "system": system, "prompt": prompt, "schema": schema})
        return self.json_responses.pop(0) if self.json_responses else {}
