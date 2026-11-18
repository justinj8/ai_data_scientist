"""The adapter interface every provider implements, and the errors adapters raise."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol

from adsci.llm.types import Message, ModelInfo, StreamEvent, ToolSpec

UserAction = Literal["update_credentials", "switch_model", "retry", "none"]


class ProviderError(Exception):
    user_action: UserAction = "none"
    retryable: bool = False

    def __init__(self, message: str, provider: str, *, user_action: UserAction | None = None,
                 retryable: bool | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        if user_action is not None:
            self.user_action = user_action
        if retryable is not None:
            self.retryable = retryable


class AuthError(ProviderError):
    user_action = "update_credentials"


class AccessDenied(ProviderError):
    """Authenticated, but this credential may not use the API (e.g. a subscription tier without API access)."""

    user_action = "update_credentials"


class RateLimitError(ProviderError):
    """Rate limit or subscription usage limit reached."""

    user_action = "switch_model"
    retryable = True


class ProviderUnavailable(ProviderError):
    user_action = "retry"
    retryable = True


class LLMAdapter(Protocol):
    provider: str

    def stream(
        self, *, model: str, system: str, messages: list[Message], tools: list[ToolSpec], max_tokens: int = 8192
    ) -> AsyncIterator[StreamEvent]: ...

    async def list_models(self) -> list[ModelInfo]: ...

    async def complete_json(self, *, model: str, system: str, prompt: str, schema: dict) -> dict: ...
