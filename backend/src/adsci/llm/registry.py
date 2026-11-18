"""Which model is "latest" for each choice, resolved at runtime from each provider's model list."""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import BaseModel

from adsci.auth.store import ApiKeyCredential, CredentialStore, OAuthCredential
from adsci.llm.anthropic_adapter import AnthropicAdapter
from adsci.llm.base import AuthError, LLMAdapter
from adsci.llm.openai_adapter import OpenAICompatAdapter
from adsci.llm.types import ModelInfo

ChoiceKey = Literal["opus", "sonnet", "gpt", "grok"]
CHOICE_KEYS: tuple[ChoiceKey, ...] = ("opus", "sonnet", "gpt", "grok")
DEFAULT_KEY: ChoiceKey = "opus"
CHOICE_PROVIDER = {"opus": "anthropic", "sonnet": "anthropic", "gpt": "openai", "grok": "xai"}
PROVIDER_NAMES = {"anthropic": "Anthropic", "openai": "OpenAI", "xai": "xAI"}
CHOICE_LABELS = {"opus": "Claude Opus", "sonnet": "Claude Sonnet", "gpt": "OpenAI GPT", "grok": "xAI Grok"}
# Used when a provider's model list can't be fetched. Verified 2026-11-17.
FALLBACK_MODELS: dict[str, str] = {
    "opus": "claude-opus-5-5",
    "sonnet": "claude-sonnet-5-5",
    "gpt": "gpt-6.2-astra",
    "grok": "grok-4.7",
}
OPENAI_TIERS = {"astra": 4, "sol": 3, "terra": 2, "luna": 1}


def _version(model_id: str) -> tuple[int, ...]:
    m = re.match(r"^[a-z]+-(?:[a-z]+-)?(\d+(?:[.-]\d+)*)", model_id)
    if not m:
        return ()
    return tuple(int(x) for x in re.split(r"[.-]", m.group(1)) if len(x) <= 2)


def latest(key: str, models: list[ModelInfo]) -> ModelInfo | None:
    candidates = [m for m in models if m.family == key and m.supports_tools]
    if not candidates:
        return None
    if key == "gpt":
        # Newest version, then the flagship tier (release dates of tiers within a version vary).
        return max(candidates, key=lambda m: (_version(m.id), max((v for t, v in OPENAI_TIERS.items() if t in m.id), default=0)))
    # Anthropic and xAI: newest release; version numbers like "4.20" vs "4.7" don't sort reliably.
    return max(candidates, key=lambda m: (m.created.timestamp() if m.created else 0, _version(m.id)))


def provider_for_model(model_id: str) -> str:
    if model_id.startswith("claude-"):
        return "anthropic"
    if model_id.startswith("grok-"):
        return "xai"
    return "openai"


def fallback_info(key: str) -> ModelInfo:
    mid = FALLBACK_MODELS[key]
    return ModelInfo(id=mid, provider=CHOICE_PROVIDER[key], family=key, display_name=mid)


class ModelChoice(BaseModel):
    key: ChoiceKey
    label: str
    model: ModelInfo
    available: bool
    reason: str | None = None
    billed_to: str | None = None


TokenProviderFactory = Callable[[OAuthCredential], Callable[[], Awaitable[str]]]


class AdapterFactory:
    """Builds the right adapter for a provider from whatever credential is stored for it."""

    def __init__(self, store: CredentialStore) -> None:
        self.store = store
        # Filled by the auth module: kind -> function(credential) -> async token provider (handles refresh).
        self.token_providers: dict[str, TokenProviderFactory] = {}

    def for_api_key(self, provider: str, key: str) -> LLMAdapter:
        if provider == "anthropic":
            return AnthropicAdapter(api_key=key)
        return OpenAICompatAdapter(provider, _static(key), billed_to="api_key")

    def __call__(self, provider: str) -> LLMAdapter:
        cred = self.store.get(provider)
        if cred is None:
            raise AuthError(
                f"No {PROVIDER_NAMES[provider]} credentials. Add an API key"
                + (" or sign in with your subscription" if provider != "anthropic" else "")
                + " in Settings.",
                provider,
            )
        if isinstance(cred, ApiKeyCredential):
            return self.for_api_key(provider, cred.api_key.get_secret_value())
        make = self.token_providers.get(cred.kind)
        if make is None:
            raise AuthError(f"Sign-in method {cred.kind} is not available.", provider)
        return OpenAICompatAdapter(provider, make(cred), billed_to=cred.kind)


def _static(token: str):
    async def provide() -> str:
        return token

    return provide


class ModelRegistry:
    def __init__(self, store: CredentialStore, adapter_factory: Callable[[str], LLMAdapter], ttl_s: float = 86400):
        self.store = store
        self.adapter_factory = adapter_factory
        self.ttl_s = ttl_s
        self._cache: dict[str, tuple[float, list[ModelInfo]]] = {}

    def invalidate(self, provider: str | None = None) -> None:
        if provider is None:
            self._cache.clear()
        else:
            self._cache.pop(provider, None)

    async def models(self, provider: str) -> list[ModelInfo] | None:
        hit = self._cache.get(provider)
        if hit and time.monotonic() - hit[0] < self.ttl_s:
            return hit[1]
        try:
            models = await self.adapter_factory(provider).list_models()
        except Exception:  # noqa: BLE001 - one provider's failure must never hide the others
            return None
        self._cache[provider] = (time.monotonic(), models)
        return models

    async def choices(self) -> list[ModelChoice]:
        out = []
        for key in CHOICE_KEYS:
            provider = CHOICE_PROVIDER[key]
            cred = self.store.get(provider)
            if cred is None:
                hint = "an API key" if provider == "anthropic" else "an API key or sign in with your subscription"
                out.append(ModelChoice(key=key, label=CHOICE_LABELS[key], model=fallback_info(key), available=False,
                                       reason=f"Add {hint} for {PROVIDER_NAMES[provider]} in Settings."))
                continue
            listed = await self.models(provider)
            model = (latest(key, listed) if listed else None) or fallback_info(key)
            out.append(ModelChoice(key=key, label=CHOICE_LABELS[key], model=model, available=True, billed_to=cred.kind))
        return out

    async def all_models(self) -> list[ModelInfo]:
        out = []
        for provider in ("anthropic", "openai", "xai"):
            if self.store.get(provider) is not None:
                out += [m for m in (await self.models(provider) or []) if m.supports_tools and m.family != "other"]
        return out

    async def resolve(self, key_or_model_id: str | None) -> tuple[LLMAdapter, ModelInfo]:
        key = key_or_model_id or DEFAULT_KEY
        if key in CHOICE_KEYS:
            choice = next(c for c in await self.choices() if c.key == key)
            if not choice.available:
                raise AuthError(choice.reason or "Model unavailable.", CHOICE_PROVIDER[key])
            return self.adapter_factory(choice.model.provider), choice.model
        provider = provider_for_model(key)
        listed = await self.models(provider) if self.store.get(provider) else None
        info = next((m for m in listed or [] if m.id == key), None) or ModelInfo(
            id=key, provider=provider, family="other", display_name=key
        )
        return self.adapter_factory(provider), info
