from datetime import UTC, datetime

import pytest
from cryptography.fernet import Fernet

from adsci.auth.store import ApiKeyCredential, CredentialStore
from adsci.llm.base import AuthError, ProviderUnavailable
from adsci.llm.registry import FALLBACK_MODELS, ModelRegistry, latest
from adsci.llm.types import ModelInfo


def mi(id, provider, family, created=None):
    return ModelInfo(id=id, provider=provider, family=family, display_name=id,
                     created=datetime(*created, tzinfo=UTC) if created else None)


class FakeListAdapter:
    def __init__(self, provider, models=None, fail=False):
        self.provider, self.models, self.fail, self.calls = provider, models or [], fail, 0

    async def list_models(self):
        self.calls += 1
        if self.fail:
            raise ProviderUnavailable("down", self.provider)
        return self.models


def test_latest_rules():
    assert latest("opus", [mi("claude-opus-5", "anthropic", "opus", (2026, 5, 1)),
                           mi("claude-opus-5-5", "anthropic", "opus", (2026, 9, 1))]).id == "claude-opus-5-5"
    gpts = [mi("gpt-6.2-sol", "openai", "gpt", (2026, 10, 1)), mi("gpt-6.2-astra", "openai", "gpt", (2026, 9, 1)),
            mi("gpt-6.1-terra", "openai", "gpt", (2026, 11, 1)), mi("gpt-6-luna", "openai", "gpt", (2026, 6, 1))]
    assert latest("gpt", gpts).id == "gpt-6.2-astra"  # newest version, flagship tier
    groks = [mi("grok-4.20-0309-reasoning", "xai", "grok", (2026, 3, 9)), mi("grok-4.7", "xai", "grok", (2026, 9, 21))]
    assert latest("grok", groks).id == "grok-4.7"  # by release date, not by "4.20" > "4.7"
    assert latest("sonnet", gpts) is None


@pytest.fixture
def store(tmp_path):
    return CredentialStore(tmp_path / "c.enc", Fernet.generate_key())


async def test_choices_resolve_latest_and_mark_unavailable(store):
    store.set(ApiKeyCredential(provider="anthropic", api_key="k"))
    fake = FakeListAdapter("anthropic", [mi("claude-opus-5-5", "anthropic", "opus", (2026, 9, 1)),
                                         mi("claude-opus-6", "anthropic", "opus", (2026, 11, 1)),
                                         mi("claude-sonnet-5-5", "anthropic", "sonnet", (2026, 10, 1))])
    reg = ModelRegistry(store, lambda provider: fake)
    choices = {c.key: c for c in await reg.choices()}
    assert choices["opus"].model.id == "claude-opus-6" and choices["opus"].available
    assert choices["sonnet"].model.id == "claude-sonnet-5-5"
    assert not choices["gpt"].available and "OpenAI" in choices["gpt"].reason
    assert choices["gpt"].model.id == FALLBACK_MODELS["gpt"]
    await reg.choices()
    assert fake.calls == 1  # cached


async def test_listing_failure_falls_back(store):
    store.set(ApiKeyCredential(provider="anthropic", api_key="k"))
    reg = ModelRegistry(store, lambda provider: FakeListAdapter(provider, fail=True))
    opus = next(c for c in await reg.choices() if c.key == "opus")
    assert opus.available and opus.model.id == FALLBACK_MODELS["opus"] == "claude-opus-5-5"


async def test_resolve_key_and_raw_id(store):
    store.set(ApiKeyCredential(provider="xai", api_key="k"))
    fake = FakeListAdapter("xai", [mi("grok-4.7", "xai", "grok", (2026, 9, 21))])
    reg = ModelRegistry(store, lambda provider: fake)
    adapter, model = await reg.resolve("grok")
    assert adapter is fake and model.id == "grok-4.7"
    _, raw = await reg.resolve("grok-4.6")
    assert raw.id == "grok-4.6" and raw.provider == "xai"
    with pytest.raises(AuthError, match="Anthropic"):
        await reg.resolve("opus")


async def test_broken_subscription_does_not_break_other_models(store):
    from adsci.auth.chatgpt import SignInError

    store.set(ApiKeyCredential(provider="anthropic", api_key="k"))

    class Broken:
        provider = "openai"

        async def list_models(self):
            raise SignInError("Your ChatGPT sign-in expired. Please sign in again.")

    good = FakeListAdapter("anthropic", [mi("claude-opus-5-5", "anthropic", "opus", (2026, 9, 1))])
    from adsci.auth.store import OAuthCredential

    store.set(OAuthCredential(provider="openai", kind="chatgpt_plan", access_token="a", refresh_token="r",
                              expires_at=datetime(2020, 1, 1, tzinfo=UTC)))
    reg = ModelRegistry(store, lambda p: good if p == "anthropic" else Broken())
    choices = {c.key: c for c in await reg.choices()}
    assert choices["opus"].available and choices["gpt"].model.id == FALLBACK_MODELS["gpt"]
    adapter, model = await reg.resolve("opus")
    assert model.id == "claude-opus-5-5"


def test_sign_in_errors_are_provider_auth_errors():
    from adsci.auth.chatgpt import SignInError
    from adsci.auth.grok import GrokSignInUnavailable

    assert issubclass(SignInError, AuthError) and issubclass(GrokSignInUnavailable, AuthError)
    assert SignInError("x").user_action == "update_credentials"
