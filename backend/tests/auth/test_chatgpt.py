from urllib.parse import parse_qs, urlparse

import pytest
from cryptography.fernet import Fernet

from adsci.auth.chatgpt import ChatGPTAuth, PlanUsageNotEnabled, SignInError
from adsci.auth.store import CredentialStore

from .mock_oauth import MockOAuth


@pytest.fixture
def server():
    return MockOAuth()


@pytest.fixture
def auth(tmp_path, settings, server):
    store = CredentialStore(tmp_path / "c.enc", Fernet.generate_key())
    return ChatGPTAuth(store, settings, http=server.client())


def _params(url):
    return {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}


async def test_start_builds_spec_compliant_url(auth):
    url = auth.start()
    assert url.startswith("https://auth.openai.com/api/accounts/authorize?")
    p = _params(url)
    assert p["client_id"] == "dynamic_agent_client" and p["agent_name_hint"] == "AI Data Scientist"
    assert p["ext_agent_host_id"].startswith("urn:uuid:") and p["code_challenge_method"] == "S256"
    assert set(p["scope"].split()) == {"openid", "profile", "email", "offline_access", "resource.invoke",
                                       "chatgpt.tokens.use.direct"}
    assert p["redirect_uri"] == "http://127.0.0.1:8000/auth/callback" and p["resource"] == "https://api.openai.com/v1"
    assert p["state"] and p["nonce"] and len(p["code_challenge"]) == 43
    assert _params(auth.start())["ext_agent_host_id"] == p["ext_agent_host_id"]  # stable host id


async def test_finish_stores_verified_credential(auth, server):
    p = _params(auth.start())
    server.nonce = p["nonce"]
    cred = await auth.finish(code="c1", state=p["state"], client_id="oaiapp_123")
    assert cred.kind == "chatgpt_plan" and cred.client_id == "oaiapp_123" and cred.account_label == "me@example.com"
    req = server.token_requests[-1]
    assert req["grant_type"] == "authorization_code" and req["client_id"] == "oaiapp_123"
    assert req["code_verifier"] and req["redirect_uri"] == p["redirect_uri"] and req["resource"] == p["resource"]
    assert auth.store.get("openai").kind == "chatgpt_plan"
    p2 = _params(auth.start())  # returning sign-in reuses the issued client id
    assert p2["client_id"] == "oaiapp_123" and "agent_name_hint" not in p2


async def test_wrong_state_rejected(auth):
    auth.start()
    with pytest.raises(SignInError, match="state"):
        await auth.finish(code="c", state="forged", client_id="oaiapp_1")


async def test_bad_nonce_rejected(auth, server):
    p = _params(auth.start())
    server.nonce = "other"
    with pytest.raises(SignInError, match="nonce"):
        await auth.finish(code="c", state=p["state"], client_id="oaiapp_1")
    assert auth.store.get("openai") is None


async def test_missing_plan_scope_not_saved(tmp_path, settings):
    server = MockOAuth(scope="openid profile email offline_access")
    auth = ChatGPTAuth(CredentialStore(tmp_path / "c.enc", Fernet.generate_key()), settings, http=server.client())
    p = _params(auth.start())
    server.nonce = p["nonce"]
    with pytest.raises(PlanUsageNotEnabled):
        await auth.finish(code="c", state=p["state"], client_id="oaiapp_1")
    assert auth.store.get("openai") is None


async def test_token_refreshes_when_near_expiry(auth, server):
    p = _params(auth.start())
    server.nonce = p["nonce"]
    server.access_ttl = 60  # expires within the 5-minute refresh window
    await auth.finish(code="c", state=p["state"], client_id="oaiapp_1")
    tok = await auth.token()
    assert tok == "at-2" and server.token_requests[-1]["grant_type"] == "refresh_token"
    assert server.token_requests[-1]["refresh_token"] == "rt-1" and server.token_requests[-1]["client_id"] == "oaiapp_1"
    server.access_ttl = 3600
    assert await auth.token() == "at-3"  # rotated refresh token is saved and used
    assert await auth.token() == "at-3"  # fresh now, no refresh


async def test_logout_revokes_and_clears(auth, server):
    p = _params(auth.start())
    server.nonce = p["nonce"]
    await auth.finish(code="c", state=p["state"], client_id="oaiapp_1")
    await auth.logout()
    assert server.revoked == ["rt-1"] and auth.store.get("openai") is None


async def test_failed_refresh_drops_subscription_so_api_key_is_used(auth, server):
    from adsci.auth.store import ApiKeyCredential

    p = _params(auth.start())
    server.nonce = p["nonce"]
    server.access_ttl = 60
    await auth.finish(code="c", state=p["state"], client_id="oaiapp_1")
    auth.store.set(ApiKeyCredential(provider="openai", api_key="sk-fallback-1234"))
    server.fail_refresh = True
    with pytest.raises(SignInError, match="expired"):
        await auth.token()
    assert auth.store.get("openai").kind == "api_key"
