from datetime import UTC, datetime, timedelta

import pytest
from cryptography.fernet import Fernet, InvalidToken

from adsci.auth.store import ApiKeyCredential, CredentialStore, OAuthCredential, load_store_key


@pytest.fixture
def store(tmp_path):
    return CredentialStore(tmp_path / "creds.enc", Fernet.generate_key())


def test_roundtrip_and_encrypted_at_rest(store, tmp_path):
    store.set(ApiKeyCredential(provider="anthropic", api_key="sk-ant-secret-1234"))
    assert store.get("anthropic").api_key.get_secret_value() == "sk-ant-secret-1234"
    raw = (tmp_path / "creds.enc").read_bytes()
    assert b"sk-ant-secret" not in raw


def test_wrong_key_cannot_decrypt(store, tmp_path):
    store.set(ApiKeyCredential(provider="openai", api_key="sk-x"))
    with pytest.raises(InvalidToken):
        CredentialStore(tmp_path / "creds.enc", Fernet.generate_key()).get("openai")


def test_env_fallback(store, monkeypatch):
    assert store.get("xai") is None
    monkeypatch.setenv("XAI_API_KEY", "xai-from-env-9999")
    cred = store.get("xai")
    assert cred.api_key.get_secret_value() == "xai-from-env-9999"
    assert store.status()["xai"].from_env is True


def test_status_masks_and_prefers_subscription(store):
    store.set(ApiKeyCredential(provider="openai", api_key="sk-proj-abcdefgh1234"))
    st = store.status()["openai"]
    assert st.configured and st.kind == "api_key" and st.masked == "sk-…1234" and "abcdefgh" not in st.masked
    store.set(OAuthCredential(provider="openai", kind="chatgpt_plan", access_token="at", refresh_token="rt",
                              expires_at=datetime.now(UTC) + timedelta(hours=1), account_label="me@example.com"))
    assert store.get("openai").kind == "chatgpt_plan"
    st = store.status()["openai"]
    assert st.kind == "chatgpt_plan" and st.has_api_key and st.has_subscription and st.account_label == "me@example.com"
    store.delete("openai", "chatgpt_plan")
    assert store.get("openai").kind == "api_key"
    store.delete("openai")
    assert store.get("openai") is None and not store.status()["openai"].configured


def test_load_store_key_from_file_is_private_and_stable(tmp_path, settings):
    k1 = load_store_key(settings)
    k2 = load_store_key(settings)
    assert k1 == k2 and (settings.secret_key_file.stat().st_mode & 0o777) == 0o600
