import pytest
from cryptography.fernet import Fernet

from adsci.auth.grok import GrokAuth, GrokSignInUnavailable
from adsci.auth.store import CredentialStore

from .mock_oauth import MockOAuth


def make(tmp_path, client_id="xai-client-1"):
    server = MockOAuth()
    store = CredentialStore(tmp_path / "c.enc", Fernet.generate_key())
    return GrokAuth(store, client_id=client_id, http=server.client(), discovery_url="https://auth.x.ai/.well-known/openid-configuration"), server


async def test_requires_an_xai_issued_client_id(tmp_path):
    auth, _ = make(tmp_path, client_id=None)
    assert auth.available is False
    with pytest.raises(GrokSignInUnavailable, match="API key"):
        await auth.start()


async def test_device_flow_pending_then_approved(tmp_path):
    auth, server = make(tmp_path)
    dc = await auth.start()
    assert dc.user_code == "ABCD-EFGH" and dc.verification_uri.startswith("https://accounts.x.ai")
    assert await auth.poll() == "pending"
    server.device_status = "ok"
    assert await auth.poll() == "approved"
    cred = auth.store.get("xai")
    assert cred.kind == "supergrok_plan" and await auth.token() == "at-1"


async def test_device_flow_denied(tmp_path):
    auth, server = make(tmp_path)
    await auth.start()
    server.device_status = "access_denied"
    assert await auth.poll() == "denied"
    assert auth.store.get("xai") is None
