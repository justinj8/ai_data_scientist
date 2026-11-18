from adsci.llm.base import AuthError
from adsci.llm.types import ModelInfo


class FakeAdapter:
    def __init__(self, ok):
        self.ok = ok

    async def list_models(self):
        if not self.ok:
            raise AuthError("Anthropic rejected the API key: invalid", "anthropic")
        return [ModelInfo(id="claude-opus-5-5", provider="anthropic", family="opus", display_name="Opus")]


def test_put_key_validates_before_saving(client):
    factory = client.app.state.adapters
    factory.for_api_key = lambda provider, key: FakeAdapter(ok=key == "good-key-1234")
    bad = client.put("/api/credentials/anthropic/api-key", json={"api_key": "bad"})
    assert bad.status_code == 400 and "rejected" in bad.json()["detail"]
    assert client.get("/api/credentials").json()["providers"]["anthropic"]["configured"] is False
    ok = client.put("/api/credentials/anthropic/api-key", json={"api_key": "good-key-1234"})
    assert ok.status_code == 200
    st = client.get("/api/credentials").json()["providers"]["anthropic"]
    assert st["configured"] and st["masked"].endswith("1234") and "good-key" not in str(st)
    assert client.delete("/api/credentials/anthropic").status_code == 204


def test_credentials_explain_claude_subscription(client):
    p = client.get("/api/credentials").json()["providers"]
    assert p["anthropic"]["subscription_supported"] is False and "terms" in p["anthropic"]["subscription_note"]
    assert p["openai"]["subscription_supported"] is True
    assert p["xai"]["subscription_supported"] is True and p["xai"]["subscription_experimental"] is True


def test_unknown_provider_404(client):
    assert client.put("/api/credentials/gemini/api-key", json={"api_key": "x"}).status_code == 404


def test_models_endpoint_lists_four_choices_with_default(client):
    body = client.get("/api/models").json()
    assert body["default"] == "opus"
    assert [c["key"] for c in body["choices"]] == ["opus", "sonnet", "gpt", "grok"]
    assert all(c["available"] is False for c in body["choices"])  # nothing configured in tests
