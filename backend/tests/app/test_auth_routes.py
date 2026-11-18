from urllib.parse import parse_qs, urlparse


def test_chatgpt_start_and_bad_callback(client):
    r = client.post("/api/auth/chatgpt/start")
    assert r.status_code == 200 and r.json()["authorize_url"].startswith("https://auth.openai.com/")
    bad = client.get("/auth/callback", params={"code": "x", "state": "forged", "client_id": "oaiapp_1"})
    assert bad.status_code == 400 and "text/html" in bad.headers["content-type"] and "try again" in bad.text.lower()


def test_callback_access_denied_page(client):
    url = client.post("/api/auth/chatgpt/start").json()["authorize_url"]
    state = parse_qs(urlparse(url).query)["state"][0]
    r = client.get("/auth/callback", params={"error": "access_denied", "state": state})
    assert r.status_code == 400 and "cancel" in r.text.lower()


def test_grok_start_unavailable_without_client_id(client):
    r = client.post("/api/auth/grok/start")
    assert r.status_code == 409 and "API key" in r.json()["detail"]
    assert client.get("/api/credentials").json()["providers"]["xai"]["subscription_available"] is False


def test_factory_has_token_providers(client):
    assert {"chatgpt_plan", "supergrok_plan"} <= set(client.app.state.adapters.token_providers)
