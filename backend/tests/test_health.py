def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_data_dir_is_private(settings, client):
    assert (settings.data_dir.stat().st_mode & 0o777) == 0o700


def test_rejects_foreign_host_header(client):
    # DNS rebinding: a web page re-points its own hostname at 127.0.0.1; the Host header still names it.
    assert client.get("/api/health", headers={"host": "attacker.example"}).status_code == 400
    assert client.get("/api/health", headers={"host": "127.0.0.1:8000"}).status_code == 200
    assert client.get("/api/health", headers={"host": "localhost:5173"}).status_code == 200
