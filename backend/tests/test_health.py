def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_data_dir_is_private(settings, client):
    assert (settings.data_dir.stat().st_mode & 0o777) == 0o700
