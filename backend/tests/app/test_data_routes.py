from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SAMPLE = REPO / "sample_data" / "out"


def _new(client):
    return client.post("/api/sessions").json()["session_id"]


def _sources(client, sid):
    return {s["name"]: s for s in client.get(f"/api/sessions/{sid}/schema").json()["sources"]}


def test_upload_csv_shows_in_schema(client):
    sid = _new(client)
    with open(SAMPLE / "customers.csv", "rb") as f:
        r = client.post(f"/api/sessions/{sid}/files", files={"file": ("customers.csv", f, "text/csv")})
    assert r.status_code == 200
    assert r.json()["tables"][0]["table"] == "customers"
    src = _sources(client, sid)["files"]
    assert src["kind"] == "file" and src["dialect"] == "duckdb"
    assert [t["name"] for t in src["tables"]] == ["customers"]


def test_upload_rejects_other_types(client):
    sid = _new(client)
    r = client.post(f"/api/sessions/{sid}/files", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_add_sqlite_connection_and_duplicate_and_delete(client):
    sid = _new(client)
    body = {"name": "shop", "config": {"engine": "sqlite", "database": str(SAMPLE / "retail.db")}}
    r = client.post(f"/api/sessions/{sid}/connections", json=body)
    assert r.status_code == 200, r.text
    assert {"orders", "customers"} <= {t["name"] for t in r.json()["tables"]}
    assert client.post(f"/api/sessions/{sid}/connections", json=body).status_code == 409
    assert "shop" in _sources(client, sid)
    assert client.delete(f"/api/sessions/{sid}/sources/shop").status_code == 204
    assert "shop" not in _sources(client, sid)


def test_bad_connection_returns_400_without_secret(client):
    sid = _new(client)
    body = {
        "name": "pg",
        "config": {"engine": "postgresql", "host": "127.0.0.1", "port": 1, "database": "x", "username": "u", "password": "hunter2"},
    }
    r = client.post(f"/api/sessions/{sid}/connections", json=body)
    assert r.status_code == 400 and "hunter2" not in r.text


def test_test_connection_endpoint(client):
    ok = client.post("/api/connections/test", json={"engine": "sqlite", "database": str(SAMPLE / "retail.db")})
    assert ok.json() == {"ok": True, "error": None}
    bad = client.post("/api/connections/test", json={"engine": "postgresql", "host": "127.0.0.1", "port": 1})
    assert bad.json()["ok"] is False and bad.json()["error"]


def test_invalid_source_name_rejected(client):
    sid = _new(client)
    body = {"name": "Bad Name!", "config": {"engine": "sqlite", "database": str(SAMPLE / "retail.db")}}
    assert client.post(f"/api/sessions/{sid}/connections", json=body).status_code == 422


def test_sample_retail_loads_files_and_sqlite(client):
    sid = _new(client)
    assert client.post(f"/api/sessions/{sid}/samples/retail").status_code == 200
    srcs = _sources(client, sid)
    assert {"files", "retail_sqlite"} <= set(srcs)
    assert {"orders", "customers"} <= {t["name"] for t in srcs["files"]["tables"]}
    assert client.post(f"/api/sessions/{sid}/samples/retail").status_code == 200  # idempotent
    assert client.post(f"/api/sessions/{sid}/samples/nope").status_code == 404


def test_unknown_session_404(client):
    assert client.get("/api/sessions/nope/schema").status_code == 404


def test_reap_idle_stops_kernels_but_keeps_session_data(client):
    manager = client.app.state.sessions
    sid = _new(client)
    session = manager.get(sid)
    stopped = []

    class FakeKernel:
        def close_sync(self):
            stopped.append(True)

    session.kernel = FakeKernel()
    session.last_active = datetime.now(UTC) - timedelta(hours=5)
    assert manager.reap_idle(datetime.now(UTC)) == 1
    assert stopped == [True] and session.kernel is None
    assert client.get(f"/api/sessions/{sid}/schema").status_code == 200  # data, history and board survive
