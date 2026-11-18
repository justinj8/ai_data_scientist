import pytest
from fastapi.testclient import TestClient

from adsci.app.main import create_app
from adsci.config import Settings


@pytest.fixture
def sandboxed(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", secret_key_file=tmp_path / "k", kernel_users=["kernel0"],
                        allowed_hosts=["127.0.0.1", "localhost", "testserver"])
    return create_app(settings)


def test_in_container_callers_cannot_use_the_control_api(sandboxed):
    with TestClient(sandboxed, client=("127.0.0.1", 50000)) as local:
        r = local.put("/api/credentials/anthropic/api-key", json={"api_key": "attacker"})
        assert r.status_code == 403 and "inside the container" in r.json()["detail"]
        assert local.delete("/api/credentials/openai").status_code == 403
        assert local.post("/api/auth/openai/logout").status_code == 403
        assert local.get("/auth/callback", params={"state": "x", "code": "y"}).status_code == 403
        assert local.get("/api/health").status_code == 200  # container healthcheck
        # The kernel's data bridge is still reachable (and protected by its own token).
        assert local.post("/internal/sessions/nope/sql", json={"query": "select 1"}).status_code == 404


def test_browser_via_port_mapping_is_allowed(sandboxed):
    with TestClient(sandboxed, client=("172.21.0.1", 50000)) as browser:
        assert browser.get("/api/credentials").status_code == 200


def test_boundary_off_without_sandbox(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", secret_key_file=tmp_path / "k",
                        allowed_hosts=["127.0.0.1", "localhost", "testserver"])
    with TestClient(create_app(settings), client=("127.0.0.1", 50000)) as local:
        assert local.get("/api/credentials").status_code == 200  # native runs: the browser is local too


def test_forwarded_for_header_cannot_spoof_the_client_address(tmp_path):
    """Through the real server stack: a local caller claiming to be the browser via X-Forwarded-For is still local."""
    import socket
    import threading
    import time

    import httpx
    import uvicorn

    from adsci.app.main import server_config

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    settings = Settings(data_dir=tmp_path / "data", secret_key_file=tmp_path / "k", kernel_users=["kernel0"],
                        host="127.0.0.1", port=port)
    server = uvicorn.Server(server_config(create_app(settings), settings))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    while not server.started:
        time.sleep(0.05)
    try:
        r = httpx.put(f"http://127.0.0.1:{port}/api/credentials/anthropic/api-key", json={"api_key": "x"},
                      headers={"X-Forwarded-For": "172.21.0.1", "X-Real-IP": "172.21.0.1"})
        assert r.status_code == 403
    finally:
        server.should_exit = True
        t.join(timeout=10)
