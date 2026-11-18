import socket
import threading
import time

import pytest
import uvicorn

from adsci.app.main import create_app
from adsci.config import Settings
from adsci.kernel.manager import KernelSession


@pytest.fixture
async def kernel(tmp_path):
    k = KernelSession("test", tmp_path, internal_url="http://127.0.0.1:9", token="t")
    await k.start()
    yield k
    await k.shutdown()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_app(tmp_path):
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    app = create_app(Settings(data_dir=tmp_path / "data", internal_base_url=base))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield app, base
    server.should_exit = True
    t.join(timeout=10)
