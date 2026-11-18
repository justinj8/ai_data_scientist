import pytest

from adsci.kernel.manager import KernelSession


@pytest.fixture
async def kernel(tmp_path):
    k = KernelSession("test", tmp_path, internal_url="http://127.0.0.1:9", token="t")
    await k.start()
    yield k
    await k.shutdown()


