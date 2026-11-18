import pytest

from adsci.kernel.pool import KernelUserPool, PoolExhausted


def test_each_kernel_gets_its_own_user():
    pool = KernelUserPool(["k0", "k1"])
    a, b = pool.acquire(), pool.acquire()
    assert {a, b} == {"k0", "k1"}
    with pytest.raises(PoolExhausted, match="2 analysis kernels"):
        pool.acquire()
    pool.release(a)
    assert pool.acquire() == a
    pool.release("unknown")  # no-op


def test_empty_pool_means_no_sandbox():
    pool = KernelUserPool([])
    assert not pool.enabled and pool.acquire() is None


def test_stopping_a_session_kernel_returns_its_user(settings):
    from adsci.app.sessions import SessionManager

    settings.kernel_users = ["k0"]
    mgr = SessionManager(settings)
    s = mgr.create()
    s.kernel_user = mgr.kernel_pool.acquire()

    class FakeKernel:
        def close_sync(self):
            pass

    s.kernel = FakeKernel()
    assert s.stop_kernel()
    assert s.kernel_user is None and mgr.kernel_pool.acquire() == "k0"
