import pytest

from adsci.kernel.pool import KernelUserPool, PoolExhausted


def test_each_kernel_gets_its_own_user():
    pool = KernelUserPool(["k0", "k1"], reaper=None)
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
    mgr.kernel_pool.reaper = None
    s = mgr.create()
    s.kernel_user = mgr.kernel_pool.acquire()

    class FakeKernel:
        def close_sync(self):
            pass

    s.kernel = FakeKernel()
    assert s.stop_kernel()
    assert s.kernel_user is None and mgr.kernel_pool.acquire() == "k0"


def test_release_kills_the_users_leftover_processes():
    reaped = []
    pool = KernelUserPool(["k0"], reaper=reaped.append)
    u = pool.acquire()
    pool.release(u)
    assert reaped == ["k0"]
    pool.release("unknown")
    assert reaped == ["k0"]


def test_sandboxed_kernel_env_uses_a_fresh_home(tmp_path):
    from adsci.kernel.manager import KernelSession

    k = KernelSession("s1", tmp_path / "work", internal_url="http://x", token="t", run_as="kernel0", group="kernel0")
    env = k._env()
    assert env["HOME"] == str(tmp_path / "work")
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["IPYTHONDIR"].startswith(str(tmp_path / "work"))
    assert env["MPLCONFIGDIR"].startswith(str(tmp_path / "work"))
    plain = KernelSession("s2", tmp_path / "w2", internal_url="http://x", token="t")._env()
    assert plain.get("HOME") != str(tmp_path / "w2")


def test_failed_cleanup_is_logged_not_silent(monkeypatch, caplog):
    import subprocess

    from adsci.kernel import pool as pool_mod

    def fake_run(*a, **k):
        return subprocess.CompletedProcess(a[0], 1, b"", b"sudo: /usr/bin/pkill: command not found")

    monkeypatch.setattr(pool_mod.subprocess, "run", fake_run)
    with caplog.at_level("WARNING"):
        pool_mod.kill_user_processes("kernel0")
    assert "could not stop" in caplog.text and "command not found" in caplog.text
