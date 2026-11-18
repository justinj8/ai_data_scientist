import httpx

from adsci.kernel.manager import KernelSession


async def test_state_persists(kernel):
    await kernel.execute("x = 41", 10)
    r = await kernel.execute("print(x + 1)", 10)
    assert r.stdout.strip() == "42" and r.error is None


async def test_error_returned(kernel):
    r = await kernel.execute("1/0", 10)
    assert r.error is not None and r.error.ename == "ZeroDivisionError"
    assert "ZeroDivisionError" in r.error.traceback


async def test_timeout_interrupts(kernel):
    r = await kernel.execute("import time; time.sleep(30)", 2)
    assert r.error is not None and "timed out" in r.error.evalue.lower()
    assert (await kernel.execute("print(1)", 10)).stdout.strip() == "1"


async def test_kernel_death_restarts(kernel):
    r = await kernel.execute("import os; os._exit(1)", 10)
    assert r.error is not None and r.restarted
    after = await kernel.execute("print('alive'); print(type(pd).__name__)", 20)
    assert after.stdout.split() == ["alive", "module"]  # bootstrap re-ran after restart


async def test_plotly_and_table_capture(kernel):
    r = await kernel.execute(
        "import plotly.express as px\nshow(px.bar(x=[1, 2], y=[3, 4]))\npd.DataFrame({'a': [1, 2]})", 30
    )
    kinds = [o.type for o in r.outputs]
    assert "plotly" in kinds and "table" in kinds
    table = next(o for o in r.outputs if o.type == "table")
    assert table.columns == ["a"] and table.rows == [[1], [2]] and table.total_rows == 2


async def test_large_frame_preview_is_capped(kernel):
    r = await kernel.execute("pd.DataFrame({'a': range(1000)})", 20)
    table = next(o for o in r.outputs if o.type == "table")
    assert len(table.rows) == 50 and table.total_rows == 1000


async def test_stdout_is_capped(kernel):
    r = await kernel.execute("print('x' * 500_000)", 20)
    assert len(r.stdout) <= 200_100 and "truncated" in r.stdout


async def test_sql_helper_roundtrip(live_app, tmp_path):
    app, base = live_app
    async with httpx.AsyncClient(base_url=base) as c:
        sid = (await c.post("/api/sessions")).json()["session_id"]
        assert (await c.post(f"/api/sessions/{sid}/samples/retail")).status_code == 200
    session = app.state.sessions.get(sid)
    k = KernelSession(sid, tmp_path / "k", internal_url=base, token=session.internal_token)
    await k.start()
    try:
        r = await k.execute("df = sql('select * from orders limit 7')\nprint(len(df), list(df.columns)[:2])", 30)
        assert r.error is None, r.error
        assert r.stdout.strip() == "7 ['order_id', 'order_date']"
        r2 = await k.execute("sql('select count(*) as n from customers', source='retail_sqlite')", 30)
        assert r2.error is None and r2.outputs[0].rows == [[5000]]
        bad = await k.execute("sql('drop table orders')", 30)
        assert bad.error is not None and "read-only" in bad.error.evalue
    finally:
        await k.shutdown()


async def test_internal_sql_rejects_bad_token(live_app):
    app, base = live_app
    async with httpx.AsyncClient(base_url=base) as c:
        sid = (await c.post("/api/sessions")).json()["session_id"]
        r = await c.post(f"/internal/sessions/{sid}/sql", json={"query": "select 1", "source": "files"},
                         headers={"X-ADSCI-Token": "wrong"})
    assert r.status_code == 403


async def test_close_sync_kills_kernel(tmp_path):
    k = KernelSession("c", tmp_path, internal_url="http://127.0.0.1:9", token="t")
    await k.start()
    k.close_sync()
    import asyncio

    await asyncio.sleep(0.5)
    assert not await k._km.is_alive()


def test_kernel_env_is_an_allowlist(monkeypatch):
    from adsci.kernel.manager import kernel_env

    monkeypatch.setenv("ADSCI_SECRET_KEY_FILE", "/app/data/secret.key")
    monkeypatch.setenv("ADSCI_DATA_DIR", "/app/data")
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "t0k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    env = kernel_env({"ADSCI_TOKEN": "session-token", "ADSCI_SESSION_ID": "s"})
    assert env["ADSCI_TOKEN"] == "session-token" and "PATH" in env
    for leaked in ("ADSCI_SECRET_KEY_FILE", "ADSCI_DATA_DIR", "SOME_SERVICE_TOKEN", "ANTHROPIC_API_KEY"):
        assert leaked not in env


def test_sandbox_kernelspec_runs_as_other_user():
    import sys

    from adsci.kernel.manager import sandbox_kernelspec

    spec = sandbox_kernelspec("kernel")
    assert spec["argv"][:4] == ["sudo", "-n", "-u", "kernel"]
    assert spec["argv"][4:] == [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]


async def test_kernel_process_does_not_inherit_secret_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ADSCI_SECRET_KEY_FILE", str(tmp_path / "secret.key"))
    k = KernelSession("e", tmp_path, internal_url="http://127.0.0.1:9", token="t")
    await k.start()
    try:
        r = await k.execute("import os; print(os.environ.get('ADSCI_SECRET_KEY_FILE'))", 20)
        assert r.stdout.strip() == "None"
    finally:
        await k.shutdown()


async def test_uninterruptible_code_restarts_the_kernel(kernel):
    r = await kernel.execute("import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\ntime.sleep(60)", 2)
    assert r.error is not None and r.restarted
    assert (await kernel.execute("print('ok')", 20)).stdout.strip() == "ok"


async def test_cancelled_step_interrupts_the_kernel(kernel):
    import asyncio
    import time

    task = asyncio.create_task(kernel.execute("import time; time.sleep(60)", 120))
    await asyncio.sleep(1.5)
    task.cancel()
    t0 = time.monotonic()
    r = await kernel.execute("print('next')", 15)
    assert r.stdout.strip() == "next" and time.monotonic() - t0 < 10
