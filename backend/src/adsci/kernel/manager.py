"""One stateful Jupyter kernel per session, with structured output capture."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Annotated, Literal

from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client.manager import AsyncKernelManager
from pydantic import BaseModel, Field

from adsci.kernel.bootstrap import KERNEL_STARTUP, PLOTLY_MIME, TABLE_MIME
from adsci.kernel.plotly_json import figure_to_plain

MAX_STDOUT = 200_000
# Only these variables reach analysis code. Everything else (secret-key paths, tokens, provider keys) stays out.
ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR", "HOME", "USER", "SYSTEMROOT")
SANDBOX_KERNEL = "adsci-sandbox"


def kernel_env(extra: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k in ENV_ALLOWLIST}
    if os.environ.get("ADSCI_KERNEL_MEMORY_BYTES"):
        env["ADSCI_KERNEL_MEMORY_BYTES"] = os.environ["ADSCI_KERNEL_MEMORY_BYTES"]
    env.update(extra)
    return env


def sandbox_kernelspec(user: str) -> dict:
    """A kernelspec that starts the kernel as another OS user, so it can't read the app's files (credentials)."""
    return {
        "argv": ["sudo", "-n", "-u", user, sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "AI Data Scientist (sandboxed)",
        "language": "python",
        "interrupt_mode": "signal",
    }


class _SharedConnectionFileKM(AsyncKernelManager):
    """Writes the connection file group-readable so a kernel running as another user (same group) can read it."""

    def write_connection_file(self, **kwargs):
        result = super().write_connection_file(**kwargs)
        os.chmod(self.connection_file, 0o640)
        return result
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class TextOutput(BaseModel):
    type: Literal["text"] = "text"
    text: str


class TableOutput(BaseModel):
    type: Literal["table"] = "table"
    columns: list[str]
    rows: list[list]
    total_rows: int


class PlotlyOutput(BaseModel):
    type: Literal["plotly"] = "plotly"
    figure: dict


class ImageOutput(BaseModel):
    type: Literal["image"] = "image"
    png_b64: str


Output = Annotated[TextOutput | TableOutput | PlotlyOutput | ImageOutput, Field(discriminator="type")]


class ExecError(BaseModel):
    ename: str
    evalue: str
    traceback: str


class ExecutionResult(BaseModel):
    stdout: str = ""
    outputs: list[Output] = []
    error: ExecError | None = None
    duration_s: float = 0.0
    restarted: bool = False


def _convert(data: dict) -> Output | None:
    if PLOTLY_MIME in data:
        return PlotlyOutput(figure=figure_to_plain(data[PLOTLY_MIME]))
    if TABLE_MIME in data:
        return TableOutput(**data[TABLE_MIME])
    if "image/png" in data:
        return ImageOutput(png_b64=data["image/png"])
    if "text/plain" in data:
        return TextOutput(text=data["text/plain"])
    return None


class KernelSession:
    def __init__(
        self,
        session_id: str,
        workdir: Path,
        internal_url: str,
        token: str,
        preview_rows: int = 50,
        run_as: str | None = None,
        group: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.workdir = Path(workdir)
        self.internal_url = internal_url
        self.token = token
        self.preview_rows = preview_rows
        self.run_as = run_as
        self.group = group
        self._km: AsyncKernelManager | None = None
        self._kc = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        env = kernel_env({
            "ADSCI_INTERNAL_URL": self.internal_url,
            "ADSCI_SESSION_ID": self.session_id,
            "ADSCI_TOKEN": self.token,
            "ADSCI_PREVIEW_ROWS": str(self.preview_rows),
        })
        if self.run_as:
            self._prepare_shared_workdir()
            spec_root = Path(tempfile.mkdtemp(prefix="adsci-kernelspec-"))
            (spec_root / SANDBOX_KERNEL).mkdir()
            (spec_root / SANDBOX_KERNEL / "kernel.json").write_text(json.dumps(sandbox_kernelspec(self.run_as)))
            self._km = _SharedConnectionFileKM(
                kernel_name=SANDBOX_KERNEL,
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(spec_root)]),
                connection_file=str(self.workdir / f"kernel-{self.session_id}.json"),
            )
        else:
            self._km = AsyncKernelManager(kernel_name="python3")
        await self._km.start_kernel(cwd=str(self.workdir), env=env)
        await self._connect()

    def _prepare_shared_workdir(self) -> None:
        """Let the sandbox user (a member of ``group``) use the work directory, and nothing else of ours."""
        if self.group:
            import grp

            os.chown(self.workdir, -1, grp.getgrnam(self.group).gr_gid)
        os.chmod(self.workdir, 0o2770)

    async def _connect(self) -> None:
        assert self._km is not None
        self._kc = self._km.client()
        self._kc.start_channels()
        await self._kc.wait_for_ready(timeout=60)
        result = await self._run(KERNEL_STARTUP, timeout_s=120)
        if result.error:
            raise RuntimeError(f"Kernel bootstrap failed: {result.error.evalue}")

    async def execute(self, code: str, timeout_s: int) -> ExecutionResult:
        async with self._lock:
            return await self._run(code, timeout_s)

    async def _run(self, code: str, timeout_s: int) -> ExecutionResult:
        assert self._km is not None and self._kc is not None
        started = time.monotonic()
        msg_id = self._kc.execute(code, store_history=True, allow_stdin=False)
        result = ExecutionResult()
        stdout: list[str] = []
        deadline = started + timeout_s
        while True:
            if time.monotonic() > deadline:
                await self._km.interrupt_kernel()
                await self._drain(msg_id, 5)
                result.error = ExecError(
                    ename="TimeoutError",
                    evalue=f"Execution timed out after {timeout_s}s and was interrupted.",
                    traceback="",
                )
                break
            try:
                msg = await self._kc.get_iopub_msg(timeout=0.5)
            except Exception:  # noqa: BLE001 - queue.Empty
                if not await self._km.is_alive():
                    await self._restart()
                    result.restarted = True
                    result.error = ExecError(
                        ename="KernelDied",
                        evalue="The Python kernel crashed (possibly out of memory) and was restarted. "
                        "Variables from earlier steps were lost; reload any data you need.",
                        traceback="",
                    )
                    break
                continue
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            mtype, content = msg["msg_type"], msg["content"]
            if mtype == "stream":
                stdout.append(content.get("text", ""))
            elif mtype in ("execute_result", "display_data"):
                out = _convert(content.get("data", {}))
                if out is not None:
                    result.outputs.append(out)
            elif mtype == "error":
                result.error = ExecError(
                    ename=content.get("ename", "Error"),
                    evalue=content.get("evalue", ""),
                    traceback=_ANSI.sub("", "\n".join(content.get("traceback", []))),
                )
            elif mtype == "status" and content.get("execution_state") == "idle":
                break
        text = "".join(stdout)
        if len(text) > MAX_STDOUT:
            text = text[:MAX_STDOUT] + f"\n... [output truncated, {len(text) - MAX_STDOUT} more characters]"
        result.stdout = text
        result.duration_s = round(time.monotonic() - started, 3)
        return result

    async def _drain(self, msg_id: str, seconds: float) -> None:
        assert self._kc is not None
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            try:
                msg = await self._kc.get_iopub_msg(timeout=0.5)
            except Exception:  # noqa: BLE001
                continue
            if (
                msg.get("parent_header", {}).get("msg_id") == msg_id
                and msg["msg_type"] == "status"
                and msg["content"].get("execution_state") == "idle"
            ):
                return

    async def _restart(self) -> None:
        assert self._km is not None
        if self._kc is not None:
            self._kc.stop_channels()
        await self._km.restart_kernel(now=True)
        await self._connect()

    async def restart(self) -> None:
        async with self._lock:
            await self._restart()

    def close_sync(self) -> None:
        """Kill the kernel process without an event loop (used when a session is closed)."""
        if self._kc is not None:
            self._kc.stop_channels()
        proc = getattr(getattr(self._km, "provisioner", None), "process", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()  # sudo relays SIGTERM to a sandboxed kernel; SIGKILL would orphan it
            try:
                proc.wait(timeout=3)
            except Exception:  # noqa: BLE001
                proc.kill()
        if self._km is not None:
            self._km.cleanup_resources()

    async def shutdown(self) -> None:
        if self._kc is not None:
            self._kc.stop_channels()
        if self._km is not None and await self._km.is_alive():
            await self._km.shutdown_kernel(now=True)
