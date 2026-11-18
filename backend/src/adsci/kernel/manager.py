"""One stateful Jupyter kernel per session, with structured output capture."""

from __future__ import annotations

import asyncio
import os
import re
import time
from pathlib import Path
from typing import Annotated, Literal

from jupyter_client.manager import AsyncKernelManager
from pydantic import BaseModel, Field

from adsci.kernel.bootstrap import KERNEL_STARTUP, PLOTLY_MIME, TABLE_MIME

MAX_STDOUT = 200_000
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
        return PlotlyOutput(figure=data[PLOTLY_MIME])
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
    ) -> None:
        self.session_id = session_id
        self.workdir = Path(workdir)
        self.internal_url = internal_url
        self.token = token
        self.preview_rows = preview_rows
        self._km: AsyncKernelManager | None = None
        self._kc = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._km = AsyncKernelManager(kernel_name="python3")
        env = dict(os.environ)
        env.update(
            ADSCI_INTERNAL_URL=self.internal_url,
            ADSCI_SESSION_ID=self.session_id,
            ADSCI_TOKEN=self.token,
            ADSCI_PREVIEW_ROWS=str(self.preview_rows),
        )
        # Provider API keys never reach user code.
        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY"):
            env.pop(key, None)
        await self._km.start_kernel(cwd=str(self.workdir), env=env)
        await self._connect()

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

    async def shutdown(self) -> None:
        if self._kc is not None:
            self._kc.stop_channels()
        if self._km is not None and await self._km.is_alive():
            await self._km.shutdown_kernel(now=True)
