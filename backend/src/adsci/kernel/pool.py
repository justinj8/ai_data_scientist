"""A pool of OS users for sandboxed kernels: every running kernel gets its own user, so one session's
analysis code can't read another session's connection file, environment or memory."""

from __future__ import annotations

import logging
import subprocess
import threading
from collections.abc import Callable

log = logging.getLogger("adsci.sandbox")


def kill_user_processes(user: str) -> None:
    """Kill everything a sandbox user left running (e.g. a background process waiting for the next session).
    Runs as that user via the narrow sudo rule, so it can only signal that user's own processes."""
    try:
        r = subprocess.run(["sudo", "-n", "-u", user, "/usr/bin/pkill", "-KILL", "-U", user],
                           timeout=10, check=False, capture_output=True)
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning("Sandbox: could not stop leftover processes of %s: %s", user, e)
        return
    # pkill exits 1 when nothing matched; anything else (or an error message) means the cleanup didn't run.
    stderr = r.stderr.decode(errors="replace").strip() if isinstance(r.stderr, bytes) else str(r.stderr or "")
    if r.returncode not in (0, 1) or stderr:
        log.warning("Sandbox: could not stop leftover processes of %s (exit %s): %s", user, r.returncode, stderr)


class PoolExhausted(RuntimeError):
    pass


class KernelUserPool:
    def __init__(self, users: list[str], reaper: Callable[[str], None] | None = kill_user_processes) -> None:
        self.users = list(users)
        self.reaper = reaper
        self._free = list(users)
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.users)

    def acquire(self) -> str | None:
        if not self.enabled:
            return None
        with self._lock:
            if not self._free:
                raise PoolExhausted(
                    f"All {len(self.users)} analysis kernels are in use. Close an idle tab or wait for idle "
                    "kernels to stop, then try again."
                )
            return self._free.pop(0)

    def release(self, user: str | None) -> None:
        if user not in self.users:
            return
        if self.reaper is not None:
            self.reaper(user)  # nothing of the previous session may survive into the next one
        with self._lock:
            if user not in self._free:
                self._free.append(user)
