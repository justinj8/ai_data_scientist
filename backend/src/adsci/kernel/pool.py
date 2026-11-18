"""A pool of OS users for sandboxed kernels: every running kernel gets its own user, so one session's
analysis code can't read another session's connection file, environment or memory."""

from __future__ import annotations

import threading


class PoolExhausted(RuntimeError):
    pass


class KernelUserPool:
    def __init__(self, users: list[str]) -> None:
        self.users = list(users)
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
        with self._lock:
            if user in self.users and user not in self._free:
                self._free.append(user)
