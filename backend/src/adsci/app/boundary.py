"""When analysis kernels are sandboxed (Docker), keep them away from the app's own control API.

The kernel runs inside the container and can open network connections, so it could call the app's HTTP API on
localhost (or the container's own IP) to replace or delete stored credentials. The browser reaches the app
through the published port, so its requests come from the Docker gateway instead. Requests from the
container's own addresses are therefore refused, except the kernel's token-protected data bridge
(``/internal/``) and the health check.
"""

from __future__ import annotations

import ipaddress
import socket

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

OPEN_PATHS = ("/internal/", "/api/health")
GUARDED_PREFIXES = ("/api/", "/auth/")


def local_addresses() -> set[str]:
    addrs = {"127.0.0.1", "::1"}
    try:
        addrs.update(socket.gethostbyname_ex(socket.gethostname())[2])
        addrs.update(info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None))
    except OSError:
        pass
    return addrs


def _is_local(host: str, own: set[str]) -> bool:
    if host in own:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class SandboxBoundaryMiddleware:
    def __init__(self, app: ASGIApp, own_addresses: set[str] | None = None) -> None:
        self.app = app
        self.own = own_addresses if own_addresses is not None else local_addresses()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            path = scope.get("path", "")
            client = scope.get("client")
            if (
                path.startswith(GUARDED_PREFIXES)
                and not path.startswith(OPEN_PATHS)
                and client is not None
                and _is_local(client[0], self.own)
            ):
                response = JSONResponse(
                    {"detail": "The app API isn't available from inside the container (analysis code runs here). "
                               "Open the app from your browser instead."},
                    status_code=403,
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
