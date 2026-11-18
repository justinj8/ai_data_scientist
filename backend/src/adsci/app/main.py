"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from fastapi import FastAPI

from adsci.app.routes import data
from adsci.app.sessions import SessionManager
from adsci.config import Settings, get_settings

REAP_INTERVAL_S = 300


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    sessions = SessionManager(settings)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async def reaper() -> None:
            while True:
                await asyncio.sleep(REAP_INTERVAL_S)
                await asyncio.to_thread(sessions.reap_idle, datetime.now(UTC))

        task = asyncio.create_task(reaper())
        try:
            yield
        finally:
            task.cancel()
            await asyncio.to_thread(sessions.close_all)

    app = FastAPI(title="AI Data Scientist", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.sessions = sessions

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(data.router)
    return app


def run() -> None:
    import uvicorn

    uvicorn.run("adsci.app.main:create_app", factory=True, host="127.0.0.1", port=8000)
