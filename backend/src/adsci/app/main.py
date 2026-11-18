"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from adsci.agent.orchestrator import Orchestrator
from adsci.agent.verifier import Verifier
from adsci.app.routes import auth, chat, data, insights, internal, models
from adsci.app.sessions import SessionManager
from adsci.auth.chatgpt import ChatGPTAuth
from adsci.auth.grok import GrokAuth
from adsci.auth.store import CredentialStore, load_store_key
from adsci.config import Settings, get_settings
from adsci.llm.registry import AdapterFactory, ModelRegistry
from adsci.tools.base import default_registry

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
    store = CredentialStore(settings.data_dir / "credentials.enc", load_store_key(settings))
    app.state.credentials = store
    app.state.adapters = AdapterFactory(store)
    app.state.models = ModelRegistry(store, lambda provider: app.state.adapters(provider))
    app.state.orchestrator = Orchestrator(default_registry(), Verifier(settings), settings)
    app.state.chatgpt = ChatGPTAuth(store, settings)
    app.state.grok = GrokAuth(store, client_id=settings.xai_oauth_client_id)
    app.state.adapters.token_providers["chatgpt_plan"] = lambda cred: app.state.chatgpt.token
    app.state.adapters.token_providers["supergrok_plan"] = lambda cred: app.state.grok.token

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(data.router)
    app.include_router(internal.router)
    app.include_router(models.router)
    app.include_router(auth.router)
    app.include_router(chat.router)
    app.include_router(insights.router)
    if settings.static_dir is not None and (settings.static_dir / "index.html").exists():
        _mount_frontend(app, settings.static_dir)
    return app


def _mount_frontend(app: FastAPI, dist) -> None:
    """Serve the built single-page app; unknown non-API paths fall back to index.html."""
    root = dist.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path.startswith(("api/", "internal/")):
            raise HTTPException(status_code=404)
        target = (root / path).resolve()
        if path and target.is_relative_to(root) and target.is_file():
            return FileResponse(target)
        return FileResponse(root / "index.html")


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run("adsci.app.main:create_app", factory=True, host=settings.host, port=settings.port)
