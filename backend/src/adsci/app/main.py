"""FastAPI application factory."""

from fastapi import FastAPI

from adsci.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="AI Data Scientist", version="0.1.0")
    app.state.settings = settings

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def run() -> None:
    import uvicorn

    uvicorn.run("adsci.app.main:create_app", factory=True, host="127.0.0.1", port=8000)
