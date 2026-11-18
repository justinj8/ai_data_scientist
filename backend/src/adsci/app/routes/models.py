"""Model choices and provider credentials."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from adsci.auth.store import PROVIDERS, ApiKeyCredential
from adsci.llm.base import ProviderError
from adsci.llm.registry import DEFAULT_KEY

router = APIRouter(prefix="/api")

SUBSCRIPTIONS = {
    "anthropic": {
        "subscription_supported": False,
        "subscription_experimental": False,
        "subscription_note": (
            "Anthropic's terms don't allow third-party apps to use Claude Pro or Max subscriptions. "
            "Use an API key from console.anthropic.com."
        ),
    },
    "openai": {
        "subscription_supported": True,
        "subscription_experimental": False,
        "subscription_note": "Sign in with ChatGPT to use your Plus or Pro plan in this local, open-source app.",
    },
    "xai": {
        "subscription_supported": True,
        "subscription_experimental": True,
        "subscription_note": (
            "Experimental: xAI only allows some SuperGrok tiers to use the API this way. "
            "If sign-in works but requests fail, add an xAI API key instead."
        ),
    },
}


class ApiKeyBody(BaseModel):
    api_key: str


def _check(provider: str) -> None:
    if provider not in PROVIDERS:
        raise HTTPException(status_code=404, detail=f"Unknown provider '{provider}'. Use one of: {', '.join(PROVIDERS)}.")


@router.get("/models")
async def list_choices(request: Request) -> dict:
    choices = await request.app.state.models.choices()
    return {"default": DEFAULT_KEY, "choices": [c.model_dump(mode="json") for c in choices]}


@router.get("/models/all")
async def list_all(request: Request) -> dict:
    return {"models": [m.model_dump(mode="json") for m in await request.app.state.models.all_models()]}


@router.get("/credentials")
def credentials(request: Request) -> dict:
    status = request.app.state.credentials.status()
    return {"providers": {p: {**status[p].model_dump(), **SUBSCRIPTIONS[p]} for p in PROVIDERS}}


@router.put("/credentials/{provider}/api-key")
async def set_api_key(request: Request, provider: str, body: ApiKeyBody) -> dict:
    _check(provider)
    key = body.api_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="The API key is empty.")
    try:
        await request.app.state.adapters.for_api_key(provider, key).list_models()
    except ProviderError as e:
        raise HTTPException(status_code=400, detail=e.message) from None
    request.app.state.credentials.set(ApiKeyCredential(provider=provider, api_key=key))
    request.app.state.models.invalidate(provider)
    return {"ok": True}


@router.delete("/credentials/{provider}", status_code=204)
def delete_credentials(request: Request, provider: str, kind: str | None = None) -> Response:
    _check(provider)
    request.app.state.credentials.delete(provider, kind)
    request.app.state.models.invalidate(provider)
    return Response(status_code=204)
