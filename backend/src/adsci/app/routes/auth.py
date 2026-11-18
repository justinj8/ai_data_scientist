"""Subscription sign-in: ChatGPT plan (supported) and SuperGrok (experimental)."""

from __future__ import annotations

import html

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from adsci.auth.chatgpt import SignInError
from adsci.auth.grok import GrokSignInUnavailable

router = APIRouter()


def _page(title: str, message: str, ok: bool) -> HTMLResponse:
    color = "#1f7a4d" if ok else "#b42318"
    body = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="font-family:system-ui,sans-serif;max-width:32rem;margin:15vh auto;padding:0 1rem;color:#1d1d1f">
<h1 style="color:{color};font-size:1.4rem">{html.escape(title)}</h1><p>{html.escape(message)}</p>
<p style="color:#666">You can close this tab and return to AI Data Scientist.</p></body></html>"""
    return HTMLResponse(body, status_code=200 if ok else 400)


@router.post("/api/auth/chatgpt/start")
def chatgpt_start(request: Request) -> dict:
    return {"authorize_url": request.app.state.chatgpt.start()}


@router.get("/auth/callback", response_class=HTMLResponse)
async def chatgpt_callback(
    request: Request, state: str = "", code: str | None = None, client_id: str | None = None, error: str | None = None
) -> HTMLResponse:
    auth = request.app.state.chatgpt
    if error:
        auth._pending.pop(state, None)
        msg = "Sign-in was cancelled." if error == "access_denied" else f"ChatGPT returned an error: {error}."
        return _page("Not connected", msg + " Return to the app to try again.", ok=False)
    if not code:
        return _page("Not connected", "The sign-in response had no code. Please try again.", ok=False)
    try:
        cred = await auth.finish(code=code, state=state, client_id=client_id)
    except SignInError as e:
        return _page("Not connected", f"{e} Please try again.", ok=False)
    request.app.state.models.invalidate("openai")
    return _page("Connected to ChatGPT", f"Signed in as {cred.account_label or 'your account'}. "
                 "Requests will use your ChatGPT plan.", ok=True)


@router.post("/api/auth/grok/start")
async def grok_start(request: Request) -> dict:
    try:
        dc = await request.app.state.grok.start()
    except GrokSignInUnavailable as e:
        raise HTTPException(status_code=409, detail=str(e)) from None
    return dc.model_dump()


@router.get("/api/auth/grok/status")
async def grok_status(request: Request) -> dict:
    status = await request.app.state.grok.poll()
    if status == "approved":
        request.app.state.models.invalidate("xai")
    return {"status": status}


@router.post("/api/auth/{provider}/logout")
async def logout(request: Request, provider: str) -> dict:
    if provider == "openai":
        confirmed = await request.app.state.chatgpt.logout()
        request.app.state.models.invalidate("openai")
        return {"ok": True, "revocation_confirmed": confirmed}
    if provider == "xai":
        await request.app.state.grok.logout()
        request.app.state.models.invalidate("xai")
        return {"ok": True}
    raise HTTPException(status_code=404, detail="Subscription sign-in exists only for openai and xai.")
