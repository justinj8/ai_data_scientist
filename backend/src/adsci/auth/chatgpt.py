"""Sign in with ChatGPT and use the ChatGPT plan (OpenAI's flow for open-source and locally run apps).

Spec: https://developers.openai.com/siwc/token-sharing-open-source/sign-in — OAuth 2.0 authorization code with
PKCE, dynamic client registration on first sign-in, a loopback redirect on 127.0.0.1 with path
``/auth/callback``, ID-token validation against OpenAI's JWKS, and the ``chatgpt.tokens.use.direct`` scope.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode, urlparse

import httpx2
import jwt
from pydantic import SecretStr

from adsci.auth.store import CredentialStore, OAuthCredential
from adsci.config import Settings
from adsci.llm.base import AuthError

AUTHORIZE_URL = "https://auth.openai.com/api/accounts/authorize"
TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
DISCOVERY_URL = "https://auth.openai.com/.well-known/openid-configuration"
RESOURCE = "https://api.openai.com/v1"
DYNAMIC_CLIENT = "dynamic_agent_client"
APP_NAME = "AI Data Scientist"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
CALLBACK_PATH = "/auth/callback"
REFRESH_MARGIN = timedelta(minutes=5)
PENDING_TTL_S = 600


class SignInError(AuthError):
    """Sign-in failed or expired; the message is safe to show the user."""

    def __init__(self, message: str, provider: str = "openai") -> None:
        super().__init__(message, provider)


class PlanUsageNotEnabled(SignInError):
    pass


@dataclass
class _Pending:
    verifier: str
    nonce: str
    redirect_uri: str
    client_id: str
    created: float


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


class ChatGPTAuth:
    def __init__(self, store: CredentialStore, settings: Settings, http: httpx2.AsyncClient | None = None) -> None:
        self.store = store
        self.settings = settings
        self.http = http or httpx2.AsyncClient(timeout=30)
        self._pending: dict[str, _Pending] = {}
        self._lock = asyncio.Lock()
        self._host_file = settings.data_dir / "chatgpt-host.json"

    # ------------------------------------------------------------- host registration (not secret)

    def _host(self) -> dict:
        if self._host_file.exists():
            return json.loads(self._host_file.read_text())
        data = {"ext_agent_host_id": f"urn:uuid:{uuid.uuid4()}", "client_id": None, "login_hint": None}
        self._save_host(data)
        return data

    def _save_host(self, data: dict) -> None:
        self._host_file.parent.mkdir(parents=True, exist_ok=True)
        self._host_file.write_text(json.dumps(data))

    def redirect_uri(self) -> str:
        port = urlparse(self.settings.public_base_url).port or 8000
        return f"http://127.0.0.1:{port}{CALLBACK_PATH}"  # spec: 127.0.0.1, never "localhost"

    # ------------------------------------------------------------- authorization

    def start(self) -> str:
        now = time.monotonic()
        self._pending = {s: p for s, p in self._pending.items() if now - p.created < PENDING_TTL_S}
        host = self._host()
        state, nonce = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
        verifier = _b64url(secrets.token_bytes(48))
        challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
        client_id = host.get("client_id") or DYNAMIC_CLIENT
        params = {"client_id": client_id}
        if client_id == DYNAMIC_CLIENT:
            params["agent_name_hint"] = APP_NAME
        elif host.get("login_hint"):
            params["login_hint"] = host["login_hint"]
        params.update(
            ext_agent_host_id=host["ext_agent_host_id"],
            response_type="code",
            redirect_uri=self.redirect_uri(),
            scope=SCOPES,
            resource=RESOURCE,
            state=state,
            nonce=nonce,
            code_challenge_method="S256",
            code_challenge=challenge,
        )
        self._pending[state] = _Pending(verifier, nonce, self.redirect_uri(), client_id, now)
        return f"{AUTHORIZE_URL}?{urlencode(params)}"

    async def finish(self, code: str, state: str, client_id: str | None) -> OAuthCredential:
        pending = self._pending.pop(state, None)
        if pending is None:
            raise SignInError("This sign-in attempt is unknown or expired (state mismatch). Please try again.")
        if pending.client_id == DYNAMIC_CLIENT:
            if not client_id or client_id == DYNAMIC_CLIENT:
                raise SignInError("ChatGPT did not return a registered client ID; registration is incomplete.")
            issued = client_id
        else:
            if client_id and client_id != pending.client_id:
                raise SignInError("ChatGPT returned a different client registration than expected.")
            issued = pending.client_id
        resp = await self.http.post(TOKEN_URL, data={
            "grant_type": "authorization_code", "client_id": issued, "code": code,
            "code_verifier": pending.verifier, "redirect_uri": pending.redirect_uri, "resource": RESOURCE,
        })
        if resp.status_code != 200:
            raise SignInError(f"Token exchange failed ({resp.status_code}). Please sign in again.")
        tokens = resp.json()
        claims = await self._verify_id_token(tokens.get("id_token"), issued, pending.nonce)

        host = self._host()
        host.update(client_id=issued, login_hint=claims.get("email"))
        self._save_host(host)

        scopes = (tokens.get("scope") or "").split()
        if PLAN_SCOPE not in scopes:
            raise PlanUsageNotEnabled(
                "You signed in, but ChatGPT plan usage wasn't allowed for this app. Sign in again and allow "
                "plan usage, or add an OpenAI API key instead."
            )
        cred = OAuthCredential(
            provider="openai", kind="chatgpt_plan",
            access_token=tokens["access_token"], refresh_token=tokens.get("refresh_token"),
            expires_at=datetime.now(UTC) + timedelta(seconds=int(tokens.get("expires_in", 3600))),
            client_id=issued, account_label=claims.get("email"), scopes=scopes,
            extra={"id_token": tokens.get("id_token"), "subject": claims.get("sub")},
        )
        self.store.set(cred)
        return cred

    async def _discovery(self) -> dict:
        r = await self.http.get(DISCOVERY_URL)
        r.raise_for_status()
        return r.json()

    async def _verify_id_token(self, id_token: str | None, client_id: str, nonce: str) -> dict:
        if not id_token:
            raise SignInError("ChatGPT did not return a verifiable identity. Please try signing in again.")
        disco = await self._discovery()
        jwks = (await self.http.get(disco["jwks_uri"])).json()
        kid = jwt.get_unverified_header(id_token).get("kid")
        jwk = next((k for k in jwks["keys"] if k.get("kid") == kid), None)
        if jwk is None:
            raise SignInError("Could not find the key that signed ChatGPT's identity token.")
        try:
            claims = jwt.decode(
                id_token, jwt.PyJWK(jwk).key, algorithms=["RS256"], audience=client_id, issuer=disco["issuer"]
            )
        except jwt.PyJWTError as e:
            raise SignInError(f"ChatGPT's identity token is invalid: {e}") from None
        if claims.get("nonce") != nonce:
            raise SignInError("ChatGPT's identity token has the wrong nonce; sign-in was rejected.")
        return claims

    # ------------------------------------------------------------- tokens

    async def token(self) -> str:
        async with self._lock:
            cred = self.store.get("openai", "chatgpt_plan")
            if not isinstance(cred, OAuthCredential):
                raise SignInError("Not signed in with ChatGPT.")
            if cred.expires_at - REFRESH_MARGIN > datetime.now(UTC):
                return cred.access_token.get_secret_value()
            if cred.refresh_token is None:
                raise SignInError("Your ChatGPT sign-in expired. Please sign in again.")
            r = await self.http.post(TOKEN_URL, data={
                "grant_type": "refresh_token", "client_id": cred.client_id,
                "refresh_token": cred.refresh_token.get_secret_value(), "resource": RESOURCE,
            })
            if r.status_code != 200:
                # The refresh token is dead: drop the subscription so a saved API key (if any) takes over.
                self.store.delete("openai", "chatgpt_plan")
                raise SignInError("Your ChatGPT sign-in expired. Please sign in again.")
            t = r.json()
            cred.access_token = SecretStr(t["access_token"])
            if t.get("refresh_token"):
                cred.refresh_token = SecretStr(t["refresh_token"])
            cred.expires_at = datetime.now(UTC) + timedelta(seconds=int(t.get("expires_in", 3600)))
            if t.get("scope"):
                cred.scopes = t["scope"].split()
            self.store.set(cred)
            return cred.access_token.get_secret_value()

    async def logout(self) -> bool:
        """Revoke the refresh token (best effort) and clear local credentials. Returns whether revocation was confirmed."""
        cred = self.store.get("openai", "chatgpt_plan")
        confirmed = False
        if isinstance(cred, OAuthCredential) and cred.refresh_token is not None:
            try:
                disco = await self._discovery()
                r = await self.http.post(disco["revocation_endpoint"], data={
                    "token": cred.refresh_token.get_secret_value(), "token_type_hint": "refresh_token",
                    "client_id": cred.client_id,
                })
                confirmed = r.status_code == 200
            except httpx2.HTTPError:
                confirmed = False
        self.store.delete("openai", "chatgpt_plan")
        return confirmed
