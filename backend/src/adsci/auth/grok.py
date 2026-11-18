"""Experimental: use a SuperGrok subscription through xAI's OAuth device-code flow.

xAI does not publish a client registration for third-party apps, and this app will not borrow another
product's client ID. The flow is therefore off unless the user configures an OAuth client ID that xAI issued
to them (``ADSCI_XAI_OAUTH_CLIENT_ID``). xAI also gates API access by subscription tier; a 403 at inference
time is surfaced with advice to use an API key instead.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Literal

import httpx2
from pydantic import BaseModel, SecretStr

from adsci.auth.store import CredentialStore, OAuthCredential

DISCOVERY_URL = "https://auth.x.ai/.well-known/openid-configuration"
DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
SCOPES = "openid profile email offline_access api:access"
REFRESH_MARGIN = timedelta(minutes=5)


class GrokSignInUnavailable(Exception):
    pass


class DeviceCode(BaseModel):
    verification_uri: str
    verification_uri_complete: str | None = None
    user_code: str
    expires_in: int
    interval: int


class GrokAuth:
    def __init__(
        self,
        store: CredentialStore,
        client_id: str | None,
        http: httpx2.AsyncClient | None = None,
        discovery_url: str = DISCOVERY_URL,
    ) -> None:
        self.store = store
        self.client_id = client_id
        self.http = http or httpx2.AsyncClient(timeout=30)
        self.discovery_url = discovery_url
        self._device: dict | None = None
        self._lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        return bool(self.client_id)

    async def _discovery(self) -> dict:
        r = await self.http.get(self.discovery_url)
        r.raise_for_status()
        return r.json()

    async def start(self) -> DeviceCode:
        if not self.available:
            raise GrokSignInUnavailable(
                "SuperGrok sign-in needs an OAuth client ID issued by xAI (set ADSCI_XAI_OAUTH_CLIENT_ID). "
                "Until xAI offers one for third-party apps, use an xAI API key."
            )
        disco = await self._discovery()
        r = await self.http.post(disco["device_authorization_endpoint"], data={"client_id": self.client_id,
                                                                                 "scope": SCOPES})
        r.raise_for_status()
        body = r.json()
        self._device = {"device_code": body["device_code"], "token_endpoint": disco["token_endpoint"]}
        return DeviceCode(**{k: body[k] for k in ("verification_uri", "user_code", "expires_in", "interval")},
                          verification_uri_complete=body.get("verification_uri_complete"))

    async def poll(self) -> Literal["pending", "approved", "denied", "expired"]:
        if self._device is None:
            return "expired"
        r = await self.http.post(self._device["token_endpoint"], data={
            "grant_type": DEVICE_GRANT, "client_id": self.client_id, "device_code": self._device["device_code"],
        })
        if r.status_code == 200:
            t = r.json()
            self.store.set(OAuthCredential(
                provider="xai", kind="supergrok_plan", access_token=t["access_token"],
                refresh_token=t.get("refresh_token"), client_id=self.client_id,
                expires_at=datetime.now(UTC) + timedelta(seconds=int(t.get("expires_in", 3600))),
                scopes=(t.get("scope") or "").split(),
            ))
            self._device = None
            return "approved"
        error = r.json().get("error") if r.headers.get("content-type", "").startswith("application/json") else None
        if error in ("authorization_pending", "slow_down"):
            return "pending"
        self._device = None
        return "expired" if error == "expired_token" else "denied"

    async def token(self) -> str:
        async with self._lock:
            cred = self.store.get("xai", "supergrok_plan")
            if not isinstance(cred, OAuthCredential):
                raise GrokSignInUnavailable("Not signed in with SuperGrok.")
            if cred.expires_at - REFRESH_MARGIN > datetime.now(UTC):
                return cred.access_token.get_secret_value()
            disco = await self._discovery()
            r = await self.http.post(disco["token_endpoint"], data={
                "grant_type": "refresh_token", "client_id": cred.client_id,
                "refresh_token": cred.refresh_token.get_secret_value() if cred.refresh_token else "",
            })
            if r.status_code != 200:
                raise GrokSignInUnavailable("Your SuperGrok sign-in expired. Please sign in again.")
            t = r.json()
            cred.access_token = SecretStr(t["access_token"])
            if t.get("refresh_token"):
                cred.refresh_token = SecretStr(t["refresh_token"])
            cred.expires_at = datetime.now(UTC) + timedelta(seconds=int(t.get("expires_in", 3600)))
            self.store.set(cred)
            return cred.access_token.get_secret_value()

    async def logout(self) -> None:
        self.store.delete("xai", "supergrok_plan")
