"""Credentials stored only on this machine, encrypted at rest with Fernet.

The encryption key lives in the OS keychain (via ``keyring``) when running natively, or in a private file
(``ADSCI_SECRET_KEY_FILE``, mode 0600) in Docker where there is no keychain.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from cryptography.fernet import Fernet
from pydantic import BaseModel, Field, SecretStr, TypeAdapter

from adsci.config import Settings

ProviderId = Literal["anthropic", "openai", "xai"]
PROVIDERS: tuple[str, ...] = ("anthropic", "openai", "xai")
ENV_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "xai": "XAI_API_KEY"}
KEYRING_SERVICE = "adsci"
KEYRING_USER = "credential-store-key"


class ApiKeyCredential(BaseModel):
    kind: Literal["api_key"] = "api_key"
    provider: ProviderId
    api_key: SecretStr


class OAuthCredential(BaseModel):
    kind: Literal["chatgpt_plan", "supergrok_plan"]
    provider: ProviderId
    access_token: SecretStr
    refresh_token: SecretStr | None = None
    expires_at: datetime
    client_id: str | None = None
    account_label: str | None = None
    scopes: list[str] = []
    extra: dict = {}

    @property
    def expired(self) -> bool:
        return self.expires_at <= datetime.now(UTC)


Credential = Annotated[ApiKeyCredential | OAuthCredential, Field(discriminator="kind")]
_CRED = TypeAdapter(Credential)


class ProviderStatus(BaseModel):
    configured: bool
    kind: str | None = None
    masked: str | None = None
    account_label: str | None = None
    has_api_key: bool = False
    has_subscription: bool = False
    from_env: bool = False


def mask(secret: str) -> str:
    head = secret.split("-")[0] if "-" in secret[:8] else secret[:2]
    return f"{head}-…{secret[-4:]}" if len(secret) > 8 else "…"


class CredentialStore:
    def __init__(self, path: Path, key: bytes) -> None:
        self.path = Path(path)
        self._fernet = Fernet(key)
        self._lock = threading.Lock()

    def _load(self) -> dict[str, dict[str, dict]]:
        if not self.path.exists():
            return {}
        return json.loads(self._fernet.decrypt(self.path.read_bytes()))

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(self._fernet.encrypt(json.dumps(data).encode()))
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)

    def _entries(self, provider: str) -> dict[str, ApiKeyCredential | OAuthCredential]:
        raw = self._load().get(provider, {})
        return {kind: _CRED.validate_python(v) for kind, v in raw.items()}

    def set(self, cred: ApiKeyCredential | OAuthCredential) -> None:
        with self._lock:
            data = self._load()
            dumped = cred.model_dump(mode="json")
            for k in list(dumped):  # SecretStr dumps as '**********' — store real values
                attr = getattr(cred, k)
                if isinstance(attr, SecretStr):
                    dumped[k] = attr.get_secret_value()
            data.setdefault(cred.provider, {})[cred.kind] = dumped
            self._save(data)

    def delete(self, provider: str, kind: str | None = None) -> None:
        with self._lock:
            data = self._load()
            if kind is None:
                data.pop(provider, None)
            else:
                data.get(provider, {}).pop(kind, None)
            self._save(data)

    def get(self, provider: str, kind: str | None = None) -> ApiKeyCredential | OAuthCredential | None:
        """The credential to use: a requested ``kind``, else a subscription if present, else the API key,
        else the provider's environment variable."""
        entries = self._entries(provider)
        if kind:
            return entries.get(kind)
        for c in entries.values():
            if isinstance(c, OAuthCredential):
                return c
        if "api_key" in entries:
            return entries["api_key"]
        env = os.environ.get(ENV_KEYS.get(provider, ""), "")
        return ApiKeyCredential(provider=provider, api_key=env) if env else None

    def status(self) -> dict[str, ProviderStatus]:
        out = {}
        for p in PROVIDERS:
            entries = self._entries(p)
            active = self.get(p)
            st = ProviderStatus(
                configured=active is not None,
                has_api_key="api_key" in entries,
                has_subscription=any(isinstance(c, OAuthCredential) for c in entries.values()),
                from_env=active is not None and "api_key" not in entries and not any(
                    isinstance(c, OAuthCredential) for c in entries.values()),
            )
            if active is not None:
                st.kind = active.kind
                if isinstance(active, ApiKeyCredential):
                    st.masked = mask(active.api_key.get_secret_value())
                else:
                    st.account_label = active.account_label
            out[p] = st
        return out


def load_store_key(settings: Settings) -> bytes:
    if settings.secret_key_file is not None:
        f = Path(settings.secret_key_file)
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(Fernet.generate_key())
        os.chmod(f, 0o600)
        return f.read_bytes().strip()
    import keyring

    key = keyring.get_password(KEYRING_SERVICE, KEYRING_USER)
    if not key:
        key = Fernet.generate_key().decode()
        keyring.set_password(KEYRING_SERVICE, KEYRING_USER, key)
    return key.encode()
