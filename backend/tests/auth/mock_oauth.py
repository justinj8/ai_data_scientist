"""A tiny OAuth/OIDC server for tests, served through httpx2.MockTransport."""

import json
import time
from urllib.parse import parse_qs

import httpx2
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

ISSUER = "https://auth.openai.com"


class MockOAuth:
    def __init__(self, scope="openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"):
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(self.key.public_key()))
        jwk.update(kid="k1", alg="RS256", use="sig")
        self.jwks = {"keys": [jwk]}
        self.scope = scope
        self.nonce = None
        self.token_requests: list[dict] = []
        self.revoked: list[str] = []
        self.issued = 0
        self.access_ttl = 3600
        self.device_status = "authorization_pending"

    def id_token(self, aud):
        now = int(time.time())
        return jwt.encode({"iss": ISSUER, "aud": aud, "sub": "user-1", "email": "me@example.com", "iat": now,
                           "exp": now + 3600, "nonce": self.nonce}, self.key, algorithm="RS256", headers={"kid": "k1"})

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        if path.endswith("/.well-known/openid-configuration"):
            base = f"{request.url.scheme}://{request.url.host}"
            return httpx2.Response(200, json={
                "issuer": ISSUER, "authorization_endpoint": f"{base}/api/accounts/authorize",
                "token_endpoint": f"{base}/api/accounts/oauth/token", "jwks_uri": f"{base}/jwks",
                "revocation_endpoint": f"{base}/oauth/revoke",
                "device_authorization_endpoint": f"{base}/oauth2/device/code",
            })
        if path == "/jwks":
            return httpx2.Response(200, json=self.jwks)
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        if path == "/oauth/revoke":
            self.revoked.append(form.get("token"))
            return httpx2.Response(200)
        if path == "/oauth2/device/code":
            return httpx2.Response(200, json={"device_code": "dev-1", "user_code": "ABCD-EFGH",
                                              "verification_uri": "https://accounts.x.ai/device",
                                              "verification_uri_complete": "https://accounts.x.ai/device?code=ABCD-EFGH",
                                              "expires_in": 600, "interval": 1})
        if path.endswith("/token"):
            self.token_requests.append(form)
            if form.get("grant_type") == "urn:ietf:params:oauth:grant-type:device_code":
                if self.device_status != "ok":
                    return httpx2.Response(400, json={"error": self.device_status})
            self.issued += 1
            body = {"access_token": f"at-{self.issued}", "refresh_token": f"rt-{self.issued}",
                    "token_type": "Bearer", "expires_in": self.access_ttl, "scope": self.scope}
            if form.get("grant_type") == "authorization_code":
                body["id_token"] = self.id_token(form["client_id"])
            return httpx2.Response(200, json=body)
        return httpx2.Response(404, json={"error": "not_found"})

    def client(self):
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self))
