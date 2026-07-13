from starlette.testclient import TestClient

from ypotheto_compchem_mcp.artifacts import sign_artifact_url
from ypotheto_compchem_mcp.config import settings
from ypotheto_compchem_mcp.http_app import app
from ypotheto_compchem_mcp.workspace import get_workspace_id_from_token


def test_healthz():
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "version" in response.json()

def test_auth_middleware():
    # Set api_token in settings for testing
    original_token = settings.api_token
    settings.api_token = "test_secret_token"
    client = TestClient(app)
    
    try:
        # Request without token should fail with 401
        response = client.get("/mcp")
        assert response.status_code == 401
        
        # Request with invalid token should fail with 401
        response = client.get("/mcp", headers={"Authorization": "Bearer bad_token"})
        assert response.status_code == 401
        
        # Request with valid token in header should pass the auth middleware
        # (It will then hit the mounted FastMCP app; GET /mcp might return 404/405, but NOT 401)
        response = client.get("/mcp", headers={"Authorization": "Bearer test_secret_token"})
        assert response.status_code != 401

        # Query-param token auth (?t=) was removed once signed artifact URLs
        # landed - the raw shared secret must no longer be accepted this way.
        response = client.get("/mcp?t=test_secret_token")
        assert response.status_code == 401
    finally:
        # Restore original token
        settings.api_token = original_token

def test_serve_artifact_blocks_cross_workspace_access():
    from ypotheto_compchem_mcp.storage import storage

    original_token = settings.api_token
    settings.api_token = ""  # unauthenticated mode: caller's Bearer value only selects their workspace
    client = TestClient(app)

    try:
        token_a = "workspace-a-secret"
        token_b = "workspace-b-secret"
        workspace_a = get_workspace_id_from_token(token_a)
        workspace_b = get_workspace_id_from_token(token_b)
        assert workspace_a != workspace_b

        storage.write_file(workspace_a, "artifacts/abc123/result.txt", b"secret data")

        # Workspace B's caller must not be able to read workspace A's artifact by
        # substituting workspace A's id into the URL path.
        response = client.get(
            f"/artifacts/{workspace_a}/abc123/result.txt",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert response.status_code == 404

        # The owning workspace's caller can still read its own artifact.
        response = client.get(
            f"/artifacts/{workspace_a}/abc123/result.txt",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert response.status_code == 200
        assert response.content == b"secret data"
    finally:
        settings.api_token = original_token

def test_signed_artifact_url_grants_access_without_bearer_token():
    from ypotheto_compchem_mcp.storage import storage

    original_token = settings.api_token
    settings.api_token = "shared_secret"
    client = TestClient(app)

    try:
        workspace_id = get_workspace_id_from_token("shared_secret")
        storage.write_file(workspace_id, "artifacts/xyz789/plot.png", b"plot bytes")

        query = sign_artifact_url(workspace_id, "xyz789", "plot.png")
        url = f"/artifacts/{workspace_id}/xyz789/plot.png?{query}"

        # No Authorization header at all - the signature alone authenticates.
        response = client.get(url)
        assert response.status_code == 200
        assert response.content == b"plot bytes"

        # Tampering with the signed filename must invalidate the signature and
        # fall back to requiring a Bearer token, which isn't present -> 401.
        tampered = f"/artifacts/{workspace_id}/xyz789/other.png?{query}"
        response = client.get(tampered)
        assert response.status_code == 401

        # An expired signature must also fall back and be rejected.
        expired_query = query.replace(query.split("&")[0], "exp=1")
        response = client.get(f"/artifacts/{workspace_id}/xyz789/plot.png?{expired_query}")
        assert response.status_code == 401
    finally:
        settings.api_token = original_token

def test_timeout_middleware_504s_slow_post_but_not_slow_get():
    import asyncio

    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    from ypotheto_compchem_mcp.http_app import TimeoutMiddleware

    async def slow_handler(request):
        await asyncio.sleep(0.2)
        return PlainTextResponse("done")

    slow_routes = [Route("/slow", endpoint=slow_handler, methods=["POST", "GET"])]
    wrapped = TimeoutMiddleware(Starlette(routes=slow_routes), timeout_seconds=0.05)
    client = TestClient(wrapped)

    response = client.post("/slow")
    assert response.status_code == 504

    # Scoped to POST only - a slow GET (health check, artifact download, or a
    # long-lived streamable-HTTP push) must not be interrupted.
    response = client.get("/slow")
    assert response.status_code == 200

def test_auth_mode_none_allows_all_requests_regardless_of_api_token():
    original_mode = settings.auth_mode
    original_token = settings.api_token
    settings.auth_mode = "none"
    settings.api_token = "some_secret_that_should_be_ignored"
    client = TestClient(app)

    try:
        response = client.get("/mcp")
        assert response.status_code != 401

        response = client.get("/mcp", headers={"Authorization": "Bearer totally_wrong"})
        assert response.status_code != 401
    finally:
        settings.auth_mode = original_mode
        settings.api_token = original_token

def test_auth_mode_keys_uses_key_store(tmp_path):
    from ypotheto_compchem_mcp.apikeys import SqliteKeyStore

    original_mode = settings.auth_mode
    original_db_url = settings.database_url
    original_data_dir = settings.data_dir
    settings.auth_mode = "keys"
    settings.database_url = ""
    settings.data_dir = tmp_path
    client = TestClient(app)

    try:
        raw_key = SqliteKeyStore(tmp_path / "keys.db").issue_key("ws_from_key")

        response = client.get("/mcp")
        assert response.status_code == 401

        response = client.get("/mcp", headers={"Authorization": "Bearer not_a_real_key"})
        assert response.status_code == 401

        response = client.get("/mcp", headers={"Authorization": f"Bearer {raw_key}"})
        assert response.status_code != 401
    finally:
        settings.auth_mode = original_mode
        settings.database_url = original_db_url
        settings.data_dir = original_data_dir

def test_auth_mode_oauth_wiring_rejects_and_accepts_and_sets_www_authenticate():
    import time

    import jwt
    from cryptography.hazmat.primitives.asymmetric import rsa

    from ypotheto_compchem_mcp import oauth as oauth_module

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    class _StaticSigningKey:
        def __init__(self, key):
            self.key = key

    class _StaticJWKClient:
        def get_signing_key_from_jwt(self, token):
            return _StaticSigningKey(public_key)

    issuer = "https://test-tenant.example.com"
    audience = "https://ypotheto-compchem-mcp.example/mcp"
    permission = "access:ypotheto-compchem-mcp"

    def _make_token(expired=False):
        now = time.time()
        claims = {
            "iss": issuer,
            "aud": audience,
            "sub": "kp_http_test",
            "exp": now - 5 if expired else now + 300,
            "permissions": [permission],
        }
        return jwt.encode(claims, private_key, algorithm="RS256")

    original_mode = settings.auth_mode
    original_issuer = settings.oauth_issuer
    original_audience = settings.oauth_audience
    original_permission = settings.oauth_required_permission
    original_build_verifier = oauth_module.build_oauth_verifier
    settings.auth_mode = "oauth"
    settings.oauth_issuer = issuer
    settings.oauth_audience = audience
    settings.oauth_required_permission = permission
    client = TestClient(app)

    try:
        # No token at all -> 401 with a WWW-Authenticate pointing at discovery.
        # This case never even reaches build_oauth_verifier (resolve_workspace_id_for_token
        # short-circuits on an empty token), so no patching is needed yet.
        response = client.get("/mcp")
        assert response.status_code == 401
        assert "resource_metadata" in response.headers.get("www-authenticate", "")

        # http_app.py resolves build_oauth_verifier via a fresh `from
        # ypotheto_compchem_mcp.oauth import build_oauth_verifier` on every
        # call, so patching the module attribute here is picked up
        # immediately - this exercises the exact middleware wiring a real
        # deployment would use, just swapping in a mocked JWK client instead
        # of a real JWKS endpoint.
        real_verifier = oauth_module.OAuthVerifier(
            issuer=issuer, audience=audience, required_permission=permission, jwk_client=_StaticJWKClient()
        )
        oauth_module.build_oauth_verifier = lambda _settings: real_verifier

        expired_token = _make_token(expired=True)
        response = client.get("/mcp", headers={"Authorization": f"Bearer {expired_token}"})
        assert response.status_code == 401

        valid_token = _make_token()
        response = client.get("/mcp", headers={"Authorization": f"Bearer {valid_token}"})
        assert response.status_code != 401
    finally:
        settings.auth_mode = original_mode
        settings.oauth_issuer = original_issuer
        settings.oauth_audience = original_audience
        settings.oauth_required_permission = original_permission
        oauth_module.build_oauth_verifier = original_build_verifier

def test_oauth_protected_resource_metadata_endpoint():
    original_issuer = settings.oauth_issuer
    original_audience = settings.oauth_audience
    settings.oauth_issuer = "https://test-tenant.example.com"
    settings.oauth_audience = "https://ypotheto-compchem-mcp.example/mcp"
    client = TestClient(app)

    try:
        response = client.get("/.well-known/oauth-protected-resource")
        assert response.status_code == 200
        body = response.json()
        assert body["resource"] == settings.oauth_audience
        assert body["authorization_servers"] == [settings.oauth_issuer]
    finally:
        settings.oauth_issuer = original_issuer
        settings.oauth_audience = original_audience

def test_dns_rebinding_protection_rejects_forged_host_header():
    original_token = settings.api_token
    settings.api_token = ""
    mcp_request = {"jsonrpc": "2.0", "method": "ping", "id": 1}
    mcp_headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

    try:
        # TestClient() must be entered as a context manager here: that's what
        # sends the ASGI lifespan.startup event, which is what starts the
        # streamable-HTTP session manager's task group in the first place.
        with TestClient(app) as client:
            # A Host header naming neither localhost/127.0.0.1 nor the
            # configured public_base_url is rejected before reaching the MCP
            # session handler at all.
            response = client.post(
                "/mcp/mcp", headers={**mcp_headers, "Host": "evil.example.com"}, json=mcp_request
            )
            assert response.status_code == 421

            # An allowed host is let through to the real MCP protocol handler
            # (a 400 here is the *protocol* layer complaining about a missing
            # session id, which only happens once past DNS-rebinding checks).
            response = client.post(
                "/mcp/mcp", headers={**mcp_headers, "Host": "localhost"}, json=mcp_request
            )
            assert response.status_code != 421
    finally:
        settings.api_token = original_token

def test_cors_lockdown_default_emits_no_cors_headers():
    client = TestClient(app)
    response = client.get("/healthz", headers={"Origin": "https://evil.example.com"})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers

def test_cors_allows_only_configured_origins():
    original_origins = settings.allowed_origins
    settings.allowed_origins = ["https://allowed.example.com"]
    client = TestClient(app)

    try:
        response = client.get("/healthz", headers={"Origin": "https://allowed.example.com"})
        assert response.headers.get("access-control-allow-origin") == "https://allowed.example.com"

        response = client.get("/healthz", headers={"Origin": "https://not-allowed.example.com"})
        assert "access-control-allow-origin" not in response.headers
    finally:
        settings.allowed_origins = original_origins

def test_query_param_token_no_longer_used_in_generated_urls():
    from pathlib import Path

    import ypotheto_compchem_mcp

    src_dir = Path(ypotheto_compchem_mcp.__file__).parent
    offenders = [
        str(path) for path in src_dir.rglob("*.py")
        if '"?t="' in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
