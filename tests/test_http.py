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
