import asyncio

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from ypotheto_compchem_mcp import __version__
from ypotheto_compchem_mcp.config import settings
from ypotheto_compchem_mcp.server import mcp


def _extract_bearer_token(request: Request) -> str:
    auth_header = request.headers.get("Authorization")
    return auth_header.split(" ", 1)[1] if auth_header and auth_header.startswith("Bearer ") else ""


def resolve_workspace_id_for_token(token: str) -> str | None:
    """Resolve a bearer token to a workspace id per the live `settings.auth_mode`,
    or None if the token is missing/invalid for that mode. Shared by
    `AuthMiddleware` and `serve_artifact`'s Bearer-fallback path so the two auth
    checks can never silently diverge (e.g. one getting updated for a new
    auth_mode while the other still only knows about the shared-secret check).

    Reads `settings.auth_mode`/`settings.api_token`/etc. live on every call
    rather than baking them in at construction time, matching this module's
    existing settings-singleton-is-mutable pattern (see CorsLockdownMiddleware)
    - this is what lets tests flip `settings.auth_mode`/`settings.api_token`
    around a shared `app` object without rebuilding it."""
    from ypotheto_compchem_mcp.workspace import get_workspace_id_from_token

    mode = settings.auth_mode
    if mode == "none":
        # Explicit opt-out: no credential required at all, regardless of
        # whatever api_token happens to be set to. Whatever bearer token the
        # caller does supply (if any) still selects their own workspace.
        return get_workspace_id_from_token(token)

    if mode == "keys":
        if not token:
            return None
        from ypotheto_compchem_mcp.apikeys import build_key_store
        return build_key_store(settings).verify_key(token)

    if mode == "oauth":
        if not token:
            return None
        from ypotheto_compchem_mcp.oauth import build_oauth_verifier
        return build_oauth_verifier(settings).verify(token)

    # "token" (default) and any unrecognized value: preserve the original
    # shared-secret behavior exactly - an unset api_token means auth is
    # effectively open in this mode too.
    if settings.api_token and token != settings.api_token:
        return None
    return get_workspace_id_from_token(token)


def _oauth_unauthorized_headers() -> dict[str, str]:
    if settings.auth_mode != "oauth":
        return {}
    base_url = settings.public_base_url.rstrip("/")
    resource_metadata_url = f"{base_url}/.well-known/oauth-protected-resource"
    return {"WWW-Authenticate": f'Bearer resource_metadata="{resource_metadata_url}"'}


class AuthMiddleware:
    """Plain ASGI middleware (not `BaseHTTPMiddleware`, which buffers the whole
    response and breaks streaming - the MCP mount at /mcp is a streamable-HTTP app)."""

    def __init__(self, app):
        self._app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        path = scope["path"]
        # /healthz, /.well-known/oauth-protected-resource, and /artifacts/* are
        # excluded from Bearer auth here: the well-known route has to be
        # reachable *before* a client has a token (it's how one is discovered
        # in the first place), and artifacts authenticate themselves (signed
        # URL, or a Bearer fallback scoped to the caller's own workspace - see
        # serve_artifact) since a signed link has to work for a recipient who
        # never had the shared secret/token in the first place.
        if path in ("/healthz", "/.well-known/oauth-protected-resource") or path.startswith("/artifacts/"):
            await self._app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        token = _extract_bearer_token(request)
        workspace_id = resolve_workspace_id_for_token(token)

        if workspace_id is None:
            response = JSONResponse(
                {"detail": "Unauthorized"}, status_code=401, headers=_oauth_unauthorized_headers()
            )
            await response(scope, receive, send)
            return

        from ypotheto_compchem_mcp.workspace import current_workspace_id
        token_var = current_workspace_id.set(workspace_id)
        try:
            await self._app(scope, receive, send)
        finally:
            current_workspace_id.reset(token_var)

class TimeoutMiddleware:
    """Bounds the worst-case duration of a single tool-call request (POST /mcp).
    Scoped to POST only so it never interrupts a GET (health check, artifact
    download, or a long-lived streamable-HTTP server-push stream). Long
    computations are expected to go through the job queue instead, so a hard
    HTTP timeout here is safe."""

    def __init__(self, app, timeout_seconds: float):
        self._app = app
        self._timeout_seconds = timeout_seconds

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or self._timeout_seconds <= 0:
            await self._app(scope, receive, send)
            return

        try:
            async with asyncio.timeout(self._timeout_seconds):
                await self._app(scope, receive, send)
        except TimeoutError:
            response = JSONResponse({"detail": "Request timed out"}, status_code=504)
            await response(scope, receive, send)

class CorsLockdownMiddleware:
    """Wraps Starlette's CORSMiddleware but re-reads settings.allowed_origins on
    every request (like Auth/Timeout above) instead of baking it in at import
    time, matching the rest of this module's settings-singleton-is-mutable
    pattern. An empty list - the default - means no CORS headers at all
    (same-origin only); owner confirmed no browser clients exist, so nothing
    should be seeded here."""

    def __init__(self, app):
        self._app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not settings.allowed_origins:
            await self._app(scope, receive, send)
            return
        cors_app = CORSMiddleware(
            self._app,
            allow_origins=settings.allowed_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        await cors_app(scope, receive, send)

async def healthz(request):
    return JSONResponse({"status": "ok", "version": __version__})

async def oauth_protected_resource_metadata(request):
    # RFC 9728 Protected Resource Metadata: tells an MCP client where the
    # authorization server is, so it knows where to log in. Always registered
    # (not just under auth_mode="oauth") since it's a public discovery
    # endpoint - a client only ever fetches it after seeing a 401 whose
    # WWW-Authenticate header points here, which only happens in oauth mode.
    return JSONResponse(
        {
            "resource": settings.oauth_audience or "",
            "authorization_servers": [settings.oauth_issuer] if settings.oauth_issuer else [],
        }
    )

async def serve_artifact(request):
    workspace_id = request.path_params.get("workspace_id")
    artifact_id = request.path_params.get("artifact_id")
    filename = request.path_params.get("filename")

    from ypotheto_compchem_mcp.artifacts import verify_artifact_signature
    is_signed = verify_artifact_signature(
        workspace_id, artifact_id, filename,
        request.query_params.get("exp"), request.query_params.get("sig"),
    )
    if not is_signed:
        # Fall back to Bearer auth. Shares resolve_workspace_id_for_token with
        # AuthMiddleware so keys/oauth mode is honored here too, not just the
        # legacy shared-secret check. Two distinct failure cases, matching the
        # original behavior: a credential that's invalid outright (wrong
        # shared secret / unknown key / bad OAuth token) -> 401; a validly
        # authenticated caller whose OWN workspace just isn't this artifact's
        # -> 404 (not 401/403) so a guessed/reused workspace_id in the URL
        # doesn't even confirm the artifact exists.
        token = _extract_bearer_token(request)
        resolved_workspace_id = resolve_workspace_id_for_token(token)
        if resolved_workspace_id is None:
            return JSONResponse(
                {"detail": "Unauthorized"}, status_code=401, headers=_oauth_unauthorized_headers()
            )
        if workspace_id != resolved_workspace_id:
            return JSONResponse({"detail": "Not Found"}, status_code=404)

    path = f"artifacts/{artifact_id}/{filename}"
    try:
        from ypotheto_compchem_mcp.storage import storage
        data = storage.read_file(workspace_id, path)
        import mimetypes
        media_type, _ = mimetypes.guess_type(filename)
        return Response(data, media_type=media_type)
    except (FileNotFoundError, ValueError):
        return Response("Artifact not found", status_code=404)
    except Exception as e:
        return Response(f"Internal storage error: {str(e)}", status_code=500)

# Build ASGI routes
routes = [
    Route("/healthz", endpoint=healthz, methods=["GET"]),
    Route(
        "/.well-known/oauth-protected-resource",
        endpoint=oauth_protected_resource_metadata,
        methods=["GET"],
    ),
    Route(
        "/artifacts/{workspace_id}/{artifact_id}/{filename}",
        endpoint=serve_artifact,
        methods=["GET"]
    ),
    Mount("/mcp", app=mcp.streamable_http_app()),
]

# Wrap the routing app in plain-ASGI middleware, outermost first: CORS (so
# preflight is handled before auth/timeout even run) -> Timeout -> Auth -> routes.
#
# lifespan is wired explicitly to mcp.session_manager.run(): Starlette's default
# lifespan handler does NOT recurse into a Mount()'d sub-app's own lifespan, so
# without this the streamable-HTTP session manager's task group is never
# started and every real request to /mcp/mcp fails with "Task group is not
# initialized. Make sure to use run()." - this was a pre-existing bug (present
# since this Mount was first added), not introduced by this middleware rewrite.
app = Starlette(routes=routes, lifespan=lambda _app: mcp.session_manager.run())
app = AuthMiddleware(app)
app = TimeoutMiddleware(app, timeout_seconds=settings.request_timeout_seconds)
app = CorsLockdownMiddleware(app)
