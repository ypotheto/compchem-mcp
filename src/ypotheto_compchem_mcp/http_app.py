import asyncio

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from ypotheto_compchem_mcp import __version__
from ypotheto_compchem_mcp.config import settings
from ypotheto_compchem_mcp.server import mcp


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
        # /healthz and /artifacts/* are excluded from Bearer auth here: artifacts
        # authenticate themselves (signed URL, or a Bearer fallback scoped to the
        # caller's own workspace - see serve_artifact) since a signed link has to
        # work for a recipient who never had the shared secret in the first place.
        if path == "/healthz" or path.startswith("/artifacts/"):
            await self._app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        auth_header = request.headers.get("Authorization")
        token = auth_header.split(" ", 1)[1] if auth_header and auth_header.startswith("Bearer ") else ""

        if settings.api_token and token != settings.api_token:
            response = JSONResponse({"detail": "Unauthorized"}, status_code=401)
            await response(scope, receive, send)
            return

        from ypotheto_compchem_mcp.workspace import (
            current_workspace_id,
            get_workspace_id_from_token,
        )
        workspace_id = get_workspace_id_from_token(token)
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
        # Fall back to Bearer auth, scoped to the caller's own workspace - never
        # the workspace_id named in the URL. 404 (not 403/401) on mismatch so a
        # guessed/reused workspace_id doesn't even confirm it exists.
        auth_header = request.headers.get("Authorization")
        token = auth_header.split(" ", 1)[1] if auth_header and auth_header.startswith("Bearer ") else ""
        if settings.api_token and token != settings.api_token:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        from ypotheto_compchem_mcp.workspace import get_workspace_id_from_token
        if workspace_id != get_workspace_id_from_token(token):
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
