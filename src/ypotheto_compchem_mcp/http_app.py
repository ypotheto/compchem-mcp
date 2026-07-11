from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, FileResponse
from starlette.routing import Route, Mount
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from ypotheto_compchem_mcp.config import settings
from ypotheto_compchem_mcp import __version__
from ypotheto_compchem_mcp.server import mcp

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Exclude healthz from authentication
        if request.url.path == "/healthz":
            return await call_next(request)
            
        token = ""
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
        else:
            token = request.query_params.get("t", "")
            
        # Validate API token if one is set in config
        if settings.api_token:
            if auth_header != f"Bearer {settings.api_token}" and token != settings.api_token:
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
                
        # Resolve and set the current workspace ID ContextVar
        from ypotheto_compchem_mcp.workspace import current_workspace_id, get_workspace_id_from_token
        workspace_id = get_workspace_id_from_token(token)
        token_var = current_workspace_id.set(workspace_id)
        try:
            return await call_next(request)
        finally:
            current_workspace_id.reset(token_var)

async def healthz(request):
    return JSONResponse({"status": "ok", "version": __version__})

async def serve_artifact(request):
    workspace_id = request.path_params.get("workspace_id")
    artifact_id = request.path_params.get("artifact_id")
    filename = request.path_params.get("filename")
    
    # Simple directory traversal check
    if ".." in workspace_id or ".." in artifact_id or ".." in filename:
        return JSONResponse({"detail": "Forbidden"}, status_code=403)
        
    filepath = settings.data_dir / "workspaces" / workspace_id / "artifacts" / artifact_id / filename
    if not filepath.exists() or not filepath.is_file():
        return Response("Artifact not found", status_code=404)
        
    return FileResponse(filepath)

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

# Instantiate Starlette application with CORS and Auth middleware
app = Starlette(
    routes=routes,
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"]
        ),
        Middleware(AuthMiddleware)
    ]
)
