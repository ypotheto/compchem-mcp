import argparse
from pathlib import Path

from ypotheto_compchem_mcp.config import settings


def main():
    parser = argparse.ArgumentParser(description="Ypotheto Computational Chemistry MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport mechanism (stdio or http)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help="Port to run the HTTP/SSE server on"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(settings.data_dir),
        help="Directory to store datasets/artifacts"
    )
    
    args = parser.parse_args()
    
    # Update settings
    settings.port = args.port
    settings.data_dir = Path(args.data_dir).expanduser()
    
    # Ensure data directory exists
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize database tables
    from ypotheto_compchem_mcp.database import initialize_database
    initialize_database()
    
    # Start background job queue workers
    from ypotheto_compchem_mcp.jobs import job_manager
    job_manager.start_workers()
    
    from ypotheto_compchem_mcp.server import create_server
    bundle = create_server(settings)

    if args.transport == "stdio":
        # Run standard FastMCP server via STDIO
        bundle.mcp.run()
    else:
        # Run HTTP/SSE Starlette app via uvicorn
        import uvicorn

        from ypotheto_compchem_mcp.http_app import create_app
        app = create_app(bundle)
        uvicorn.run(app, host="0.0.0.0", port=settings.port)

if __name__ == "__main__":
    main()
