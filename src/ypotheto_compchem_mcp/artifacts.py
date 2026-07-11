import uuid
from pathlib import Path
from ypotheto_compchem_mcp.config import settings
from ypotheto_compchem_mcp.workspace import get_workspace_id, workspace_manager
from ypotheto_compchem_mcp.envelope import ArtifactInfo

def register_artifact(
    filename: str,
    data: bytes,
    kind: str,  # "structure" | "plot" | "report"
    description: str
) -> ArtifactInfo:
    """
    Save files (e.g. XYZ structures, Matplotlib plots, HTML frames) to the workspace artifact directory.
    Returns an ArtifactInfo instance with a signed/tokenized download URL.
    """
    workspace_id = get_workspace_id()
    artifact_id = uuid.uuid4().hex[:8]
    
    # Workspace artifacts location: workspaces/{workspace_id}/artifacts/{artifact_id}/{filename}
    artifacts_dir = workspace_manager.get_artifacts_dir(workspace_id)
    target_dir = artifacts_dir / artifact_id
    target_dir.mkdir(parents=True, exist_ok=True)
    
    target_file = target_dir / filename
    target_file.write_bytes(data)
    
    # Public URL construction: {public_base_url}/artifacts/{workspace_id}/{artifact_id}/{filename}
    url = f"{settings.public_base_url}/artifacts/{workspace_id}/{artifact_id}/{filename}"
    
    # Automatically append authentication query parameter if token auth is active
    if settings.api_token:
        url += f"?t={settings.api_token}"
        
    return ArtifactInfo(
        kind=kind,
        description=description,
        url=url
    )
