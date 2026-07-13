import hashlib
import hmac
import time
import uuid

from ypotheto_compchem_mcp.config import settings
from ypotheto_compchem_mcp.envelope import ArtifactInfo
from ypotheto_compchem_mcp.workspace import get_workspace_id, workspace_manager


def _artifact_signature(workspace_id: str, artifact_id: str, filename: str, expiry_ts: int) -> str:
    message = f"{workspace_id}/{artifact_id}/{filename}:{expiry_ts}"
    return hmac.new(settings.api_token.encode(), message.encode(), hashlib.sha256).hexdigest()

def sign_artifact_url(workspace_id: str, artifact_id: str, filename: str) -> str:
    """Build the `exp=...&sig=...` query string for a per-artifact signed URL.
    The signature IS the auth for this URL - anyone holding it can fetch this
    exact workspace/artifact/filename until it expires, without a Bearer token."""
    expiry_ts = int(time.time()) + settings.artifact_url_expiry_seconds
    sig = _artifact_signature(workspace_id, artifact_id, filename, expiry_ts)
    return f"exp={expiry_ts}&sig={sig}"

def verify_artifact_signature(
    workspace_id: str, artifact_id: str, filename: str, exp_param: str | None, sig_param: str | None
) -> bool:
    if not settings.api_token or exp_param is None or sig_param is None:
        return False
    try:
        expiry_ts = int(exp_param)
    except ValueError:
        return False
    if time.time() > expiry_ts:
        return False
    expected = _artifact_signature(workspace_id, artifact_id, filename, expiry_ts)
    return hmac.compare_digest(expected, sig_param)

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
    
    # Save to local workspace cache directory
    artifacts_dir = workspace_manager.get_artifacts_dir(workspace_id)
    target_dir = artifacts_dir / artifact_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / filename
    target_file.write_bytes(data)
    
    # Save to global persistent storage backend
    from ypotheto_compchem_mcp.storage import storage
    path = f"artifacts/{artifact_id}/{filename}"
    storage.write_file(workspace_id, path, data)
    
    # Public URL construction: {public_base_url}/artifacts/{workspace_id}/{artifact_id}/{filename}
    url = f"{settings.public_base_url}/artifacts/{workspace_id}/{artifact_id}/{filename}"

    # When token auth is active, sign the URL so it's self-authenticating without
    # leaking the shared secret itself into chat transcripts/logs/referrers.
    if settings.api_token:
        url += f"?{sign_artifact_url(workspace_id, artifact_id, filename)}"

    return ArtifactInfo(
        kind=kind,
        description=description,
        url=url
    )
