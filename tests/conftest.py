import pytest

from ypotheto_compchem_mcp.config import settings
from ypotheto_compchem_mcp.storage import storage
from ypotheto_compchem_mcp.workspace import workspace_manager


@pytest.fixture(scope="session", autouse=True)
def disable_db_for_tests():
    """Disable PostgreSQL database connection, Spaces bucket, and API token auth
    for standard test runs to prevent test delays, mock incompatibilities, and
    production pollution. A real .env with live DB/Spaces credentials and a real
    shared secret sits in the working tree - without this, tests would silently
    inherit it (statistician-mcp had tests hit production credentials this way).
    Individual tests are still free to set settings.api_token etc. themselves
    for the duration of one test; this only fixes the baseline they start from."""
    original_url = settings.database_url
    original_bucket = settings.spaces_bucket
    original_token = settings.api_token

    settings.database_url = ""
    settings.spaces_bucket = None
    settings.api_token = ""
    storage.reset()

    yield

    settings.database_url = original_url
    settings.spaces_bucket = original_bucket
    settings.api_token = original_token
    storage.reset()

@pytest.fixture(scope="session", autouse=True)
def isolate_data_dir_for_tests(tmp_path_factory):
    """
    The default local workspace directory (~/.compchem-mcp) is NOT test-isolated:
    workspace_manager is a module-level singleton that captures settings.data_dir
    once at import time, so simply patching settings.data_dir here wouldn't be
    enough on its own - workspace_manager.data_dir must be patched directly too.

    Without this, every test session on a given machine shares (and can corrupt)
    the same molecules/index.json file - concurrent test runs (e.g. from multiple
    sessions/worktrees on the same machine) can race on it and produce invalid
    JSON, as happened in practice. Point both at a fresh per-session temp dir.
    """
    original_settings_data_dir = settings.data_dir
    original_manager_data_dir = workspace_manager.data_dir

    tmp_dir = tmp_path_factory.mktemp("compchem_data_dir")
    settings.data_dir = tmp_dir
    workspace_manager.data_dir = tmp_dir
    storage.reset()

    yield

    settings.data_dir = original_settings_data_dir
    workspace_manager.data_dir = original_manager_data_dir
    storage.reset()
