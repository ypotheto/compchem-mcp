import pytest
from ypotheto_compchem_mcp.config import settings
from ypotheto_compchem_mcp.storage import storage

@pytest.fixture(scope="session", autouse=True)
def disable_db_for_tests():
    """Disable PostgreSQL database connection and Spaces bucket for standard test runs to prevent test delays, mock incompatibilities, and production pollution."""
    original_url = settings.database_url
    original_bucket = settings.spaces_bucket
    
    settings.database_url = ""
    settings.spaces_bucket = None
    storage.reset()
    
    yield
    
    settings.database_url = original_url
    settings.spaces_bucket = original_bucket
    storage.reset()
