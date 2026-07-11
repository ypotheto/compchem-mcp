from starlette.testclient import TestClient
from ypotheto_compchem_mcp.http_app import app
from ypotheto_compchem_mcp.config import settings

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
        
        # Request with valid token in query parameter (for artifacts) should pass auth middleware
        response = client.get("/mcp?t=test_secret_token")
        assert response.status_code != 401
    finally:
        # Restore original token
        settings.api_token = original_token
