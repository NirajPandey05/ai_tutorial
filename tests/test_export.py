"""
Tests for the export router.
"""

import base64
import json
import pytest
from fastapi.testclient import TestClient

from src.ai_tutorial.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestDownloadEndpoint:
    """Tests for code download functionality."""
    
    def test_download_code_basic(self, client):
        """Test basic code download."""
        response = client.post(
            "/api/export/download",
            json={
                "code": "print('hello')",
                "lab_id": "first-api-call",
                "lab_title": "First API Call"
            }
        )
        
        assert response.status_code == 200
        assert "text/x-python" in response.headers["content-type"]
        assert "attachment" in response.headers["content-disposition"]
        assert "first_api_call.py" in response.headers["content-disposition"]
    
    def test_download_includes_header(self, client):
        """Test that download includes header comments."""
        response = client.post(
            "/api/export/download",
            json={
                "code": "print('hello')",
                "lab_id": "test-lab",
                "lab_title": "Test Lab",
                "include_comments": True
            }
        )
        
        content = response.content.decode()
        assert "AI Engineering Tutorial" in content
        assert "Test Lab" in content
        assert "print('hello')" in content
    
    def test_download_without_header(self, client):
        """Test download without header comments."""
        response = client.post(
            "/api/export/download",
            json={
                "code": "print('hello')",
                "lab_id": "test-lab",
                "lab_title": "Test Lab",
                "include_comments": False
            }
        )
        
        content = response.content.decode()
        assert content.strip() == "print('hello')"


class TestShareEndpoint:
    """Tests for share link functionality."""
    
    def test_create_share_link(self, client):
        """Test creating a share link."""
        response = client.post(
            "/api/export/share",
            json={
                "code": "print('hello')",
                "lab_id": "first-api-call"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "share_id" in data
        assert "share_url" in data
        assert len(data["share_id"]) == 12
    
    def test_retrieve_shared_code(self, client):
        """Test retrieving shared code."""
        # First create a share
        create_response = client.post(
            "/api/export/share",
            json={
                "code": "test_code_here",
                "lab_id": "test-lab",
                "result": "test result"
            }
        )
        
        share_id = create_response.json()["share_id"]
        
        # Then retrieve it
        get_response = client.get(f"/api/export/share/{share_id}")
        
        assert get_response.status_code == 200
        data = get_response.json()
        assert data["code"] == "test_code_here"
        assert data["lab_id"] == "test-lab"
        assert data["result"] == "test result"
    
    def test_retrieve_shared_code_raw(self, client):
        """Test retrieving raw shared code."""
        # First create a share
        create_response = client.post(
            "/api/export/share",
            json={
                "code": "raw_code_content",
                "lab_id": "test-lab"
            }
        )
        
        share_id = create_response.json()["share_id"]
        
        # Get raw content
        get_response = client.get(f"/api/export/share/{share_id}/raw")
        
        assert get_response.status_code == 200
        assert get_response.text == "raw_code_content"
    
    def test_share_not_found(self, client):
        """Test that invalid share ID returns 404."""
        response = client.get("/api/export/share/nonexistent123")
        assert response.status_code == 404


class TestEncodeDecodeEndpoint:
    """Tests for URL-based code encoding."""
    
    def test_encode_code(self, client):
        """Test encoding code for URL."""
        response = client.post(
            "/api/export/encode",
            json={
                "code": "print('hello')",
                "lab_id": "test-lab"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "encoded" in data
        assert "share_url" in data
        assert "size_bytes" in data
    
    def test_decode_code(self, client):
        """Test decoding code from URL."""
        # First encode
        encode_response = client.post(
            "/api/export/encode",
            json={
                "code": "test_code",
                "lab_id": "test-lab",
                "result": "test_result"
            }
        )
        
        encoded = encode_response.json()["encoded"]
        
        # Then decode
        decode_response = client.get(f"/api/export/decode?data={encoded}")
        
        assert decode_response.status_code == 200
        data = decode_response.json()
        assert data["code"] == "test_code"
        assert data["lab_id"] == "test-lab"
        assert data["result"] == "test_result"
    
    def test_decode_invalid_data(self, client):
        """Test that invalid encoded data returns error."""
        response = client.get("/api/export/decode?data=invalid_base64!")
        assert response.status_code == 400


class TestSocialShareEndpoint:
    """Tests for social media sharing."""
    
    def test_generate_social_links(self, client):
        """Test generating social share links."""
        response = client.post(
            "/api/export/social-share",
            json={
                "code": "print('hello')",
                "lab_id": "first-api-call"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "share_id" in data
        assert "share_url" in data
        assert "social_links" in data
        
        social_links = data["social_links"]
        assert "twitter" in social_links
        assert "linkedin" in social_links
        assert "facebook" in social_links
        assert "reddit" in social_links
        
        # Check URLs are valid
        assert "twitter.com" in social_links["twitter"]
        assert "linkedin.com" in social_links["linkedin"]
        assert "facebook.com" in social_links["facebook"]
        assert "reddit.com" in social_links["reddit"]
    
    def test_social_share_includes_embed(self, client):
        """Test that social share includes embed code."""
        response = client.post(
            "/api/export/social-share",
            json={
                "code": "print('hello')",
                "lab_id": "test-lab"
            }
        )
        
        data = response.json()
        assert "embed_code" in data
        assert "<iframe" in data["embed_code"]


class TestGistUrlEndpoint:
    """Tests for GitHub Gist URL generation."""
    
    def test_generate_gist_url(self, client):
        """Test generating Gist URL data."""
        response = client.post(
            "/api/export/gist-url",
            json={
                "code": "print('hello world')",
                "filename": "my_script.py",
                "description": "My test gist"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "gist_url" in data
        assert "gist_data" in data
        assert "api_endpoint" in data
        
        gist_data = data["gist_data"]
        assert gist_data["description"] == "My test gist"
        assert "my_script.py" in gist_data["files"]
        assert gist_data["files"]["my_script.py"]["content"] == "print('hello world')"
