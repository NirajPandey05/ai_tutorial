"""
Tests for FastAPI Routes

Tests cover:
- Page routing
- Content delivery
- Learning path pages
- API endpoints
- Error handling
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai_tutorial.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHomeRoutes:
    """Tests for home and main pages."""
    
    def test_home_page(self, client):
        """Home page should load."""
        response = client.get("/")
        assert response.status_code == 200
        assert "AI Engineering Tutorial" in response.text
    
    def test_learning_paths_page(self, client):
        """Learning paths page should load."""
        response = client.get("/learning-paths")
        assert response.status_code == 200
        assert "Learning Path" in response.text
    
    def test_glossary_page(self, client):
        """Glossary page should load."""
        response = client.get("/glossary")
        assert response.status_code == 200
    
    def test_settings_page(self, client):
        """Settings page should load."""
        response = client.get("/settings")
        assert response.status_code == 200
        assert "API" in response.text or "Settings" in response.text


class TestModuleRoutes:
    """Tests for module pages."""
    
    def test_module_overview(self, client):
        """Module overview page should load."""
        response = client.get("/learn/llm-fundamentals")
        assert response.status_code == 200
        assert "LLM Fundamentals" in response.text
    
    def test_nonexistent_module(self, client):
        """Nonexistent module should return 404."""
        response = client.get("/learn/nonexistent-module")
        assert response.status_code == 404
    
    def test_section_page(self, client):
        """Section page should load."""
        response = client.get("/learn/llm-fundamentals/intro-to-llms")
        assert response.status_code == 200
    
    def test_nonexistent_section(self, client):
        """Nonexistent section should return 404."""
        response = client.get("/learn/llm-fundamentals/nonexistent-section")
        assert response.status_code == 404


class TestContentPages:
    """Tests for content pages (lessons, labs, quizzes)."""
    
    def test_lesson_page(self, client):
        """Lesson page should load."""
        response = client.get("/learn/llm-fundamentals/intro-to-llms/what-are-llms")
        assert response.status_code == 200
        assert "Large Language Models" in response.text or "LLM" in response.text
    
    def test_nonexistent_page(self, client):
        """Nonexistent page should return 404."""
        response = client.get("/learn/llm-fundamentals/intro-to-llms/nonexistent-page")
        assert response.status_code == 404
    
    def test_content_has_navigation(self, client):
        """Content page should have navigation."""
        response = client.get("/learn/llm-fundamentals/intro-to-llms/what-are-llms")
        assert response.status_code == 200
        # Should have some form of navigation
        assert "Next" in response.text or "Previous" in response.text or "nav" in response.text.lower()


class TestLearningPathRoutes:
    """Tests for learning path pages."""
    
    def test_path_detail_page(self, client):
        """Learning path detail page should load."""
        response = client.get("/path/quick-start")
        assert response.status_code == 200
        assert "Quick Start" in response.text
    
    def test_nonexistent_path(self, client):
        """Nonexistent path should return 404."""
        response = client.get("/path/nonexistent-path")
        assert response.status_code == 404
    
    def test_path_shows_modules(self, client):
        """Path page should show modules."""
        response = client.get("/path/rag-engineer")
        assert response.status_code == 200
        # RAG Engineer path includes these modules
        assert "LLM Fundamentals" in response.text or "Embeddings" in response.text or "RAG" in response.text


class TestPathAwareNavigation:
    """Tests for path-aware navigation via query parameter."""
    
    def test_content_with_path_param(self, client):
        """Content page with path parameter should load."""
        response = client.get("/learn/llm-fundamentals/intro-to-llms/what-are-llms?path=quick-start")
        assert response.status_code == 200
    
    def test_path_param_preserves_context(self, client):
        """Navigation should preserve path context."""
        response = client.get("/learn/llm-fundamentals/intro-to-llms/what-are-llms?path=rag-engineer")
        assert response.status_code == 200
        # Path should be referenced somewhere in the page
        assert "rag-engineer" in response.text or "RAG Engineer" in response.text


class TestAPIEndpoints:
    """Tests for API endpoints."""
    
    def test_path_content_api(self, client):
        """Path content API should return JSON."""
        response = client.get("/api/path/quick-start/content")
        assert response.status_code == 200
        data = response.json()
        assert "path_id" in data
        assert "content" in data
        assert data["path_id"] == "quick-start"
    
    def test_path_content_api_nonexistent(self, client):
        """API should return 404 for nonexistent path."""
        response = client.get("/api/path/nonexistent/content")
        assert response.status_code == 404


class TestStaticFiles:
    """Tests for static file serving."""
    
    def test_css_loads(self, client):
        """CSS file should load."""
        response = client.get("/static/css/styles.css")
        assert response.status_code == 200
        assert "text/css" in response.headers.get("content-type", "")
    
    def test_js_loads(self, client):
        """JavaScript file should load."""
        response = client.get("/static/js/main.js")
        assert response.status_code == 200
    
    def test_progress_tracker_loads(self, client):
        """Progress tracker JS should load."""
        response = client.get("/static/js/progress-tracker.js")
        assert response.status_code == 200


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_404_for_invalid_route(self, client):
        """Invalid routes should return 404."""
        response = client.get("/this/route/does/not/exist")
        assert response.status_code == 404
    
    def test_module_404_message(self, client):
        """Module 404 should have helpful message."""
        response = client.get("/learn/fake-module")
        assert response.status_code == 404


class TestBreadcrumbs:
    """Tests for breadcrumb navigation."""
    
    def test_module_has_breadcrumbs(self, client):
        """Module page should have breadcrumbs."""
        response = client.get("/learn/llm-fundamentals")
        assert response.status_code == 200
        assert "Home" in response.text
    
    def test_content_has_breadcrumbs(self, client):
        """Content page should have breadcrumbs."""
        response = client.get("/learn/llm-fundamentals/intro-to-llms/what-are-llms")
        assert response.status_code == 200
        # Should have module name in breadcrumb
        assert "LLM Fundamentals" in response.text


class TestResponsiveness:
    """Tests for responsive design elements."""
    
    def test_viewport_meta(self, client):
        """Pages should have viewport meta tag."""
        response = client.get("/")
        assert response.status_code == 200
        assert "viewport" in response.text
    
    def test_mobile_nav_elements(self, client):
        """Should have mobile navigation elements."""
        response = client.get("/")
        assert response.status_code == 200
        # Should have some mobile-friendly classes or elements
        assert "md:" in response.text or "lg:" in response.text or "mobile" in response.text.lower()
