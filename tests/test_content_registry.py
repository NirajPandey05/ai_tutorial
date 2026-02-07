"""
Tests for Content Registry and Navigation

Tests cover:
- Module registration and retrieval
- Section and page lookups
- Navigation (adjacent content)
- Learning path functionality
- Path-aware navigation
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai_tutorial.content.registry import ContentRegistry, get_registry
from ai_tutorial.content.models import (
    Module, Section, Page, Lab, Quiz, LearningPath,
    Difficulty, PageType
)


class TestContentRegistry:
    """Tests for ContentRegistry singleton and basic operations."""
    
    @pytest.fixture
    def registry(self):
        """Get the content registry."""
        return ContentRegistry.get_instance()
    
    def test_singleton_pattern(self):
        """Registry should be a singleton."""
        r1 = ContentRegistry.get_instance()
        r2 = ContentRegistry.get_instance()
        assert r1 is r2
    
    def test_get_registry_function(self):
        """get_registry() should return the singleton."""
        registry = get_registry()
        assert registry is ContentRegistry.get_instance()
    
    def test_has_modules(self, registry):
        """Registry should have modules registered."""
        modules = registry.get_all_modules()
        assert len(modules) > 0
    
    def test_modules_sorted_by_order(self, registry):
        """Modules should be sorted by order."""
        modules = registry.get_all_modules()
        orders = [m.order for m in modules]
        assert orders == sorted(orders)


class TestModuleOperations:
    """Tests for module-related operations."""
    
    @pytest.fixture
    def registry(self):
        return ContentRegistry.get_instance()
    
    def test_get_module_by_id(self, registry):
        """Should retrieve module by ID."""
        module = registry.get_module("llm-fundamentals")
        assert module is not None
        assert module.id == "llm-fundamentals"
        assert module.title == "LLM Fundamentals"
    
    def test_get_nonexistent_module(self, registry):
        """Should return None for nonexistent module."""
        module = registry.get_module("nonexistent-module")
        assert module is None
    
    def test_module_has_sections(self, registry):
        """Module should have sections."""
        module = registry.get_module("llm-fundamentals")
        assert len(module.sections) > 0
    
    def test_module_difficulty(self, registry):
        """Module should have difficulty level."""
        module = registry.get_module("llm-fundamentals")
        assert module.difficulty in [Difficulty.BEGINNER, Difficulty.INTERMEDIATE, Difficulty.ADVANCED]


class TestSectionOperations:
    """Tests for section-related operations."""
    
    @pytest.fixture
    def registry(self):
        return ContentRegistry.get_instance()
    
    def test_get_section(self, registry):
        """Should retrieve section by module and section ID."""
        section = registry.get_section("llm-fundamentals", "intro-to-llms")
        assert section is not None
        assert section.id == "intro-to-llms"
    
    def test_get_nonexistent_section(self, registry):
        """Should return None for nonexistent section."""
        section = registry.get_section("llm-fundamentals", "nonexistent")
        assert section is None
    
    def test_section_has_pages(self, registry):
        """Section should have pages."""
        section = registry.get_section("llm-fundamentals", "intro-to-llms")
        assert len(section.pages) > 0


class TestPageOperations:
    """Tests for page-related operations."""
    
    @pytest.fixture
    def registry(self):
        return ContentRegistry.get_instance()
    
    def test_get_page(self, registry):
        """Should retrieve page by full path."""
        page = registry.get_page("llm-fundamentals", "intro-to-llms", "what-are-llms")
        assert page is not None
        assert page.id == "what-are-llms"
    
    def test_get_nonexistent_page(self, registry):
        """Should return None for nonexistent page."""
        page = registry.get_page("llm-fundamentals", "intro-to-llms", "nonexistent")
        assert page is None
    
    def test_page_has_required_fields(self, registry):
        """Page should have all required fields."""
        page = registry.get_page("llm-fundamentals", "intro-to-llms", "what-are-llms")
        assert page.title
        assert page.id
        assert page.estimated_time >= 0


class TestNavigationOperations:
    """Tests for navigation functionality."""
    
    @pytest.fixture
    def registry(self):
        return ContentRegistry.get_instance()
    
    def test_get_adjacent_content(self, registry):
        """Should get previous and next content items."""
        prev_content, next_content = registry.get_adjacent_content(
            "llm-fundamentals", "intro-to-llms", "what-are-llms"
        )
        # First item in first section should have no previous
        assert prev_content is None
        # Should have next content
        assert next_content is not None
    
    def test_adjacent_content_structure(self, registry):
        """Adjacent content should have correct structure."""
        _, next_content = registry.get_adjacent_content(
            "llm-fundamentals", "intro-to-llms", "what-are-llms"
        )
        assert "type" in next_content
        assert "item" in next_content
        assert "section" in next_content
        assert "module" in next_content
        assert "id" in next_content
    
    def test_adjacent_sections(self, registry):
        """Should get adjacent sections."""
        module = registry.get_module("llm-fundamentals")
        if len(module.sections) > 1:
            first_section = module.sections[0]
            prev_sec, next_sec = registry.get_adjacent_sections(
                "llm-fundamentals", first_section.id
            )
            assert prev_sec is None  # First section has no previous
            assert next_sec is not None


class TestLearningPaths:
    """Tests for learning path functionality."""
    
    @pytest.fixture
    def registry(self):
        return ContentRegistry.get_instance()
    
    def test_has_learning_paths(self, registry):
        """Registry should have learning paths."""
        paths = registry.get_all_learning_paths()
        assert len(paths) > 0
    
    def test_get_learning_path_by_id(self, registry):
        """Should retrieve learning path by ID."""
        path = registry.get_learning_path("quick-start")
        assert path is not None
        assert path.id == "quick-start"
        assert path.title == "Quick Start"
    
    def test_get_nonexistent_path(self, registry):
        """Should return None for nonexistent path."""
        path = registry.get_learning_path("nonexistent-path")
        assert path is None
    
    def test_path_has_modules(self, registry):
        """Learning path should have modules."""
        path = registry.get_learning_path("quick-start")
        assert len(path.modules) > 0
    
    def test_path_modules_exist(self, registry):
        """All modules in path should exist."""
        path = registry.get_learning_path("rag-engineer")
        for module_id in path.modules:
            module = registry.get_module(module_id)
            assert module is not None, f"Module {module_id} should exist"


class TestPathAwareNavigation:
    """Tests for path-aware navigation."""
    
    @pytest.fixture
    def registry(self):
        return ContentRegistry.get_instance()
    
    def test_get_path_content_list(self, registry):
        """Should get flat list of all content in path."""
        content = registry.get_path_content_list("quick-start")
        assert len(content) > 0
        # Each item should have required fields
        for item in content:
            assert "id" in item
            assert "title" in item
            assert "type" in item
            assert "module_id" in item
            assert "section_id" in item
    
    def test_get_first_content_in_path(self, registry):
        """Should get first content item in path."""
        first = registry.get_first_content_in_path("quick-start")
        assert first is not None
        assert "id" in first
        # Should be from the first module in the path
        path = registry.get_learning_path("quick-start")
        assert first["module_id"] == path.modules[0]
    
    def test_path_aware_navigation(self, registry):
        """Path-aware navigation should work across modules."""
        path = registry.get_learning_path("rag-engineer")
        first_content = registry.get_first_content_in_path("rag-engineer")
        
        prev, next_item, path_info = registry.get_adjacent_content_for_path(
            "rag-engineer",
            first_content["module_id"],
            first_content["section_id"],
            first_content["id"]
        )
        
        # First item should have no previous
        assert prev is None
        # Should have next
        assert next_item is not None
        # Should have path info
        assert path_info is not None
        assert path_info["path_id"] == "rag-engineer"
        assert path_info["is_first_in_path"] is True
    
    def test_path_info_structure(self, registry):
        """Path info should have correct structure."""
        first = registry.get_first_content_in_path("rag-engineer")
        _, _, path_info = registry.get_adjacent_content_for_path(
            "rag-engineer",
            first["module_id"],
            first["section_id"],
            first["id"]
        )
        
        assert "path_id" in path_info
        assert "path_title" in path_info
        assert "path_icon" in path_info
        assert "current_module_index" in path_info
        assert "total_modules" in path_info
        assert "is_first_in_path" in path_info
        assert "is_last_in_path" in path_info
        assert "current_item_index" in path_info
        assert "total_items" in path_info


class TestLabOperations:
    """Tests for lab-related operations."""
    
    @pytest.fixture
    def registry(self):
        return ContentRegistry.get_instance()
    
    def test_get_all_labs(self, registry):
        """Should get all labs."""
        labs = registry.get_all_labs()
        assert len(labs) > 0
    
    def test_lab_has_required_fields(self, registry):
        """Lab should have required fields."""
        labs = registry.get_all_labs()
        for lab in labs[:3]:  # Check first 3
            assert lab.id
            assert lab.title
            assert lab.difficulty in [Difficulty.BEGINNER, Difficulty.INTERMEDIATE, Difficulty.ADVANCED]


class TestNavigationData:
    """Tests for navigation data structure."""
    
    @pytest.fixture
    def registry(self):
        return ContentRegistry.get_instance()
    
    def test_get_navigation_data(self, registry):
        """Should get navigation data for sidebar."""
        nav_data = registry.get_navigation_data()
        assert "modules" in nav_data
        assert "learning_paths" in nav_data
        assert len(nav_data["modules"]) > 0
        assert len(nav_data["learning_paths"]) > 0
