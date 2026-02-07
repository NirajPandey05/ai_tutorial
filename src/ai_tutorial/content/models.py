"""
Content Models - Data structures for the tutorial content.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class PageType(str, Enum):
    """Type of content page."""
    LESSON = "lesson"
    LAB = "lab"
    QUIZ = "quiz"
    REFERENCE = "reference"
    OVERVIEW = "overview"


class Difficulty(str, Enum):
    """Difficulty level of content."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class Page:
    """A single page of content within a section."""
    id: str
    title: str
    slug: str
    page_type: PageType = PageType.LESSON
    description: str = ""
    content_file: str = ""  # Path to markdown file (deprecated)
    content_path: str = ""  # Path to markdown file relative to content/
    estimated_time: int = 5  # Minutes
    difficulty: Difficulty = Difficulty.BEGINNER
    prerequisites: List[str] = field(default_factory=list)  # Page IDs
    related_pages: List[str] = field(default_factory=list)  # Page IDs
    tags: List[str] = field(default_factory=list)
    order: int = 0
    # Sandbox settings for labs
    sandbox_preset: str = "basic"  # Preset for Pyodide sandbox: basic, dataScience, llmBasics, rag, etc.
    use_sandbox: bool = False  # Whether to use client-side Pyodide sandbox
    starter_code: str = ""  # Starter code for lab exercises
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "slug": self.slug,
            "page_type": self.page_type.value,
            "description": self.description,
            "content_path": self.content_path or self.content_file,
            "estimated_time": self.estimated_time,
            "difficulty": self.difficulty.value,
            "prerequisites": self.prerequisites,
            "related_pages": self.related_pages,
            "tags": self.tags,
            "order": self.order,
            "sandbox_preset": self.sandbox_preset,
            "use_sandbox": self.use_sandbox,
            "starter_code": self.starter_code,
        }


@dataclass
class Lab:
    """An interactive lab exercise."""
    id: str
    title: str
    slug: str
    description: str = ""
    instructions: str = ""  # Markdown content (inline)
    content_path: str = ""  # Path to markdown file relative to content/
    starter_code: str = ""
    solution_code: str = ""
    estimated_time: int = 15  # Minutes
    difficulty: Difficulty = Difficulty.BEGINNER
    provider_required: bool = True  # Requires API key
    supported_providers: List[str] = field(default_factory=lambda: ["openai", "anthropic", "google", "xai"])
    hints: List[str] = field(default_factory=list)
    success_criteria: str = ""
    order: int = 0
    # Sandbox settings
    sandbox_preset: str = "basic"  # Preset for Pyodide sandbox
    use_sandbox: bool = True  # Whether to use client-side Pyodide sandbox
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "slug": self.slug,
            "description": self.description,
            "content_path": self.content_path,
            "estimated_time": self.estimated_time,
            "difficulty": self.difficulty.value,
            "provider_required": self.provider_required,
            "supported_providers": self.supported_providers,
            "order": self.order,
            "sandbox_preset": self.sandbox_preset,
            "use_sandbox": self.use_sandbox,
        }


@dataclass
class Quiz:
    """A quiz for testing understanding."""
    id: str
    title: str
    slug: str
    description: str = ""
    questions: List[Dict[str, Any]] = field(default_factory=list)
    passing_score: int = 70  # Percentage
    estimated_time: int = 10  # Minutes
    order: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "slug": self.slug,
            "description": self.description,
            "question_count": len(self.questions),
            "passing_score": self.passing_score,
            "estimated_time": self.estimated_time,
            "order": self.order,
        }


@dataclass
class Section:
    """A section within a module containing pages, labs, and quizzes."""
    id: str
    title: str
    slug: str
    description: str = ""
    icon: str = "📖"
    pages: List[Page] = field(default_factory=list)
    labs: List[Lab] = field(default_factory=list)
    quizzes: List[Quiz] = field(default_factory=list)
    order: int = 0
    
    @property
    def total_time(self) -> int:
        """Calculate total estimated time for section."""
        time = sum(p.estimated_time for p in self.pages)
        time += sum(l.estimated_time for l in self.labs)
        time += sum(q.estimated_time for q in self.quizzes)
        return time
    
    @property
    def total_items(self) -> int:
        """Total number of content items."""
        return len(self.pages) + len(self.labs) + len(self.quizzes)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "slug": self.slug,
            "description": self.description,
            "icon": self.icon,
            "pages": [p.to_dict() for p in self.pages],
            "labs": [l.to_dict() for l in self.labs],
            "quizzes": [q.to_dict() for q in self.quizzes],
            "total_time": self.total_time,
            "total_items": self.total_items,
            "order": self.order,
        }


@dataclass
class Module:
    """A major learning module containing multiple sections."""
    id: str
    title: str
    slug: str
    description: str = ""
    icon: str = "📚"
    color: str = "primary"  # Tailwind color name
    sections: List[Section] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)  # Module IDs
    difficulty: Difficulty = Difficulty.BEGINNER
    order: int = 0
    is_coming_soon: bool = False
    
    @property
    def total_time(self) -> int:
        """Calculate total estimated time for module."""
        return sum(s.total_time for s in self.sections)
    
    @property
    def total_items(self) -> int:
        """Total number of content items."""
        return sum(s.total_items for s in self.sections)
    
    @property
    def total_sections(self) -> int:
        return len(self.sections)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "slug": self.slug,
            "description": self.description,
            "icon": self.icon,
            "color": self.color,
            "sections": [s.to_dict() for s in self.sections],
            "prerequisites": self.prerequisites,
            "difficulty": self.difficulty.value,
            "total_time": self.total_time,
            "total_items": self.total_items,
            "total_sections": self.total_sections,
            "order": self.order,
            "is_coming_soon": self.is_coming_soon,
        }


@dataclass
class LearningPath:
    """A curated learning path through modules."""
    id: str
    title: str
    slug: str
    description: str = ""
    icon: str = "🎯"
    modules: List[str] = field(default_factory=list)  # Module IDs in order
    estimated_hours: float = 0
    target_audience: str = ""
    outcomes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "slug": self.slug,
            "description": self.description,
            "icon": self.icon,
            "modules": self.modules,
            "estimated_hours": self.estimated_hours,
            "target_audience": self.target_audience,
            "outcomes": self.outcomes,
        }
