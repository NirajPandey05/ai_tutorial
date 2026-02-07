# Contributing to AI Engineering Tutorial

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Adding Content](#adding-content)
- [Testing](#testing)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Be kind, constructive, and professional in all interactions.

## Getting Started

### Prerequisites

- Python 3.11+
- UV package manager
- Git
- A code editor (VS Code recommended)

### Development Setup

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ai-tutorial.git
   cd ai-tutorial
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

4. **Install dev dependencies**:
   ```bash
   uv pip install -e ".[dev]"
   ```

5. **Run the application**:
   ```bash
   uv run uvicorn src.ai_tutorial.main:app --reload --port 8080
   ```

6. **Run tests** to verify setup:
   ```bash
   uv run pytest
   ```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-agent-memory-lab` - New features
- `fix/navigation-bug` - Bug fixes
- `docs/update-readme` - Documentation
- `refactor/provider-abstraction` - Code refactoring

### Commit Messages

Follow conventional commit format:
```
type(scope): description

[optional body]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
- `feat(labs): add RAG evaluation lab`
- `fix(navigation): correct path-aware prev/next links`
- `docs(readme): update installation instructions`

## Pull Request Process

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature
   ```

3. **Make your changes** and commit them

4. **Run tests and linting**:
   ```bash
   uv run pytest
   uv run black src/ tests/
   uv run ruff check src/ tests/
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature
   ```

6. **Open a Pull Request** on GitHub

### PR Requirements

- [ ] Tests pass (`uv run pytest`)
- [ ] Code is formatted (`uv run black src/`)
- [ ] No linting errors (`uv run ruff check src/`)
- [ ] Documentation updated if needed
- [ ] Descriptive PR title and description

## Coding Standards

### Python Style

- Follow PEP 8 guidelines
- Use Black for formatting (line length: 100)
- Use Ruff for linting
- Add type hints where practical
- Write docstrings for public functions/classes

```python
def process_content(content: str, options: dict = None) -> str:
    """Process content with optional transformations.
    
    Args:
        content: The raw content to process
        options: Optional processing options
        
    Returns:
        Processed content string
    """
    ...
```

### JavaScript Style

- Use ES6+ features
- Prefer `const` over `let`, avoid `var`
- Use template literals for string formatting
- Document complex functions

### HTML/Templates

- Use semantic HTML5 elements
- Include ARIA labels for accessibility
- Follow Tailwind CSS conventions
- Keep templates focused and readable

## Adding Content

### Adding a New Lesson

1. **Create the markdown file** in `content/{module}/{section}/`:
   ```markdown
   # Lesson Title
   
   Introduction paragraph...
   
   ## Section 1
   
   Content...
   
   ## Summary
   
   Key takeaways...
   ```

2. **Register the page** in `src/ai_tutorial/content/registry.py`:
   ```python
   Page(
       id="lesson-id",
       title="Lesson Title",
       slug="lesson-id",
       description="Brief description",
       estimated_time=10,
       order=1,
   )
   ```

3. **Add tests** if applicable

### Adding a New Lab

1. **Create the lab content** in `content/{module}/{section}/`:
   ```markdown
   # Lab: Hands-on Exercise
   
   ## Objective
   What you'll learn...
   
   ## Instructions
   Step-by-step guide...
   
   ## Code
   \```python
   # Starter code
   \```
   
   ## Challenge
   Additional exercises...
   ```

2. **Register the lab** in registry:
   ```python
   Lab(
       id="lab-id",
       title="Lab Title",
       slug="lab-id",
       description="What you'll build",
       difficulty=Difficulty.INTERMEDIATE,
       estimated_time=20,
       order=1,
   )
   ```

### Adding a New Module

1. Create directory structure under `content/`
2. Add module registration in `registry.py`
3. Add sections and pages
4. Update learning paths if applicable
5. Add tests for new content

## Testing

### Running Tests

```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov=src/ai_tutorial --cov-report=html

# Specific test file
uv run pytest tests/test_routes.py

# Verbose output
uv run pytest -v

# Stop on first failure
uv run pytest -x
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use pytest fixtures for setup
- Mock external dependencies

```python
def test_module_retrieval():
    """Test that modules can be retrieved by ID."""
    registry = ContentRegistry.get_instance()
    module = registry.get_module("llm-fundamentals")
    
    assert module is not None
    assert module.id == "llm-fundamentals"
    assert len(module.sections) > 0
```

### Test Categories

- **Unit tests**: Test individual functions/classes
- **Integration tests**: Test component interactions
- **Route tests**: Test API endpoints

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Tag maintainers for urgent issues

---

Thank you for contributing! 🙏
