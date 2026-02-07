"""
Pages router - Serves HTML pages for the tutorial website.
Uses ContentRegistry for dynamic content management.
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..content.registry import ContentRegistry
from ..content.renderer import MarkdownRenderer
from ..content.models import PageType

router = APIRouter()

# Set up templates
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Content directory for markdown files
CONTENT_DIR = BASE_DIR / "content"

# Initialize content systems
registry = ContentRegistry()
renderer = MarkdownRenderer()


@router.get("/learn/{module_id}", response_class=HTMLResponse)
async def module_overview(request: Request, module_id: str):
    """Render a module overview page."""
    module = registry.get_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")
    
    # Build navigation context
    all_modules = registry.get_all_modules()
    
    return templates.TemplateResponse(
        "content/module.html",
        {
            "request": request,
            "module": module,
            "title": f"{module.title} - AI Engineering Tutorial",
            "all_modules": all_modules,
            "breadcrumbs": [
                {"title": "Home", "url": "/"},
                {"title": module.title, "url": None},
            ],
        },
    )


@router.get("/learn/{module_id}/{section_id}", response_class=HTMLResponse)
async def section_page(request: Request, module_id: str, section_id: str):
    """Render a section page."""
    module = registry.get_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")
    
    section = registry.get_section(module_id, section_id)
    if not section:
        raise HTTPException(status_code=404, detail=f"Section '{section_id}' not found in module '{module_id}'")
    
    # Try to load and render markdown content
    content_path = CONTENT_DIR / module_id / f"{section_id}.md"
    rendered_content = None
    if content_path.exists():
        markdown_content = content_path.read_text(encoding="utf-8")
        rendered_content = renderer.render(markdown_content)
    
    # Build navigation context
    all_modules = registry.get_all_modules()
    
    # Get prev/next for navigation
    prev_section, next_section = registry.get_adjacent_sections(module_id, section_id)
    
    return templates.TemplateResponse(
        "content/lesson.html",
        {
            "request": request,
            "module": module,
            "section": section,
            "rendered_content": rendered_content,
            "title": f"{section.title} - {module.title}",
            "all_modules": all_modules,
            "prev_section": prev_section,
            "next_section": next_section,
            "breadcrumbs": [
                {"title": "Home", "url": "/"},
                {"title": module.title, "url": f"/learn/{module_id}"},
                {"title": section.title, "url": None},
            ],
        },
    )


@router.get("/learn/{module_id}/{section_id}/{page_id}", response_class=HTMLResponse)
async def content_page(request: Request, module_id: str, section_id: str, page_id: str, path: str = None):
    """Render a content page (lesson, lab, or quiz).
    
    Args:
        path: Optional learning path ID for path-aware navigation.
              When provided, navigation will cross module boundaries within the path.
    """
    from ..content.models import Page, Lab, Quiz
    
    module = registry.get_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")
    
    section = registry.get_section(module_id, section_id)
    if not section:
        raise HTTPException(status_code=404, detail=f"Section '{section_id}' not found")
    
    # First, try to find a lab directly
    lab = registry.get_lab_from_section(module_id, section_id, page_id)
    page = None
    quiz = None
    
    if not lab:
        # Try to find a page
        item = registry.get_page(module_id, section_id, page_id)
        if item:
            # get_page can return Page, Lab, or Quiz - check the type
            if isinstance(item, Lab):
                lab = item
            elif isinstance(item, Quiz):
                quiz = item
            elif isinstance(item, Page):
                page = item
        
        if not (page or lab or quiz):
            raise HTTPException(status_code=404, detail=f"Page '{page_id}' not found")
    
    # Determine content source and template
    content_path = None
    template_name = "content/lesson.html"
    
    if lab:
        # It's a lab - check if it should use the sandbox template
        if hasattr(lab, 'use_sandbox') and lab.use_sandbox:
            template_name = "content/sandbox-lab.html"
        else:
            template_name = "content/lab.html"
        if lab.content_path:
            content_path = CONTENT_DIR / lab.content_path
        else:
            content_path = CONTENT_DIR / module_id / section_id / f"{page_id}.md"
    elif quiz:
        # It's a quiz
        template_name = "content/quiz.html"
        content_path = CONTENT_DIR / module_id / section_id / f"{page_id}.md"
    elif page:
        # It's a page - check page_type for template
        if page.page_type == PageType.LAB:
            # Check if it should use the sandbox template
            if hasattr(page, 'use_sandbox') and page.use_sandbox:
                template_name = "content/sandbox-lab.html"
            else:
                template_name = "content/lab.html"
        elif page.page_type == PageType.QUIZ:
            template_name = "content/quiz.html"
        
        # Check for content_path field first, then fall back to convention
        if hasattr(page, 'content_path') and page.content_path:
            content_path = CONTENT_DIR / page.content_path
        else:
            content_path = CONTENT_DIR / module_id / section_id / f"{page_id}.md"
    
    # Try to load and render markdown content
    rendered_content = None
    if content_path and content_path.exists():
        markdown_content = content_path.read_text(encoding="utf-8")
        rendered_content = renderer.render(markdown_content)
    
    # Build navigation context
    all_modules = registry.get_all_modules()
    
    # Get previous/next content for navigation
    # If a learning path is specified, use path-aware navigation (cross-module)
    path_info = None
    if path:
        prev_content, next_content, path_info = registry.get_adjacent_content_for_path(
            path, module_id, section_id, page_id
        )
    else:
        prev_content, next_content = registry.get_adjacent_content(module_id, section_id, page_id)
    
    # Context based on page, lab, or quiz
    item = page or lab or quiz
    
    # Build breadcrumbs - include path context if navigating within a path
    breadcrumbs = [{"title": "Home", "url": "/"}]
    if path_info:
        breadcrumbs.append({"title": path_info['path_title'], "url": f"/path/{path}"})
    breadcrumbs.extend([
        {"title": module.title, "url": f"/learn/{module_id}"},
        {"title": section.title, "url": f"/learn/{module_id}/{section_id}"},
        {"title": item.title if item else "Content", "url": None},
    ])
    
    context = {
        "request": request,
        "module": module,
        "section": section,
        "rendered_content": rendered_content,
        "title": f"{item.title} - {module.title}" if item else f"Content - {module.title}",
        "all_modules": all_modules,
        "prev_content": prev_content,
        "next_content": next_content,
        "current_path": path,  # Pass path ID for navigation links
        "path_info": path_info,  # Pass path progress info
        "breadcrumbs": breadcrumbs,
    }
    
    if page:
        context["page"] = page
    if lab:
        context["lab"] = lab
    if quiz:
        context["quiz"] = quiz
    
    return templates.TemplateResponse(template_name, context)


@router.get("/lab/{lab_id}", response_class=HTMLResponse)
async def standalone_lab(request: Request, lab_id: str):
    """Render a standalone lab page."""
    lab = registry.get_lab(lab_id)
    if not lab:
        raise HTTPException(status_code=404, detail=f"Lab '{lab_id}' not found")
    
    # Get lab context for proper navigation
    lab_context = registry.get_lab_context(lab_id)
    module_id, section_id = lab_context if lab_context else (None, None)
    
    # Try to load lab content from the lab's content_path
    rendered_content = None
    if lab.content_path:
        content_path = CONTENT_DIR / lab.content_path
        if content_path.exists():
            markdown_content = content_path.read_text(encoding="utf-8")
            rendered_content = renderer.render(markdown_content)
    
    all_modules = registry.get_all_modules()
    
    # Build breadcrumbs with module/section context if available
    breadcrumbs = [{"title": "Home", "url": "/"}]
    if module_id and section_id:
        module = registry.get_module(module_id)
        section = registry.get_section(module_id, section_id)
        if module:
            breadcrumbs.append({"title": module.title, "url": f"/learn/{module_id}"})
        if section:
            breadcrumbs.append({"title": section.title, "url": f"/learn/{module_id}/{section_id}"})
    else:
        breadcrumbs.append({"title": "Labs", "url": "/labs"})
    breadcrumbs.append({"title": lab.title, "url": None})
    
    # Determine template based on sandbox setting
    template_name = "content/sandbox-lab.html" if (hasattr(lab, 'use_sandbox') and lab.use_sandbox) else "content/lab.html"
    
    return templates.TemplateResponse(
        template_name,
        {
            "request": request,
            "lab": lab,
            "rendered_content": rendered_content,
            "title": f"{lab.title} - Interactive Lab",
            "all_modules": all_modules,
            "module_id": module_id,
            "section_id": section_id,
            "breadcrumbs": breadcrumbs,
        },
    )


@router.get("/path/{path_id}", response_class=HTMLResponse)
async def learning_path_detail(request: Request, path_id: str):
    """Render a learning path detail page."""
    learning_path = registry.get_learning_path(path_id)
    if not learning_path:
        raise HTTPException(status_code=404, detail=f"Learning path '{path_id}' not found")
    
    # Get module details for the path
    path_modules = [registry.get_module(mid) for mid in learning_path.modules]
    path_modules = [m for m in path_modules if m]  # Filter None
    
    # Get the first content item for the "Start Path" button
    first_content = registry.get_first_content_in_path(path_id)
    
    # Get the full content list for progress tracking
    path_content = registry.get_path_content_list(path_id)
    
    all_modules = registry.get_all_modules()
    all_paths = registry.get_all_learning_paths()
    
    return templates.TemplateResponse(
        "content/learning-path.html",
        {
            "request": request,
            "learning_path": learning_path,
            "path_modules": path_modules,
            "first_content": first_content,
            "path_content": path_content,
            "title": f"{learning_path.title} - Learning Path",
            "all_modules": all_modules,
            "all_paths": all_paths,
            "breadcrumbs": [
                {"title": "Home", "url": "/"},
                {"title": "Learning Paths", "url": "/learning-paths"},
                {"title": learning_path.title, "url": None},
            ],
        },
    )


from fastapi.responses import JSONResponse

@router.get("/api/path/{path_id}/content")
async def get_path_content_api(path_id: str):
    """API endpoint to get all content items in a learning path."""
    learning_path = registry.get_learning_path(path_id)
    if not learning_path:
        raise HTTPException(status_code=404, detail=f"Learning path '{path_id}' not found")
    
    path_content = registry.get_path_content_list(path_id)
    
    return JSONResponse({
        "path_id": path_id,
        "path_title": learning_path.title,
        "modules": learning_path.modules,
        "content": path_content,
        "total_items": len(path_content),
    })


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Render the settings page for API key configuration."""
    all_modules = registry.get_all_modules()
    return templates.TemplateResponse(
        "pages/settings.html",
        {
            "request": request,
            "title": "Settings - API Keys",
            "all_modules": all_modules,
        },
    )


@router.get("/learning-paths", response_class=HTMLResponse)
async def learning_paths_page(request: Request):
    """Render the learning paths page."""
    all_modules = registry.get_all_modules()
    all_paths = registry.get_all_learning_paths()
    
    # Build a dict of modules for easy lookup in templates
    all_modules_dict = {m.id: m for m in all_modules}
    
    return templates.TemplateResponse(
        "pages/learning-paths.html",
        {
            "request": request,
            "title": "Learning Paths",
            "all_modules": all_modules,
            "all_modules_dict": all_modules_dict,
            "all_paths": all_paths,
        },
    )


@router.get("/glossary", response_class=HTMLResponse)
async def glossary_page(request: Request):
    """Render the glossary page."""
    all_modules = registry.get_all_modules()
    return templates.TemplateResponse(
        "pages/glossary.html",
        {
            "request": request,
            "title": "AI/ML Glossary",
            "all_modules": all_modules,
        },
    )


@router.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    """Render the about page."""
    all_modules = registry.get_all_modules()
    return templates.TemplateResponse(
        "pages/about.html",
        {
            "request": request,
            "title": "About",
            "all_modules": all_modules,
        },
    )


@router.get("/labs", response_class=HTMLResponse)
async def labs_index(request: Request):
    """Render the labs index page."""
    all_modules = registry.get_all_modules()
    all_labs = registry.get_all_labs()
    return templates.TemplateResponse(
        "pages/labs.html",
        {
            "request": request,
            "title": "Interactive Labs",
            "all_modules": all_modules,
            "all_labs": all_labs,
        },
    )


@router.get("/progress", response_class=HTMLResponse)
async def progress_page(request: Request):
    """Render the progress tracking page."""
    all_modules = registry.get_all_modules()
    all_paths = registry.get_all_learning_paths()
    return templates.TemplateResponse(
        "pages/progress.html",
        {
            "request": request,
            "title": "My Progress",
            "all_modules": all_modules,
            "all_paths": all_paths,
        },
    )


@router.get("/sandbox", response_class=HTMLResponse)
async def sandbox_demo(request: Request):
    """Render the sandbox demo page - for testing the live code sandbox."""
    from ..content.models import Page, PageType, Difficulty
    
    all_modules = registry.get_all_modules()
    
    # Create a demo page object for the sandbox
    demo_page = Page(
        id="sandbox-demo",
        title="Live Code Sandbox",
        slug="sandbox",
        page_type=PageType.LAB,
        description="Test the in-browser Python execution environment",
        use_sandbox=True,
        sandbox_preset="basic",
        starter_code='''# Welcome to the Live Code Sandbox! 🐍
# This Python code runs directly in your browser using Pyodide.

# Try some basic Python
print("Hello from the browser!")

# Math operations work
result = sum(range(1, 11))
print(f"Sum of 1 to 10: {result}")

# Lists and comprehensions
squares = [x**2 for x in range(1, 6)]
print(f"Squares: {squares}")

# Try the helper functions
print_section("Demo Section")
print_success("This is a success message!")
print_info("This is an info message!")

# You can also try different presets from the dropdown:
# - Data Science: numpy, pandas
# - LLM Concepts: mock OpenAI/Anthropic clients
# - RAG: vector similarity functions
# - Agents: tool registry and agent loop demos
''',
        difficulty=Difficulty.BEGINNER,
    )
    
    return templates.TemplateResponse(
        "content/sandbox-lab.html",
        {
            "request": request,
            "page": demo_page,
            "title": "Live Code Sandbox - AI Tutorial",
            "all_modules": all_modules,
            "breadcrumbs": [
                {"title": "Home", "url": "/"},
                {"title": "Sandbox", "url": None},
            ],
        },
    )


@router.get("/tools/token-visualizer", response_class=HTMLResponse)
async def token_visualizer(request: Request):
    """Render the Token Visualizer tool page."""
    all_modules = registry.get_all_modules()
    
    return templates.TemplateResponse(
        "pages/token-visualizer.html",
        {
            "request": request,
            "title": "Token Visualizer - AI Tutorial",
            "all_modules": all_modules,
            "breadcrumbs": [
                {"title": "Home", "url": "/"},
                {"title": "Tools", "url": "/tools"},
                {"title": "Token Visualizer", "url": None},
            ],
        },
    )


@router.get("/tools/embedding-playground", response_class=HTMLResponse)
async def embedding_playground(request: Request):
    """Render the Embedding Playground tool page."""
    all_modules = registry.get_all_modules()
    
    return templates.TemplateResponse(
        "pages/embedding-playground.html",
        {
            "request": request,
            "title": "Embedding Playground - AI Tutorial",
            "all_modules": all_modules,
            "breadcrumbs": [
                {"title": "Home", "url": "/"},
                {"title": "Tools", "url": "/tools"},
                {"title": "Embedding Playground", "url": None},
            ],
        },
    )


@router.get("/tools/comparison", response_class=HTMLResponse)
async def provider_comparison(request: Request):
    """Render the Provider Comparison tool page."""
    all_modules = registry.get_all_modules()
    
    return templates.TemplateResponse(
        "pages/comparison.html",
        {
            "request": request,
            "title": "Provider Comparison - AI Tutorial",
            "all_modules": all_modules,
            "breadcrumbs": [
                {"title": "Home", "url": "/"},
                {"title": "Tools", "url": "/tools"},
                {"title": "Provider Comparison", "url": None},
            ],
        },
    )


@router.get("/tools/cost-calculator", response_class=HTMLResponse)
async def cost_calculator(request: Request):
    """Render the Cost Calculator tool page."""
    all_modules = registry.get_all_modules()
    
    return templates.TemplateResponse(
        "pages/cost-calculator.html",
        {
            "request": request,
            "title": "Cost Calculator - AI Tutorial",
            "all_modules": all_modules,
            "breadcrumbs": [
                {"title": "Home", "url": "/"},
                {"title": "Tools", "url": "/tools"},
                {"title": "Cost Calculator", "url": None},
            ],
        },
    )


@router.get("/tools", response_class=HTMLResponse)
async def tools_index(request: Request):
    """Render the Tools index page."""
    all_modules = registry.get_all_modules()
    
    return templates.TemplateResponse(
        "pages/tools.html",
        {
            "request": request,
            "title": "Interactive Tools - AI Tutorial",
            "all_modules": all_modules,
            "breadcrumbs": [
                {"title": "Home", "url": "/"},
                {"title": "Tools", "url": None},
            ],
        },
    )


@router.get("/quizzes", response_class=HTMLResponse)
async def quizzes_index(request: Request):
    """Render the Knowledge Quizzes index page."""
    all_modules = registry.get_all_modules()
    
    # All available quizzes
    quizzes = [
        {
            "id": "llmBasics",
            "title": "LLM Fundamentals Quiz",
            "description": "Test your knowledge of Large Language Models basics including architecture, tokenization, and capabilities.",
            "icon": "🧠",
            "category": "fundamentals",
            "question_count": 5,
            "time_estimate": "5-10",
            "topics": ["Transformers", "Tokens", "Parameters", "Context Windows"]
        },
        {
            "id": "rag",
            "title": "RAG Knowledge Check",
            "description": "Test your understanding of Retrieval-Augmented Generation concepts and implementation.",
            "icon": "📚",
            "category": "intermediate",
            "question_count": 5,
            "time_estimate": "5-10",
            "topics": ["Embeddings", "Vector Search", "Chunking", "Retrieval"]
        },
        {
            "id": "agents",
            "title": "AI Agents Quiz",
            "description": "Test your knowledge of AI Agents, tool use, and autonomous decision-making.",
            "icon": "🤖",
            "category": "advanced",
            "question_count": 5,
            "time_estimate": "5-10",
            "topics": ["Tool Calling", "ReAct", "Planning", "Memory"]
        },
        {
            "id": "embeddings",
            "title": "Embeddings & Vectors Quiz",
            "description": "Test your understanding of embeddings, vector databases, and similarity search.",
            "icon": "🔢",
            "category": "intermediate",
            "question_count": 5,
            "time_estimate": "5-10",
            "topics": ["Vector Math", "Embedding Models", "Similarity", "Indexing"]
        },
        {
            "id": "finetuning",
            "title": "Fine-tuning Fundamentals",
            "description": "Test your knowledge of fine-tuning LLMs, including LoRA, datasets, and best practices.",
            "icon": "🎯",
            "category": "advanced",
            "question_count": 5,
            "time_estimate": "5-10",
            "topics": ["LoRA", "Dataset Prep", "Training", "Evaluation"]
        },
        {
            "id": "prompting",
            "title": "Prompt Engineering Quiz",
            "description": "Test your skills in crafting effective prompts and understanding prompting techniques.",
            "icon": "✍️",
            "category": "fundamentals",
            "question_count": 5,
            "time_estimate": "5-10",
            "topics": ["Few-shot", "Chain-of-Thought", "System Prompts", "Formatting"]
        }
    ]
    
    return templates.TemplateResponse(
        "pages/quizzes.html",
        {
            "request": request,
            "title": "Knowledge Quizzes - AI Tutorial",
            "quizzes": quizzes,
            "all_modules": all_modules,
            "breadcrumbs": [
                {"title": "Home", "url": "/"},
                {"title": "Quizzes", "url": None},
            ],
        },
    )


@router.get("/quiz/{quiz_type}", response_class=HTMLResponse)
async def quiz_page(request: Request, quiz_type: str):
    """Render a quiz page."""
    all_modules = registry.get_all_modules()
    
    # Quiz metadata
    quiz_info = {
        "llmBasics": {
            "title": "LLM Fundamentals Quiz",
            "description": "Test your knowledge of Large Language Models basics",
            "question_count": 5,
            "time_estimate": "5-10"
        },
        "rag": {
            "title": "RAG Knowledge Check",
            "description": "Test your understanding of Retrieval-Augmented Generation",
            "question_count": 5,
            "time_estimate": "5-10"
        },
        "agents": {
            "title": "AI Agents Quiz",
            "description": "Test your knowledge of AI Agents and tool use",
            "question_count": 5,
            "time_estimate": "5-10"
        }
    }
    
    info = quiz_info.get(quiz_type, {
        "title": "Knowledge Check",
        "description": "Test your understanding",
        "question_count": 5,
        "time_estimate": "5-10"
    })
    
    return templates.TemplateResponse(
        "pages/quiz.html",
        {
            "request": request,
            "title": f"{info['title']} - AI Tutorial",
            "quiz_title": info["title"],
            "quiz_description": info["description"],
            "question_count": info["question_count"],
            "time_estimate": info["time_estimate"],
            "quiz_type": quiz_type,
            "quiz_id": quiz_type,
            "passing_score": 70,
            "all_modules": all_modules,
            "breadcrumbs": [
                {"title": "Home", "url": "/"},
                {"title": "Quiz", "url": None},
            ],
        },
    )


@router.get("/challenges", response_class=HTMLResponse)
async def challenges_index(request: Request):
    """Render the Coding Challenges index page."""
    all_modules = registry.get_all_modules()
    
    return templates.TemplateResponse(
        "pages/challenges.html",
        {
            "request": request,
            "title": "Coding Challenges - AI Tutorial",
            "all_modules": all_modules,
            "breadcrumbs": [
                {"title": "Home", "url": "/"},
                {"title": "Challenges", "url": None},
            ],
        },
    )


@router.get("/challenge/{challenge_type}", response_class=HTMLResponse)
async def challenge_page(request: Request, challenge_type: str):
    """Render a coding challenge page."""
    all_modules = registry.get_all_modules()
    
    # Challenge metadata
    challenge_info = {
        "twoSum": "Two Sum",
        "reverseString": "Reverse String",
        "countTokens": "Simple Token Counter",
        "cosineSimilarity": "Cosine Similarity"
    }
    
    title = challenge_info.get(challenge_type, "Coding Challenge")
    
    return templates.TemplateResponse(
        "pages/challenge.html",
        {
            "request": request,
            "title": f"{title} - AI Tutorial",
            "challenge_type": challenge_type,
            "all_modules": all_modules,
            "breadcrumbs": [
                {"title": "Home", "url": "/"},
                {"title": "Challenges", "url": "/challenges"},
                {"title": title, "url": None},
            ],
        },
    )


@router.get("/achievements", response_class=HTMLResponse)
async def achievements_page(request: Request):
    """Render the Achievements & Progress page."""
    all_modules = registry.get_all_modules()
    
    return templates.TemplateResponse(
        "pages/achievements.html",
        {
            "request": request,
            "title": "Achievements - AI Tutorial",
            "all_modules": all_modules,
            "breadcrumbs": [
                {"title": "Home", "url": "/"},
                {"title": "Achievements", "url": None},
            ],
        },
    )


@router.get("/case-studies", response_class=HTMLResponse)
async def case_studies_page(request: Request):
    """Render the Case Studies page."""
    all_modules = registry.get_all_modules()
    
    return templates.TemplateResponse(
        "pages/case-studies.html",
        {
            "request": request,
            "title": "Case Studies - AI Tutorial",
            "all_modules": all_modules,
            "breadcrumbs": [
                {"title": "Home", "url": "/"},
                {"title": "Case Studies", "url": None},
            ],
        },
    )


@router.get("/decision-guides", response_class=HTMLResponse)
async def decision_guides_page(request: Request):
    """Render the Decision Guides page."""
    all_modules = registry.get_all_modules()
    
    return templates.TemplateResponse(
        "pages/decision-guides.html",
        {
            "request": request,
            "title": "Decision Guides - AI Tutorial",
            "all_modules": all_modules,
            "breadcrumbs": [
                {"title": "Home", "url": "/"},
                {"title": "Decision Guides", "url": None},
            ],
        },
    )


@router.get("/failure-modes", response_class=HTMLResponse)
async def failure_modes_page(request: Request):
    """Render the Failure Modes Gallery page."""
    all_modules = registry.get_all_modules()
    
    return templates.TemplateResponse(
        "pages/failure-modes.html",
        {
            "request": request,
            "title": "Failure Modes Gallery - AI Tutorial",
            "all_modules": all_modules,
            "breadcrumbs": [
                {"title": "Home", "url": "/"},
                {"title": "Failure Modes", "url": None},
            ],
        },
    )
