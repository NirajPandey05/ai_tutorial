"""
AI Engineering Tutorial - Main FastAPI Application

This is the entry point for the AI Engineering Tutorial website.
Run with: uvicorn src.ai_tutorial.main:app --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import settings
from .routers import api, labs, pages, export
from .content.registry import ContentRegistry

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Initialize content registry
registry = ContentRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    # Startup
    print(f"🚀 Starting AI Engineering Tutorial v{settings.APP_VERSION}")
    print(f"📚 Debug mode: {settings.DEBUG}")
    yield
    # Shutdown
    print("👋 Shutting down AI Engineering Tutorial")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Interactive AI Engineering Tutorial - Learn LLMs, RAG, Agents, MCP and more",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/api/docs" if settings.DEBUG else None,
    redoc_url="/api/redoc" if settings.DEBUG else None,
)

# Mount static files
static_path = BASE_DIR / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Set up Jinja2 templates
templates_path = BASE_DIR / "templates"
templates = Jinja2Templates(directory=str(templates_path))

# Include routers
app.include_router(pages.router, tags=["Pages"])
app.include_router(api.router, prefix="/api", tags=["API"])
app.include_router(labs.router, prefix="/api/labs", tags=["Labs"])
app.include_router(export.router, prefix="/api/export", tags=["Export"])


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page."""
    all_modules = registry.get_all_modules()
    all_paths = registry.get_all_learning_paths()
    
    return templates.TemplateResponse(
        "pages/index.html",
        {
            "request": request,
            "title": "AI Engineering Tutorial",
            "description": "Learn AI Engineering from fundamentals to advanced concepts",
            "all_modules": all_modules,
            "all_paths": all_paths,
        },
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


# Export for uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.ai_tutorial.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
