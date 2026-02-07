"""
Labs router - Handles lab execution endpoints.
"""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class LabExecuteRequest(BaseModel):
    """Request model for lab execution."""
    lab_id: str
    code: str
    provider: str
    api_key: str
    parameters: Optional[dict] = None


class LabExecuteResponse(BaseModel):
    """Response model for lab execution."""
    lab_id: str
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None


# Available labs organized by module
LABS = {
    "llm": {
        "first_api_call": {
            "title": "First API Call",
            "description": "Make your first call to an LLM API",
            "difficulty": "beginner",
        },
        "prompt_playground": {
            "title": "Prompt Playground",
            "description": "Experiment with different prompting techniques",
            "difficulty": "beginner",
        },
        "parameter_explorer": {
            "title": "Parameter Explorer",
            "description": "Understand temperature, top-p, and other parameters",
            "difficulty": "beginner",
        },
    },
    "rag": {
        "document_processor": {
            "title": "Document Processor",
            "description": "Build a document processing pipeline",
            "difficulty": "intermediate",
        },
        "rag_chatbot": {
            "title": "RAG Chatbot",
            "description": "Build a complete RAG chatbot",
            "difficulty": "intermediate",
        },
    },
    "agents": {
        "react_agent": {
            "title": "ReAct Agent",
            "description": "Build a ReAct agent from scratch",
            "difficulty": "advanced",
        },
        "research_agent": {
            "title": "Research Agent",
            "description": "Build an agent that can research topics",
            "difficulty": "advanced",
        },
    },
}


@router.get("/")
async def list_labs():
    """List all available labs."""
    return {"labs": LABS}


@router.get("/{module}")
async def list_module_labs(module: str):
    """List labs for a specific module."""
    if module not in LABS:
        raise HTTPException(status_code=404, detail=f"Module '{module}' not found")
    return {"module": module, "labs": LABS[module]}


@router.get("/{module}/{lab_id}")
async def get_lab_info(module: str, lab_id: str):
    """Get information about a specific lab."""
    if module not in LABS:
        raise HTTPException(status_code=404, detail=f"Module '{module}' not found")
    if lab_id not in LABS[module]:
        raise HTTPException(status_code=404, detail=f"Lab '{lab_id}' not found in module '{module}'")
    
    return {
        "module": module,
        "lab_id": lab_id,
        **LABS[module][lab_id],
    }


@router.post("/execute", response_model=LabExecuteResponse)
async def execute_lab(request: LabExecuteRequest):
    """Execute a lab with user-provided code."""
    # This will be fully implemented in later phases
    # For now, return a placeholder response
    return LabExecuteResponse(
        lab_id=request.lab_id,
        success=True,
        output="Lab execution will be fully implemented in Phase 4+",
        execution_time_ms=0,
    )
