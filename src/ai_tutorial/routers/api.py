"""
API router - Handles API endpoints for the tutorial.
"""

import json
import logging
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..providers import (
    ProviderFactory,
    LLMMessage,
    MessageRole,
    AuthenticationError,
    RateLimitError,
    ProviderError,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class ProgressSyncRequest(BaseModel):
    """Request model for progress sync."""
    progress: Dict[str, Any]


class ProgressSyncResponse(BaseModel):
    """Response model for progress sync."""
    success: bool
    message: str


class APIKeyValidation(BaseModel):
    """Request model for API key validation."""
    provider: str
    api_key: str


class APIKeyResponse(BaseModel):
    """Response model for API key validation."""
    valid: bool
    provider: str
    message: str
    model_info: Optional[dict] = None


class Message(BaseModel):
    """A single message in a conversation."""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Request model for chat completion."""
    provider: str
    model: str
    messages: List[Message]
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = False


class ChatResponse(BaseModel):
    """Response model for chat completion."""
    content: str
    provider: str
    model: str
    usage: Optional[dict] = None


def _convert_messages(messages: List[Message]) -> List[LLMMessage]:
    """Convert API messages to LLMMessage format."""
    result = []
    for msg in messages:
        role = MessageRole(msg.role)
        result.append(LLMMessage(role=role, content=msg.content))
    return result


@router.post("/validate-key", response_model=APIKeyResponse)
async def validate_api_key(request: APIKeyValidation):
    """Validate an API key for a specific provider."""
    try:
        provider = ProviderFactory.create(
            name=request.provider,
            api_key=request.api_key,
        )
        is_valid, message = await provider.validate_api_key()
        
        return APIKeyResponse(
            valid=is_valid,
            provider=request.provider,
            message=message,
            model_info={
                "default_model": provider.default_model,
                "available_models": provider.available_models,
            } if is_valid else None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error validating API key: {e}")
        return APIKeyResponse(
            valid=False,
            provider=request.provider,
            message=f"Error validating key: {str(e)}",
        )


@router.post("/chat")
async def chat_completion(request: ChatRequest):
    """Send a chat completion request to the specified provider."""
    try:
        provider = ProviderFactory.create(
            name=request.provider,
            api_key=request.api_key,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        
        messages = _convert_messages(request.messages)
        
        if request.stream:
            # Return streaming response
            async def generate():
                try:
                    async for chunk in provider.stream_chat(messages):
                        data = {
                            "content": chunk.content,
                            "is_final": chunk.is_final,
                            "finish_reason": chunk.finish_reason,
                            "usage": chunk.usage,
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                except AuthenticationError as e:
                    yield f"data: {json.dumps({'error': 'Invalid API key', 'details': str(e)})}\n\n"
                except RateLimitError as e:
                    yield f"data: {json.dumps({'error': 'Rate limit exceeded', 'details': str(e)})}\n\n"
                except ProviderError as e:
                    yield f"data: {json.dumps({'error': 'Provider error', 'details': str(e)})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': 'Unknown error', 'details': str(e)})}\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # Non-streaming response
            response = await provider.chat(messages)
            return ChatResponse(
                content=response.content,
                provider=response.provider,
                model=response.model,
                usage=response.usage,
            )
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=f"Invalid API key: {e.message}")
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {e.message}")
    except ProviderError as e:
        raise HTTPException(status_code=500, detail=f"Provider error: {e.message}")
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/providers")
async def list_providers():
    """List available LLM providers and their models."""
    providers_info = ProviderFactory.list_providers()
    
    providers_list = []
    for provider_id, info in providers_info.items():
        providers_list.append({
            "id": provider_id,
            "name": info["display_name"],
            "models": info["available_models"],
            "default_model": info["default_model"],
            "supports_streaming": info["supports_streaming"],
            "supports_system_message": info["supports_system_message"],
        })
    
    return {"providers": providers_list}


@router.get("/progress")
async def get_progress():
    """Get user progress through the tutorial."""
    # This will be implemented later with proper storage
    return {
        "completed_sections": [],
        "current_section": None,
        "total_sections": 10,
        "percentage": 0,
    }


@router.post("/progress/{section}/{page}")
async def update_progress(section: str, page: str, completed: bool = True):
    """Update user progress for a specific page."""
    # This will be implemented later with proper storage
    return {
        "section": section,
        "page": page,
        "completed": completed,
        "message": "Progress tracking will be fully implemented in Phase 16",
    }


class LabExecuteRequest(BaseModel):
    """Request model for lab code execution."""
    code: str
    provider: str
    lab_id: str
    api_key: Optional[str] = None


class LabExecuteResponse(BaseModel):
    """Response model for lab code execution."""
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    completed: bool = False


@router.post("/lab/execute", response_model=LabExecuteResponse)
async def execute_lab_code(request: LabExecuteRequest):
    """Execute lab code with LLM integration.
    
    This endpoint provides a sandboxed environment for running lab exercises.
    The code is not actually executed - instead, we parse the intent and make
    appropriate API calls.
    """
    try:
        # Get API key from request or return error
        if not request.api_key:
            return LabExecuteResponse(
                success=False,
                error="API key required. Please configure your API key in Settings.",
            )
        
        # For the "first-api-call" lab, we intercept the code and run it safely
        if request.lab_id in ["first-api-call", "first_api_call"]:
            return await _execute_first_api_call_lab(request)
        
        # For prompt playground lab
        if request.lab_id in ["prompt-playground", "prompt_playground"]:
            return await _execute_prompt_lab(request)
        
        # For compare prompts lab
        if request.lab_id in ["compare-prompts", "compare_prompts"]:
            return await _execute_compare_prompts_lab(request)
        
        # Default: extract message from code and make API call
        return await _execute_generic_lab(request)
        
    except AuthenticationError as e:
        return LabExecuteResponse(
            success=False,
            error=f"Authentication failed: {e.message}",
        )
    except RateLimitError as e:
        return LabExecuteResponse(
            success=False,
            error=f"Rate limit exceeded: {e.message}. Please wait and try again.",
        )
    except ProviderError as e:
        return LabExecuteResponse(
            success=False,
            error=f"Provider error: {e.message}",
        )
    except Exception as e:
        logger.error(f"Lab execution error: {e}")
        return LabExecuteResponse(
            success=False,
            error=f"Error: {str(e)}",
        )


async def _execute_first_api_call_lab(request: LabExecuteRequest) -> LabExecuteResponse:
    """Execute the first API call lab."""
    import re
    
    # Extract the question from the code
    question_match = re.search(r'question\s*=\s*["\'](.+?)["\']', request.code)
    if question_match:
        question = question_match.group(1)
    else:
        # Try to find any string that looks like a prompt
        string_match = re.search(r'["\']([^"\']{10,})["\']', request.code)
        question = string_match.group(1) if string_match else "Hello! Can you introduce yourself?"
    
    # Create provider and make the call
    provider = ProviderFactory.create(
        name=request.provider,
        api_key=request.api_key,
        temperature=0.7,
        max_tokens=500,
    )
    
    messages = [LLMMessage(role=MessageRole.USER, content=question)]
    response = await provider.chat(messages)
    
    output = f"Question: {question}\n\nAnswer: {response.content}\n\n"
    output += f"---\nModel: {response.model}\n"
    if response.usage:
        output += f"Tokens used: {response.usage.get('total_tokens', 'N/A')}"
    
    return LabExecuteResponse(
        success=True,
        output=output,
        completed=True,
    )


async def _execute_prompt_lab(request: LabExecuteRequest) -> LabExecuteResponse:
    """Execute the prompt playground lab."""
    import re
    
    # Extract prompt and any system message
    prompt_match = re.search(r'prompt\s*=\s*["\'](.+?)["\']', request.code, re.DOTALL)
    system_match = re.search(r'system\s*=\s*["\'](.+?)["\']', request.code, re.DOTALL)
    
    prompt = prompt_match.group(1) if prompt_match else "Explain machine learning"
    system = system_match.group(1) if system_match else None
    
    provider = ProviderFactory.create(
        name=request.provider,
        api_key=request.api_key,
        temperature=0.7,
        max_tokens=800,
    )
    
    messages = []
    if system:
        messages.append(LLMMessage(role=MessageRole.SYSTEM, content=system))
    messages.append(LLMMessage(role=MessageRole.USER, content=prompt))
    
    response = await provider.chat(messages)
    
    output = f"Prompt: {prompt}\n"
    if system:
        output += f"System: {system}\n"
    output += f"\n{'-'*50}\n\nResponse:\n{response.content}"
    
    return LabExecuteResponse(
        success=True,
        output=output,
        completed=True,
    )


async def _execute_compare_prompts_lab(request: LabExecuteRequest) -> LabExecuteResponse:
    """Execute the compare prompts lab."""
    import re
    
    prompt_match = re.search(r'prompt\s*=\s*["\'](.+?)["\']', request.code, re.DOTALL)
    prompt = prompt_match.group(1) if prompt_match else "What is AI?"
    
    provider = ProviderFactory.create(
        name=request.provider,
        api_key=request.api_key,
        temperature=0.7,
        max_tokens=500,
    )
    
    messages = [LLMMessage(role=MessageRole.USER, content=prompt)]
    response = await provider.chat(messages)
    
    output = f"Provider: {request.provider.upper()}\n"
    output += f"Model: {response.model}\n"
    output += f"Prompt: {prompt}\n"
    output += f"\n{'-'*50}\n\nResponse:\n{response.content}"
    
    return LabExecuteResponse(
        success=True,
        output=output,
        completed=True,
    )


async def _execute_generic_lab(request: LabExecuteRequest) -> LabExecuteResponse:
    """Execute a generic lab with code containing a prompt."""
    import re
    
    # Try to extract any reasonable prompt from the code
    patterns = [
        r'prompt\s*=\s*["\'](.+?)["\']',
        r'question\s*=\s*["\'](.+?)["\']',
        r'message\s*=\s*["\'](.+?)["\']',
        r'text\s*=\s*["\'](.+?)["\']',
        r'["\']([^"\']{15,})["\']',  # Any string > 15 chars
    ]
    
    prompt = None
    for pattern in patterns:
        match = re.search(pattern, request.code, re.DOTALL)
        if match:
            prompt = match.group(1)
            break
    
    if not prompt:
        return LabExecuteResponse(
            success=False,
            error="Could not find a prompt in your code. Make sure you have a variable like 'prompt = \"your question\"'",
        )
    
    provider = ProviderFactory.create(
        name=request.provider,
        api_key=request.api_key,
        temperature=0.7,
        max_tokens=800,
    )
    
    messages = [LLMMessage(role=MessageRole.USER, content=prompt)]
    response = await provider.chat(messages)
    
    output = f"Detected prompt: {prompt}\n"
    output += f"\n{'-'*50}\n\nResponse:\n{response.content}"
    
    return LabExecuteResponse(
        success=True,
        output=output,
        completed=True,
    )


@router.post("/progress/sync", response_model=ProgressSyncResponse)
async def sync_progress(request: Dict[str, Any]):
    """Sync user progress from client.
    
    Note: This is a placeholder that accepts progress data.
    In production, this would store progress in a database.
    """
    logger.info(f"Progress sync received")
    
    # For now, just acknowledge the sync
    # In production, this would save to a database
    return ProgressSyncResponse(
        success=True,
        message="Progress synced successfully (client-side storage)"
    )
