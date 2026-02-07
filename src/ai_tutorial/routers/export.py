"""
Export router - Handles code export and sharing features.
"""

import hashlib
import json
import base64
from datetime import datetime
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from ..config import settings

router = APIRouter()


class CodeExportRequest(BaseModel):
    """Request model for code export."""
    code: str = Field(..., max_length=100000)  # 100KB limit
    lab_id: str
    lab_title: str
    language: str = "python"
    include_comments: bool = True


class GistExportRequest(BaseModel):
    """Request model for GitHub Gist creation."""
    code: str
    filename: str
    description: str
    public: bool = True
    github_token: Optional[str] = None


class ShareLinkRequest(BaseModel):
    """Request model for shareable link generation."""
    code: str
    lab_id: str
    result: Optional[str] = None


class ShareLinkResponse(BaseModel):
    """Response model for shareable link."""
    share_id: str
    share_url: str
    expires_at: Optional[str] = None


# In-memory storage for shared code (replace with database in production)
_shared_code_store: dict = {}


def generate_file_header(lab_title: str, lab_id: str) -> str:
    """Generate a header comment for exported code."""
    return f'''"""
AI Engineering Tutorial - {lab_title}
Lab ID: {lab_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Learn more at: https://ai-tutorial.example.com/labs/{lab_id}
"""

'''


def generate_share_id(code: str, lab_id: str) -> str:
    """Generate a unique share ID for the code."""
    content = f"{code}:{lab_id}:{datetime.now().isoformat()}"
    return hashlib.sha256(content.encode()).hexdigest()[:12]


@router.post("/download")
async def download_code(request: CodeExportRequest):
    """Generate downloadable Python file."""
    # Build the file content
    content = ""
    
    if request.include_comments:
        content += generate_file_header(request.lab_title, request.lab_id)
    
    content += request.code
    
    # Ensure file ends with newline
    if not content.endswith("\n"):
        content += "\n"
    
    # Create filename
    safe_name = request.lab_id.replace("-", "_").replace(" ", "_")
    filename = f"{safe_name}.py"
    
    return Response(
        content=content,
        media_type="text/x-python",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": "text/x-python; charset=utf-8",
        }
    )


@router.post("/gist-url")
async def generate_gist_url(request: GistExportRequest):
    """Generate a URL for creating a GitHub Gist with the code.
    
    This returns a URL that users can click to create a Gist manually,
    or use the GitHub API if a token is provided.
    """
    # Encode the content for URL
    encoded_content = quote(request.code)
    encoded_filename = quote(request.filename)
    encoded_description = quote(request.description)
    
    # GitHub Gist web URL (manual creation)
    # Note: GitHub doesn't have a direct "pre-fill" URL, so we return
    # instructions and the data for clipboard copy
    
    gist_data = {
        "description": request.description,
        "public": request.public,
        "files": {
            request.filename: {
                "content": request.code
            }
        }
    }
    
    return {
        "success": True,
        "gist_url": "https://gist.github.com/",
        "gist_data": gist_data,
        "instructions": "Click 'Create Gist' and paste your code, or use the GitHub API with the provided data.",
        "api_endpoint": "https://api.github.com/gists",
    }


@router.post("/share", response_model=ShareLinkResponse)
async def create_share_link(request: ShareLinkRequest):
    """Create a shareable link for lab code.
    
    Note: This is a simple in-memory implementation.
    For production, use a database with expiration.
    """
    share_id = generate_share_id(request.code, request.lab_id)
    
    # Store the shared code
    _shared_code_store[share_id] = {
        "code": request.code,
        "lab_id": request.lab_id,
        "result": request.result,
        "created_at": datetime.now().isoformat(),
    }
    
    # Generate share URL (frontend will handle this)
    share_url = f"/share/{share_id}"
    
    return ShareLinkResponse(
        share_id=share_id,
        share_url=share_url,
        expires_at=None,  # No expiration in this simple implementation
    )


@router.get("/share/{share_id}")
async def get_shared_code(share_id: str):
    """Retrieve shared code by ID."""
    if share_id not in _shared_code_store:
        raise HTTPException(status_code=404, detail="Shared code not found or expired")
    
    return _shared_code_store[share_id]


@router.get("/share/{share_id}/raw", response_class=PlainTextResponse)
async def get_shared_code_raw(share_id: str):
    """Get raw code content for a shared link."""
    if share_id not in _shared_code_store:
        raise HTTPException(status_code=404, detail="Shared code not found or expired")
    
    return _shared_code_store[share_id]["code"]


@router.post("/encode")
async def encode_code_for_url(request: ShareLinkRequest):
    """Encode code into a shareable URL parameter.
    
    This creates a self-contained URL with the code embedded,
    no server storage required.
    """
    # Create data payload
    payload = {
        "c": request.code,  # code
        "l": request.lab_id,  # lab_id
        "r": request.result,  # result (optional)
    }
    
    # Encode as JSON and then base64
    json_str = json.dumps(payload, separators=(',', ':'))  # Compact JSON
    encoded = base64.urlsafe_b64encode(json_str.encode()).decode()
    
    return {
        "encoded": encoded,
        "share_url": f"/share/code?data={encoded}",
        "size_bytes": len(encoded),
    }


@router.get("/decode")
async def decode_code_from_url(data: str):
    """Decode code from a URL parameter."""
    try:
        # Decode base64
        json_str = base64.urlsafe_b64decode(data.encode()).decode()
        payload = json.loads(json_str)
        
        return {
            "code": payload.get("c", ""),
            "lab_id": payload.get("l", ""),
            "result": payload.get("r"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid encoded data: {str(e)}")


@router.post("/social-share")
async def generate_social_share_links(request: ShareLinkRequest):
    """Generate social media share links for a lab completion."""
    # First create a share link
    share_id = generate_share_id(request.code, request.lab_id)
    
    # Store the shared code
    _shared_code_store[share_id] = {
        "code": request.code,
        "lab_id": request.lab_id,
        "result": request.result,
        "created_at": datetime.now().isoformat(),
    }
    
    # Base URL from configuration
    base_url = settings.BASE_URL
    share_url = f"{base_url}/share/{share_id}"
    
    # Generate share text
    share_text = f"I just completed the {request.lab_id} lab on AI Engineering Tutorial! 🚀 #AIEngineering #LLM"
    encoded_text = quote(share_text)
    encoded_url = quote(share_url)
    
    return {
        "share_id": share_id,
        "share_url": share_url,
        "social_links": {
            "twitter": f"https://twitter.com/intent/tweet?text={encoded_text}&url={encoded_url}",
            "linkedin": f"https://www.linkedin.com/sharing/share-offsite/?url={encoded_url}",
            "facebook": f"https://www.facebook.com/sharer/sharer.php?u={encoded_url}",
            "reddit": f"https://reddit.com/submit?url={encoded_url}&title={quote(share_text)}",
        },
        "embed_code": f'<iframe src="{share_url}/embed" width="100%" height="400"></iframe>',
    }
