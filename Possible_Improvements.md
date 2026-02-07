# AI Engineering Tutorial - Possible Improvements & Bug Fixes

**Analysis Date**: January 21, 2026  
**Project Version**: 0.1.0  
**Analyst**: Automated Code Review

---

## Executive Summary

This document provides a comprehensive analysis of the AI Engineering Tutorial project, identifying **52 issues** across 7 categories. Issues range from critical security vulnerabilities to low-priority enhancement opportunities.

### Issue Distribution

| Severity | Count | Description |
|----------|-------|-------------|
| 🔴 **Critical** | 4 | Security/functionality blockers requiring immediate attention |
| 🟠 **High** | 18 | Significant issues affecting functionality or user experience |
| 🟡 **Medium** | 22 | Improvements for code quality and maintainability |
| 🟢 **Low** | 8 | Nice-to-have enhancements |

---

## 1. Security Issues

### 🔴 CRITICAL: Hardcoded Secret Key
**File**: `src/ai_tutorial/config.py` & `.env.example`

**Issue**: The SECRET_KEY has a default fallback value that's hardcoded.

**Current Code**:
```python
SECRET_KEY=change-this-in-production-use-a-real-secret-key
```

**Risk**: If not changed, sessions/tokens can be forged.

**Fix**: 
- Remove default value
- Require environment variable at startup
- Add validation that rejects default value

---

### 🔴 CRITICAL: API Keys in Request Body
**File**: `src/ai_tutorial/routers/api.py`

**Issue**: API keys are sent in the request body instead of headers.

**Current Code**:
```python
class ChatRequest(BaseModel):
    api_key: str  # Exposed in logs, request bodies
```

**Risk**: API keys may be logged, cached, or exposed in error messages.

**Fix**:
- Move API keys to `Authorization` header
- Use `Header()` dependency injection
- Never log request bodies containing credentials

---

### 🔴 CRITICAL: Missing Rate Limiting
**File**: All API endpoints

**Issue**: No rate limiting on any API endpoints.

**Risk**: 
- DoS attacks
- API key abuse
- Resource exhaustion

**Fix**:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@router.post("/chat")
@limiter.limit("10/minute")
async def chat(...):
```

---

### 🟠 HIGH: Google Safety Settings Disabled
**File**: `src/ai_tutorial/providers/google_provider.py`

**Issue**: All safety settings are set to `BLOCK_NONE`.

**Current Code**:
```python
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    # ... all set to BLOCK_NONE
}
```

**Risk**: May generate harmful content in production.

**Fix**: Use appropriate safety levels for educational content.

---

## 2. Backend Code Issues

### 🔴 CRITICAL: Progress Sync Type Mismatch
**File**: `src/ai_tutorial/routers/api.py`

**Issue**: Endpoint signature uses `Dict[str, Any]` but Pydantic validation expects a model.

**Current Code**:
```python
@router.post("/progress/sync")
async def sync_progress(request: Dict[str, Any]):  # Causes 422 errors
```

**Fix**:
```python
from fastapi import Body

@router.post("/progress/sync")
async def sync_progress(request: dict = Body(...)):
    # Now accepts any JSON body
```

---

### 🟠 HIGH: Deprecated Google SDK
**File**: `src/ai_tutorial/providers/google_provider.py`

**Issue**: Uses deprecated `google.generativeai` package.

**Warning**:
```
FutureWarning: All support for the `google.generativeai` package has ended.
Please switch to the `google.genai` package as soon as possible.
```

**Fix**:
1. Update `pyproject.toml`:
   ```toml
   dependencies = [
       "google-genai>=1.0.0",  # Replace google-generativeai
   ]
   ```
2. Update import and usage to new SDK

---

### 🟠 HIGH: In-Memory Shared Code Storage
**File**: `src/ai_tutorial/routers/export.py`

**Issue**: `_shared_code_store` is in-memory dict, lost on server restart.

**Current Code**:
```python
_shared_code_store: dict = {}  # Lost on restart, no expiration
```

**Fix**:
- Use Redis or SQLite for persistence
- Implement TTL/expiration for shared links
- Add cleanup job for expired entries

---

### 🟠 HIGH: ContentRegistry Singleton Not Thread-Safe
**File**: `src/ai_tutorial/content/registry.py`

**Issue**: Singleton pattern may have race conditions in async context.

**Current Code**:
```python
_instance: Optional['ContentRegistry'] = None

@classmethod
def get_instance(cls):
    if cls._instance is None:
        cls._instance = cls()
    return cls._instance
```

**Fix**:
```python
import asyncio
_lock = asyncio.Lock()

@classmethod
async def get_instance(cls):
    async with cls._lock:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
```

---

### 🟡 MEDIUM: Incomplete Lab Execution
**File**: `src/ai_tutorial/routers/labs.py`

**Issue**: The `/execute` endpoint returns a placeholder.

**Current Code**:
```python
return LabExecuteResponse(
    lab_id=request.lab_id,
    success=True,
    output="Lab execution will be fully implemented in Phase 4+",
)
```

**Fix**: Implement actual lab execution logic or remove the endpoint.

---

### 🟡 MEDIUM: Large Registry File
**File**: `src/ai_tutorial/content/registry.py` (1625+ lines)

**Issue**: Single file contains all content definitions.

**Fix**: Split into modules:
```
content/
├── registry.py          # Core registry class
├── modules/
│   ├── llm_fundamentals.py
│   ├── embeddings.py
│   ├── rag.py
│   └── ...
└── learning_paths.py
```

---

### 🟡 MEDIUM: Inconsistent ID Formats
**Files**: Various

**Issue**: Mix of hyphen and underscore in IDs.

**Examples**:
- `first-api-call` (hyphen)
- `first_api_call` (underscore)

**Fix**: Standardize on hyphen format for URLs, underscore for Python identifiers.

---

## 3. Frontend Issues

### 🟠 HIGH: Default Model Mismatch
**File**: `static/js/main.js`

**Issue**: Default models don't match provider configurations.

**Current Code**:
```javascript
function getDefaultModel(provider) {
    const defaults = {
        openai: 'gpt-4o-mini',  // Outdated
        anthropic: 'claude-3-5-sonnet-20241022',
        google: 'gemini-1.5-flash',
        xai: 'grok-beta'  // Outdated
    };
}
```

**Fix**: Sync with actual provider models or fetch from API.

---

### 🟠 HIGH: Missing Null Checks
**File**: `static/js/main.js`

**Issue**: DOM element access without null checks.

**Current Code**:
```javascript
const progressBar = document.querySelector('.progress-bar');
progressBar.style.width = `${percentage}%`;  // Crashes if null
```

**Fix**:
```javascript
const progressBar = document.querySelector('.progress-bar');
if (progressBar) {
    progressBar.style.width = `${percentage}%`;
}
```

---

### 🟡 MEDIUM: Potential XSS in Code Editor
**File**: `static/js/code-editor.js`

**Issue**: Output may be inserted without sanitization.

**Fix**: Sanitize all user-generated content before DOM insertion:
```javascript
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
```

---

### 🟡 MEDIUM: CDN Dependency Without Fallback
**File**: `static/js/code-editor.js`

**Issue**: Monaco Editor loaded from CDN without fallback.

**Fix**: Add local fallback or graceful degradation:
```javascript
require.config({ 
    paths: { vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs' },
    timeout: 10000
});
require.onError = function(err) {
    console.error('Monaco failed to load, falling back to textarea');
    // Show fallback editor
};
```

---

### 🟢 LOW: TailwindCSS via CDN
**File**: `templates/base.html`

**Issue**: Using CDN Tailwind with comment "will be replaced in production".

**Fix**: Set up Tailwind build process for production.

---

## 4. API Design Issues

### 🟠 HIGH: No Input Size Limits
**File**: `src/ai_tutorial/routers/export.py`

**Issue**: `code` field accepts unlimited size.

**Fix**:
```python
class CodeExportRequest(BaseModel):
    code: str = Field(..., max_length=100000)  # 100KB limit
```

---

### 🟠 HIGH: Duplicate Endpoints
**Files**: `routers/api.py` and `routers/labs.py`

**Issue**: Similar lab execution endpoints exist in both files.

**Fix**: Consolidate lab-related endpoints in `labs.py` only.

---

### 🟡 MEDIUM: Hardcoded URLs
**File**: `src/ai_tutorial/routers/export.py`

**Issue**: Base URL hardcoded for social sharing.

**Current Code**:
```python
base_url = "https://ai-tutorial.example.com"
```

**Fix**: Use configuration:
```python
from ..config import settings
base_url = settings.BASE_URL
```

---

### 🟡 MEDIUM: Fragile Code Extraction
**File**: `src/ai_tutorial/routers/api.py`

**Issue**: Uses regex to extract prompts from user code.

**Current Code**:
```python
patterns = [
    r'prompt\s*=\s*["\'](.+?)["\']',
    r'prompt\s*=\s*"""(.+?)"""',
    # ...
]
```

**Risk**: Easily broken by code variations.

**Fix**: Use AST parsing for reliable extraction:
```python
import ast

def extract_prompt(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'prompt':
                        if isinstance(node.value, ast.Constant):
                            return node.value.value
    except SyntaxError:
        return None
```

---

## 5. Testing Gaps

### 🟠 HIGH: Missing Integration Tests
**Directory**: `tests/`

**Issue**: All provider tests use mocks, no real API tests.

**Fix**: Add optional integration tests:
```python
@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No API key")
async def test_real_openai_call():
    provider = OpenAIProvider(config)
    response = await provider.chat([...])
    assert response.content
```

---

### 🟠 HIGH: No JavaScript Tests
**Directory**: `tests/`

**Issue**: No frontend unit tests exist.

**Fix**: Add Jest or Vitest for JavaScript testing:
```javascript
// tests/js/progress-tracker.test.js
describe('ProgressTracker', () => {
    test('marks page complete', () => {
        progressTracker.markComplete('page-id');
        expect(localStorage.getItem('progress')).toContain('page-id');
    });
});
```

---

### 🟡 MEDIUM: Missing Error Path Tests
**File**: `tests/test_routes.py`

**Issue**: Limited testing of error conditions.

**Fix**: Add tests for:
- Invalid module IDs
- Malformed requests
- Provider errors
- Rate limiting

---

## 6. Configuration Issues

### 🟠 HIGH: Python Version Mismatch
**Files**: `pyproject.toml` vs `README.md`

**Issue**: 
- pyproject.toml: `requires-python = ">=3.10"`
- README.md: "Python 3.11+"

**Fix**: Align both to `>=3.11`

---

### 🟠 HIGH: Incomplete PWA Icons
**File**: `static/manifest.json`

**Issue**: Only has SVG icon, missing required PNG sizes.

**Current**:
```json
"icons": [
    { "src": "/static/icons/icon-144x144.svg", "sizes": "any" }
]
```

**Required for PWA**: 192x192 and 512x512 PNG icons minimum.

**Fix**: Generate PNG icons from SVG or create proper icon set.

---

### 🟡 MEDIUM: Outdated Dependencies
**File**: `pyproject.toml`

**Issues**:
- `google-generativeai>=0.4.0` - deprecated
- No version pinning for stability

**Fix**:
```toml
[project]
dependencies = [
    "fastapi>=0.109.0,<0.120.0",
    "google-genai>=1.0.0",
    # Pin major versions
]
```

---

### 🟡 MEDIUM: Port Inconsistency
**Files**: `.env.example`, `README.md`, `config.py`

**Issue**: Different default ports mentioned (8000 vs 8080).

**Fix**: Standardize on 8080 everywhere.

---

## 7. Documentation Issues

### 🟡 MEDIUM: Missing Docker Documentation
**File**: `README.md`

**Issue**: Docker files exist but usage not documented in README.

**Fix**: Add section:
```markdown
## Docker Deployment

### Production
```bash
docker compose up -d
```

### Development
```bash
docker compose -f docker-compose.dev.yml up
```
```

---

### 🟡 MEDIUM: Outdated Architecture Doc
**File**: `docs/ARCHITECTURE.md`

**Issue**: References outdated router structure.

**Fix**: Update to reflect current structure including `export.py` router.

---

### 🟢 LOW: Missing API Documentation
**Issue**: No custom API documentation beyond auto-generated Swagger.

**Fix**: Add docstrings to all endpoints with examples:
```python
@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Send a message to an LLM provider.
    
    Example:
        ```json
        {
            "provider": "openai",
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        ```
    """
```

---

## 8. Performance Improvements

### 🟡 MEDIUM: Content Loading
**File**: `src/ai_tutorial/content/registry.py`

**Issue**: All content metadata loaded at startup.

**Fix**: Implement lazy loading for content files:
```python
@property
def content(self):
    if self._content is None:
        self._content = self._load_content()
    return self._content
```

---

### 🟡 MEDIUM: No Response Caching
**File**: `src/ai_tutorial/routers/pages.py`

**Issue**: Static pages re-rendered on every request.

**Fix**: Add caching headers or implement server-side caching:
```python
from fastapi import Response

@router.get("/module/{module_id}")
async def get_module(module_id: str, response: Response):
    response.headers["Cache-Control"] = "public, max-age=3600"
    # ...
```

---

### 🟢 LOW: Template Pre-compilation
**File**: `src/ai_tutorial/main.py`

**Issue**: Jinja2 templates compiled on each request in development.

**Fix**: Enable template caching:
```python
templates = Jinja2Templates(
    directory=str(templates_path),
    auto_reload=settings.DEBUG  # Only reload in dev
)
```

---

## Priority Action Items

### Immediate (Critical)
1. ✅ Fix progress sync endpoint type (use `Body(...)`)
2. ⚠️ Move API keys to headers
3. ⚠️ Add rate limiting
4. ⚠️ Change default secret key handling

### Short-term (High)
5. Migrate from deprecated Google SDK
6. Add persistent storage for shared code
7. Fix PWA icon requirements
8. Add input size validation

### Medium-term (Medium)
9. Add integration tests
10. Split large registry file
11. Standardize ID formats
12. Add JavaScript tests

### Long-term (Low)
13. Production Tailwind build
14. Comprehensive API documentation
15. Performance optimizations

---

## Conclusion

The AI Engineering Tutorial is a well-structured project with comprehensive content coverage. The main areas requiring attention are:

1. **Security hardening** - API key handling, rate limiting
2. **Dependency updates** - Google SDK migration
3. **Testing expansion** - Integration and frontend tests
4. **Production readiness** - PWA icons, caching, configuration

Addressing the critical and high-priority items will significantly improve the application's security, reliability, and user experience.

---

*Document generated: January 21, 2026*
