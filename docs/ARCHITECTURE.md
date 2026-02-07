# Architecture Overview

This document describes the technical architecture of the AI Engineering Tutorial platform.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client (Browser)                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   HTML/CSS   │  │  JavaScript  │  │  Progress Tracking   │   │
│  │  (Tailwind)  │  │   (main.js)  │  │    (localStorage)    │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/HTTPS
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Application                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                      Routers                              │   │
│  │  ┌────────────┐  ┌────────────┐  ┌──────────────────┐    │   │
│  │  │   pages    │  │    api     │  │     content      │    │   │
│  │  │ (HTML UI)  │  │(JSON/REST) │  │   (markdown)     │    │   │
│  │  └────────────┘  └────────────┘  └──────────────────┘    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Core Services                           │   │
│  │  ┌────────────────────┐  ┌─────────────────────────────┐ │   │
│  │  │  Content Registry  │  │     LLM Provider Factory    │ │   │
│  │  │    (Singleton)     │  │     (Abstract Factory)      │ │   │
│  │  └────────────────────┘  └─────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    LLM Providers                          │   │
│  │  ┌────────┐  ┌──────────┐  ┌────────┐  ┌────────────┐   │   │
│  │  │ OpenAI │  │ Anthropic│  │ Google │  │    xAI     │   │   │
│  │  └────────┘  └──────────┘  └────────┘  └────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ API Calls
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    External Services                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐  │
│  │ OpenAI API │  │ Claude API │  │ Gemini API │  │  Grok API│  │
│  └────────────┘  └────────────┘  └────────────┘  └──────────┘  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Ollama (Local)                          │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
ai_tutorial/
├── content/                    # Markdown content files
│   ├── llm-fundamentals/       # Module: LLM Basics
│   ├── embeddings-vectors/     # Module: Embeddings & Vectors
│   ├── rag/                    # Module: RAG Patterns
│   ├── fine-tuning/            # Module: Fine-tuning
│   ├── advanced-llm/           # Module: Advanced Topics
│   └── self-hosting/           # Module: Self-Hosting
├── src/
│   └── ai_tutorial/
│       ├── __init__.py
│       ├── main.py             # FastAPI app entry point
│       ├── config.py           # Configuration management
│       ├── content/
│       │   └── registry.py     # Content registry singleton
│       ├── providers/
│       │   ├── base.py         # Abstract provider interface
│       │   ├── factory.py      # Provider factory
│       │   ├── openai_provider.py
│       │   ├── anthropic_provider.py
│       │   ├── google_provider.py
│       │   └── xai_provider.py
│       └── routers/
│           ├── pages.py        # HTML page routes
│           ├── api.py          # API endpoints
│           └── content.py      # Content serving
├── static/
│   ├── css/
│   │   └── styles.css          # Global styles
│   └── js/
│       └── main.js             # Client-side JavaScript
├── templates/
│   ├── base.html               # Base layout template
│   ├── content/                # Content-specific templates
│   │   ├── lesson.html
│   │   ├── lab.html
│   │   ├── quiz.html
│   │   ├── module.html
│   │   └── learning-path.html
│   └── pages/                  # Page templates
│       ├── index.html
│       ├── about.html
│       ├── labs.html
│       └── ...
└── tests/
    ├── conftest.py             # Pytest configuration
    ├── test_providers.py       # Provider tests
    ├── test_content_registry.py
    └── test_routes.py
```

## Core Components

### Content Registry

The Content Registry is a singleton that manages all tutorial content:

```python
class ContentRegistry:
    _instance: Optional['ContentRegistry'] = None
    
    @classmethod
    def get_instance(cls) -> 'ContentRegistry':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
```

**Key Features:**
- Singleton pattern ensures single source of truth
- Lazy loading of content
- Path-aware navigation within learning paths
- Hierarchical content structure (Module → Section → Page)

**Content Hierarchy:**
```
Module
├── id, title, slug, description
├── sections: List[Section]
└── learning_path: LearningPath (optional)

Section
├── id, title, slug
└── pages: List[Page | Lab | Quiz]

Page/Lab/Quiz
├── id, title, slug, description
├── estimated_time
└── content (loaded from markdown)
```

### LLM Provider System

The provider system uses the Abstract Factory pattern:

```python
# Base interface
class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(self, messages: List[LLMMessage], 
                       model: str, **kwargs) -> LLMResponse:
        pass
    
    @abstractmethod
    def available_models(self) -> List[str]:
        pass

# Factory
class LLMProviderFactory:
    _providers: Dict[str, Type[BaseLLMProvider]] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: Type[BaseLLMProvider]):
        cls._providers[name] = provider_class
    
    @classmethod
    def create(cls, name: str, config: ProviderConfig) -> BaseLLMProvider:
        return cls._providers[name](config)
```

**Supported Providers:**
| Provider | Class | Models |
|----------|-------|--------|
| OpenAI | OpenAIProvider | gpt-4o, gpt-4o-mini, gpt-4-turbo, o1 |
| Anthropic | AnthropicProvider | claude-sonnet-4-20250514, claude-3.5 |
| Google | GoogleProvider | gemini-3-pro, gemini-2-flash |
| xAI | XAIProvider | grok-3, grok-3-mini |

### Router Architecture

**Page Router** (`/`):
- Serves HTML pages with Jinja2 templates
- Handles navigation state (learning path context)
- Renders markdown content

**API Router** (`/api`):
- RESTful JSON endpoints
- LLM interaction endpoints
- Progress tracking API

**Export Router** (`/api/export`):
- Code download as Python files
- Shareable link generation
- Social media share link generation
- Base64 URL encoding for code sharing

**Labs Router** (`/api/labs`):
- Lab execution endpoints
- Lab progress tracking

**Content Router**:
- Static content serving
- Markdown file access

## Data Flow

### Page Request Flow

```
1. User navigates to /module/llm-fundamentals/intro-to-llms/what-are-llms
2. Pages router extracts: module_id, section_id, page_id
3. ContentRegistry retrieves module, section, page
4. Markdown content loaded from content/ directory
5. Template rendered with navigation context
6. HTML returned to browser
7. Client-side JS initializes progress tracking
```

### LLM Chat Flow

```
1. User submits prompt in lab interface
2. JavaScript sends POST to /api/chat
3. API router receives ChatRequest
4. Provider factory creates appropriate provider
5. Provider sends request to external API
6. Response streamed back to client
7. UI updates with response
```

### Progress Tracking Flow

```
1. User completes a page/lab
2. JavaScript calls progressTracker.markComplete(pageId)
3. Progress saved to localStorage
4. progress-updated event dispatched
5. UI updates completion indicators
6. Optional: Sync to server (future feature)
```

## Design Patterns

### Singleton (Content Registry)
Ensures consistent content state across the application.

### Abstract Factory (LLM Providers)
Enables adding new providers without modifying existing code.

### Strategy Pattern (Navigation)
Different navigation strategies for learning paths vs free browsing.

### Observer Pattern (Progress Events)
Components subscribe to progress updates via CustomEvents.

## API Endpoints

### Content Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Home page |
| GET | `/module/{id}` | Module overview |
| GET | `/module/{mid}/{sid}/{pid}` | Page content |
| GET | `/path/{id}` | Learning path view |
| GET | `/labs` | Labs listing |

### API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat` | Send chat message to LLM |
| GET | `/api/models` | List available models |
| GET | `/api/providers` | List configured providers |
| POST | `/api/run-code` | Execute code (sandboxed) |

## Configuration

### Environment Variables

```bash
# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
XAI_API_KEY=xai-...

# Local Models
OLLAMA_HOST=http://localhost:11434

# Application
APP_ENV=development  # or production
DEBUG=true
LOG_LEVEL=INFO
```

### Config Management

```python
class Settings(BaseSettings):
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    xai_api_key: str = ""
    ollama_host: str = "http://localhost:11434"
    
    model_config = SettingsConfigDict(env_file=".env")
```

## Security Considerations

1. **API Keys**: Never exposed to client, stored in environment
2. **Code Execution**: Sandboxed with timeout limits
3. **Input Validation**: Pydantic models validate all inputs
4. **Content Sanitization**: Markdown rendered with safe HTML
5. **CORS**: Configured for appropriate origins

## Performance Optimizations

1. **Content Caching**: Registry caches parsed content
2. **Lazy Loading**: Content loaded on-demand
3. **Async Operations**: All LLM calls are async
4. **Static Files**: Served with caching headers
5. **Template Caching**: Jinja2 template compilation cached

## Extending the Platform

### Adding a New LLM Provider

1. Create provider class in `providers/`:
```python
class NewProvider(BaseLLMProvider):
    def __init__(self, config: ProviderConfig):
        self.client = NewAPIClient(api_key=config.api_key)
    
    async def generate(self, messages, model, **kwargs):
        # Implementation
        pass
    
    def available_models(self):
        return ["model-1", "model-2"]
```

2. Register in factory:
```python
LLMProviderFactory.register("new-provider", NewProvider)
```

### Adding New Content Types

1. Define data class in registry:
```python
@dataclass
class Workshop:
    id: str
    title: str
    # ...
```

2. Add template in `templates/content/`
3. Add route handler in `routers/pages.py`

## Testing Strategy

### Unit Tests
- Test individual functions and classes
- Mock external dependencies
- Fast execution

### Integration Tests
- Test component interactions
- Test route handlers with test client
- Test content loading

### End-to-End Tests (Future)
- Playwright/Selenium browser tests
- Full user journey testing

---

For more details, see the [README](README.md) and [Contributing Guide](CONTRIBUTING.md).
