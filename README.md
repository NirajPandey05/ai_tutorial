# 🤖 AI Engineering Tutorial

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive, comprehensive tutorial website for learning AI Engineering - from LLM fundamentals to production multi-agent systems. Built with FastAPI, featuring hands-on labs, progress tracking, and support for multiple LLM providers.

## ✨ Features

### 📚 Comprehensive Curriculum
- **LLM Fundamentals** - Tokens, prompts, parameters, models
- **Advanced LLM Concepts** - Attention mechanisms, architectures, multimodal
- **Embeddings & Vectors** - Vector databases, semantic search, similarity
- **RAG (Retrieval-Augmented Generation)** - Document loading, chunking, retrieval strategies
- **AI Agents** - ReAct, tool use, planning, memory
- **MCP (Model Context Protocol)** - Building MCP servers and integrations
- **Multi-Agent Systems** - Orchestration, collaboration, patterns
- **Fine-tuning** - LoRA, dataset preparation, training workflows
- **Self-Hosting** - Ollama, vLLM, Docker deployments
- **Advanced Topics** - Production architecture, security, evaluation

### 🧪 Hands-on Learning
- **Interactive Labs** - Code directly in the browser
- **Quizzes** - Test your understanding
- **Real API Calls** - Use actual LLM APIs with your own keys
- **Progress Tracking** - Track completion, streaks, achievements

### 🔑 Bring Your Own Keys
Use any combination of providers:
- **OpenAI** - GPT-5.2, GPT-4o, o3 series
- **Anthropic** - Claude 4, Claude 3.5 Sonnet
- **Google** - Gemini 3 Pro, Gemini 2.5
- **xAI** - Grok 4, Grok 3
- **Local** - Ollama (llama, mistral, phi)

### 🎯 Structured Learning Paths

| Path | Duration | Level | Focus |
|------|----------|-------|-------|
| 🚀 Quick Start | 2 hours | Beginner | Get started with LLMs fast |
| 📚 RAG Engineer | 8 hours | Intermediate | Build knowledge-based apps |
| 🤖 Agent Developer | 12 hours | Intermediate | Create autonomous agents |
| 🎓 Complete AI Engineer | 50+ hours | All Levels | Master everything |

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** ([Download](https://www.python.org/downloads/))
- **UV** package manager ([Install](https://github.com/astral-sh/uv))
- **Git** ([Download](https://git-scm.com/downloads))

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-tutorial.git
cd ai-tutorial

# Install dependencies with UV
uv sync

# Copy environment template
cp .env.example .env

# Start the development server
uv run uvicorn src.ai_tutorial.main:app --reload --port 8080
```

Open **http://localhost:8080** in your browser 🎉

### Configure API Keys

1. Navigate to **Settings** (⚙️) in the sidebar
2. Add API keys for providers you want to use
3. Keys are stored **locally in your browser** (never sent to servers)

## 📁 Project Structure

```
ai-tutorial/
├── src/ai_tutorial/           # Application source code
│   ├── main.py               # FastAPI entry point
│   ├── config.py             # Configuration management
│   ├── content/              # Content registry & models
│   │   ├── registry.py       # Module/section/page definitions
│   │   ├── models.py         # Data models
│   │   └── renderer.py       # Markdown rendering
│   ├── providers/            # LLM provider adapters
│   │   ├── base.py          # Abstract base class
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   ├── google_provider.py
│   │   └── xai_provider.py
│   └── routers/              # API routes
│       └── pages.py          # Page rendering routes
├── content/                   # Markdown content files
│   ├── llm-fundamentals/
│   ├── rag/
│   ├── agents/
│   └── ...
├── static/                    # Static assets
│   ├── css/styles.css
│   └── js/
│       ├── main.js           # Core functionality
│       └── progress-tracker.js
├── templates/                 # Jinja2 templates
│   ├── base.html
│   ├── content/              # Content templates
│   └── pages/                # Static pages
├── tests/                     # Test suite
│   ├── test_providers.py
│   ├── test_content_registry.py
│   └── test_routes.py
├── pyproject.toml            # Project configuration
└── README.md
```

## 🔧 Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/ai_tutorial

# Run specific test file
uv run pytest tests/test_routes.py -v
```

### Code Quality

```bash
# Format code
uv run black src/

# Lint code
uv run ruff check src/

# Type checking (optional)
uv run mypy src/
```

### Local Development

```bash
# Start with auto-reload
uv run uvicorn src.ai_tutorial.main:app --reload --port 8080

# Run with debug logging
LOG_LEVEL=debug uv run uvicorn src.ai_tutorial.main:app --reload
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t ai-tutorial .

# Run the container
docker run -p 8080:8080 ai-tutorial
```

### Docker Compose with Ollama

```bash
# Start all services (includes Ollama for local models)
docker-compose up -d

# View logs
docker-compose logs -f
```

## ☁️ Production Deployment

### Deploy to Fly.io (Recommended)

**Fastest way to deploy - takes ~5 minutes:**

```bash
# Install Flyctl
# Windows:
iwr https://fly.io/install.ps1 -useb | iex

# macOS/Linux:
curl -L https://fly.io/install.sh | sh

# Authenticate
flyctl auth login

# Setup and deploy (Windows)
pwsh scripts/deploy-flyio.ps1 setup
pwsh scripts/deploy-flyio.ps1 deploy

# Setup and deploy (macOS/Linux)
bash scripts/deploy-flyio.sh setup
bash scripts/deploy-flyio.sh deploy
```

**Cost:** ~$5-15/month with auto-scaling to zero when idle.

📚 **Full guide:** [QUICKSTART_FLYIO.md](QUICKSTART_FLYIO.md) | [DEPLOYMENT.md](docs/DEPLOYMENT.md)

### Other Platforms

The project includes Docker support and can be deployed to:
- **Railway.app** - One-click from GitHub
- **Render.com** - $7/month web service tier
- **Google Cloud Run** - Serverless, pay-per-request
- **AWS Fargate** - Production scale
- **Azure Container Instances** - Enterprise-ready

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for platform-specific guides.

## 📖 API Documentation

When running, API documentation is available at:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

### Key Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Home page |
| `GET /learn/{module}` | Module overview |
| `GET /learn/{module}/{section}/{page}` | Content page |
| `GET /path/{path_id}` | Learning path detail |
| `GET /api/path/{path_id}/content` | Path content JSON |
| `GET /settings` | API key configuration |

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Guidelines

- Follow existing code style (Black + Ruff)
- Add tests for new functionality
- Update documentation as needed
- Keep PRs focused and atomic

## 🔑 API Key Requirements

| Provider | Required For | Get Keys |
|----------|-------------|----------|
| OpenAI | GPT models, embeddings | [platform.openai.com](https://platform.openai.com) |
| Anthropic | Claude models | [console.anthropic.com](https://console.anthropic.com) |
| Google | Gemini models | [ai.google.dev](https://ai.google.dev) |
| xAI | Grok models | [x.ai](https://x.ai) |

**Note**: All keys are stored in your browser's localStorage and are never transmitted to any server. Labs use client-side API calls only.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- UI components from [Tailwind CSS](https://tailwindcss.com/)
- Reactivity with [Alpine.js](https://alpinejs.dev/)
- Icons from [Heroicons](https://heroicons.com/)

---

**Happy Learning!** 🚀

If you find this project helpful, please consider giving it a ⭐ on GitHub!
