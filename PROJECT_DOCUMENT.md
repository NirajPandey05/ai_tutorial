# AI Engineering Tutorial Website - Project Document

## Project Overview
A comprehensive, interactive tutorial website teaching AI Engineering concepts with hands-on labs. Users provide their own API keys (OpenAI, Claude, Grok, Google Gemini) to run practical exercises.

---

## ✅ Completed Phases Summary

| Phase | Title | Status | Files Created |
|-------|-------|--------|---------------|
| 1 | Project Foundation & Infrastructure | ✅ Complete | Core project structure |
| 2 | API Key Management & LLM Integration | ✅ Complete | Provider abstractions |
| 3 | Content Architecture & Navigation | ✅ Complete | Registry, routing |
| 4 | LLM Fundamentals | ✅ Complete | 16 content files |
| 5 | Advanced LLM Concepts | ✅ Complete | 5 content files |
| 6 | Self-Hosting LLMs | ✅ Complete | 14 content files |
| 7 | Embeddings & Vector Databases | ✅ Complete | 13 content files |
| 8 | RAG (Retrieval-Augmented Generation) | ✅ Complete | 24 content files |
| - | Post-Phase 8 Maintenance & Bug Fixes | ✅ Complete | Labs system fixes |
| 9 | MCP (Model Context Protocol) | ✅ Complete | 14 content files |
| 10 | AI Agents | ✅ Complete | 16 content files |
| 11 | Multi-Agent Workflows | ✅ Complete | 16 content files |
| - | Post-Phase 11 Navigation Bug Fix | ✅ Complete | 50+ files patched |
| 12 | Interactive Features & Polish | ✅ Complete | 10 files (5 JS, 5 templates) |
| 13 | Assessment & Gamification | ✅ Complete | 6 files (3 JS, 3 templates) |
| 14 | Content Enrichment | ✅ Complete | 5 files (1 JS, 4 templates) |
| - | Post-Phase 14 Fixes & Updates | ✅ Complete | Multiple bug fixes |
| 15 | Advanced Topics | ✅ Complete | 16 content files |
| 16 | Progress Tracking & User Experience | ✅ Complete | 8 files (4 JS, 4 templates/static) |
| 17 | Developer Experience & Deployment | ✅ Complete | 15+ files (tests, docs, Docker, export) |

---

## 🔄 Development Workflow Rules

### Agent Workflow Protocol
**CRITICAL**: These rules must be followed for every phase and sub-phase execution.

#### Before Starting Any Phase/Sub-Phase:
1. **READ** the PROJECT_DOCUMENT.md completely
2. **REVIEW** the current phase objectives and deliverables
3. **CHECK** dependencies on previous phases
4. **CONFIRM** with user before proceeding

#### During Phase Execution:
1. **UPDATE** todo list with current progress
2. **DOCUMENT** any challenges encountered immediately
3. **COMMIT** code changes with meaningful messages
4. **TEST** functionality before marking tasks complete

#### After Completing Phase/Sub-Phase:
1. **UPDATE** PROJECT_DOCUMENT.md with:
   - Summary of work completed
   - Code files added/modified
   - Challenges encountered and solutions
   - Any deviations from original plan
2. **PRESENT** changes to user for review
3. **WAIT** for user acceptance before marking phase complete

#### Phase Completion Criteria:
- [ ] All sub-phase tasks completed
- [ ] Code tested and working
- [ ] Documentation updated
- [ ] **USER ACCEPTANCE RECEIVED** ← Required for completion

---

## Technology Stack
- **Package Manager**: UV (Python package manager)
- **Backend**: FastAPI (Python)
- **Frontend**: HTML/CSS/JavaScript with HTMX for interactivity
- **Database**: SQLite (for user progress tracking)
- **Vector Database**: ChromaDB (for RAG demonstrations)
- **Styling**: TailwindCSS
- **Code Highlighting**: Prism.js / Highlight.js
- **In-Browser Python**: Pyodide (for live code sandbox)
- **Visualizations**: D3.js / Plotly.js (for embeddings, diagrams)
- **PWA**: Service Workers (for offline mode)

---

## 🎯 Learning Paths

Pre-defined tracks for different learning goals:

### 🚀 Quick Start Path (2 hours)
```
LLM Basics → First API Call → Simple Chatbot
```
**Target**: Developers wanting to get started quickly

### 📚 RAG Engineer Path (8 hours)
```
Embeddings → Vector DB → Basic RAG → Advanced RAG → Evaluation
```
**Target**: Developers building knowledge-based applications

### 🤖 Agent Developer Path (10 hours)
```
Tool Use → Function Calling → Agent Loop → MCP → Multi-Agent
```
**Target**: Developers building autonomous AI systems

### 🏠 Self-Hosting Specialist Path (6 hours)
```
Ollama Setup → Quantization → vLLM → Production Deployment
```
**Target**: Developers prioritizing privacy and cost control

### 🔧 Fine-Tuning Expert Path (8 hours)
```
Dataset Preparation → Cloud Fine-tuning → Local LoRA → Evaluation
```
**Target**: Developers customizing models for specific domains

### 🎓 Complete AI Engineer Path (40+ hours)
```
All modules in optimal dependency order
```
**Target**: Comprehensive mastery of AI Engineering

---

## ✨ Enhanced Features

### Interactive Learning Features
| Feature | Description | Priority |
|---------|-------------|----------|
| **Live Code Sandbox** | Pyodide-based in-browser Python execution | High |
| **Token Visualizer** | Real-time tokenization display as users type | High |
| **Embedding Playground** | 2D/3D visualization with dimensionality reduction | High |
| **Comparison Mode** | Side-by-side LLM response comparison | Medium |
| **Visual Debugger** | Step-through for agent loops and RAG pipelines | Medium |
| **Cost Calculator** | Interactive API cost estimation tool | High |
| **Decision Trees** | Interactive "when to use what" guides | Medium |

### Gamification System
| Feature | Description | Priority |
|---------|-------------|----------|
| **Achievement Badges** | Unlock badges for milestones | Medium |
| **Skill Tree** | Visual progression through concepts | Medium |
| **Streak Tracking** | Consecutive learning days | Low |
| **Daily Challenges** | Optional prompt engineering challenges | Low |
| **Leaderboard** | Anonymous completion leaderboard | Low |

### Assessment & Certification
| Feature | Description | Priority |
|---------|-------------|----------|
| **Concept Quizzes** | Multiple-choice after each section | High |
| **Coding Challenges** | Auto-graded exercises | Medium |
| **Capstone Projects** | End-of-module projects | Medium |
| **Completion Certificate** | Downloadable PDF certificate | High |
| **Skills Assessment** | Pre-assessment for personalized path | Low |

### Content Enrichment
| Feature | Description | Priority |
|---------|-------------|----------|
| **Glossary** | Searchable AI/ML terminology | High |
| **Case Studies** | Real-world industry examples | Medium |
| **Failure Modes Gallery** | Common mistakes and fixes | Medium |
| **Architecture Diagrams** | Interactive system diagrams | Medium |
| **Code Export** | Export to GitHub Gist or download | High |

### Technical Features
| Feature | Description | Priority |
|---------|-------------|----------|
| **Response Caching** | Cache LLM responses for demos | High |
| **Offline Mode (PWA)** | Progressive Web App support | Medium |
| **API Cost Tracking** | Track estimated spend across labs | High |
| **Rate Limit Handling** | Graceful retry UI | High |
| **Environment Snapshots** | Save/restore lab state | Low |

### Accessibility & Reach
| Feature | Description | Priority |
|---------|-------------|----------|
| **Screen Reader Support** | Full ARIA labels | High |
| **Keyboard Navigation** | Complete keyboard support | High |
| **Dark/Light Mode** | Theme toggle | High |
| **Mobile Responsive** | Works on all devices | High |
| **i18n Ready** | Internationalization structure | Low |

### Developer Experience
| Feature | Description | Priority |
|---------|-------------|----------|
| **Docker Compose** | One-command full stack | High |
| **GitHub Codespaces** | Zero local setup config | Medium |
| **CLI Tool** | `ai-tutorial run lab-name` | Low |
| **VS Code Extension** | Run labs in IDE | Low |

### Additional Topics
| Topic | Description | Priority |
|-------|-------------|----------|
| **Structured Outputs** | JSON mode, schemas | High |
| **Guardrails & Safety** | Prompt injection prevention | High |
| **Observability** | LangSmith, Phoenix integration | Medium |
| **Semantic Caching** | Cost optimization | Medium |
| **Voice & Audio** | Whisper, TTS integration | Low |
| **Agentic RAG** | Agents deciding retrieval | Medium |

---

## Project Phases

### Phase 1: Project Foundation & Infrastructure
**Objective**: Set up the project structure, dependencies, and basic server infrastructure.

#### Sub-Phase 1.1: Project Initialization
- [x] Initialize UV project with `uv init`
- [x] Create project directory structure
- [x] Set up virtual environment
- [x] Configure pyproject.toml with dependencies

#### Sub-Phase 1.2: Core Dependencies Setup
- [x] Install FastAPI, Uvicorn
- [x] Install Jinja2 for templating
- [x] Install LLM SDKs (openai, anthropic, google-generativeai)
- [x] Install ChromaDB for vector operations
- [x] Install additional utilities (python-dotenv, pydantic)

#### Sub-Phase 1.3: Basic Server Structure
- [x] Create FastAPI application entry point
- [x] Set up routing structure
- [x] Configure static files serving
- [x] Set up Jinja2 templates
- [x] Create base HTML template with TailwindCSS

#### Sub-Phase 1.4: Configuration Management
- [x] Create environment configuration system
- [x] Implement API key management (secure session storage)
- [x] Create settings validation

**Deliverables**: Working FastAPI server with basic routing and templates

---

### Phase 2: API Key Management & LLM Integration Layer
**Objective**: Create a unified interface for multiple LLM providers.

#### Sub-Phase 2.1: API Key Input Interface
- [x] Create API key configuration page
- [x] Implement secure client-side key storage (session/localStorage)
- [x] Add key validation endpoints for each provider
- [x] Create provider selection UI

#### Sub-Phase 2.2: LLM Provider Abstraction
- [x] Create base LLM interface/protocol
- [x] Implement OpenAI provider adapter
- [x] Implement Anthropic (Claude) provider adapter
- [x] Implement Google Gemini provider adapter
- [x] Implement Grok (xAI) provider adapter
- [x] Create provider factory pattern

#### Sub-Phase 2.3: Unified Chat Interface
- [x] Create streaming response handler
- [x] Implement error handling for API failures
- [x] Add rate limiting awareness
- [x] Create response formatting utilities

**Deliverables**: Unified LLM interface supporting 4+ providers

---

### Phase 3: Content Architecture & Navigation System
**Objective**: Design the learning path and content organization.

#### Sub-Phase 3.1: Learning Path Design
- [x] Define concept dependency graph
- [x] Create learning modules structure
- [x] Design prerequisite system
- [x] Plan lab exercises for each concept

#### Sub-Phase 3.2: Navigation & UI Components
- [x] Create sidebar navigation component
- [x] Implement breadcrumb navigation
- [x] Add progress tracking UI
- [x] Create concept linking system (organic connections)
- [x] Design "Related Concepts" component

#### Sub-Phase 3.3: Content Rendering System
- [x] Create Markdown content renderer
- [x] Implement code block rendering with syntax highlighting
- [x] Add interactive code playground component
- [x] Create diagram/visualization components

**Deliverables**: Complete navigation system with learning path visualization

---

### Phase 4: Core Concepts - LLM Fundamentals ✅
**Objective**: Build tutorial content and labs for LLM basics.
**Status**: ✅ Complete - Awaiting User Acceptance

#### Sub-Phase 4.1: Introduction to LLMs ✅
- [x] Content: What are Large Language Models? (`content/llm-fundamentals/intro-to-llms/what-are-llms.md`)
- [x] Content: History and evolution of LLMs (`content/llm-fundamentals/intro-to-llms/history.md`)
- [x] Content: How LLMs work (Transformers, Attention) (`content/llm-fundamentals/intro-to-llms/how-they-work.md`)
- [x] Lab: First API call to an LLM (`content/llm-fundamentals/intro-to-llms/first-api-call.md`)
- [x] API Endpoint: `/api/lab/execute` for lab code execution

#### Sub-Phase 4.2: Prompt Engineering ✅
- [x] Content: Prompt Fundamentals & CRISPE Framework (`content/llm-fundamentals/prompt-engineering/prompt-fundamentals.md`)
- [x] Content: Prompt Patterns & Techniques (`content/llm-fundamentals/prompt-engineering/prompt-patterns.md`)
- [x] Content: Zero-shot, One-shot, Few-shot Learning (`content/llm-fundamentals/prompt-engineering/zero-few-shot.md`)
- [x] Lab: Prompt Playground (`content/llm-fundamentals/prompt-engineering/prompt-playground.md`)

#### Sub-Phase 4.3: LLM Parameters & Configuration ✅
- [x] Content: Temperature & Sampling (`content/llm-fundamentals/parameters/temperature.md`)
- [x] Content: Token Limits & Max Tokens (`content/llm-fundamentals/parameters/tokens-limits.md`)
- [x] Content: Stop Sequences & Output Control (`content/llm-fundamentals/parameters/stop-sequences.md`)
- [x] Lab: Parameter Explorer (`content/llm-fundamentals/parameters/parameter-explorer.md`)

#### Sub-Phase 4.4: Tokens & Context Windows ✅
- [x] Content: Deep Dive Tokenization (`content/llm-fundamentals/tokens/tokenization.md`)
- [x] Content: Understanding Context Windows (`content/llm-fundamentals/tokens/context-windows.md`)
- [x] Content: Managing Context Effectively (`content/llm-fundamentals/tokens/managing-context.md`)
- [x] Lab: Token Counter & Context Analyzer (`content/llm-fundamentals/tokens/token-counter.md`)

**Deliverables**: ✅ Complete LLM fundamentals module with 4 sections and 4 labs
- 12 lesson pages (comprehensive markdown content with code examples, diagrams, tables)
- 4 interactive labs with hands-on exercises
- Lab execution API endpoint added to api.py
- Content registry updated with proper content_path references

**User Acceptance**: ⏳ Pending

---

### Phase 5: Advanced LLM Concepts & Fine-tuning
**Objective**: Deep dive into advanced LLM techniques and fine-tuning.

#### Sub-Phase 5.1: Advanced LLM Concepts
- [ ] Content: Understanding model architectures (GPT, LLaMA, Mistral)
- [ ] Content: Attention mechanisms deep dive
- [ ] Content: Mixture of Experts (MoE) models
- [ ] Content: Multimodal LLMs (vision, audio)
- [ ] Lab: Working with multimodal models

#### Sub-Phase 5.2: Fine-tuning Fundamentals
- [ ] Content: Why fine-tune? When to fine-tune vs RAG vs prompting
- [ ] Content: Types of fine-tuning (full, LoRA, QLoRA)
- [ ] Content: Dataset preparation for fine-tuning
- [ ] Content: Training data formats (instruction, conversation, completion)
- [ ] Lab: Prepare a fine-tuning dataset

#### Sub-Phase 5.3: Fine-tuning with Cloud Providers
- [ ] Content: OpenAI fine-tuning API
- [ ] Content: Anthropic fine-tuning (when available)
- [ ] Content: Google Vertex AI fine-tuning
- [ ] Lab: Fine-tune a model with OpenAI API
- [ ] Lab: Evaluate fine-tuned model performance

#### Sub-Phase 5.4: Local Fine-tuning
- [ ] Content: Introduction to Hugging Face Transformers
- [ ] Content: LoRA and QLoRA explained
- [ ] Content: Using PEFT library
- [ ] Content: Training configurations and hyperparameters
- [ ] Lab: Fine-tune a small model locally with LoRA
- [ ] Lab: Merge and export fine-tuned adapters

#### Sub-Phase 5.5: Fine-tuning Best Practices
- [ ] Content: Evaluation metrics for fine-tuned models
- [ ] Content: Avoiding catastrophic forgetting
- [ ] Content: Data quality and curation
- [ ] Content: Cost optimization strategies
- [ ] Lab: A/B testing fine-tuned vs base models

**Deliverables**: Complete fine-tuning module with cloud and local options

---

### Phase 6: Self-Hosting LLMs
**Objective**: Teach how to run LLMs locally and in production.

#### Sub-Phase 6.1: Introduction to Self-Hosting
- [ ] Content: Why self-host? (Privacy, cost, customization)
- [ ] Content: Hardware requirements (GPU, RAM, storage)
- [ ] Content: Model formats (GGUF, GPTQ, AWQ, safetensors)
- [ ] Content: Quantization explained (4-bit, 8-bit, 16-bit)

#### Sub-Phase 6.2: Local Inference Tools
- [ ] Content: Ollama introduction and setup
- [ ] Content: llama.cpp and its ecosystem
- [ ] Content: LM Studio for beginners
- [ ] Content: vLLM for production
- [ ] Lab: Set up Ollama and run local models
- [ ] Lab: Compare inference speeds across tools

#### Sub-Phase 6.3: Model Selection & Optimization
- [ ] Content: Open-source model landscape (LLaMA, Mistral, Phi, Qwen)
- [ ] Content: Choosing the right model size
- [ ] Content: Quantization trade-offs (speed vs quality)
- [ ] Content: Context length considerations
- [ ] Lab: Benchmark different model variants

#### Sub-Phase 6.4: Production Deployment
- [ ] Content: API servers (Ollama API, vLLM, TGI)
- [ ] Content: Load balancing and scaling
- [ ] Content: Docker deployment for LLMs
- [ ] Content: GPU memory management
- [ ] Lab: Deploy a model with Docker and vLLM
- [ ] Lab: Set up OpenAI-compatible API endpoint

#### Sub-Phase 6.5: Self-Hosted Integration
- [ ] Content: Connecting self-hosted models to applications
- [ ] Content: Using Ollama with LangChain/LlamaIndex
- [ ] Content: Hybrid architectures (local + cloud)
- [ ] Lab: Build an app using self-hosted LLM
- [ ] Link to: RAG with local models, Agents with local models

**Deliverables**: Complete self-hosting guide with deployment labs

---

### Phase 7: Vector Databases & Embeddings ✅
**Objective**: Teach embedding concepts and vector database operations.
**Status**: ✅ Complete

#### Sub-Phase 7.1: Understanding Embeddings ✅
- [x] Content: What are embeddings? (`content/embeddings-vectors/what-are-embeddings.md`)
- [x] Content: Semantic meaning in vector space (`content/embeddings-vectors/semantic-vectors.md`)
- [x] Content: Embedding models comparison (`content/embeddings-vectors/embedding-models.md`)
- [x] Lab: Generate and visualize embeddings (`content/embeddings-vectors/embeddings-lab.md`)

#### Sub-Phase 7.2: Vector Database Concepts ✅
- [x] Content: Introduction to vector databases (`content/embeddings-vectors/vector-db-intro.md`)
- [x] Content: Similarity search algorithms (`content/embeddings-vectors/similarity-search.md`)
- [x] Content: Indexing strategies (`content/embeddings-vectors/indexing-strategies.md`)
- [x] Content: Vector DB options (`content/embeddings-vectors/vector-db-options.md`)
- [x] Lab: Set up ChromaDB locally (`content/embeddings-vectors/chromadb-lab.md`)

#### Sub-Phase 7.3: Practical Vector Operations ✅
- [x] Content: Working with collections (`content/embeddings-vectors/collections.md`)
- [x] Content: Querying documents (`content/embeddings-vectors/querying-documents.md`)
- [x] Content: Metadata filtering (`content/embeddings-vectors/metadata-filtering.md`)
- [x] Lab: Build a semantic search engine (`content/embeddings-vectors/semantic-search-lab.md`)

**Deliverables**: ✅ Complete - 13 content files covering embeddings and vector databases

---

### Phase 8: RAG (Retrieval-Augmented Generation) ✅
**Objective**: Comprehensive RAG implementation tutorial from basics to production.
**Status**: ✅ Complete

#### Sub-Phase 8.1: RAG Fundamentals ✅
- [x] Content: What is RAG? (`content/rag/what-is-rag.md`)
- [x] Content: RAG vs Fine-tuning (`content/rag/rag-vs-finetuning.md`)
- [x] Content: RAG Architecture (`content/rag/rag-architecture.md`)

#### Sub-Phase 8.2: Document Processing Pipeline ✅
- [x] Content: Document loading strategies (`content/rag/document-loading.md`)
- [x] Content: Chunking strategies (`content/rag/chunking-strategies.md`)
- [x] Content: Chunk sizing guide (`content/rag/chunk-sizing.md`)
- [x] Content: Document parsing (`content/rag/document-parsing.md`)
- [x] Lab: Document processor pipeline (`content/rag/document-processor-lab.md`)

#### Sub-Phase 8.3: Retrieval Strategies ✅
- [x] Content: Basic similarity search (`content/rag/basic-retrieval.md`)
- [x] Content: Hybrid search (`content/rag/hybrid-search.md`)
- [x] Content: Reranking techniques (`content/rag/reranking.md`)
- [x] Content: Multi-query retrieval (`content/rag/multi-query.md`)
- [x] Content: Contextual compression (`content/rag/contextual-compression.md`)
- [x] Lab: Retrieval strategies comparison (`content/rag/retrieval-strategies-lab.md`)

#### Sub-Phase 8.4: Generation with Context ✅
- [x] Content: RAG prompt engineering (`content/rag/rag-prompts.md`)
- [x] Content: Handling context overflow (`content/rag/context-overflow.md`)
- [x] Content: Citation tracking (`content/rag/citation-tracking.md`)
- [x] Content: Streaming with sources (`content/rag/streaming-sources.md`)
- [x] Lab: RAG chatbot (`content/rag/rag-chatbot-lab.md`)

#### Sub-Phase 8.5-8.7: Advanced RAG ✅
- [x] Content: Advanced patterns (HyDE, RAG Fusion, Self-RAG, CRAG) (`content/rag/advanced-patterns.md`)
- [x] Content: Production architecture (`content/rag/production-architecture.md`)
- [x] Content: Evaluation & optimization (`content/rag/evaluation-optimization.md`)
- [x] Lab: Advanced RAG lab (`content/rag/advanced-rag-lab.md`)

**Deliverables**: ✅ Complete - 23 content files covering RAG from fundamentals to production

---

### Phase 9: MCP (Model Context Protocol) ✅
**Objective**: Teach MCP concepts and implementation.

#### Sub-Phase 9.1: MCP Introduction
- [x] Content: What is Model Context Protocol?
- [x] Content: MCP architecture and components
- [x] Content: When to use MCP
- [x] Link to: Agents, Tool Use

#### Sub-Phase 9.2: MCP Core Concepts
- [x] Content: Resources in MCP
- [x] Content: Tools in MCP
- [x] Content: Prompts in MCP
- [x] Content: Transport layer (stdio, HTTP)

#### Sub-Phase 9.3: Building MCP Servers
- [x] Content: MCP server structure
- [x] Content: Implementing tools
- [x] Content: Implementing resources
- [x] Lab: Build a simple MCP server

#### Sub-Phase 9.4: MCP Clients & Integration
- [x] Content: MCP client implementation
- [x] Content: Connecting to MCP servers
- [x] Lab: Build MCP client and connect to server
- [x] Lab: Create a file system MCP server

**Deliverables**: MCP tutorial with server and client labs ✅

---

### Phase 10: AI Agents ✅
**Objective**: Comprehensive agent development tutorial.
**Status**: COMPLETE

#### Sub-Phase 10.1: Agent Fundamentals ✅
- [x] Content: What are AI Agents? (what-are-agents.md)
- [x] Content: Agent Architectures (agent-architectures.md)
- [x] Content: Agent Components (agent-components.md)

#### Sub-Phase 10.2: Tool Use & Function Calling ✅
- [x] Content: Function calling basics (function-calling.md)
- [x] Content: Tool definition schemas (tool-schemas.md)
- [x] Content: Tool execution patterns (tool-execution.md)
- [x] Lab: Build tools for an agent (tools-lab.md)

#### Sub-Phase 10.3: Agent Loop Implementation ✅
- [x] Content: The agent loop pattern (agent-loop.md)
- [x] Content: Observation-Thought-Action cycle (observation-thought-action.md)
- [x] Content: Memory and state management (memory-state.md)
- [x] Lab: Build a ReAct agent from scratch (react-agent-lab.md)

#### Sub-Phase 10.4: Agent Capabilities ✅
- [x] Content: Web browsing agents (web-agents.md)
- [x] Content: Code execution agents (code-agents.md)
- [x] Content: Multi-tool agents (multi-tool-agents.md)
- [x] Lab: Build a research agent (research-agent-lab.md)
- [x] Lab: Build a code assistant agent (code-assistant-lab.md)

**Deliverables**: Agent development module with multiple agent labs ✅

**Files Created (16 total)**:
- content/agents/what-are-agents.md
- content/agents/agent-architectures.md
- content/agents/agent-components.md
- content/agents/function-calling.md
- content/agents/tool-schemas.md
- content/agents/tool-execution.md
- content/agents/tools-lab.md
- content/agents/agent-loop.md
- content/agents/observation-thought-action.md
- content/agents/memory-state.md
- content/agents/react-agent-lab.md
- content/agents/web-agents.md
- content/agents/code-agents.md
- content/agents/multi-tool-agents.md
- content/agents/research-agent-lab.md
- content/agents/code-assistant-lab.md

---

### Phase 11: Multi-Agent Workflows ✅ COMPLETED
**Objective**: Teach multi-agent orchestration patterns.

#### Sub-Phase 11.1: Multi-Agent Concepts ✅
- [x] Content: Why multi-agent systems? (`content/multi-agents/why-multi-agent.md`)
- [x] Content: Agent communication patterns (`content/multi-agents/communication-patterns.md`)
- [x] Content: Coordination strategies (`content/multi-agents/coordination-strategies.md`)
- [x] Link to: Agents, MCP

#### Sub-Phase 11.2: Orchestration Patterns ✅
- [x] Content: Sequential workflows (`content/multi-agents/sequential-workflows.md`)
- [x] Content: Parallel workflows (`content/multi-agents/parallel-workflows.md`)
- [x] Content: Hierarchical agents (`content/multi-agents/hierarchical-agents.md`)
- [x] Content: Debate/collaboration patterns (`content/multi-agents/debate-patterns.md`)

#### Sub-Phase 11.3: Building Multi-Agent Systems ✅
- [x] Content: Agent role definition (`content/multi-agents/agent-roles.md`)
- [x] Content: Message passing between agents (`content/multi-agents/message-passing.md`)
- [x] Content: State sharing and handoffs (`content/multi-agents/state-sharing.md`)
- [x] Lab: Build a writer-reviewer system (`content/multi-agents/writer-reviewer-lab.md`)

#### Sub-Phase 11.4: Advanced Multi-Agent Patterns ✅
- [x] Content: Agent swarms (`content/multi-agents/agent-swarms.md`)
- [x] Content: Autonomous agent teams (`content/multi-agents/autonomous-teams.md`)
- [x] Content: Human-in-the-loop patterns (`content/multi-agents/human-in-the-loop.md`)
- [x] Lab: Build a research team (`content/multi-agents/research-team-lab.md`)
- [x] Lab: Build a coding team (`content/multi-agents/coding-team-lab.md`)

**Deliverables**: ✅ Complete multi-agent module with team-based labs
- 12 lesson pages (comprehensive markdown content)
- 4 hands-on labs (writer-reviewer, research-team, coding-team)
- Content registry updated with all multi-agent content

**User Acceptance**: ✅ Accepted

#### Post-Phase 11: Navigation Link Bug Fix ✅
**Issue Encountered**: Navigation links in lesson content (e.g., "Next: Agent Communication Patterns") were returning 404 errors.

**Root Cause Analysis**:
1. **Initial Problem**: Links were using relative `.md` file references (e.g., `(communication-patterns.md)`) instead of absolute URL paths
2. **Secondary Problem**: After fixing to absolute URLs, links used section `slug` values instead of section `id` values in the URL path

**Technical Details**:
- Router pattern: `/learn/{module_id}/{section_id}/{page_id}`
- The `get_section()` method in registry.py (line 1184-1192) matches on `section.id`, NOT `section.slug`
- Example: URL used `/learn/advanced-llm/architectures/attention` but should be `/learn/advanced-llm/model-architectures/attention`

**Solution Applied**:
- Fixed 50+ content files across 5 modules (multi-agents, agents, mcp, fine-tuning, advanced-llm)
- Changed all navigation links from `.md` file references to absolute URL paths
- Corrected all section slugs to section IDs in URLs

**Files Modified**:
- `content/multi-agents/*.md` - 16 files
- `content/agents/*.md` - 15 files  
- `content/mcp/*.md` - 12 files
- `content/advanced-llm/*.md` - 3 files
- `content/fine-tuning/*.md` - 6 files

**Lesson Learned**: Always use section `id` (not `slug`) when constructing internal navigation URLs

---

### Phase 12: Interactive Learning Features ✅
**Objective**: Implement advanced interactive learning tools.
**Status**: ✅ Complete

#### Sub-Phase 12.1: Live Code Sandbox ✅
- [x] Integrate Pyodide for in-browser Python execution
- [x] Create code editor component with syntax highlighting (Monaco Editor)
- [x] Implement code execution with output display
- [x] Add error handling and timeout management
- [x] Create sandbox presets for each lab type (basic, dataScience, llmBasics, rag, tokenization, promptEngineering, agents)

#### Sub-Phase 12.2: Token Visualizer ✅
- [x] Create real-time tokenization display
- [x] Implement token counting for multiple providers (OpenAI, Anthropic, Google, xAI)
- [x] Add cost estimation based on token count
- [x] Visualize token boundaries in text with color coding
- [x] Provider comparison view

#### Sub-Phase 12.3: Embedding Playground ✅
- [x] Implement embedding generation interface (mock embeddings for demo)
- [x] Add dimensionality reduction (PCA, t-SNE, UMAP simplified implementations)
- [x] Create 2D interactive visualization with Canvas rendering
- [x] Show semantic similarity calculations (cosine similarity, similarity matrix)
- [x] Add clustering visualization (k-means)
- [x] Quick-add example datasets

#### Sub-Phase 12.4: Comparison Mode ✅
- [x] Create side-by-side provider comparison UI
- [x] Implement parallel API calls (simulated for demo)
- [x] Display response differences
- [x] Add response time and cost comparison
- [x] Multiple model selection per provider

#### Sub-Phase 12.5: Cost Calculator ✅
- [x] Create interactive pricing tool with sliders
- [x] Add provider pricing database (13 models)
- [x] Implement usage estimation (requests, tokens)
- [x] Show daily/monthly/yearly cost projections
- [x] Preset scenarios (chatbot, RAG, agent, batch, startup, enterprise)
- [x] Cost optimization tips

**Files Created:**
- `static/js/pyodide-sandbox.js` - Pyodide integration with presets
- `static/js/code-editor.js` - Monaco Editor wrapper with fallback
- `static/js/token-visualizer.js` - Tokenization and cost estimation
- `static/js/embedding-playground.js` - Embedding visualization and clustering
- `templates/content/sandbox-lab.html` - Full sandbox lab template
- `templates/pages/token-visualizer.html` - Token visualizer tool page
- `templates/pages/embedding-playground.html` - Embedding playground page
- `templates/pages/comparison.html` - Provider comparison page
- `templates/pages/cost-calculator.html` - Cost calculator page
- `templates/pages/tools.html` - Tools index page

**Routes Added:**
- `/sandbox` - Live code sandbox demo
- `/tools` - Tools index page
- `/tools/token-visualizer` - Token visualizer
- `/tools/embedding-playground` - Embedding playground
- `/tools/comparison` - Provider comparison
- `/tools/cost-calculator` - Cost calculator

**Deliverables**: ✅ Full interactive learning toolkit

---

### Phase 13: Assessment & Gamification ✅
**Objective**: Implement assessment system and gamification features.
**Status**: ✅ Complete

#### Sub-Phase 13.1: Quiz System ✅
- [x] Create quiz component with multiple question types (`static/js/quiz-engine.js`)
  - Multiple choice, multiple select, true/false, fill-in-blank, ordering
  - Progress bar and question navigation
  - Immediate feedback with explanations
- [x] Implement quiz for each concept section (`QuizData` in quiz-engine.js)
  - LLM Basics quiz (5 questions)
  - RAG quiz (4 questions)
  - Agents quiz (4 questions)
- [x] Add immediate feedback with explanations
- [x] Track quiz scores and progress (localStorage: quizResults)
- [x] Create quiz page template (`templates/pages/quiz.html`)
- [x] Add quiz routes (`/quiz/{quiz_type}`)
- [x] Create quizzes index page (`templates/pages/quizzes.html`)
- [x] Add quizzes route (`/quizzes`) with 6 available quizzes

#### Sub-Phase 13.2: Coding Challenges ✅
- [x] Create auto-graded coding exercise framework (`static/js/coding-challenge.js`)
  - Pyodide-based in-browser Python execution
  - Monaco editor with syntax highlighting
- [x] Implement test case runner with visual feedback
  - Pass/fail indicators for each test
  - Expected vs actual output comparison
- [x] Add hints system (3 hints per challenge with reveal progression)
- [x] Create challenge templates:
  - `templates/pages/challenge.html` - Individual challenge page
  - `templates/pages/challenges.html` - Challenges index page
- [x] Add challenge routes (`/challenges`, `/challenge/{challenge_type}`)
- [x] Sample challenges: twoSum, reverseString, countTokens, cosineSimilarity

#### Sub-Phase 13.3: Gamification ✅
- [x] Design achievement badge system (`static/js/gamification.js`)
  - 20+ badge definitions across 5 categories
  - Rarity tiers: common, uncommon, rare, legendary
  - Badge categories: getting-started, modules, streaks, achievements, mastery
- [x] Implement skill tree visualization
  - 4 tiers: Foundations, Intermediate, Advanced, Expert
  - Module-based skill progression
- [x] Add streak tracking with activity heatmap
  - Current streak, max streak, total days
  - Activity history visualization
- [x] Create progress milestones and tracking
- [x] Create achievements page (`templates/pages/achievements.html`)
  - Stats overview (lessons, labs, quizzes, challenges, badges, modules)
  - Streak display with fire animation
  - Activity heatmap
  - Badge gallery with unlock status
  - Skill tree visualization
- [x] Add achievements route (`/achievements`)
- [x] Add Achievements and Challenges to sidebar navigation
- [x] Add Assessment & Gamification section to home page

#### Sub-Phase 13.4: Certification
- [ ] Create completion certificate generator
- [ ] Design certificate template (PDF)
- [ ] Add verification system
- [ ] Implement LinkedIn sharing

**Deliverables**: ✅ Complete assessment and gamification system (except certification)

---

### Phase 14: Content Enrichment ✅
**Objective**: Add supplementary content and reference materials.
**Status**: ✅ Complete

#### Sub-Phase 14.1: Glossary System ✅
- [x] Create searchable glossary database (`static/js/glossary.js`)
  - 50+ AI/ML terms with full definitions
  - 13 categories (basics, prompting, tokens, parameters, embeddings, rag, agents, training, architecture, models, infrastructure, frameworks, challenges)
  - Related terms and cross-references
  - Links to relevant learning modules
- [x] Implement inline term definitions (tooltips)
  - Hover tooltips for glossary terms
  - data-glossary attribute support
- [x] Add cross-references to concepts
  - Related terms navigation
  - "Learn More" links to module content
- [x] Create glossary search UI (`templates/pages/glossary.html`)
  - Full-text search across terms and definitions
  - Category filtering with pill buttons
  - A-Z letter navigation with clickable letters
  - Term detail modal with full info

#### Sub-Phase 14.2: Case Studies ✅
- [x] Create case studies page (`templates/pages/case-studies.html`)
- [x] Write customer support bot case study
  - RAG + Agents architecture
  - 73% auto-resolution rate, $0.02/ticket
- [x] Create code review agent case study
  - Multi-agent system (Security, Style, Logic agents)
  - 89% issues caught, -65% review time
- [x] Add RAG knowledge base case study
  - Hybrid search with reranking
  - 94% retrieval accuracy, 100% citation rate
- [x] Add document processing case study
  - Multimodal (GPT-4 Vision + OCR)
  - 96% extraction accuracy
- [x] Include cost/performance analysis table
- [x] Add common success patterns section
- [x] Expandable detail sections for each case study

#### Sub-Phase 14.3: Decision Guides ✅
- [x] Create decision guides page (`templates/pages/decision-guides.html`)
- [x] Create "RAG vs Fine-tuning vs Prompting" decision tree
  - 4 question flow with dynamic recommendations
  - Based on data freshness, volume, citations, budget
- [x] Build "Which LLM Provider" selector
  - Considers use case, cost priority, multimodal needs, privacy
  - Recommends specific models and alternatives
  - Updated with January 2026 model leaderboard data
- [x] Add "Vector DB Comparison" guide
  - ChromaDB, Pinecone, Weaviate, pgvector recommendations
  - Based on scale, deployment, filtering needs
- [x] Add "Self-host vs API" decision guide
  - Based on volume, data sensitivity, team expertise
- [x] Implement interactive decision flows with Alpine.js
- [x] Add quick reference comparison table

#### Sub-Phase 14.4: Failure Modes Gallery ✅
- [x] Create failure modes page (`templates/pages/failure-modes.html`)
- [x] Document common prompt injection attacks
  - Direct injection with detection/prevention code
  - Indirect injection via external data
  - Jailbreak attempts with defense layers
- [x] Show hallucination examples and fixes
  - Factual hallucination with RAG mitigation
  - Citation hallucination with verification
  - Confident nonsense detection
- [x] Create RAG failure modes guide
  - Poor retrieval quality (hybrid search fix)
  - Context window overflow (compression, ordering)
  - Bad chunking (semantic chunking, parent retrieval)
- [x] Add agent failure documentation
  - Infinite loops with detection/prevention
  - Tool misuse with parameter validation
  - Goal drift with explicit tracking
- [x] Add failure prevention checklist
- [x] Tabbed navigation for failure mode categories

**Deliverables**: ✅ Comprehensive reference content with 4 major resource pages

---

### Post-Phase 14: Fixes & Updates ✅
**Status**: ✅ Complete
**Date**: January 21, 2026

#### Summary
After Phase 14 completion, addressed several issues and enhancements:

#### Issues Fixed

**1. Glossary Letter Navigation**
- **Problem**: A-Z letter buttons were not clickable
- **Root Cause**: `allAvailableLetters` was computed in Alpine `init()` after template render
- **Solution**: Compute at module load time using IIFE

**2. Alpine.js Script Loading Order**
- **Problem**: Glossary component not registering
- **Root Cause**: `{% block head %}` was after Alpine.js CDN in base.html
- **Solution**: Moved `{% block head %}` before Alpine.js so component scripts load first

**3. Failure Modes Tabs Not Working**
- **Problem**: Tab content not showing when clicking tab buttons
- **Root Cause**: `x-data` was on tabs div, but content sections were outside that scope
- **Solution**: Moved `x-data` to parent container

**4. Case Studies Broken Links**
- **Problem**: Links like `/case-studies/customer-support` had no routes
- **Solution**: Replaced links with expandable detail buttons using Alpine.js

**5. Knowledge Quizzes Navigation**
- **Problem**: No way to navigate to quizzes from sidebar
- **Solution**: Created `/quizzes` listing page with 6 available quizzes, added sidebar link

**6. Home Page Quiz Link Inconsistency**
- **Problem**: Home page linked directly to `/quiz/llmBasics` instead of `/quizzes`
- **Solution**: Updated to link to `/quizzes` for consistency

**7. LLM Model and Pricing Updates**
- **Problem**: Decision guides had outdated model information
- **Solution**: Updated with January 2026 leaderboard data from llm-stats.com
  - Claude Opus 4.5 (#1, 87% GPQA, $5/$25 per 1M tokens)
  - Gemini 3 Pro (#2, 91.9% GPQA, $2/$12)
  - GPT-5.2 (#3, 92.4% GPQA, $1.75/$14)
  - DeepSeek-V3.2 (budget option, $0.28/$0.42)
- Updated files: token-visualizer.js, comparison.html, decision-guides.html, glossary.js, case-studies.html

**8. Dark/Light Mode Toggle**
- **Problem**: Toggle button existed but didn't work properly
- **Attempted Solution**: Added dark mode classes to all components
- **Issue**: Many page templates still had hardcoded dark styles
- **Final Decision**: Reverted to dark mode only for consistency

#### Files Modified
- `templates/base.html` - Script loading order, sidebar link, reverted to dark mode only
- `static/js/glossary.js` - Letter navigation fix, model definitions update
- `templates/pages/glossary.html` - Letter nav UI
- `templates/pages/failure-modes.html` - Alpine.js scope fix
- `templates/pages/case-studies.html` - Expandable details, model updates
- `templates/pages/decision-guides.html` - LLM provider recommendations with leaderboard data
- `templates/pages/quizzes.html` - New quizzes listing page
- `templates/pages/index.html` - Quiz link fix
- `static/js/token-visualizer.js` - New models and pricing
- `templates/pages/comparison.html` - Updated model lists and pricing
- `src/ai_tutorial/routers/pages.py` - Added `/quizzes` route

---

### Phase 15: Advanced Topics ✅ COMPLETE
**Objective**: Add advanced AI engineering topics.

#### Sub-Phase 15.1: Structured Outputs ✅
- [x] Content: JSON mode and schemas
- [x] Content: Function calling best practices
- [x] Content: Pydantic integration
- [x] Lab: Build type-safe LLM outputs

#### Sub-Phase 15.2: Guardrails & Safety ✅
- [x] Content: Prompt injection prevention
- [x] Content: Content filtering
- [x] Content: Output validation
- [x] Lab: Implement safety guardrails

#### Sub-Phase 15.3: Observability ✅
- [x] Content: LangSmith integration
- [x] Content: Phoenix/Arize monitoring
- [x] Content: Custom logging and tracing
- [x] Lab: Set up LLM observability

#### Sub-Phase 15.4: Cost Optimization ✅
- [x] Content: Semantic caching strategies
- [x] Content: Prompt compression
- [x] Content: Model routing (cheap vs expensive)
- [x] Lab: Implement semantic cache

**Deliverables**: Advanced topics module with 16 content files

**Files Created**:
- `content/advanced-topics/json-mode-schemas.md` - JSON mode and structured outputs
- `content/advanced-topics/function-calling-practices.md` - Function calling best practices
- `content/advanced-topics/pydantic-integration.md` - Type-safe outputs with Pydantic
- `content/advanced-topics/structured-outputs-lab.md` - Hands-on lab for structured extraction
- `content/advanced-topics/prompt-injection-prevention.md` - Security against injection attacks
- `content/advanced-topics/content-filtering.md` - Multi-layer content filtering
- `content/advanced-topics/output-validation.md` - Validating LLM outputs
- `content/advanced-topics/safety-guardrails-lab.md` - Comprehensive safety implementation lab
- `content/advanced-topics/langsmith-observability.md` - LangSmith tracing and monitoring
- `content/advanced-topics/phoenix-arize-monitoring.md` - Phoenix/Arize observability
- `content/advanced-topics/custom-logging-tracing.md` - Custom logging systems
- `content/advanced-topics/observability-lab.md` - Full observability system lab
- `content/advanced-topics/semantic-caching.md` - Semantic caching for cost savings
- `content/advanced-topics/prompt-compression.md` - Token reduction techniques
- `content/advanced-topics/model-routing.md` - Smart model routing strategies
- `content/advanced-topics/cost-optimization-lab.md` - Cost-optimized LLM system lab

**Registry Updated**: Added "Advanced Topics" module with 4 sections to `registry.py`

---

### Phase 16: Progress Tracking & User Experience ✅ COMPLETE
**Objective**: Implement user progress and accessibility.

#### Sub-Phase 16.1: Progress Persistence ✅
- [x] Implemented comprehensive progress-tracker.js (~600 lines)
- [x] Local storage with structured data (completedItems, labResults, quizScores, etc.)
- [x] Achievement system with 20+ achievements across 4 categories
- [x] Streak tracking with daily/longest streak calculation
- [x] Lab and quiz result persistence
- [x] Export/import progress functionality
- [x] Server sync with offline queue

**Files Created/Modified**:
- `static/js/progress-tracker.js` - Comprehensive progress tracking module
- `templates/pages/progress.html` - Enhanced progress dashboard
- `templates/content/lesson.html` - Updated to use ProgressTracker
- `templates/content/lab.html` - Updated to use ProgressTracker

#### Sub-Phase 16.2: Learning Paths UI ✅
- [x] Created enhanced learning paths page with filtering
- [x] Added grid/list view toggle
- [x] Implemented difficulty filtering (beginner/intermediate/advanced)
- [x] Real-time progress tracking per module
- [x] Active path banner with continue button
- [x] Expandable path details with module list

**Files Created/Modified**:
- `templates/pages/learning-paths.html` - Enhanced learning paths interface
- `src/ai_tutorial/routers/pages.py` - Updated to pass all_modules_dict

#### Sub-Phase 16.3: Accessibility ✅
- [x] Created accessibility.js with full keyboard navigation
- [x] Implemented skip links (Alt+1 main, Alt+2 nav, Alt+A accessibility panel)
- [x] Added ARIA landmarks (role=main, role=navigation, role=banner)
- [x] Added proper aria-label and aria-hidden attributes
- [x] Implemented high contrast mode
- [x] Added reduce motion support
- [x] Created accessibility settings panel
- [x] Added focus indicators and visible focus states
- [x] Screen reader announcements via live regions

**Files Created/Modified**:
- `static/js/accessibility.js` - Accessibility manager module
- `templates/base.html` - Added ARIA roles, skip links, accessibility button

#### Sub-Phase 16.4: Polish & Responsiveness ✅
- [x] Created theme-manager.js for dark/light mode
- [x] Added theme toggle button in header
- [x] Implemented PWA service worker for offline support
- [x] Created offline.html fallback page
- [x] Added manifest.json for installable PWA
- [x] Added print styles for clean printing
- [x] Light mode CSS styles
- [x] Responsive sidebar behavior on mobile
- [x] Progress indicator from ProgressTracker

**Files Created/Modified**:
- `static/js/theme-manager.js` - Theme manager with dark/light toggle
- `static/sw.js` - Service worker for caching and offline support
- `static/offline.html` - Offline fallback page
- `static/manifest.json` - PWA manifest
- `templates/base.html` - PWA meta tags, theme toggle, responsive styles

**Deliverables**: ✅ Complete user experience with accessibility

---

### Phase 17: Developer Experience & Deployment
**Objective**: Ensure quality and prepare for deployment.

#### Sub-Phase 17.1: Testing ✅
- [x] Write unit tests for LLM adapters (24 tests)
- [x] Test content registry (29 tests)
- [x] Test route handlers (28 tests)
- [x] Test export functionality (13 tests)
- [x] **Total: 94 tests passing**

#### Sub-Phase 17.2: Documentation ✅
- [x] Create README.md with setup instructions
- [x] Document API key requirements
- [x] Create CONTRIBUTING.md guidelines
- [x] Write docs/ARCHITECTURE.md documentation

#### Sub-Phase 17.3: Docker & Deployment ✅
- [x] Create Dockerfile (multi-stage production build)
- [x] Create Dockerfile.dev (hot reload development)
- [x] Set up docker-compose.yml with Ollama
- [x] Set up docker-compose.dev.yml for development
- [x] Create .devcontainer for GitHub Codespaces
- [x] Create deployment scripts (deploy.sh, deploy.ps1)
- [x] Create .dockerignore

#### Sub-Phase 17.4: Code Export & Sharing ✅
- [x] Implement GitHub Gist URL generation
- [x] Add download as .py option
- [x] Create shareable lab result links
- [x] Add social sharing options (Twitter, LinkedIn, Reddit, Facebook)
- [x] URL-based code encoding/decoding
- [x] Share dialog UI in main.js

**Deliverables**: ✅ Production-ready application with full DX

---

## Concept Dependency Graph

```
                         ┌─────────────────────────────────────────┐
                         │         LLM Fundamentals                │
                         │   (Tokens, Prompts, Parameters)         │
                         └─────────────────┬───────────────────────┘
                                           │
          ┌────────────────────────────────┼────────────────────────────────┐
          │                                │                                │
          ▼                                ▼                                ▼
  ┌───────────────────┐          ┌─────────────────┐            ┌─────────────────┐
  │  Advanced LLM     │          │  Tool Use &     │            │    Prompt       │
  │  Concepts         │          │  Function       │            │   Engineering   │
  └────────┬──────────┘          │   Calling       │            └─────────────────┘
           │                     └────────┬────────┘
           ▼                              │
  ┌───────────────────┐                   │
  │   Fine-tuning     │                   │
  │  (LoRA, QLoRA)    │                   │
  └────────┬──────────┘                   │
           │                              │
           ▼                              │
  ┌───────────────────┐                   │
  │  Self-Hosting     │                   │
  │  LLMs (Ollama,    │                   │
  │  vLLM, llama.cpp) │                   │
  └────────┬──────────┘                   │
           │                              │
           │    ┌─────────────────────────┘
           │    │
           ▼    ▼
  ┌───────────────────┐          ┌─────────────────┐
  │   Embeddings      │          │                 │
  │                   │          │                 │
  └────────┬──────────┘          │                 │
           │                     │                 │
           ▼                     │                 │
  ┌───────────────────┐          │                 │
  │  Vector Databases │          │                 │
  │  (ChromaDB, etc.) │          │                 │
  └────────┬──────────┘          │                 │
           │                     │                 │
           ▼                     │                 │
  ┌───────────────────┐          │                 │
  │     Basic RAG     │◄─────────┘                 │
  │                   │                            │
  └────────┬──────────┘                            │
           │                                       │
           ▼                                       │
  ┌───────────────────┐                            │
  │   Advanced RAG    │                            │
  │  (HyDE, Self-RAG, │                            │
  │   RAG Fusion)     │                            │
  └────────┬──────────┘                            │
           │                                       │
           │              ┌───────────────┐        │
           │              │     MCP       │        │
           │              │               │        │
           │              └───────┬───────┘        │
           │                      │                │
           ▼                      ▼                │
  ┌─────────────────────────────────────────────┐  │
  │              AI Agents                      │◄─┘
  │       (ReAct, Tool Use, Memory)             │
  └───────────────────┬─────────────────────────┘
                      │
                      ▼
  ┌─────────────────────────────────────────────┐
  │         Multi-Agent Workflows               │
  │      (Orchestration, Collaboration)         │
  └─────────────────────────────────────────────┘
```

---

## Directory Structure (Planned)

```
agent_tutorial/
├── pyproject.toml
├── uv.lock
├── README.md
├── PROJECT_DOCUMENT.md
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── .devcontainer/                  # GitHub Codespaces
│   └── devcontainer.json
├── src/
│   └── ai_tutorial/
│       ├── __init__.py
│       ├── main.py                 # FastAPI entry point
│       ├── config.py               # Configuration management
│       ├── providers/              # LLM provider adapters
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── openai_provider.py
│       │   ├── anthropic_provider.py
│       │   ├── google_provider.py
│       │   └── grok_provider.py
│       ├── routers/                # API routes
│       │   ├── __init__.py
│       │   ├── pages.py
│       │   ├── api.py
│       │   ├── labs.py
│       │   ├── quiz.py
│       │   └── certificates.py
│       ├── services/               # Business logic
│       │   ├── __init__.py
│       │   ├── llm_service.py
│       │   ├── rag_service.py
│       │   ├── agent_service.py
│       │   ├── quiz_service.py
│       │   ├── gamification_service.py
│       │   ├── cost_calculator.py
│       │   └── cache_service.py
│       ├── models/                 # Pydantic models
│       │   ├── __init__.py
│       │   ├── schemas.py
│       │   ├── quiz_models.py
│       │   └── progress_models.py
│       ├── tools/                  # Interactive tools
│       │   ├── __init__.py
│       │   ├── token_visualizer.py
│       │   ├── embedding_playground.py
│       │   ├── cost_calculator.py
│       │   └── comparison_tool.py
│       └── utils/                  # Utilities
│           ├── __init__.py
│           ├── helpers.py
│           └── certificate_generator.py
├── static/
│   ├── css/
│   │   ├── styles.css
│   │   └── themes/
│   │       ├── light.css
│   │       └── dark.css
│   ├── js/
│   │   ├── main.js
│   │   ├── code-runner.js
│   │   ├── pyodide-worker.js       # In-browser Python
│   │   ├── progress.js
│   │   ├── quiz.js
│   │   ├── gamification.js
│   │   ├── token-visualizer.js
│   │   ├── embedding-viz.js
│   │   └── cost-calculator.js
│   ├── images/
│   │   └── badges/                 # Achievement badges
│   └── data/
│       ├── glossary.json
│       ├── pricing.json            # LLM provider pricing
│       └── quiz-questions/
├── templates/
│   ├── base.html
│   ├── components/
│   │   ├── sidebar.html
│   │   ├── code-block.html
│   │   ├── lab-runner.html
│   │   ├── concept-link.html
│   │   ├── quiz.html
│   │   ├── badge.html
│   │   ├── skill-tree.html
│   │   ├── token-visualizer.html
│   │   ├── embedding-playground.html
│   │   ├── cost-calculator.html
│   │   ├── comparison-mode.html
│   │   ├── decision-tree.html
│   │   └── glossary-tooltip.html
│   ├── pages/
│   │   ├── index.html
│   │   ├── settings.html
│   │   ├── about.html
│   │   ├── learning-paths.html
│   │   ├── achievements.html
│   │   ├── glossary.html
│   │   └── certificate.html
│   └── content/
│       ├── llm/
│       │   ├── introduction.html
│       │   ├── prompt-engineering.html
│       │   ├── parameters.html
│       │   └── tokens.html
│       │   ├── introduction.html
│       │   ├── prompt-engineering.html
│       │   ├── parameters.html
│       │   └── tokens.html
│       ├── advanced-llm/
│       │   ├── architectures.html
│       │   ├── multimodal.html
│       │   └── moe.html
│       ├── fine-tuning/
│       │   ├── introduction.html
│       │   ├── cloud-finetuning.html
│       │   ├── local-finetuning.html
│       │   ├── lora-qlora.html
│       │   └── best-practices.html
│       ├── self-hosting/
│       │   ├── introduction.html
│       │   ├── ollama.html
│       │   ├── vllm.html
│       │   ├── quantization.html
│       │   └── deployment.html
│       ├── embeddings/
│       │   ├── introduction.html
│       │   └── visualization.html
│       ├── vector-db/
│       │   ├── introduction.html
│       │   └── operations.html
│       ├── rag/
│       │   ├── introduction.html
│       │   ├── chunking.html
│       │   ├── retrieval.html
│       │   ├── advanced.html
│       │   ├── hyde-selfrag.html
│       │   ├── production.html
│       │   └── evaluation.html
│       ├── mcp/
│       │   ├── introduction.html
│       │   ├── servers.html
│       │   └── clients.html
│       ├── agents/
│       │   ├── introduction.html
│       │   ├── tool-use.html
│       │   └── agent-loop.html
│       └── multi-agents/
│           ├── introduction.html
│           ├── orchestration.html
│           └── patterns.html
├── labs/
│   ├── __init__.py
│   ├── lab_runner.py
│   ├── llm_labs/
│   │   ├── first_api_call.py
│   │   ├── prompt_playground.py
│   │   ├── parameter_explorer.py
│   │   └── multimodal_lab.py
│   ├── finetuning_labs/
│   │   ├── prepare_dataset.py
│   │   ├── openai_finetune.py
│   │   ├── local_lora_finetune.py
│   │   └── evaluate_model.py
│   ├── selfhosting_labs/
│   │   ├── ollama_setup.py
│   │   ├── benchmark_models.py
│   │   ├── vllm_deploy.py
│   │   └── local_app.py
│   ├── embedding_labs/
│   │   └── visualize_embeddings.py
│   ├── rag_labs/
│   │   ├── document_processor.py
│   │   ├── retrieval_comparison.py
│   │   ├── rag_chatbot.py
│   │   ├── reranking_pipeline.py
│   │   ├── self_correcting_rag.py
│   │   ├── multi_source_rag.py
│   │   └── rag_evaluation.py
│   ├── mcp_labs/
│   │   ├── simple_server.py
│   │   └── mcp_client.py
│   ├── agent_labs/
│   │   ├── react_agent.py
│   │   └── research_agent.py
│   ├── multi_agent_labs/
│   │   ├── writer_reviewer.py
│   │   ├── research_team.py
│   │   └── coding_team.py
│   └── advanced_labs/
│       ├── guardrails_lab.py
│       ├── observability_lab.py
│       └── semantic_cache_lab.py
├── quizzes/
│   ├── llm_quiz.json
│   ├── rag_quiz.json
│   ├── agents_quiz.json
│   └── ...
├── case_studies/
│   ├── customer_support_bot.md
│   ├── code_review_agent.md
│   └── rag_knowledge_base.md
├── certificates/
│   └── template.html
└── tests/
    ├── __init__.py
    ├── test_providers.py
    ├── test_labs.py
    ├── test_services.py
    ├── test_quizzes.py
    └── test_accessibility.py
```

---

## Phase Execution Log

### Phase 1: Project Foundation & Infrastructure
**Status**: ✅ Completed
**Start Date**: January 20, 2026
**End Date**: January 20, 2026
**User Acceptance**: ✅ Approved

#### Summary
Successfully created the complete project foundation with FastAPI server, TailwindCSS templates, and multi-LLM provider support structure. The server runs on port 8080 and includes a full navigation system with learning paths, glossary, settings page for API key management, and a responsive sidebar layout with dark mode support.

#### Code Added
**Core Application Files:**
- `pyproject.toml` - Project configuration with all dependencies (FastAPI, LLM SDKs, ChromaDB, etc.)
- `src/ai_tutorial/__init__.py` - Package initialization with version info
- `src/ai_tutorial/main.py` - FastAPI application entry point with lifespan handler
- `src/ai_tutorial/config.py` - Pydantic-settings based configuration system

**Routers:**
- `src/ai_tutorial/routers/__init__.py` - Router package initialization
- `src/ai_tutorial/routers/pages.py` - HTML page routes (learn, settings, paths, glossary, about)
- `src/ai_tutorial/routers/api.py` - API endpoints (validate-key, chat, providers, progress)
- `src/ai_tutorial/routers/labs.py` - Lab listing and execution endpoints

**Templates:**
- `templates/base.html` - Base template with TailwindCSS, HTMX, Alpine.js, responsive sidebar, dark mode
- `templates/pages/index.html` - Home page with learning path overview
- `templates/pages/settings.html` - API key management with localStorage persistence
- `templates/pages/learning-paths.html` - 6 predefined learning tracks with time estimates
- `templates/pages/glossary.html` - Searchable A-Z AI/ML glossary
- `templates/pages/about.html` - About page with project information

**Static Files:**
- `static/css/styles.css` - Custom CSS styles
- `static/js/main.js` - JavaScript utilities (clipboard, progress tracking, API key management)

**Configuration Files:**
- `.env.example` - Environment variable template
- `README.md` - Comprehensive project documentation
- `.python-version` - Python version pinned to 3.12.10

#### Challenges & Solutions
1. **Python Version Compatibility**: Initial attempt with Python 3.11 failed due to network certificate issues when downloading Python. Used existing Python 3.12.10 installation instead.
2. **ONNX Runtime Compatibility**: Python 3.9 incompatible with chromadb's onnxruntime dependency. Updated to Python 3.10+ requirement.
3. **Port 8000 Blocked**: Default port 8000 was blocked by Windows. Changed to port 8080 for development server.

---

### Phase 2: API Key Management & LLM Integration Layer
**Status**: ✅ Completed
**Start Date**: January 20, 2026
**End Date**: January 20, 2026
**User Acceptance**: ✅ Approved

#### Summary
Implemented a comprehensive LLM provider abstraction layer supporting 4 providers (OpenAI, Anthropic, Google, xAI) with a unified interface. Created a factory pattern for provider instantiation, streaming response handling with Server-Sent Events, and real API key validation. Enhanced the settings page with test buttons, status indicators, and model selection. Updated to latest 2026 models (GPT-5.2, Claude 4.5, Gemini 3, Grok 4).

#### Code Added
**Provider Abstraction Layer (`src/ai_tutorial/providers/`):**
- `__init__.py` - Module exports with all classes and errors
- `base.py` - Abstract base class `LLMProvider`, dataclasses (`LLMMessage`, `LLMResponse`, `StreamChunk`, `ProviderConfig`), `MessageRole` enum, and custom exceptions (`ProviderError`, `AuthenticationError`, `RateLimitError`, etc.)
- `openai_provider.py` - OpenAI adapter with GPT-4o, GPT-4, GPT-3.5 support
- `anthropic_provider.py` - Anthropic adapter with Claude 3.5, Claude 3 support
- `google_provider.py` - Google Gemini adapter with Gemini 1.5 Pro/Flash support
- `xai_provider.py` - xAI Grok adapter using OpenAI-compatible API
- `factory.py` - `ProviderFactory` class with provider registration and creation

**API Router Updates (`src/ai_tutorial/routers/api.py`):**
- Real `/api/validate-key` endpoint with actual provider validation
- `/api/chat` endpoint with streaming SSE support
- `/api/providers` endpoint returning dynamic provider metadata
- Error handling for authentication, rate limits, and provider errors

**Settings Page Enhancement (`templates/pages/settings.html`):**
- Dynamic provider cards with test buttons
- Real-time status indicators (Configured, Valid, Invalid, Testing)
- Test result messages display
- "Test All Keys" and "Clear All Keys" buttons
- Default model selection per provider
- Model preferences panel

#### Challenges & Solutions
1. **Import Error**: `MessageRole` not exported from providers module. Fixed by adding all necessary exports to `__init__.py`.
2. **Google GenAI Deprecation Warning**: The `google-generativeai` package shows deprecation warning. Library still works; will migrate to `google.genai` in future phase.

---

### Phase 3: Content Architecture & Navigation System
**Status**: ✅ Completed
**Start Date**: Phase 2 completion
**End Date**: Current
**User Acceptance**: ✅ Approved

#### Summary
Implemented comprehensive content architecture with dynamic navigation system:
- Created content data models (Module, Section, Page, Lab, Quiz, LearningPath)
- Built ContentRegistry with 9 learning modules fully defined
- Implemented Markdown renderer with Pygments syntax highlighting
- Updated pages router with dynamic content serving
- Created new templates for modules, lessons, labs, quizzes, and learning paths
- Added progress tracking page with localStorage persistence
- Implemented breadcrumb navigation
- Added interactive labs index page
- Updated sidebar navigation to dynamically render modules from registry
- Home page now shows all modules dynamically

#### Code Added
**Content System**:
- `src/ai_tutorial/content/__init__.py` - Module exports
- `src/ai_tutorial/content/models.py` - Data models (Module, Section, Page, Lab, Quiz, LearningPath, PageType, Difficulty)
- `src/ai_tutorial/content/registry.py` - ContentRegistry singleton with 9 learning modules and 4 learning paths
- `src/ai_tutorial/content/renderer.py` - MarkdownRenderer with Pygments highlighting

**Templates**:
- `templates/content/module.html` - Module overview page
- `templates/content/lesson.html` - Lesson/section page with ToC
- `templates/content/lab.html` - Interactive lab page with code editor
- `templates/content/quiz.html` - Quiz page with progress tracking
- `templates/content/learning-path.html` - Learning path detail page
- `templates/pages/labs.html` - Labs index page
- `templates/pages/progress.html` - Progress tracking dashboard

**Updated Files**:
- `src/ai_tutorial/routers/pages.py` - Dynamic content routing
- `src/ai_tutorial/main.py` - Registry integration for home page
- `templates/base.html` - Dynamic sidebar navigation with modules from registry
- `templates/pages/index.html` - Dynamic modules grid
- `templates/pages/learning-paths.html` - Dynamic learning paths list

#### Learning Modules Defined
1. 🧠 LLM Fundamentals (Beginner)
2. 🔢 Embeddings & Vector Databases (Intermediate)
3. 📚 RAG: Retrieval Augmented Generation (Intermediate)
4. 🔧 Tool Use & Function Calling (Intermediate)
5. 🤖 AI Agents (Advanced)
6. 🔌 Model Context Protocol (MCP) (Advanced)
7. 👥 Multi-Agent Systems (Advanced)
8. ⚙️ Fine-tuning LLMs (Advanced)
9. 🏠 Self-Hosting LLMs (Intermediate)

#### Learning Paths Defined
1. 🚀 Quick Start (2 hours) - Beginner-friendly introduction
2. 📚 RAG Engineer (8 hours) - Build knowledge-based AI apps
3. 🤖 Agent Developer (12 hours) - Create autonomous AI agents
4. 🎓 Complete AI Engineer (40+ hours) - Master everything

#### Challenges & Solutions
1. **Module ID Mismatch**: Learning path model used `modules` but router expected `module_ids`. Fixed by using consistent naming.
2. **Google GenAI Warning**: Still showing deprecation warning (cosmetic only, works fine).

---

### Phase 4: Core Concepts - LLM Fundamentals
**Status**: ✅ Completed
**Start Date**: January 20, 2026
**End Date**: January 20, 2026
**User Acceptance**: ✅ Approved

#### Summary
Successfully created comprehensive LLM Fundamentals tutorial content covering 4 major sections with 12 lessons and 4 interactive labs:

1. **Introduction to LLMs** - What LLMs are, history/evolution, transformer architecture, and first API call lab
2. **Prompt Engineering** - CRISPE framework, prompt patterns, zero/few-shot learning, prompt playground lab
3. **LLM Parameters** - Temperature/sampling, token limits, stop sequences, parameter explorer lab
4. **Tokens & Context Windows** - Tokenization deep dive, context windows, context management, token counter lab

Each lesson includes comprehensive explanations, code examples, diagrams, and practical exercises.

#### Code Added
**Content Files (`content/llm-fundamentals/`):**
- `intro-to-llms/what-are-llms.md` - Introduction to Large Language Models
- `intro-to-llms/history.md` - History and evolution of LLMs
- `intro-to-llms/how-they-work.md` - Transformer architecture and attention mechanisms
- `intro-to-llms/first-api-call.md` - Lab: First API call to an LLM
- `prompt-engineering/prompt-fundamentals.md` - CRISPE framework and prompt basics
- `prompt-engineering/prompt-patterns.md` - Common prompt patterns and techniques
- `prompt-engineering/zero-few-shot.md` - Zero-shot, one-shot, few-shot learning
- `prompt-engineering/prompt-playground.md` - Lab: Interactive prompt playground
- `parameters/temperature.md` - Temperature and sampling parameters
- `parameters/tokens-limits.md` - Token limits and max tokens
- `parameters/stop-sequences.md` - Stop sequences and output control
- `parameters/parameter-explorer.md` - Lab: Parameter explorer
- `tokens/tokenization.md` - Deep dive into tokenization
- `tokens/context-windows.md` - Understanding context windows
- `tokens/managing-context.md` - Managing context effectively
- `tokens/token-counter.md` - Lab: Token counter and context analyzer

**API Updates:**
- Added `/api/lab/execute` endpoint to `api.py` for lab code execution

**Registry Updates:**
- Updated content registry with all Phase 4 sections and pages with proper content_path references

#### Challenges & Solutions
1. **Content Organization**: Structured content with clear progression from theory to practice
2. **Lab Integration**: Designed labs to be interactive with the unified LLM provider interface

---

### Phase 5: Advanced LLM Concepts & Fine-tuning
**Status**: ✅ Completed
**Start Date**: January 20, 2026
**End Date**: January 20, 2026
**User Acceptance**: ✅ Approved

#### Summary
Successfully created comprehensive content for Advanced LLM Concepts and Fine-tuning modules:

**Advanced LLM Concepts (New Module 2):**
- Model architectures deep dive (decoder-only, encoder-only, encoder-decoder)
- Attention mechanisms (self-attention, MHA, MQA, GQA, RoPE, ALiBi)
- Mixture of Experts (MoE) models with routing strategies
- Multimodal LLMs (GPT-4V, Claude Vision, Gemini) with practical API usage
- Interactive multimodal vision lab

**Fine-tuning LLMs (Module 9):**
- Decision framework for fine-tuning vs RAG vs prompting
- Types of fine-tuning (full, LoRA, QLoRA, adapters, prefix tuning)
- Dataset preparation with JSONL formats for different platforms
- Cloud fine-tuning with OpenAI, Google Vertex, and Together AI
- Local fine-tuning with PEFT, QLoRA, and Hugging Face SFTTrainer
- Best practices for evaluation, debugging, and cost optimization
- Hands-on OpenAI fine-tuning lab

#### Code Added
**Advanced LLM Content (`content/advanced-llm/`):**
- `model-architectures.md` - LLM architectures deep dive (decoder-only GPT, encoder-only BERT, encoder-decoder T5, modern innovations like GQA, RoPE, Flash Attention)
- `attention-mechanisms.md` - Self-attention, Multi-Head Attention, MQA, GQA, positional encodings (RoPE, ALiBi), Flash Attention optimizations
- `moe-models.md` - Mixture of Experts architecture, routing mechanisms, expert parallelism, models like Mixtral, Grok, DeepSeek V2
- `multimodal-llms.md` - Vision models (GPT-4V, Claude Vision, Gemini), image understanding, API usage patterns for all major providers
- `multimodal-lab.md` - Interactive lab for vision API usage with image analysis exercises

**Fine-tuning Content (`content/fine-tuning/`):**
- `why-finetune.md` - Decision framework comparing fine-tuning, RAG, and prompt engineering with cost/benefit analysis
- `types-of-finetuning.md` - Full fine-tuning, LoRA, QLoRA, adapters, prefix tuning with memory/GPU requirements
- `dataset-preparation.md` - JSONL formats (OpenAI, Alpaca, ShareGPT), data quality guidelines, validation scripts
- `cloud-finetuning.md` - OpenAI fine-tuning API, Google Vertex AI, Together AI with complete code examples
- `local-finetuning.md` - QLoRA/LoRA setup with PEFT, Hugging Face Transformers, SFTTrainer, 4-bit quantization
- `best-practices.md` - Evaluation metrics, catastrophic forgetting prevention, cost optimization, debugging techniques
- `openai-finetune-lab.md` - Hands-on lab for fine-tuning GPT models via OpenAI API

**Registry Updates (`src/ai_tutorial/content/registry.py`):**
- Added new "Advanced LLM Concepts" module (Module 2) with 3 sections and multimodal lab
- Updated Fine-tuning module (Module 9) with 4 detailed sections and proper content_path references
- Renumbered all modules (2→10 total modules)
- Updated "Complete AI Engineer" learning path to include advanced-llm module (45 hours)

#### Module Structure After Phase 5
1. 🧠 LLM Fundamentals (Beginner) - order=1
2. 🧠 Advanced LLM Concepts (Advanced) - order=2 **[NEW]**
3. 🔢 Embeddings & Vector Databases (Intermediate) - order=3
4. 📚 RAG: Retrieval Augmented Generation (Intermediate) - order=4
5. 🔧 Tool Use & Function Calling (Intermediate) - order=5
6. 🤖 AI Agents (Advanced) - order=6
7. 🔌 Model Context Protocol (MCP) (Advanced) - order=7
8. 👥 Multi-Agent Systems (Expert) - order=8
9. 🎛️ Fine-tuning LLMs (Advanced) - order=9 **[EXPANDED]**
10. 🏠 Self-Hosting LLMs (Advanced) - order=10

#### Challenges & Solutions
1. **Module Ordering**: Added new advanced-llm module required renumbering all subsequent modules (2-10)
2. **Content Depth**: Balanced technical depth with accessibility for advanced learners
3. **Provider Coverage**: Included code examples for OpenAI, Anthropic, and Google multimodal APIs

---

### Phase 6: Self-Hosting LLMs
**Status**: ✅ Completed
**Start Date**: Session 6
**End Date**: Session 6
**User Acceptance**: ✅ Approved

#### Summary
Completed comprehensive self-hosting module covering local LLM deployment from fundamentals to production. Created 12 content files across 5 sections covering why self-host, hardware requirements, model formats, quantization, inference tools (Ollama, llama.cpp, vLLM), open-source models guide, Docker deployment, and hybrid architectures.

#### Code Added
**Self-Hosting Content (`content/self-hosting/`):**
- `why-self-host.md` - Decision framework comparing self-hosting vs cloud APIs (privacy, cost, latency, customization)
- `hardware-requirements.md` - GPU/CPU/RAM requirements matrix, VRAM calculator, build examples ($1.5K-$20K)
- `model-formats.md` - GGUF, GPTQ, AWQ, Safetensors formats with conversion scripts and comparison matrix
- `quantization.md` - K-quants, I-quants explained with quality/size/speed tradeoffs and decision framework
- `ollama-setup.md` - Complete Ollama guide: installation, API usage, Modelfile creation, Python integration
- `llama-cpp.md` - Building from source, Python bindings (llama-cpp-python), server mode, grammar constraints
- `vllm.md` - PagedAttention explained, tensor parallelism, continuous batching, Docker/Kubernetes deployment
- `ollama-lab.md` - Hands-on 7-part lab: installation → API → chatbot → benchmarking
- `open-source-models.md` - Guide to Llama 3, Mistral/Mixtral, Qwen, Phi, Gemma, DeepSeek with use case recommendations
- `docker-deployment.md` - Ollama/vLLM Docker setup, multi-GPU, production stack with Nginx/Prometheus
- `integration-patterns.md` - Provider abstraction, connection pooling, retry logic, caching, queue management
- `hybrid-architectures.md` - Privacy-based routing, complexity routing, cost optimization, fallback chains

**Registry Updates (`src/ai_tutorial/content/registry.py`):**
- Updated Self-Hosting module (Module 10) with 5 sections and proper content_path references
- Section 1: Self-Hosting Introduction (4 pages)
- Section 2: Local Inference Tools (3 pages + 1 lab)
- Section 3: Model Selection (1 page)
- Section 4: Production Deployment (1 page)
- Section 5: Self-Hosted Integration (2 pages)

#### Self-Hosting Module Structure
```
🏠 Self-Hosting LLMs (Advanced) - order=10
├── 🖥️ Self-Hosting Introduction
│   ├── Why Self-Host?
│   ├── Hardware Requirements
│   ├── Model Formats
│   └── Model Quantization
├── 🔧 Local Inference Tools
│   ├── Ollama Setup & Usage
│   ├── llama.cpp Deep Dive
│   ├── vLLM for Production
│   └── [Lab] Hands-on Ollama Lab
├── 🎯 Model Selection
│   └── Open Source Model Guide
├── 🚀 Production Deployment
│   └── Docker Deployment
└── 🔌 Self-Hosted Integration
    ├── Integration Patterns
    └── Hybrid Architectures
```

#### Technical Content Highlights
- **Hardware Matrix**: VRAM requirements for 1B to 405B parameter models
- **Quantization Guide**: Q2_K through Q8_0 with quality metrics
- **Inference Tools**: Ollama (easy), llama.cpp (flexible), vLLM (production)
- **Production Patterns**: Connection pooling, caching, retry logic, load balancing

#### Challenges & Solutions
1. **Content Scope**: Balanced beginner-friendly Ollama content with advanced vLLM/llama.cpp material
2. **Hardware Complexity**: Created clear VRAM calculator and GPU recommendation tiers
3. **Tool Variety**: Provided consistent API examples across Ollama, llama.cpp, and vLLM

---

### Phase 7: Vector Databases & Embeddings
**Status**: ✅ Completed
**Start Date**: Session 7
**End Date**: Session 7
**User Acceptance**: ✅ Approved

#### Summary
Completed comprehensive embeddings and vector databases module covering semantic vectors, embedding models, vector database concepts, indexing strategies, and practical operations. Created 13 content files across 3 sections covering embeddings fundamentals, vector database architecture, similarity search algorithms, ChromaDB hands-on, and a complete semantic search engine lab.

#### Code Added
**Embeddings Content (`content/embeddings-vectors/`):**
- `what-are-embeddings.md` - Introduction to vector representations, OpenAI embedding API, similarity concepts
- `semantic-vectors.md` - Geometry of meaning, word analogies, clustering, t-SNE/UMAP visualization
- `embedding-models.md` - Model comparison (OpenAI, Cohere, Voyage, BGE, E5), selection criteria, benchmarks
- `embeddings-lab.md` - 7-part hands-on lab: generate embeddings → visualize → semantic search → analogies
- `vector-db-intro.md` - Why vector databases, ANN vs exact search, architecture components, CRUD operations
- `similarity-search.md` - Distance metrics (cosine, euclidean, dot product, Manhattan), benchmarks, selection guide
- `indexing-strategies.md` - HNSW algorithm, IVF clustering, product quantization, tuning parameters
- `vector-db-options.md` - ChromaDB, Pinecone, Qdrant, Weaviate, Milvus, pgvector with code examples
- `chromadb-lab.md` - 7-part lab: setup → collections → embeddings → querying → RAG retriever pattern
- `collections.md` - CRUD operations, multi-collection strategies, schemas, batch operations
- `querying-documents.md` - Query patterns, reranking, MMR diversity, hybrid search, caching
- `metadata-filtering.md` - Where clauses, operators ($eq, $in, $and, $or), faceted search, access control
- `semantic-search-lab.md` - Complete search engine lab: document processor, vector store, search interface

**Registry Updates (`src/ai_tutorial/content/registry.py`):**
- Updated Embeddings & Vectors module (Module 3) with 3 sections and proper content_path references
- Section 1: Understanding Embeddings (3 pages + 1 lab)
- Section 2: Vector Database Concepts (4 pages + 1 lab)
- Section 3: Practical Vector Operations (3 pages + 1 lab)

#### Embeddings Module Structure
```
🔢 Embeddings & Vector Databases (Intermediate) - order=3
├── 📊 Understanding Embeddings
│   ├── What are Embeddings?
│   ├── Semantic Vectors
│   ├── Embedding Models Comparison
│   └── [Lab] Generate & Visualize Embeddings
├── 🗄️ Vector Database Concepts
│   ├── Introduction to Vector Databases
│   ├── Similarity Search Algorithms
│   ├── Indexing Strategies
│   ├── Vector Database Options
│   └── [Lab] Set Up ChromaDB
└── ⚡ Practical Vector Operations
    ├── Working with Collections
    ├── Querying Documents
    ├── Metadata Filtering
    └── [Lab] Build Semantic Search
```

#### Technical Content Highlights
- **Embedding Models**: Coverage of OpenAI, Cohere, Voyage AI, BGE, E5 with dimension/context comparisons
- **Similarity Metrics**: Cosine (unit vectors), Euclidean (magnitude-sensitive), Dot Product (performance)
- **Indexing Algorithms**: HNSW (graph-based, best recall), IVF (cluster-based), PQ (compression)
- **Production Patterns**: Caching, batch operations, hybrid search, multi-tenancy filtering

#### Challenges & Solutions
1. **Content Depth Balance**: Covered both conceptual understanding and practical implementation with code
2. **Database Comparisons**: Provided fair comparison of 6 vector database options with use case guidance
3. **Lab Complexity**: Built progressive labs from basic embeddings to complete search engine

---

### Phase 8: RAG (Retrieval-Augmented Generation)
**Status**: ✅ Completed
**Start Date**: Session 8
**End Date**: Session 8
**User Acceptance**: ✅ Approved

#### Summary
Completed comprehensive RAG module covering fundamentals, document processing, retrieval strategies, generation with context, and advanced patterns. Created 23 content files across 5 sections covering everything from basic RAG concepts to production-ready systems with evaluation frameworks.

#### Code Added
**RAG Fundamentals (`content/rag/`):**
- `what-is-rag.md` - Introduction to RAG, when to use it, core architecture
- `rag-vs-finetuning.md` - Decision guide for RAG vs fine-tuning
- `rag-architecture.md` - Complete system design with components

**Document Processing:**
- `document-loading.md` - Multi-format loaders (PDF, HTML, Markdown, DOCX)
- `chunking-strategies.md` - Fixed, recursive, semantic, code-aware chunking
- `chunk-sizing.md` - Sizing guidelines, adaptive chunking, testing
- `document-parsing.md` - Advanced PDF/HTML parsing, OCR, JS-rendered content
- `document-processor-lab.md` - Complete pipeline lab

**Retrieval Strategies:**
- `basic-retrieval.md` - Similarity search, MMR, parent document retrieval
- `hybrid-search.md` - BM25 + vector search, RRF fusion, adaptive weights
- `reranking.md` - Cohere, cross-encoders, LLM reranking, cascading
- `multi-query.md` - Query decomposition, step-back, HyDE, RAG Fusion
- `contextual-compression.md` - LLM compression, extraction-based, quality checks
- `retrieval-strategies-lab.md` - Comparison lab with evaluation framework

**Generation with Context:**
- `rag-prompts.md` - Structured prompts, role-based, edge case handling
- `context-overflow.md` - Token budgets, smart truncation, map-reduce
- `citation-tracking.md` - Inline citations, verification, rich display
- `streaming-sources.md` - Streaming with citations, FastAPI SSE
- `rag-chatbot-lab.md` - Complete chatbot with memory and streaming

**Advanced RAG:**
- `advanced-patterns.md` - HyDE, RAG Fusion, Self-RAG, CRAG implementations
- `production-architecture.md` - Scaling, caching, monitoring, deployment
- `evaluation-optimization.md` - RAGAS metrics, retrieval evaluation, optimization
- `advanced-rag-lab.md` - Production-ready system with full evaluation

**Registry Updates (`src/ai_tutorial/content/registry.py`):**
- Updated RAG module (Module 4) with 5 sections and proper content_path references
- Section 1: RAG Fundamentals (3 pages)
- Section 2: Document Processing (4 pages + 1 lab)
- Section 3: Retrieval Strategies (5 pages + 1 lab)
- Section 4: Generation with Context (4 pages + 1 lab)
- Section 5: Advanced RAG (3 pages + 1 lab)

#### RAG Module Structure
```
📚 RAG: Retrieval Augmented Generation (Intermediate) - order=4
├── 📖 RAG Fundamentals
│   ├── What is RAG?
│   ├── RAG vs Fine-Tuning
│   └── RAG Architecture
├── 📄 Document Processing
│   ├── Document Loading
│   ├── Chunking Strategies
│   ├── Chunk Sizing Guide
│   ├── Advanced Document Parsing
│   └── [Lab] Document Processor Pipeline
├── 🔍 Retrieval Strategies
│   ├── Basic Retrieval
│   ├── Hybrid Search
│   ├── Reranking Results
│   ├── Multi-Query Retrieval
│   ├── Contextual Compression
│   └── [Lab] Retrieval Strategies Lab
├── ✨ Generation with Context
│   ├── RAG Prompt Engineering
│   ├── Handling Context Overflow
│   ├── Citation Tracking
│   ├── Streaming with Sources
│   └── [Lab] RAG Chatbot Lab
└── 🚀 Advanced RAG Patterns
    ├── Advanced RAG Patterns (HyDE, Fusion, Self-RAG, CRAG)
    ├── Production RAG Architecture
    ├── RAG Evaluation & Optimization
    └── [Lab] Advanced RAG Lab
```

#### Technical Content Highlights
- **Retrieval Patterns**: Basic similarity, hybrid BM25+vector, MMR diversity, parent document
- **Advanced Retrieval**: Multi-query decomposition, HyDE, RAG Fusion with RRF
- **Reranking**: Cohere API, cross-encoders, LLM-based, cascading rerankers
- **Generation**: Structured prompts, token budgets, map-reduce, streaming SSE
- **Evaluation**: RAGAS-style metrics, retrieval precision/recall/MRR, faithfulness scoring

#### Challenges & Solutions
1. **Content Scope**: Balanced fundamentals with advanced patterns like Self-RAG and CRAG
2. **Code Examples**: Provided complete, runnable implementations with OpenAI and ChromaDB
3. **Lab Progression**: Built from simple retrieval to production-ready system with evaluation

---

### Post-Phase 8: Maintenance & Bug Fixes
**Status**: ✅ Completed
**Date**: January 21, 2026
**User Acceptance**: ✅ Approved

#### Summary
After completing Phase 8 (RAG), conducted comprehensive review and bug fixes across the Labs system. Fixed routing issues, improved difficulty level assignment for all labs, and enhanced the Labs index page with proper filtering functionality.

#### Issues Identified & Fixed

**1. Labs Page Routing Issues**
- **Problem**: Standalone lab route (`/lab/{lab_id}`) was looking for content in hardcoded `content/labs/{lab_id}.md` instead of using each lab's `content_path`
- **Solution**: Updated `pages.py` to use `lab.content_path` dynamically
- **Files Modified**: `src/ai_tutorial/routers/pages.py`

**2. Missing Lab Context for Navigation**
- **Problem**: Lab pages had generic breadcrumbs (Home > Labs > Lab) instead of showing module/section context
- **Solution**: Added `get_lab_context()` method to registry that returns (module_id, section_id) for any lab
- **Files Modified**: 
  - `src/ai_tutorial/content/registry.py` - Added `get_lab_context()` method
  - `src/ai_tutorial/routers/pages.py` - Updated breadcrumb generation

**3. Labs Page Filter Not Working**
- **Problem**: Difficulty filter buttons (All/Beginner/Intermediate/Advanced) didn't filter the lab cards - Alpine.js scope was limited to just the button container
- **Solution**: Wrapped both filter buttons and labs grid in a single `x-data` scope, added `x-show` directive to lab cards
- **Files Modified**: `templates/pages/labs.html`

**4. Missing Difficulty Levels on Labs**
- **Problem**: All labs defaulted to `Difficulty.BEGINNER` regardless of their module context
- **Solution**: Updated all 23 labs with appropriate difficulty levels based on module:
  - **Beginner (4)**: first-api-call, prompt-playground, parameter-explorer, token-counter
  - **Intermediate (10)**: embeddings-lab, chromadb-lab, semantic-search-lab, basic-rag-lab, document-processor-lab, retrieval-strategies-lab, rag-chatbot-lab, first-function-call, multi-tool
  - **Advanced (8)**: multimodal-lab, advanced-rag-lab, basic-agent, agent-with-memory, mcp-server, mcp-client, prepare-dataset, openai-finetune
  - **Expert (1)**: two-agent-system
- **Files Modified**: `src/ai_tutorial/content/registry.py`

**5. Missing Expert Filter Button**
- **Problem**: Labs page only had filters for Beginner/Intermediate/Advanced, missing Expert
- **Solution**: Added Expert filter button (red) and updated badge styling to handle Expert level
- **Files Modified**: `templates/pages/labs.html`

**6. Learning Paths Page Alignment Issue (Prior Session)**
- **Problem**: Orphaned HTML code causing alignment issues on learning-paths page
- **Solution**: Removed incomplete Self-Hosting section and stray closing tags
- **Files Modified**: `templates/pages/learning-paths.html`

**7. Missing Basic RAG Lab (Prior Session)**
- **Problem**: URL `/learn/rag/rag-fundamentals/basic-rag` returned 404 - lab existed in placeholder but not registered
- **Solution**: Created complete `basic-rag-lab.md` content file (8-part hands-on lab) and registered in registry
- **Files Added**: `content/rag/basic-rag-lab.md`
- **Files Modified**: `src/ai_tutorial/content/registry.py`

#### Code Changes Summary
```
Modified Files:
├── src/ai_tutorial/content/registry.py
│   ├── Added get_lab_context() method
│   ├── Added basic-rag-lab to RAG Fundamentals section
│   └── Updated 23 labs with proper difficulty levels
├── src/ai_tutorial/routers/pages.py
│   ├── Fixed standalone lab route to use lab.content_path
│   └── Improved breadcrumb generation with module/section context
├── templates/pages/labs.html
│   ├── Fixed Alpine.js scope for filter functionality
│   ├── Added x-show directive to lab cards for filtering
│   ├── Added Expert filter button (red)
│   └── Updated badge styling for Expert level
└── templates/pages/learning-paths.html
    └── Removed orphaned HTML causing alignment issues

Added Files:
└── content/rag/basic-rag-lab.md (8-part hands-on RAG pipeline lab)
```

#### Labs Difficulty Distribution (Post-Fix)
| Difficulty | Count | Color Badge |
|------------|-------|-------------|
| Beginner | 4 | Green |
| Intermediate | 10 | Yellow |
| Advanced | 8 | Orange |
| Expert | 1 | Red |

---

### Phase 9: MCP (Model Context Protocol)
**Status**: ✅ Completed
**Start Date**: January 21, 2026
**End Date**: January 21, 2026
**User Acceptance**: ⏳ Pending Review

#### Summary
Implemented comprehensive Model Context Protocol (MCP) tutorial content covering all aspects of MCP from introduction to production server/client development. Created 14 content files across 4 sections (10 lesson pages + 4 labs). The module teaches MCP architecture, core concepts (resources, tools, prompts, transports), server implementation, and client development with hands-on labs for building both servers and clients.

#### Content Created

**Sub-Phase 9.1: MCP Introduction (3 files)**
- `content/mcp/what-is-mcp.md` - Introduction to MCP, value proposition, comparison with alternatives
- `content/mcp/mcp-architecture.md` - Host/client/server architecture, JSON-RPC, capabilities negotiation
- `content/mcp/when-to-use-mcp.md` - Decision framework, use cases, when NOT to use MCP

**Sub-Phase 9.2: MCP Core Concepts (4 files)**
- `content/mcp/mcp-resources.md` - URI-addressable data sources, implementation patterns
- `content/mcp/mcp-tools.md` - Tool definitions, input schemas, execution patterns
- `content/mcp/mcp-prompts.md` - Prompt templates, arguments, multi-message prompts
- `content/mcp/mcp-transport.md` - stdio and HTTP+SSE transports, selection guide

**Sub-Phase 9.3: Building MCP Servers (4 files)**
- `content/mcp/server-structure.md` - Server architecture, project organization, lifecycle hooks
- `content/mcp/implementing-tools.md` - Tool implementation patterns, validation, error handling
- `content/mcp/implementing-resources.md` - Resource providers (file system, database, API)
- `content/mcp/mcp-server-lab.md` - Lab: Build a complete file operations MCP server

**Sub-Phase 9.4: MCP Clients & Integration (3 files)**
- `content/mcp/client-implementation.md` - Client patterns, multi-server, caching, events
- `content/mcp/mcp-client-lab.md` - Lab: Build an interactive MCP client REPL
- `content/mcp/filesystem-mcp-lab.md` - Lab: Build a production file system server with security

#### Registry Update
- Updated `src/ai_tutorial/content/registry.py` with complete MCP module structure:
  - 4 sections: Introduction, Core Concepts, Building Servers, Clients & Integration
  - 10 lesson pages with content_path references
  - 4 labs (mcp-server-lab, mcp-client-lab, filesystem-mcp-lab)
  - Proper tags, difficulty levels, and time estimates

#### File Structure
```
content/mcp/
├── what-is-mcp.md
├── mcp-architecture.md
├── when-to-use-mcp.md
├── mcp-resources.md
├── mcp-tools.md
├── mcp-prompts.md
├── mcp-transport.md
├── server-structure.md
├── implementing-tools.md
├── implementing-resources.md
├── mcp-server-lab.md
├── client-implementation.md
├── mcp-client-lab.md
└── filesystem-mcp-lab.md
```

#### Challenges & Solutions
1. **Content Depth Balance**: Balanced theoretical explanations with practical code examples. Included both conceptual overviews and production-ready implementations.
2. **Lab Complexity**: Labs progress from simple (basic server) to complex (production file system server) to accommodate learning curve.
3. **Python Code Consistency**: Used modern Python async patterns throughout with proper type hints and error handling.

---

### Phase 10: AI Agents
**Status**: ✅ Completed
**Start Date**: January 21, 2026
**End Date**: January 21, 2026
**User Acceptance**: ✅ Approved

#### Summary
Implemented comprehensive AI Agents tutorial content covering all aspects of building autonomous AI agents. Created 16 content files across 4 sections (12 lesson pages + 4 labs). The module teaches agent fundamentals, tool use and function calling, agent loop implementation, and advanced agent capabilities with hands-on labs for building research agents and code assistants.

#### Content Created

**Sub-Phase 10.1: Agent Fundamentals (3 files)**
- `content/agents/what-are-agents.md` - Introduction to AI agents, chatbot vs agent comparison, use cases
- `content/agents/agent-architectures.md` - ReAct, Plan-Execute, Multi-Agent patterns with ASCII diagrams
- `content/agents/agent-components.md` - Memory, planning, tools, execution engine components

**Sub-Phase 10.2: Tool Use & Function Calling (4 files)**
- `content/agents/function-calling.md` - OpenAI/Anthropic function calling basics with code examples
- `content/agents/tool-schemas.md` - JSON Schema definition, parameter types, validation
- `content/agents/tool-execution.md` - Safe execution patterns, error handling, circuit breakers
- `content/agents/tools-lab.md` - Lab: Build multi-tool agent with calculator, weather, search tools

**Sub-Phase 10.3: Agent Loop Implementation (4 files)**
- `content/agents/agent-loop.md` - Core execution loop patterns, termination conditions, streaming
- `content/agents/observation-thought-action.md` - ReAct OTA pattern implementation
- `content/agents/memory-state.md` - Short/long-term memory, conversation history, vector stores
- `content/agents/react-agent-lab.md` - Lab: Build complete ReAct agent with memory and streaming

**Sub-Phase 10.4: Agent Capabilities (5 files)**
- `content/agents/web-agents.md` - Web browsing, search tools, page extraction, Playwright automation
- `content/agents/code-agents.md` - Code generation, execution, debugging, sandboxed execution
- `content/agents/multi-tool-agents.md` - Tool orchestration, composition patterns, dependencies
- `content/agents/research-agent-lab.md` - Lab: Build research assistant with web tools and citations
- `content/agents/code-assistant-lab.md` - Lab: Build code writing and debugging agent

#### Registry Update
- Updated `src/ai_tutorial/content/registry.py` with complete Agents module structure:
  - 4 sections: Agent Fundamentals, Tool Use, Agent Loop, Agent Capabilities
  - 12 lesson pages with content_path references
  - 4 labs (tools-lab, react-agent-lab, research-agent-lab, code-assistant-lab)
  - Proper tags, difficulty levels, and time estimates

#### File Structure
```
content/agents/
├── what-are-agents.md
├── agent-architectures.md
├── agent-components.md
├── function-calling.md
├── tool-schemas.md
├── tool-execution.md
├── tools-lab.md
├── agent-loop.md
├── observation-thought-action.md
├── memory-state.md
├── react-agent-lab.md
├── web-agents.md
├── code-agents.md
├── multi-tool-agents.md
├── research-agent-lab.md
└── code-assistant-lab.md
```

#### Challenges & Solutions
1. **Content Depth Balance**: Balanced theoretical explanations with practical code examples. Included both conceptual overviews and production-ready implementations.
2. **Lab Complexity**: Labs progress from simple (multi-tool agent) to complex (research/code assistant) to accommodate learning curve.
3. **Python Code Consistency**: Used modern Python async patterns throughout with proper type hints and error handling.
4. **Comprehensive Tool Coverage**: Covered multiple tool types (web, code, multi-tool) with complete implementations.

---

### Phase 11: Multi-Agent Workflows
**Status**: ✅ Completed
**Start Date**: Session continuation
**End Date**: Current session
**User Acceptance**: ⏳ Pending

#### Summary
Created comprehensive multi-agent systems module with 16 content files covering multi-agent concepts, orchestration patterns, building multi-agent systems, and advanced patterns. Includes 4 sections with 12 lesson pages and 4 hands-on labs.

#### Content Added
1. **Sub-Phase 11.1 - Multi-Agent Concepts (3 files)**:
   - `why-multi-agent.md`: Benefits, use cases, single vs multi-agent comparison
   - `communication-patterns.md`: Pub/sub, request-response, broadcast, channels
   - `coordination-strategies.md`: Sequential, parallel, hierarchical, voting, consensus

2. **Sub-Phase 11.2 - Orchestration Patterns (4 files)**:
   - `sequential-workflows.md`: Pipelines, conditional branching, checkpointing
   - `parallel-workflows.md`: Fan-out/fan-in, map-reduce, competing agents
   - `hierarchical-agents.md`: Supervisor-worker, multi-level hierarchy, escalation
   - `debate-patterns.md`: Proposer-critic, multi-agent debate, devil's advocate

3. **Sub-Phase 11.3 - Building Multi-Agent Systems (4 files)**:
   - `agent-roles.md`: Role definitions, role registry, dynamic assignment
   - `message-passing.md`: Message types, queues, routing, priority handling
   - `state-sharing.md`: Shared state, namespaces, handoffs, checkpoints
   - `writer-reviewer-lab.md`: Complete lab with Writer and Reviewer agents

4. **Sub-Phase 11.4 - Advanced Patterns (5 files)**:
   - `agent-swarms.md`: Swarm intelligence, stigmergy, decentralization
   - `autonomous-teams.md`: Self-planning, consensus building, self-improvement
   - `human-in-the-loop.md`: Approval gates, escalation, review queues
   - `research-team-lab.md`: Multi-agent research system lab
   - `coding-team-lab.md`: Multi-agent coding team lab

5. **Registry Update**: Updated `registry.py` with complete multi-agents module containing 4 sections, 12 pages, and 4 labs

#### Challenges & Solutions
1. **Complex Pattern Coverage**: Covered diverse patterns (swarms, debates, autonomous teams) with practical implementations.
2. **Lab Progression**: Labs progress from simple (writer-reviewer) to complex (coding team) to build skills incrementally.
3. **Human Oversight**: Included human-in-the-loop patterns for safety-critical multi-agent applications.

---

### Phase 12: Interactive Learning Features
**Status**: ✅ Complete
**Start Date**: January 21, 2026
**End Date**: January 21, 2026
**User Acceptance**: ✅ Approved

#### Summary
Implemented comprehensive interactive learning tools including live code sandbox with Pyodide, token visualizer, embedding playground, provider comparison mode, and cost calculator.

#### Code Added
- `static/js/pyodide-sandbox.js` - Pyodide integration with presets
- `static/js/code-editor.js` - Monaco Editor wrapper with fallback
- `static/js/token-visualizer.js` - Tokenization and cost estimation
- `static/js/embedding-playground.js` - Embedding visualization and clustering
- `templates/content/sandbox-lab.html` - Full sandbox lab template
- `templates/pages/token-visualizer.html` - Token visualizer tool page
- `templates/pages/embedding-playground.html` - Embedding playground page
- `templates/pages/comparison.html` - Provider comparison page
- `templates/pages/cost-calculator.html` - Cost calculator page
- `templates/pages/tools.html` - Tools index page

#### Challenges & Solutions
1. **In-Browser Python**: Used Pyodide WebAssembly for secure client-side execution
2. **Monaco Editor**: Implemented fallback to textarea for unsupported browsers
3. **Token Estimation**: Used heuristic estimation since full tokenizers are heavy

---

### Phase 13: Assessment & Gamification
**Status**: ✅ Complete
**Start Date**: January 21, 2026
**End Date**: January 21, 2026
**User Acceptance**: ✅ Approved

#### Summary
Implemented comprehensive assessment and gamification system including:

1. **Quiz System**: Multi-question type quiz engine with immediate feedback
2. **Coding Challenges**: Pyodide-based auto-graded Python exercises
3. **Gamification**: Badges, streaks, skill tree, and progress tracking
4. **Quizzes Index**: Central page listing all 6 available quizzes
5. **UI Integration**: Added to home page and sidebar navigation

#### Code Added

**JavaScript Files:**
- `static/js/quiz-engine.js`: Quiz engine with 5 question types (multiple-choice, multiple-select, true/false, fill-blank, ordering), progress tracking, immediate feedback, localStorage persistence
- `static/js/coding-challenge.js`: Pyodide-based code execution, test runner, hints system, Monaco editor integration
- `static/js/gamification.js`: GamificationSystem singleton with 20+ badge definitions (4 rarity tiers), streak tracking, skill tree (4 tiers), activity heatmap

**Templates:**
- `templates/pages/quiz.html`: Quiz interface with question rendering, progress bar, results display
- `templates/pages/quizzes.html`: Quizzes index page with all 6 available quizzes
- `templates/pages/challenge.html`: Individual coding challenge page with editor and test runner
- `templates/pages/challenges.html`: Challenges index with difficulty filters and progress stats
- `templates/pages/achievements.html`: Achievements dashboard with stats, streaks, badges, and skill tree

**Routes Added (pages.py):**
- `/quizzes`: Quizzes index page (6 quizzes: LLM Basics, RAG, Agents, Embeddings, Fine-tuning, Prompting)
- `/quiz/{quiz_type}`: Quiz pages (llmBasics, rag, agents, embeddings, finetuning, prompting)
- `/challenges`: Challenges index page
- `/challenge/{challenge_type}`: Individual challenge pages (twoSum, reverseString, countTokens, cosineSimilarity)
- `/achievements`: Achievements and progress tracking page

**Navigation Updates:**
- Added "Knowledge Quizzes" to sidebar under Resources
- Added "Assessment & Gamification" section to home page
- Updated home page quiz link to go to `/quizzes`

#### Challenges & Solutions
1. **In-Browser Python Execution**: Used Pyodide WebAssembly runtime for secure client-side Python execution
2. **Progress Persistence**: Leveraged localStorage for all progress data (quizResults, challengeResults, earnedBadges, streakData)
3. **Badge System Design**: Created 20+ badges across 5 categories with 4 rarity tiers for meaningful progression

---

### Phase 14: Content Enrichment
**Status**: ✅ Complete
**Start Date**: January 21, 2026
**End Date**: January 21, 2026
**User Acceptance**: ✅ Approved

#### Summary
Implemented comprehensive reference content including searchable glossary, case studies, interactive decision guides, and failure modes gallery.

#### Code Added

**JavaScript:**
- `static/js/glossary.js`: 50+ AI/ML term definitions, category filtering, search, tooltips, A-Z letter navigation

**Templates:**
- `templates/pages/glossary.html`: Searchable glossary with letter nav, categories, term detail modal
- `templates/pages/case-studies.html`: 4 detailed case studies with expandable details
- `templates/pages/decision-guides.html`: 4 interactive decision trees with Alpine.js
- `templates/pages/failure-modes.html`: Tabbed failure mode categories with code examples

#### Challenges & Solutions
1. **Glossary Letter Navigation**: Had to compute available letters at module load time to avoid Alpine.js timing issues
2. **Alpine.js Scoping**: Fixed tab content visibility by moving `x-data` to parent container
3. **Broken Links**: Replaced case study links with expandable detail sections

---

### Post-Phase 14 Updates
**Status**: ✅ Complete
**Date**: January 21, 2026

#### Summary
Fixed various bugs discovered during testing and updated LLM model/pricing data.

#### Key Fixes
1. Script loading order in base.html (component scripts before Alpine.js)
2. Glossary letter navigation timing
3. Failure modes tab scoping
4. Case studies expandable details
5. Quizzes navigation link in sidebar
6. Home page quiz link consistency
7. LLM model and pricing updates (January 2026 leaderboard data)
8. Reverted to dark mode only (light mode incomplete)

---

### Phase 15: Advanced Topics
**Status**: ✅ Complete
**Start Date**: January 21, 2026
**End Date**: January 21, 2026
**User Acceptance**: ✅ Approved

#### Summary
Implemented advanced AI engineering topics including structured outputs, guardrails & safety, observability & monitoring, and cost optimization.

#### Files Created
- `content/advanced-topics/structured-outputs.md`: JSON mode, function calling, schema validation
- `content/advanced-topics/guardrails-safety.md`: Content filtering, jailbreak prevention, safety layers
- `content/advanced-topics/observability.md`: Logging, tracing, metrics, LangSmith integration
- `content/advanced-topics/cost-optimization.md`: Caching, batching, model selection, token optimization
- Additional content files for labs and exercises

---

### Phase 16: Progress Tracking & User Experience
**Status**: ✅ Complete
**Start Date**: January 21, 2026
**End Date**: January 21, 2026
**User Acceptance**: ✅ Approved

#### Summary
Implemented progress tracking system with learning path navigation fix, ensuring paths only navigate within their defined modules.

#### Files Created/Modified
- `static/js/progress-tracker.js`: Progress tracking with markIncomplete(), progress-updated events
- `src/ai_tutorial/content/registry.py`: Added path-aware navigation methods
- `src/ai_tutorial/routers/pages.py`: Added ?path= parameter support
- `templates/content/learning-path.html`: Full rewrite with path tracking
- `templates/pages/index.html`: Updated learning path links

---

### Phase 17: Developer Experience & Deployment
**Status**: ✅ Complete
**Start Date**: January 21, 2026
**End Date**: January 21, 2026
**User Acceptance**: ✅ Approved

#### Summary
Comprehensive developer experience improvements including testing (94 tests), documentation, Docker deployment, and code export/sharing features.

#### Files Created

**Tests (94 passing):**
- `tests/__init__.py`: Test package init
- `tests/conftest.py`: Pytest configuration with fixtures
- `tests/test_providers.py`: 24 tests for LLM provider adapters
- `tests/test_content_registry.py`: 29 tests for content registry
- `tests/test_routes.py`: 28 tests for FastAPI routes
- `tests/test_export.py`: 13 tests for export functionality

**Documentation:**
- `README.md`: Comprehensive project documentation with badges
- `CONTRIBUTING.md`: Contribution guidelines
- `docs/ARCHITECTURE.md`: System architecture documentation

**Docker & Deployment:**
- `Dockerfile`: Multi-stage production build
- `Dockerfile.dev`: Development with hot reload
- `docker-compose.yml`: Production stack with Ollama
- `docker-compose.dev.yml`: Development stack
- `.devcontainer/devcontainer.json`: GitHub Codespaces support
- `.dockerignore`: Docker build exclusions
- `scripts/deploy.sh`: Linux/Mac deployment script
- `scripts/deploy.ps1`: Windows deployment script

**Export & Sharing:**
- `src/ai_tutorial/routers/export.py`: Export API endpoints
  - `POST /api/export/download`: Download code as .py
  - `POST /api/export/share`: Create shareable links
  - `GET /api/export/share/{id}`: Retrieve shared code
  - `POST /api/export/encode`: URL-based code sharing
  - `POST /api/export/social-share`: Social media sharing
  - `POST /api/export/gist-url`: GitHub Gist preparation
- Updated `static/js/main.js`: Share dialog UI, download/share functions
- Updated `src/ai_tutorial/routers/__init__.py`: Added export router
- Updated `src/ai_tutorial/main.py`: Registered export router

---

## Key Features Summary

### For Learners
1. **Structured Learning Paths**: Multiple tracks for different goals (Quick Start, RAG Engineer, Agent Developer, etc.)
2. **Hands-on Labs**: Every concept has interactive exercises
3. **Multi-Provider Support**: Use any LLM provider (OpenAI, Claude, Gemini, Grok)
4. **Live Code Sandbox**: Run Python directly in the browser with Pyodide
5. **Progress Tracking**: Visual indication of completed sections with streaks
6. **Self-Hosted Options**: Learn to run models locally for privacy and cost savings
7. **Gamification**: Badges, skill trees, and achievements
8. **Quizzes & Challenges**: Verify understanding with assessments
9. **Certificates**: Downloadable completion certificates

### For Content
1. **LLM Fundamentals**: Complete introduction to working with LLMs
2. **Advanced LLM Concepts**: Model architectures, multimodal, MoE
3. **Fine-tuning**: LoRA, QLoRA, cloud & local fine-tuning
4. **Self-Hosting**: Ollama, vLLM, llama.cpp, quantization
5. **Vector Databases**: Embeddings, ChromaDB, semantic search
6. **RAG**: From basics to advanced patterns (HyDE, Self-RAG, RAG Fusion)
7. **MCP**: Server and client implementation
8. **Agents**: ReAct pattern, tool use, memory
9. **Multi-Agents**: Orchestration, collaboration patterns
10. **Advanced Topics**: Guardrails, Observability, Cost Optimization

### Interactive Tools
1. **Token Visualizer**: Real-time tokenization and cost display
2. **Embedding Playground**: 2D/3D semantic visualization
3. **Comparison Mode**: Side-by-side provider comparison
4. **Cost Calculator**: API cost estimation tool
5. **Decision Trees**: Interactive "when to use what" guides

### Technical Highlights
1. **UV Package Management**: Modern, fast Python package management
2. **FastAPI Backend**: Async, high-performance API
3. **Provider Abstraction**: Clean adapter pattern for LLM providers
4. **Pyodide Integration**: In-browser Python execution
5. **PWA Support**: Offline mode capability
6. **Docker Compose**: One-command setup with Ollama
7. **GitHub Codespaces**: Zero local setup option

---

## Approval Checklist

- [x] Development Workflow Rules approved
- [x] Learning Paths approved
- [x] Enhanced Features list approved
- [x] Phase structure (17 phases) approved
- [x] Technology stack approved
- [x] Content scope approved
- [x] Directory structure approved
- [x] All 17 phases completed ✅

---

*Document Version: 4.0*
*Created: January 20, 2026*
*Last Updated: January 21, 2026*
*Status: All Phases Complete* 🎉
