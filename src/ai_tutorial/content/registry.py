"""
Content Registry - Central registry for all tutorial content.
"""

from typing import Dict, List, Optional, Any
from .models import Module, Section, Page, Lab, Quiz, LearningPath, PageType, Difficulty


class ContentRegistry:
    """Central registry for managing tutorial content."""
    
    _instance: Optional["ContentRegistry"] = None
    
    def __init__(self):
        self.modules: Dict[str, Module] = {}
        self.learning_paths: Dict[str, LearningPath] = {}
        self._page_index: Dict[str, tuple] = {}  # page_id -> (module_id, section_id)
        self._initialize_content()
    
    @classmethod
    def get_instance(cls) -> "ContentRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = ContentRegistry()
        return cls._instance
    
    def _initialize_content(self):
        """Initialize all tutorial content."""
        self._register_modules()
        self._register_learning_paths()
        self._build_page_index()
    
    def _register_modules(self):
        """Register all learning modules."""
        
        # Module 1: LLM Fundamentals
        self.modules["llm-fundamentals"] = Module(
            id="llm-fundamentals",
            title="LLM Fundamentals",
            slug="llm-fundamentals",
            description="Learn the basics of Large Language Models, how they work, and how to use them effectively.",
            icon="🧠",
            color="blue",
            difficulty=Difficulty.BEGINNER,
            order=1,
            sections=[
                Section(
                    id="intro-to-llms",
                    title="Introduction to LLMs",
                    slug="intro-to-llms",
                    description="What are LLMs and how do they work?",
                    icon="📖",
                    order=1,
                    pages=[
                        Page(id="what-are-llms", title="What are Large Language Models?", slug="what-are-llms", 
                             description="Introduction to LLMs and their capabilities", estimated_time=10, order=1,
                             content_path="llm-fundamentals/intro-to-llms/what-are-llms.md",
                             tags=["llm", "basics", "introduction"]),
                        Page(id="llm-history", title="History & Evolution of LLMs", slug="history",
                             description="From N-grams to GPT-5.2 and beyond", estimated_time=12, order=2,
                             content_path="llm-fundamentals/intro-to-llms/history.md",
                             tags=["history", "evolution"]),
                        Page(id="how-llms-work", title="How LLMs Work", slug="how-they-work",
                             description="Transformers, attention, and tokenization", estimated_time=20, order=3,
                             content_path="llm-fundamentals/intro-to-llms/how-they-work.md",
                             tags=["transformers", "attention", "architecture"]),
                    ],
                    labs=[
                        Lab(id="first-api-call", title="Your First LLM API Call", slug="first-api-call",
                            description="Make your first API call to an LLM", estimated_time=15, order=1,
                            content_path="llm-fundamentals/intro-to-llms/first-api-call.md",
                            starter_code='# Your first LLM API call\nquestion = "What is the meaning of life?"\n\n# The API call will be made for you\n# Just modify the question above and run!'),
                    ],
                ),
                Section(
                    id="prompt-engineering",
                    title="Prompt Engineering",
                    slug="prompt-engineering",
                    description="Master the art of crafting effective prompts",
                    icon="✍️",
                    order=2,
                    pages=[
                        Page(id="prompt-fundamentals", title="Prompt Fundamentals", slug="prompt-fundamentals",
                             description="The anatomy of effective prompts", estimated_time=12, order=1,
                             content_path="llm-fundamentals/prompt-engineering/prompt-fundamentals.md",
                             tags=["prompts", "basics", "crispe"]),
                        Page(id="prompt-patterns", title="Prompt Patterns & Techniques", slug="prompt-patterns",
                             description="Reusable patterns for common tasks", estimated_time=15, order=2,
                             content_path="llm-fundamentals/prompt-engineering/prompt-patterns.md",
                             tags=["patterns", "templates", "techniques"]),
                        Page(id="zero-few-shot", title="Zero-Shot, One-Shot & Few-Shot Learning", slug="zero-few-shot",
                             description="When and how to use examples", estimated_time=15, order=3,
                             content_path="llm-fundamentals/prompt-engineering/zero-few-shot.md",
                             tags=["zero-shot", "few-shot", "chain-of-thought"]),
                    ],
                    labs=[
                        Lab(id="prompt-playground", title="Prompt Playground", slug="prompt-playground",
                            description="Experiment with different prompting techniques", estimated_time=20, order=1,
                            content_path="llm-fundamentals/prompt-engineering/prompt-playground.md"),
                    ],
                ),
                Section(
                    id="parameters",
                    title="LLM Parameters",
                    slug="parameters",
                    description="Understanding and tuning model parameters",
                    icon="⚙️",
                    order=3,
                    pages=[
                        Page(id="temperature", title="Temperature & Sampling", slug="temperature",
                             description="Control randomness and creativity", estimated_time=12, order=1,
                             content_path="llm-fundamentals/parameters/temperature.md",
                             tags=["temperature", "top-p", "sampling"]),
                        Page(id="tokens-limits", title="Token Limits & Max Tokens", slug="tokens-limits",
                             description="Understanding context windows and token limits", estimated_time=12, order=2,
                             content_path="llm-fundamentals/parameters/tokens-limits.md",
                             tags=["tokens", "context-window", "limits"]),
                        Page(id="stop-sequences", title="Stop Sequences & Output Control", slug="stop-sequences",
                             description="Precise control over generation", estimated_time=10, order=3,
                             content_path="llm-fundamentals/parameters/stop-sequences.md",
                             tags=["stop-sequences", "output", "control"]),
                    ],
                    labs=[
                        Lab(id="parameter-explorer", title="Parameter Explorer", slug="parameter-explorer",
                            description="Experiment with temperature, top-p, and other parameters", estimated_time=20, order=1,
                            content_path="llm-fundamentals/parameters/parameter-explorer.md"),
                    ],
                ),
                Section(
                    id="tokens",
                    title="Tokens & Context",
                    slug="tokens",
                    description="Deep dive into tokenization and context management",
                    icon="🔤",
                    order=4,
                    pages=[
                        Page(id="tokenization", title="Deep Dive: Tokenization", slug="tokenization",
                             description="How text becomes tokens", estimated_time=15, order=1,
                             content_path="llm-fundamentals/tokens/tokenization.md",
                             tags=["tokenization", "bpe", "vocabulary"]),
                        Page(id="context-windows", title="Understanding Context Windows", slug="context-windows",
                             description="How LLMs process long text", estimated_time=12, order=2,
                             content_path="llm-fundamentals/tokens/context-windows.md",
                             tags=["context", "attention", "limits"]),
                        Page(id="managing-context", title="Managing Context Effectively", slug="managing-context",
                             description="Strategies for context management", estimated_time=15, order=3,
                             content_path="llm-fundamentals/tokens/managing-context.md",
                             tags=["context-management", "summarization", "chunking"]),
                    ],
                    labs=[
                        Lab(id="token-counter", title="Token Counter & Context Analyzer", slug="token-counter",
                            description="Count tokens and analyze context usage", estimated_time=15, order=1,
                            content_path="llm-fundamentals/tokens/token-counter.md",
                            provider_required=False),
                    ],
                ),
            ],
        )
        
        # Module 2: Advanced LLM Concepts
        self.modules["advanced-llm"] = Module(
            id="advanced-llm",
            title="Advanced LLM Concepts",
            slug="advanced-llm",
            description="Deep dive into LLM architectures, attention mechanisms, and multimodal models.",
            icon="🧠",
            color="indigo",
            difficulty=Difficulty.ADVANCED,
            prerequisites=["llm-fundamentals"],
            order=2,
            sections=[
                Section(
                    id="model-architectures",
                    title="Model Architectures",
                    slug="architectures",
                    description="Understanding transformer architectures",
                    icon="🏗️",
                    order=1,
                    pages=[
                        Page(id="model-architectures", title="LLM Architectures Deep Dive", slug="model-architectures",
                             description="Decoder-only, encoder-only, and encoder-decoder models", estimated_time=20, order=1,
                             content_path="advanced-llm/model-architectures.md",
                             tags=["architectures", "transformer", "gpt", "bert"]),
                        Page(id="attention-mechanisms", title="Attention Mechanisms", slug="attention",
                             description="Self-attention, MHA, MQA, GQA explained", estimated_time=25, order=2,
                             content_path="advanced-llm/attention-mechanisms.md",
                             tags=["attention", "mha", "gqa", "rope"]),
                    ],
                ),
                Section(
                    id="advanced-architectures",
                    title="Advanced Architectures",
                    slug="advanced",
                    description="Mixture of Experts and other innovations",
                    icon="⚡",
                    order=2,
                    pages=[
                        Page(id="moe-models", title="Mixture of Experts (MoE)", slug="moe",
                             description="Sparse models and expert routing", estimated_time=20, order=1,
                             content_path="advanced-llm/moe-models.md",
                             tags=["moe", "sparse", "mixtral", "experts"]),
                    ],
                ),
                Section(
                    id="multimodal",
                    title="Multimodal LLMs",
                    slug="multimodal",
                    description="Vision, audio, and beyond",
                    icon="👁️",
                    order=3,
                    pages=[
                        Page(id="multimodal-llms", title="Multimodal LLMs", slug="multimodal-llms",
                             description="GPT-4V, Claude Vision, Gemini multimodal", estimated_time=18, order=1,
                             content_path="advanced-llm/multimodal-llms.md",
                             tags=["multimodal", "vision", "gpt-4v", "claude"]),
                    ],
                    labs=[
                        Lab(id="multimodal-lab", title="Multimodal Vision Lab", slug="vision-lab",
                            description="Build applications with vision capabilities", estimated_time=30, order=1,
                            content_path="advanced-llm/multimodal-lab.md",
                            difficulty=Difficulty.ADVANCED,
                            provider_required=True),
                    ],
                ),
            ],
        )
        
        # Module 3: Embeddings & Vector Databases
        self.modules["embeddings-vectors"] = Module(
            id="embeddings-vectors",
            title="Embeddings & Vector Databases",
            slug="embeddings-vectors",
            description="Learn about embeddings, semantic search, and vector databases.",
            icon="🔢",
            color="purple",
            difficulty=Difficulty.INTERMEDIATE,
            prerequisites=["llm-fundamentals"],
            order=3,
            sections=[
                Section(
                    id="understanding-embeddings",
                    title="Understanding Embeddings",
                    slug="embeddings",
                    description="What are embeddings and how to use them",
                    icon="📊",
                    order=1,
                    pages=[
                        Page(id="what-are-embeddings", title="What are Embeddings?", slug="what-are-embeddings",
                             description="Understanding vector representations", estimated_time=12, order=1,
                             content_path="embeddings-vectors/what-are-embeddings.md",
                             tags=["embeddings", "vectors", "similarity"]),
                        Page(id="semantic-vectors", title="Semantic Vectors", slug="semantic-vectors",
                             description="The geometry of meaning in vector space", estimated_time=15, order=2,
                             content_path="embeddings-vectors/semantic-vectors.md",
                             tags=["semantics", "vectors", "visualization"]),
                        Page(id="embedding-models", title="Embedding Models Comparison", slug="models",
                             description="Comparing different embedding models", estimated_time=12, order=3,
                             content_path="embeddings-vectors/embedding-models.md",
                             tags=["models", "openai", "cohere", "voyage"]),
                    ],
                    labs=[
                        Lab(id="embeddings-lab", title="Generate & Visualize Embeddings", slug="visualize",
                            description="Create embeddings and visualize them in 2D/3D", estimated_time=25, order=1,
                            content_path="embeddings-vectors/embeddings-lab.md",
                            difficulty=Difficulty.INTERMEDIATE),
                    ],
                ),
                Section(
                    id="vector-databases",
                    title="Vector Database Concepts",
                    slug="vector-db",
                    description="Store and query embeddings efficiently",
                    icon="🗄️",
                    order=2,
                    pages=[
                        Page(id="vector-db-intro", title="Introduction to Vector Databases", slug="intro",
                             description="Why vector databases and how they work", estimated_time=12, order=1,
                             content_path="embeddings-vectors/vector-db-intro.md",
                             tags=["vector-db", "ann", "architecture"]),
                        Page(id="similarity-search", title="Similarity Search Algorithms", slug="similarity",
                             description="Cosine similarity, euclidean, and more", estimated_time=15, order=2,
                             content_path="embeddings-vectors/similarity-search.md",
                             tags=["similarity", "cosine", "distance"]),
                        Page(id="indexing-strategies", title="Indexing Strategies", slug="indexing",
                             description="HNSW, IVF, and product quantization", estimated_time=18, order=3,
                             content_path="embeddings-vectors/indexing-strategies.md",
                             tags=["hnsw", "ivf", "indexing", "ann"]),
                        Page(id="vector-db-options", title="Vector Database Options", slug="options",
                             description="ChromaDB, Pinecone, Weaviate, Qdrant", estimated_time=12, order=4,
                             content_path="embeddings-vectors/vector-db-options.md",
                             tags=["chromadb", "pinecone", "qdrant", "weaviate"]),
                    ],
                    labs=[
                        Lab(id="chromadb-lab", title="Set Up ChromaDB", slug="chromadb",
                            description="Create your first vector database", estimated_time=20, order=1,
                            content_path="embeddings-vectors/chromadb-lab.md",
                            difficulty=Difficulty.INTERMEDIATE,
                            provider_required=False),
                    ],
                ),
                Section(
                    id="practical-operations",
                    title="Practical Vector Operations",
                    slug="practical",
                    description="Working with collections, queries, and metadata",
                    icon="⚡",
                    order=3,
                    pages=[
                        Page(id="collections", title="Working with Collections", slug="collections",
                             description="Create, manage, and organize collections", estimated_time=12, order=1,
                             content_path="embeddings-vectors/collections.md",
                             tags=["collections", "crud", "management"]),
                        Page(id="querying-documents", title="Querying Documents", slug="querying",
                             description="Search patterns and result ranking", estimated_time=15, order=2,
                             content_path="embeddings-vectors/querying-documents.md",
                             tags=["querying", "search", "ranking"]),
                        Page(id="metadata-filtering", title="Metadata Filtering", slug="filtering",
                             description="Filter results by attributes", estimated_time=12, order=3,
                             content_path="embeddings-vectors/metadata-filtering.md",
                             tags=["metadata", "filtering", "where"]),
                    ],
                    labs=[
                        Lab(id="semantic-search-lab", title="Build Semantic Search", slug="semantic-search",
                            description="Build a complete semantic search engine", estimated_time=45, order=1,
                            content_path="embeddings-vectors/semantic-search-lab.md",
                            difficulty=Difficulty.INTERMEDIATE),
                    ],
                ),
            ],
        )
        
        # Module 4: RAG (Retrieval Augmented Generation)
        self.modules["rag"] = Module(
            id="rag",
            title="RAG: Retrieval Augmented Generation",
            slug="rag",
            description="Build powerful applications by combining LLMs with external knowledge.",
            icon="📚",
            color="green",
            difficulty=Difficulty.INTERMEDIATE,
            prerequisites=["llm-fundamentals", "embeddings-vectors"],
            order=4,
            sections=[
                Section(
                    id="rag-fundamentals",
                    title="RAG Fundamentals",
                    slug="fundamentals",
                    description="Core concepts of RAG systems",
                    icon="📖",
                    order=1,
                    pages=[
                        Page(id="what-is-rag", title="What is RAG?", slug="what-is-rag",
                             description="Understanding Retrieval-Augmented Generation", estimated_time=12, order=1,
                             content_path="rag/what-is-rag.md",
                             tags=["rag", "basics", "introduction"]),
                        Page(id="rag-vs-finetuning", title="RAG vs Fine-Tuning", slug="rag-vs-finetuning",
                             description="When to use RAG vs fine-tuning", estimated_time=10, order=2,
                             content_path="rag/rag-vs-finetuning.md",
                             tags=["rag", "finetuning", "comparison"]),
                        Page(id="rag-architecture", title="RAG Architecture", slug="architecture",
                             description="Components of a RAG system", estimated_time=15, order=3,
                             content_path="rag/rag-architecture.md",
                             tags=["architecture", "components", "pipeline"]),
                    ],
                    labs=[
                        Lab(id="basic-rag-lab", title="Basic RAG Pipeline", slug="basic-rag",
                            description="Build your first RAG system from scratch", estimated_time=30, order=1,
                            content_path="rag/basic-rag-lab.md",
                            difficulty=Difficulty.INTERMEDIATE),
                    ],
                ),
                Section(
                    id="document-processing",
                    title="Document Processing",
                    slug="documents",
                    description="Load, parse, and chunk documents for RAG",
                    icon="📄",
                    order=2,
                    pages=[
                        Page(id="document-loading", title="Document Loading", slug="loading",
                             description="Load documents from various formats", estimated_time=12, order=1,
                             content_path="rag/document-loading.md",
                             tags=["documents", "loading", "formats"]),
                        Page(id="chunking-strategies", title="Chunking Strategies", slug="chunking",
                             description="Split documents into effective chunks", estimated_time=15, order=2,
                             content_path="rag/chunking-strategies.md",
                             tags=["chunking", "splitting", "strategies"]),
                        Page(id="chunk-sizing", title="Chunk Sizing Guide", slug="sizing",
                             description="Optimize chunk size for retrieval", estimated_time=12, order=3,
                             content_path="rag/chunk-sizing.md",
                             tags=["chunks", "sizing", "optimization"]),
                        Page(id="document-parsing", title="Advanced Document Parsing", slug="parsing",
                             description="Extract text from complex documents", estimated_time=15, order=4,
                             content_path="rag/document-parsing.md",
                             tags=["parsing", "pdf", "html", "ocr"]),
                    ],
                    labs=[
                        Lab(id="document-processor-lab", title="Document Processing Pipeline", slug="processor-lab",
                            description="Build a complete document processing pipeline", estimated_time=30, order=1,
                            content_path="rag/document-processor-lab.md",
                            difficulty=Difficulty.INTERMEDIATE),
                    ],
                ),
                Section(
                    id="retrieval-strategies",
                    title="Retrieval Strategies",
                    slug="retrieval",
                    description="Advanced techniques for finding relevant documents",
                    icon="🔍",
                    order=3,
                    pages=[
                        Page(id="basic-retrieval", title="Basic Retrieval", slug="basic",
                             description="Similarity search and filtering", estimated_time=12, order=1,
                             content_path="rag/basic-retrieval.md",
                             tags=["retrieval", "similarity", "basic"]),
                        Page(id="hybrid-search", title="Hybrid Search", slug="hybrid",
                             description="Combining keyword and semantic search", estimated_time=15, order=2,
                             content_path="rag/hybrid-search.md",
                             tags=["hybrid", "bm25", "fusion"]),
                        Page(id="reranking", title="Reranking Results", slug="reranking",
                             description="Improve retrieval with rerankers", estimated_time=15, order=3,
                             content_path="rag/reranking.md",
                             tags=["reranking", "cross-encoder", "cohere"]),
                        Page(id="multi-query", title="Multi-Query Retrieval", slug="multi-query",
                             description="Query decomposition and expansion", estimated_time=15, order=4,
                             content_path="rag/multi-query.md",
                             tags=["multi-query", "decomposition", "hyde"]),
                        Page(id="contextual-compression", title="Contextual Compression", slug="compression",
                             description="Extract relevant passages from documents", estimated_time=12, order=5,
                             content_path="rag/contextual-compression.md",
                             tags=["compression", "extraction", "relevant"]),
                    ],
                    labs=[
                        Lab(id="retrieval-strategies-lab", title="Retrieval Strategies Lab", slug="retrieval-lab",
                            description="Compare different retrieval approaches", estimated_time=35, order=1,
                            content_path="rag/retrieval-strategies-lab.md",
                            difficulty=Difficulty.INTERMEDIATE),
                    ],
                ),
                Section(
                    id="generation-context",
                    title="Generation with Context",
                    slug="generation",
                    description="Generate high-quality answers from retrieved context",
                    icon="✨",
                    order=4,
                    pages=[
                        Page(id="rag-prompts", title="RAG Prompt Engineering", slug="prompts",
                             description="Effective prompts for RAG systems", estimated_time=15, order=1,
                             content_path="rag/rag-prompts.md",
                             tags=["prompts", "generation", "context"]),
                        Page(id="context-overflow", title="Handling Context Overflow", slug="overflow",
                             description="Manage large amounts of context", estimated_time=12, order=2,
                             content_path="rag/context-overflow.md",
                             tags=["context", "overflow", "truncation"]),
                        Page(id="citation-tracking", title="Citation Tracking", slug="citations",
                             description="Link answers to source documents", estimated_time=12, order=3,
                             content_path="rag/citation-tracking.md",
                             tags=["citations", "sources", "attribution"]),
                        Page(id="streaming-sources", title="Streaming with Sources", slug="streaming",
                             description="Stream responses with source attribution", estimated_time=15, order=4,
                             content_path="rag/streaming-sources.md",
                             tags=["streaming", "sse", "real-time"]),
                    ],
                    labs=[
                        Lab(id="rag-chatbot-lab", title="RAG Chatbot Lab", slug="chatbot-lab",
                            description="Build a conversational RAG chatbot", estimated_time=40, order=1,
                            content_path="rag/rag-chatbot-lab.md",
                            difficulty=Difficulty.INTERMEDIATE),
                    ],
                ),
                Section(
                    id="advanced-rag",
                    title="Advanced RAG Patterns",
                    slug="advanced",
                    description="Sophisticated techniques for production RAG",
                    icon="🚀",
                    order=5,
                    pages=[
                        Page(id="advanced-patterns", title="Advanced RAG Patterns", slug="patterns",
                             description="HyDE, RAG Fusion, Self-RAG, CRAG", estimated_time=20, order=1,
                             content_path="rag/advanced-patterns.md",
                             tags=["advanced", "hyde", "fusion", "self-rag"]),
                        Page(id="production-architecture", title="Production RAG Architecture", slug="production",
                             description="Scale RAG for production deployment", estimated_time=18, order=2,
                             content_path="rag/production-architecture.md",
                             tags=["production", "architecture", "scaling"]),
                        Page(id="evaluation-optimization", title="RAG Evaluation & Optimization", slug="evaluation",
                             description="Measure and improve RAG quality", estimated_time=18, order=3,
                             content_path="rag/evaluation-optimization.md",
                             tags=["evaluation", "ragas", "metrics", "optimization"]),
                    ],
                    labs=[
                        Lab(id="advanced-rag-lab", title="Advanced RAG Lab", slug="advanced-lab",
                            description="Build a production-ready RAG system", estimated_time=90, order=1,
                            content_path="rag/advanced-rag-lab.md",
                            difficulty=Difficulty.ADVANCED),
                    ],
                ),
            ],
        )
        
        # Module 5: Tool Use & Function Calling
        self.modules["tools-functions"] = Module(
            id="tools-functions",
            title="Tool Use & Function Calling",
            slug="tools-functions",
            description="Enable LLMs to use external tools and execute functions.",
            icon="🔧",
            color="orange",
            difficulty=Difficulty.INTERMEDIATE,
            prerequisites=["llm-fundamentals"],
            order=5,
            sections=[
                Section(
                    id="function-calling-basics",
                    title="Function Calling Basics",
                    slug="basics",
                    description="Enable LLMs to call functions",
                    icon="📞",
                    order=1,
                    pages=[
                        Page(id="function-calling-intro", title="Introduction to Function Calling", slug="intro",
                             description="What is function calling and why use it?", estimated_time=10, order=1),
                        Page(id="defining-functions", title="Defining Functions for LLMs", slug="defining",
                             description="How to define and describe functions", estimated_time=12, order=2),
                        Page(id="handling-responses", title="Handling Function Call Responses", slug="responses",
                             description="Processing and executing function calls", estimated_time=10, order=3),
                    ],
                    labs=[
                        Lab(id="first-function-call", title="Your First Function Call", slug="first-call",
                            description="Implement a simple function calling example", estimated_time=20, order=1,
                            difficulty=Difficulty.INTERMEDIATE),
                        Lab(id="multi-tool", title="Multi-Tool Agent", slug="multi-tool",
                            description="Build an agent with multiple tools", estimated_time=30, order=2,
                            difficulty=Difficulty.INTERMEDIATE),
                    ],
                ),
            ],
        )
        
        # Module 6: Agents
        self.modules["agents"] = Module(
            id="agents",
            title="AI Agents",
            slug="agents",
            description="Build autonomous AI agents that can reason and take actions.",
            icon="🤖",
            color="red",
            difficulty=Difficulty.ADVANCED,
            prerequisites=["llm-fundamentals"],
            order=6,
            sections=[
                Section(
                    id="agent-fundamentals",
                    title="Agent Fundamentals",
                    slug="fundamentals",
                    description="Core concepts of AI agents",
                    icon="🎯",
                    order=1,
                    pages=[
                        Page(id="what-are-agents", title="What are AI Agents?", slug="what-are-agents",
                             description="Understanding autonomous AI agents vs chatbots", estimated_time=12, order=1,
                             content_path="agents/what-are-agents.md",
                             tags=["agents", "basics", "introduction"]),
                        Page(id="agent-architectures", title="Agent Architectures", slug="architectures",
                             description="ReAct, Plan-Execute, and Multi-Agent patterns", estimated_time=15, order=2,
                             content_path="agents/agent-architectures.md",
                             tags=["agents", "architectures", "react", "patterns"]),
                        Page(id="agent-components", title="Agent Components", slug="components",
                             description="Memory, planning, tools, and execution", estimated_time=15, order=3,
                             content_path="agents/agent-components.md",
                             tags=["agents", "components", "memory", "planning"]),
                    ],
                ),
                Section(
                    id="tool-use",
                    title="Tool Use & Function Calling",
                    slug="tool-use",
                    description="Enable agents to use external tools",
                    icon="🔧",
                    order=2,
                    pages=[
                        Page(id="function-calling", title="Function Calling Basics", slug="function-calling",
                             description="How LLMs call external functions", estimated_time=15, order=1,
                             content_path="agents/function-calling.md",
                             tags=["function-calling", "tools", "openai", "anthropic"]),
                        Page(id="tool-schemas", title="Designing Tool Schemas", slug="tool-schemas",
                             description="JSON Schema for tool definitions", estimated_time=12, order=2,
                             content_path="agents/tool-schemas.md",
                             tags=["schemas", "json-schema", "validation"]),
                        Page(id="tool-execution", title="Safe Tool Execution", slug="tool-execution",
                             description="Executing tools safely and reliably", estimated_time=15, order=3,
                             content_path="agents/tool-execution.md",
                             tags=["execution", "safety", "error-handling"]),
                    ],
                    labs=[
                        Lab(id="tools-lab", title="Build a Multi-Tool Agent", slug="tools-lab",
                            description="Create an agent with calculator, weather, and search tools", estimated_time=35, order=1,
                            content_path="agents/tools-lab.md",
                            difficulty=Difficulty.INTERMEDIATE),
                    ],
                ),
                Section(
                    id="agent-loop",
                    title="Agent Loop Implementation",
                    slug="agent-loop",
                    description="Build the core agent execution loop",
                    icon="🔄",
                    order=3,
                    pages=[
                        Page(id="agent-loop-basics", title="The Agent Loop", slug="loop-basics",
                             description="Core execution loop patterns", estimated_time=15, order=1,
                             content_path="agents/agent-loop.md",
                             tags=["agent-loop", "execution", "patterns"]),
                        Page(id="ota-pattern", title="Observation-Thought-Action", slug="ota-pattern",
                             description="Implementing the ReAct OTA pattern", estimated_time=18, order=2,
                             content_path="agents/observation-thought-action.md",
                             tags=["react", "ota", "reasoning"]),
                        Page(id="memory-state", title="Memory & State Management", slug="memory-state",
                             description="Managing agent memory and conversation state", estimated_time=15, order=3,
                             content_path="agents/memory-state.md",
                             tags=["memory", "state", "conversation"]),
                    ],
                    labs=[
                        Lab(id="react-agent-lab", title="Build a ReAct Agent", slug="react-agent-lab",
                            description="Implement a complete ReAct agent with memory", estimated_time=45, order=1,
                            content_path="agents/react-agent-lab.md",
                            difficulty=Difficulty.ADVANCED),
                    ],
                ),
                Section(
                    id="agent-capabilities",
                    title="Agent Capabilities",
                    slug="capabilities",
                    description="Specialized agent types and patterns",
                    icon="⚡",
                    order=4,
                    pages=[
                        Page(id="web-agents", title="Web Browsing Agents", slug="web-agents",
                             description="Agents that search and browse the web", estimated_time=18, order=1,
                             content_path="agents/web-agents.md",
                             tags=["web", "browsing", "search", "scraping"]),
                        Page(id="code-agents", title="Code Generation Agents", slug="code-agents",
                             description="Agents that write and execute code", estimated_time=18, order=2,
                             content_path="agents/code-agents.md",
                             tags=["code", "execution", "debugging"]),
                        Page(id="multi-tool-agents", title="Multi-Tool Agents", slug="multi-tool-agents",
                             description="Orchestrating multiple tools together", estimated_time=15, order=3,
                             content_path="agents/multi-tool-agents.md",
                             tags=["multi-tool", "orchestration", "composition"]),
                    ],
                    labs=[
                        Lab(id="research-agent-lab", title="Build a Research Agent", slug="research-agent-lab",
                            description="Create a research assistant with web tools", estimated_time=45, order=1,
                            content_path="agents/research-agent-lab.md",
                            difficulty=Difficulty.ADVANCED),
                        Lab(id="code-assistant-lab", title="Build a Code Assistant", slug="code-assistant-lab",
                            description="Create a code writing and debugging agent", estimated_time=45, order=2,
                            content_path="agents/code-assistant-lab.md",
                            difficulty=Difficulty.ADVANCED),
                    ],
                ),
            ],
        )
        
        # Module 7: Model Context Protocol (MCP)
        self.modules["mcp"] = Module(
            id="mcp",
            title="Model Context Protocol (MCP)",
            slug="mcp",
            description="Learn the standard protocol for connecting LLMs to external tools and data sources.",
            icon="🔌",
            color="indigo",
            difficulty=Difficulty.ADVANCED,
            prerequisites=["llm-fundamentals"],
            order=7,
            sections=[
                Section(
                    id="mcp-intro",
                    title="Introduction to MCP",
                    slug="intro",
                    description="Understanding the Model Context Protocol",
                    icon="📡",
                    order=1,
                    pages=[
                        Page(id="what-is-mcp", title="What is MCP?", slug="what-is-mcp",
                             description="Introduction to the Model Context Protocol", estimated_time=12, order=1,
                             content_path="mcp/what-is-mcp.md",
                             tags=["mcp", "basics", "introduction", "protocol"]),
                        Page(id="mcp-architecture", title="MCP Architecture", slug="architecture",
                             description="Hosts, clients, servers, and transports", estimated_time=15, order=2,
                             content_path="mcp/mcp-architecture.md",
                             tags=["mcp", "architecture", "json-rpc"]),
                        Page(id="when-to-use-mcp", title="When to Use MCP", slug="when-to-use",
                             description="Decision framework for adopting MCP", estimated_time=10, order=3,
                             content_path="mcp/when-to-use-mcp.md",
                             tags=["mcp", "decision", "use-cases"]),
                    ],
                ),
                Section(
                    id="mcp-core-concepts",
                    title="MCP Core Concepts",
                    slug="core-concepts",
                    description="Resources, tools, prompts, and transports",
                    icon="🧩",
                    order=2,
                    pages=[
                        Page(id="mcp-resources", title="MCP Resources", slug="resources",
                             description="Exposing data as addressable resources", estimated_time=12, order=1,
                             content_path="mcp/mcp-resources.md",
                             tags=["mcp", "resources", "uri"]),
                        Page(id="mcp-tools", title="MCP Tools", slug="tools",
                             description="Defining and implementing tools", estimated_time=15, order=2,
                             content_path="mcp/mcp-tools.md",
                             tags=["mcp", "tools", "actions"]),
                        Page(id="mcp-prompts", title="MCP Prompts", slug="prompts",
                             description="Reusable prompt templates", estimated_time=10, order=3,
                             content_path="mcp/mcp-prompts.md",
                             tags=["mcp", "prompts", "templates"]),
                        Page(id="mcp-transport", title="MCP Transport Layer", slug="transport",
                             description="stdio and HTTP+SSE transports", estimated_time=12, order=4,
                             content_path="mcp/mcp-transport.md",
                             tags=["mcp", "transport", "stdio", "http", "sse"]),
                    ],
                ),
                Section(
                    id="mcp-servers",
                    title="Building MCP Servers",
                    slug="servers",
                    description="Create production-ready MCP servers",
                    icon="🖥️",
                    order=3,
                    pages=[
                        Page(id="server-structure", title="MCP Server Structure", slug="server-structure",
                             description="Server architecture and organization", estimated_time=15, order=1,
                             content_path="mcp/server-structure.md",
                             tags=["mcp", "server", "structure"]),
                        Page(id="implementing-tools", title="Implementing Tools", slug="implementing-tools",
                             description="Building robust tool implementations", estimated_time=18, order=2,
                             content_path="mcp/implementing-tools.md",
                             tags=["mcp", "tools", "implementation"]),
                        Page(id="implementing-resources", title="Implementing Resources", slug="implementing-resources",
                             description="Building resource providers", estimated_time=15, order=3,
                             content_path="mcp/implementing-resources.md",
                             tags=["mcp", "resources", "implementation"]),
                    ],
                    labs=[
                        Lab(id="mcp-server-lab", title="Build an MCP Server", slug="server-lab",
                            description="Create a file operations MCP server", estimated_time=30, order=1,
                            content_path="mcp/mcp-server-lab.md",
                            difficulty=Difficulty.ADVANCED),
                    ],
                ),
                Section(
                    id="mcp-clients",
                    title="MCP Clients & Integration",
                    slug="clients",
                    description="Connect to and use MCP servers",
                    icon="🔗",
                    order=4,
                    pages=[
                        Page(id="client-implementation", title="MCP Client Implementation", slug="client-implementation",
                             description="Building MCP clients", estimated_time=15, order=1,
                             content_path="mcp/client-implementation.md",
                             tags=["mcp", "client", "implementation"]),
                    ],
                    labs=[
                        Lab(id="mcp-client-lab", title="Build an MCP Client", slug="client-lab",
                            description="Create an interactive MCP client REPL", estimated_time=30, order=1,
                            content_path="mcp/mcp-client-lab.md",
                            difficulty=Difficulty.ADVANCED),
                        Lab(id="filesystem-mcp-lab", title="File System MCP Server", slug="filesystem-lab",
                            description="Build a production file system server", estimated_time=45, order=2,
                            content_path="mcp/filesystem-mcp-lab.md",
                            difficulty=Difficulty.ADVANCED),
                    ],
                ),
            ],
        )
        
        # Module 8: Multi-Agent Systems
        self.modules["multi-agents"] = Module(
            id="multi-agents",
            title="Multi-Agent Systems",
            slug="multi-agents",
            description="Build systems with multiple collaborating AI agents.",
            icon="👥",
            color="pink",
            difficulty=Difficulty.EXPERT,
            prerequisites=["agents"],
            order=8,
            sections=[
                Section(
                    id="multi-agent-concepts",
                    title="Multi-Agent Concepts",
                    slug="concepts",
                    description="Why and when to use multiple agents",
                    icon="🤝",
                    order=1,
                    pages=[
                        Page(id="why-multi-agent", title="Why Multi-Agent Systems?", slug="why-multi-agent",
                             description="Benefits and use cases for multiple agents", estimated_time=12, order=1,
                             content_path="multi-agents/why-multi-agent.md",
                             tags=["multi-agent", "benefits", "patterns"]),
                        Page(id="communication-patterns", title="Agent Communication Patterns", slug="communication-patterns",
                             description="How agents communicate and share information", estimated_time=15, order=2,
                             content_path="multi-agents/communication-patterns.md",
                             tags=["communication", "messages", "channels"]),
                        Page(id="coordination-strategies", title="Coordination Strategies", slug="coordination-strategies",
                             description="Strategies for coordinating agent behavior", estimated_time=15, order=3,
                             content_path="multi-agents/coordination-strategies.md",
                             tags=["coordination", "consensus", "voting"]),
                    ],
                ),
                Section(
                    id="orchestration-patterns",
                    title="Orchestration Patterns",
                    slug="orchestration",
                    description="Common patterns for multi-agent orchestration",
                    icon="🎭",
                    order=2,
                    pages=[
                        Page(id="sequential-workflows", title="Sequential Workflows", slug="sequential-workflows",
                             description="Pipeline and sequential agent patterns", estimated_time=18, order=1,
                             content_path="multi-agents/sequential-workflows.md",
                             tags=["sequential", "pipeline", "workflow"]),
                        Page(id="parallel-workflows", title="Parallel Workflows", slug="parallel-workflows",
                             description="Fan-out/fan-in and parallel execution", estimated_time=15, order=2,
                             content_path="multi-agents/parallel-workflows.md",
                             tags=["parallel", "fan-out", "concurrent"]),
                        Page(id="hierarchical-agents", title="Hierarchical Agents", slug="hierarchical-agents",
                             description="Manager-worker and delegation patterns", estimated_time=15, order=3,
                             content_path="multi-agents/hierarchical-agents.md",
                             tags=["hierarchical", "manager", "delegation"]),
                        Page(id="debate-patterns", title="Debate Patterns", slug="debate-patterns",
                             description="Adversarial and collaborative debate", estimated_time=15, order=4,
                             content_path="multi-agents/debate-patterns.md",
                             tags=["debate", "critic", "adversarial"]),
                    ],
                ),
                Section(
                    id="building-multi-agent",
                    title="Building Multi-Agent Systems",
                    slug="building",
                    description="Implementation patterns for multi-agent systems",
                    icon="🔧",
                    order=3,
                    pages=[
                        Page(id="agent-roles", title="Defining Agent Roles", slug="agent-roles",
                             description="Specializing agents with distinct roles", estimated_time=15, order=1,
                             content_path="multi-agents/agent-roles.md",
                             tags=["roles", "specialization", "capabilities"]),
                        Page(id="message-passing", title="Message Passing", slug="message-passing",
                             description="Inter-agent communication implementation", estimated_time=18, order=2,
                             content_path="multi-agents/message-passing.md",
                             tags=["messages", "queues", "routing"]),
                        Page(id="state-sharing", title="State Sharing", slug="state-sharing",
                             description="Shared state and memory between agents", estimated_time=15, order=3,
                             content_path="multi-agents/state-sharing.md",
                             tags=["state", "shared-memory", "handoff"]),
                    ],
                    labs=[
                        Lab(id="writer-reviewer-lab", title="Writer-Reviewer System", slug="writer-reviewer",
                            description="Build a collaborative writing system", estimated_time=45, order=1,
                            content_path="multi-agents/writer-reviewer-lab.md",
                            difficulty=Difficulty.ADVANCED),
                    ],
                ),
                Section(
                    id="advanced-multi-agent",
                    title="Advanced Multi-Agent Patterns",
                    slug="advanced",
                    description="Complex multi-agent architectures",
                    icon="🚀",
                    order=4,
                    pages=[
                        Page(id="agent-swarms", title="Agent Swarms", slug="agent-swarms",
                             description="Swarm intelligence and emergent behavior", estimated_time=18, order=1,
                             content_path="multi-agents/agent-swarms.md",
                             tags=["swarms", "emergent", "decentralized"]),
                        Page(id="autonomous-teams", title="Autonomous Teams", slug="autonomous-teams",
                             description="Self-organizing agent teams", estimated_time=15, order=2,
                             content_path="multi-agents/autonomous-teams.md",
                             tags=["autonomous", "self-organizing", "teams"]),
                        Page(id="human-in-the-loop", title="Human-in-the-Loop", slug="human-in-the-loop",
                             description="Human oversight in multi-agent systems", estimated_time=15, order=3,
                             content_path="multi-agents/human-in-the-loop.md",
                             tags=["human", "oversight", "approval"]),
                    ],
                    labs=[
                        Lab(id="research-team-lab", title="Build a Research Team", slug="research-team",
                            description="Multi-agent research system", estimated_time=50, order=1,
                            content_path="multi-agents/research-team-lab.md",
                            difficulty=Difficulty.EXPERT),
                        Lab(id="coding-team-lab", title="Build a Coding Team", slug="coding-team",
                            description="Multi-agent code generation system", estimated_time=55, order=2,
                            content_path="multi-agents/coding-team-lab.md",
                            difficulty=Difficulty.EXPERT),
                    ],
                ),
            ],
        )
        
        # Module 9: Fine-tuning LLMs
        self.modules["fine-tuning"] = Module(
            id="fine-tuning",
            title="Fine-tuning LLMs",
            slug="fine-tuning",
            description="Customize LLMs for your specific use case through fine-tuning.",
            icon="🎛️",
            color="yellow",
            difficulty=Difficulty.ADVANCED,
            prerequisites=["llm-fundamentals"],
            order=9,
            sections=[
                Section(
                    id="fine-tuning-fundamentals",
                    title="Fine-tuning Fundamentals",
                    slug="fundamentals",
                    description="When and how to fine-tune",
                    icon="📖",
                    order=1,
                    pages=[
                        Page(id="why-finetune", title="Why Fine-tune?", slug="why-finetune",
                             description="Fine-tuning vs RAG vs prompting", estimated_time=15, order=1,
                             content_path="fine-tuning/why-finetune.md",
                             tags=["fine-tuning", "decision", "rag"]),
                        Page(id="types-of-finetuning", title="Types of Fine-tuning", slug="types",
                             description="Full, LoRA, QLoRA explained", estimated_time=18, order=2,
                             content_path="fine-tuning/types-of-finetuning.md",
                             tags=["lora", "qlora", "peft"]),
                        Page(id="dataset-preparation", title="Dataset Preparation", slug="dataset",
                             description="Preparing data for fine-tuning", estimated_time=20, order=3,
                             content_path="fine-tuning/dataset-preparation.md",
                             tags=["data", "jsonl", "training"]),
                    ],
                    labs=[
                        Lab(id="prepare-dataset", title="Prepare a Fine-tuning Dataset", slug="prepare",
                            description="Create and format a dataset", estimated_time=25, order=1,
                            difficulty=Difficulty.ADVANCED,
                            provider_required=False),
                    ],
                ),
                Section(
                    id="cloud-finetuning",
                    title="Cloud Fine-tuning",
                    slug="cloud",
                    description="Fine-tune using provider APIs",
                    icon="☁️",
                    order=2,
                    pages=[
                        Page(id="cloud-finetuning", title="Cloud Fine-tuning", slug="cloud-finetuning",
                             description="OpenAI, Google, and other providers", estimated_time=20, order=1,
                             content_path="fine-tuning/cloud-finetuning.md",
                             tags=["openai", "cloud", "api"]),
                    ],
                    labs=[
                        Lab(id="openai-finetune", title="Fine-tune with OpenAI", slug="openai-finetune",
                            description="Fine-tune GPT models using OpenAI API", estimated_time=45, order=1,
                            content_path="fine-tuning/openai-finetune-lab.md",
                            difficulty=Difficulty.ADVANCED,
                            provider_required=True),
                    ],
                ),
                Section(
                    id="local-finetuning",
                    title="Local Fine-tuning",
                    slug="local",
                    description="Fine-tune on your own hardware",
                    icon="🖥️",
                    order=3,
                    pages=[
                        Page(id="local-finetuning", title="Local Fine-tuning with LoRA", slug="local-finetuning",
                             description="Fine-tune with PEFT and QLoRA", estimated_time=25, order=1,
                             content_path="fine-tuning/local-finetuning.md",
                             tags=["lora", "qlora", "peft", "huggingface"]),
                    ],
                    labs=[
                        Lab(id="qlora-lab", title="QLoRA Fine-tuning", slug="qlora",
                            description="Fine-tune on consumer GPUs with QLoRA", estimated_time=60, order=1,
                            provider_required=False),
                    ],
                ),
                Section(
                    id="finetuning-practices",
                    title="Best Practices",
                    slug="best-practices",
                    description="Evaluation and optimization",
                    icon="✅",
                    order=4,
                    pages=[
                        Page(id="finetuning-best-practices", title="Fine-tuning Best Practices", slug="best-practices",
                             description="Evaluation, debugging, and optimization", estimated_time=20, order=1,
                             content_path="fine-tuning/best-practices.md",
                             tags=["evaluation", "debugging", "optimization"]),
                    ],
                ),
            ],
        )
        
        # Module 10: Self-Hosting LLMs
        self.modules["self-hosting"] = Module(
            id="self-hosting",
            title="Self-Hosting LLMs",
            slug="self-hosting",
            description="Run LLMs locally for privacy, cost savings, and customization.",
            icon="🏠",
            color="teal",
            difficulty=Difficulty.ADVANCED,
            prerequisites=["llm-fundamentals"],
            order=10,
            sections=[
                Section(
                    id="self-hosting-intro",
                    title="Self-Hosting Introduction",
                    slug="intro",
                    description="Why and how to self-host LLMs",
                    icon="🖥️",
                    order=1,
                    pages=[
                        Page(id="why-self-host", title="Why Self-Host?", slug="why",
                             description="Benefits of running models locally", estimated_time=10, order=1,
                             content_path="self-hosting/why-self-host.md",
                             tags=["self-hosting", "privacy", "cost"]),
                        Page(id="hardware-requirements", title="Hardware Requirements", slug="hardware",
                             description="GPU, RAM, and storage needs", estimated_time=12, order=2,
                             content_path="self-hosting/hardware-requirements.md",
                             tags=["gpu", "vram", "hardware"]),
                        Page(id="model-formats", title="Model Formats", slug="formats",
                             description="GGUF, GPTQ, AWQ, Safetensors explained", estimated_time=15, order=3,
                             content_path="self-hosting/model-formats.md",
                             tags=["gguf", "gptq", "awq", "formats"]),
                        Page(id="quantization", title="Model Quantization", slug="quantization",
                             description="K-quants, I-quants, and quality tradeoffs", estimated_time=15, order=4,
                             content_path="self-hosting/quantization.md",
                             tags=["quantization", "k-quants", "quality"]),
                    ],
                ),
                Section(
                    id="local-inference-tools",
                    title="Local Inference Tools",
                    slug="tools",
                    description="Tools for running LLMs locally",
                    icon="🔧",
                    order=2,
                    pages=[
                        Page(id="ollama-setup", title="Ollama Setup & Usage", slug="ollama",
                             description="Complete guide to Ollama", estimated_time=20, order=1,
                             content_path="self-hosting/ollama-setup.md",
                             tags=["ollama", "local", "inference"]),
                        Page(id="llama-cpp", title="llama.cpp Deep Dive", slug="llama-cpp",
                             description="Maximum control with llama.cpp", estimated_time=25, order=2,
                             content_path="self-hosting/llama-cpp.md",
                             tags=["llama.cpp", "ggml", "server"]),
                        Page(id="vllm", title="vLLM for Production", slug="vllm",
                             description="High-throughput inference with vLLM", estimated_time=20, order=3,
                             content_path="self-hosting/vllm.md",
                             tags=["vllm", "production", "paged-attention"]),
                    ],
                    labs=[
                        Lab(id="ollama-lab", title="Hands-on Ollama Lab", slug="ollama-lab",
                            description="Set up and use Ollama from scratch", estimated_time=30, order=1,
                            content_path="self-hosting/ollama-lab.md",
                            provider_required=False),
                    ],
                ),
                Section(
                    id="model-selection",
                    title="Model Selection",
                    slug="models",
                    description="Choosing the right open source model",
                    icon="🎯",
                    order=3,
                    pages=[
                        Page(id="open-source-models", title="Open Source Model Guide", slug="open-source",
                             description="Llama, Mistral, Qwen, and more", estimated_time=20, order=1,
                             content_path="self-hosting/open-source-models.md",
                             tags=["llama", "mistral", "qwen", "open-source"]),
                    ],
                ),
                Section(
                    id="production-deployment",
                    title="Production Deployment",
                    slug="deployment",
                    description="Deploying self-hosted LLMs",
                    icon="🚀",
                    order=4,
                    pages=[
                        Page(id="docker-deployment", title="Docker Deployment", slug="docker",
                             description="Containerize local LLM deployments", estimated_time=25, order=1,
                             content_path="self-hosting/docker-deployment.md",
                             tags=["docker", "containers", "deployment"]),
                    ],
                ),
                Section(
                    id="integration",
                    title="Self-Hosted Integration",
                    slug="integration",
                    description="Integrating self-hosted models into applications",
                    icon="🔌",
                    order=5,
                    pages=[
                        Page(id="integration-patterns", title="Integration Patterns", slug="patterns",
                             description="Production patterns for self-hosted LLMs", estimated_time=20, order=1,
                             content_path="self-hosting/integration-patterns.md",
                             tags=["integration", "patterns", "production"]),
                        Page(id="hybrid-architectures", title="Hybrid Architectures", slug="hybrid",
                             description="Combining local and cloud models", estimated_time=18, order=2,
                             content_path="self-hosting/hybrid-architectures.md",
                             tags=["hybrid", "routing", "cloud"]),
                    ],
                ),
            ],
        )
        
        # Module 11: Advanced Topics
        self.modules["advanced-topics"] = Module(
            id="advanced-topics",
            title="Advanced Topics",
            slug="advanced-topics",
            description="Production-ready techniques for structured outputs, safety, observability, and cost optimization.",
            icon="🎯",
            color="emerald",
            difficulty=Difficulty.ADVANCED,
            prerequisites=["llm-fundamentals"],
            order=11,
            sections=[
                Section(
                    id="structured-outputs",
                    title="Structured Outputs",
                    slug="structured-outputs",
                    description="Generate reliable, typed responses from LLMs",
                    icon="📋",
                    order=1,
                    pages=[
                        Page(id="json-mode-schemas", title="JSON Mode & Schemas", slug="json-mode",
                             description="Using JSON mode and structured outputs", estimated_time=15, order=1,
                             content_path="advanced-topics/json-mode-schemas.md",
                             tags=["json", "structured", "schemas"]),
                        Page(id="function-calling-practices", title="Function Calling Best Practices", slug="function-calling",
                             description="Production patterns for function calling", estimated_time=18, order=2,
                             content_path="advanced-topics/function-calling-practices.md",
                             tags=["function-calling", "tools", "patterns"]),
                        Page(id="pydantic-integration", title="Pydantic Integration", slug="pydantic",
                             description="Type-safe outputs with Pydantic", estimated_time=15, order=3,
                             content_path="advanced-topics/pydantic-integration.md",
                             tags=["pydantic", "validation", "types"]),
                    ],
                    labs=[
                        Lab(id="structured-outputs-lab", title="Build Type-Safe LLM Outputs", slug="structured-lab",
                            description="Implement reliable structured extraction", estimated_time=30, order=1,
                            content_path="advanced-topics/structured-outputs-lab.md",
                            difficulty=Difficulty.INTERMEDIATE),
                    ],
                ),
                Section(
                    id="guardrails-safety",
                    title="Guardrails & Safety",
                    slug="guardrails",
                    description="Secure and validate LLM interactions",
                    icon="🛡️",
                    order=2,
                    pages=[
                        Page(id="prompt-injection-prevention", title="Prompt Injection Prevention", slug="prompt-injection",
                             description="Defend against injection attacks", estimated_time=18, order=1,
                             content_path="advanced-topics/prompt-injection-prevention.md",
                             tags=["security", "injection", "defense"]),
                        Page(id="content-filtering", title="Content Filtering", slug="content-filtering",
                             description="Filter inputs and outputs for safety", estimated_time=15, order=2,
                             content_path="advanced-topics/content-filtering.md",
                             tags=["filtering", "moderation", "safety"]),
                        Page(id="output-validation", title="Output Validation", slug="output-validation",
                             description="Validate and verify LLM responses", estimated_time=15, order=3,
                             content_path="advanced-topics/output-validation.md",
                             tags=["validation", "verification", "quality"]),
                    ],
                    labs=[
                        Lab(id="safety-guardrails-lab", title="Implement Safety Guardrails", slug="safety-lab",
                            description="Build a comprehensive security pipeline", estimated_time=35, order=1,
                            content_path="advanced-topics/safety-guardrails-lab.md",
                            difficulty=Difficulty.ADVANCED),
                    ],
                ),
                Section(
                    id="observability",
                    title="Observability",
                    slug="observability",
                    description="Monitor, trace, and debug LLM applications",
                    icon="📊",
                    order=3,
                    pages=[
                        Page(id="langsmith-observability", title="LangSmith Observability", slug="langsmith",
                             description="Tracing with LangSmith", estimated_time=18, order=1,
                             content_path="advanced-topics/langsmith-observability.md",
                             tags=["langsmith", "tracing", "langchain"]),
                        Page(id="phoenix-arize-monitoring", title="Phoenix & Arize Monitoring", slug="phoenix",
                             description="Open-source LLM observability", estimated_time=15, order=2,
                             content_path="advanced-topics/phoenix-arize-monitoring.md",
                             tags=["phoenix", "arize", "monitoring"]),
                        Page(id="custom-logging-tracing", title="Custom Logging & Tracing", slug="custom-logging",
                             description="Build your own observability stack", estimated_time=20, order=3,
                             content_path="advanced-topics/custom-logging-tracing.md",
                             tags=["logging", "tracing", "metrics"]),
                    ],
                    labs=[
                        Lab(id="observability-lab", title="LLM Observability Lab", slug="observability-lab",
                            description="Build complete observability system", estimated_time=40, order=1,
                            content_path="advanced-topics/observability-lab.md",
                            difficulty=Difficulty.ADVANCED),
                    ],
                ),
                Section(
                    id="cost-optimization",
                    title="Cost Optimization",
                    slug="optimization",
                    description="Reduce LLM costs while maintaining quality",
                    icon="💰",
                    order=4,
                    pages=[
                        Page(id="semantic-caching", title="Semantic Caching", slug="semantic-caching",
                             description="Cache responses by semantic similarity", estimated_time=18, order=1,
                             content_path="advanced-topics/semantic-caching.md",
                             tags=["caching", "semantic", "redis"]),
                        Page(id="prompt-compression", title="Prompt Compression", slug="prompt-compression",
                             description="Reduce token usage with compression", estimated_time=15, order=2,
                             content_path="advanced-topics/prompt-compression.md",
                             tags=["compression", "tokens", "optimization"]),
                        Page(id="model-routing", title="Model Routing", slug="model-routing",
                             description="Route to cheap vs expensive models", estimated_time=18, order=3,
                             content_path="advanced-topics/model-routing.md",
                             tags=["routing", "tiering", "cost"]),
                    ],
                    labs=[
                        Lab(id="cost-optimization-lab", title="Cost Optimization Lab", slug="cost-lab",
                            description="Build a cost-optimized LLM system", estimated_time=45, order=1,
                            content_path="advanced-topics/cost-optimization-lab.md",
                            difficulty=Difficulty.ADVANCED),
                    ],
                ),
            ],
        )
    
    def _register_learning_paths(self):
        """Register curated learning paths."""
        
        self.learning_paths["quick-start"] = LearningPath(
            id="quick-start",
            title="Quick Start",
            slug="quick-start",
            description="Get started with LLMs in under 2 hours",
            icon="🚀",
            modules=["llm-fundamentals"],
            estimated_hours=2,
            target_audience="Developers new to AI",
            outcomes=["Make API calls to LLMs", "Write effective prompts", "Understand key parameters"],
        )
        
        self.learning_paths["rag-engineer"] = LearningPath(
            id="rag-engineer",
            title="RAG Engineer",
            slug="rag-engineer",
            description="Master retrieval augmented generation",
            icon="📚",
            modules=["llm-fundamentals", "embeddings-vectors", "rag"],
            estimated_hours=8,
            target_audience="Developers building knowledge applications",
            outcomes=["Build RAG pipelines", "Optimize retrieval", "Evaluate RAG quality"],
        )
        
        self.learning_paths["agent-developer"] = LearningPath(
            id="agent-developer",
            title="Agent Developer",
            slug="agent-developer",
            description="Build autonomous AI agents",
            icon="🤖",
            modules=["llm-fundamentals", "tools-functions", "agents", "mcp", "multi-agents"],
            estimated_hours=12,
            target_audience="Developers building AI automation",
            outcomes=["Build ReAct agents", "Implement MCP servers", "Create multi-agent systems"],
        )
        
        self.learning_paths["complete"] = LearningPath(
            id="complete",
            title="Complete AI Engineer",
            slug="complete",
            description="Master all AI engineering concepts",
            icon="🎓",
            modules=["llm-fundamentals", "advanced-llm", "embeddings-vectors", "rag", "tools-functions", 
                     "agents", "mcp", "multi-agents", "fine-tuning", "self-hosting", "advanced-topics"],
            estimated_hours=50,
            target_audience="Developers seeking comprehensive AI knowledge",
            outcomes=["Full-stack AI development", "Production deployment", "Advanced optimization"],
        )
    
    def _build_page_index(self):
        """Build index for quick page lookup."""
        for module_id, module in self.modules.items():
            for section in module.sections:
                for page in section.pages:
                    self._page_index[page.id] = (module_id, section.id)
                for lab in section.labs:
                    self._page_index[lab.id] = (module_id, section.id)
                for quiz in section.quizzes:
                    self._page_index[quiz.id] = (module_id, section.id)
    
    def get_module(self, module_id: str) -> Optional[Module]:
        """Get a module by ID."""
        return self.modules.get(module_id)
    
    def get_section(self, module_id: str, section_id: str) -> Optional[Section]:
        """Get a section by module and section ID."""
        module = self.get_module(module_id)
        if module:
            for section in module.sections:
                if section.id == section_id:
                    return section
        return None
    
    def get_page(self, module_id: str, section_id: str, page_id: str) -> Optional[Page]:
        """Get a page by module, section, and page ID."""
        section = self.get_section(module_id, section_id)
        if section:
            for page in section.pages:
                if page.id == page_id or page.slug == page_id:
                    return page
            for lab in section.labs:
                if lab.id == page_id or lab.slug == page_id:
                    return lab
            for quiz in section.quizzes:
                if quiz.id == page_id or quiz.slug == page_id:
                    return quiz
        return None
    
    def get_page_by_id(self, page_id: str) -> Optional[tuple]:
        """Get a page and its location by page ID."""
        if page_id in self._page_index:
            module_id, section_id = self._page_index[page_id]
            module = self.get_module(module_id)
            section = self.get_section(module_id, section_id)
            if section:
                for page in section.pages:
                    if page.id == page_id:
                        return (module, section, page)
                for lab in section.labs:
                    if lab.id == page_id:
                        return (module, section, lab)
                for quiz in section.quizzes:
                    if quiz.id == page_id:
                        return (module, section, quiz)
        return None
    
    def get_lab(self, lab_id: str) -> Optional[Lab]:
        """Get a standalone lab by ID."""
        for module in self.modules.values():
            for section in module.sections:
                for lab in section.labs:
                    if lab.id == lab_id or lab.slug == lab_id:
                        return lab
        return None
    
    def get_lab_context(self, lab_id: str) -> Optional[tuple]:
        """Get the module_id and section_id for a lab."""
        if lab_id in self._page_index:
            return self._page_index[lab_id]
        # Also check by slug
        for module in self.modules.values():
            for section in module.sections:
                for lab in section.labs:
                    if lab.id == lab_id or lab.slug == lab_id:
                        return (module.id, section.id)
        return None
    
    def get_lab_from_section(self, module_id: str, section_id: str, lab_id: str) -> Optional[Lab]:
        """Get a lab from a specific section."""
        section = self.get_section(module_id, section_id)
        if section:
            for lab in section.labs:
                if lab.id == lab_id or lab.slug == lab_id:
                    return lab
        return None
    
    def get_all_labs(self) -> List[Lab]:
        """Get all labs across all modules."""
        labs = []
        for module in self.modules.values():
            for section in module.sections:
                for lab in section.labs:
                    labs.append(lab)
        return labs
    
    def get_adjacent_sections(self, module_id: str, section_id: str) -> tuple:
        """Get previous and next sections for navigation."""
        module = self.get_module(module_id)
        if not module:
            return (None, None)
        
        sections = sorted(module.sections, key=lambda s: s.order)
        current_idx = None
        for i, section in enumerate(sections):
            if section.id == section_id:
                current_idx = i
                break
        
        if current_idx is None:
            return (None, None)
        
        prev_section = sections[current_idx - 1] if current_idx > 0 else None
        next_section = sections[current_idx + 1] if current_idx < len(sections) - 1 else None
        
        return (prev_section, next_section)
    
    def get_adjacent_content(self, module_id: str, section_id: str, page_id: str) -> tuple:
        """Get previous and next content items (pages/labs) for navigation within a module.
        
        Returns a tuple of (prev_item, next_item) where each item is a dict with:
        - type: 'page', 'lab', or 'quiz'
        - item: the actual Page, Lab, or Quiz object
        - section: the section containing the item
        - module: the module
        """
        module = self.get_module(module_id)
        if not module:
            return (None, None)
        
        # Build a flat list of all content items in order across all sections
        all_items = []
        sections = sorted(module.sections, key=lambda s: s.order)
        
        for section in sections:
            # Add pages in order
            for page in sorted(section.pages, key=lambda p: p.order):
                all_items.append({
                    'type': 'page',
                    'item': page,
                    'section': section,
                    'module': module,
                    'id': page.id
                })
            # Add labs in order
            for lab in sorted(section.labs, key=lambda l: l.order):
                all_items.append({
                    'type': 'lab',
                    'item': lab,
                    'section': section,
                    'module': module,
                    'id': lab.id
                })
            # Add quizzes in order
            for quiz in sorted(section.quizzes, key=lambda q: q.order):
                all_items.append({
                    'type': 'quiz',
                    'item': quiz,
                    'section': section,
                    'module': module,
                    'id': quiz.id
                })
        
        # Find current item
        current_idx = None
        for i, content in enumerate(all_items):
            if content['id'] == page_id:
                current_idx = i
                break
        
        if current_idx is None:
            return (None, None)
        
        prev_item = all_items[current_idx - 1] if current_idx > 0 else None
        next_item = all_items[current_idx + 1] if current_idx < len(all_items) - 1 else None
        
        return (prev_item, next_item)
    
    def get_adjacent_content_for_path(self, path_id: str, module_id: str, section_id: str, page_id: str) -> tuple:
        """Get previous and next content items for navigation within a learning path.
        
        This method provides cross-module navigation within a learning path, allowing
        users to navigate through all content in the path's modules in sequence.
        
        Returns a tuple of (prev_item, next_item, path_info) where:
        - prev_item/next_item: dict with type, item, section, module, id
        - path_info: dict with current_module_index, total_modules, is_first_in_path, is_last_in_path
        """
        learning_path = self.get_learning_path(path_id)
        if not learning_path:
            # Fall back to single module navigation
            prev_item, next_item = self.get_adjacent_content(module_id, section_id, page_id)
            return (prev_item, next_item, None)
        
        # Build a flat list of ALL content items across ALL modules in the path
        all_items = []
        for path_module_id in learning_path.modules:
            module = self.get_module(path_module_id)
            if not module:
                continue
            
            sections = sorted(module.sections, key=lambda s: s.order)
            for section in sections:
                # Add pages in order
                for page in sorted(section.pages, key=lambda p: p.order):
                    all_items.append({
                        'type': 'page',
                        'item': page,
                        'section': section,
                        'module': module,
                        'id': page.id
                    })
                # Add labs in order
                for lab in sorted(section.labs, key=lambda l: l.order):
                    all_items.append({
                        'type': 'lab',
                        'item': lab,
                        'section': section,
                        'module': module,
                        'id': lab.id
                    })
                # Add quizzes in order
                for quiz in sorted(section.quizzes, key=lambda q: q.order):
                    all_items.append({
                        'type': 'quiz',
                        'item': quiz,
                        'section': section,
                        'module': module,
                        'id': quiz.id
                    })
        
        # Find current item
        current_idx = None
        for i, content in enumerate(all_items):
            if content['id'] == page_id:
                current_idx = i
                break
        
        if current_idx is None:
            return (None, None, None)
        
        prev_item = all_items[current_idx - 1] if current_idx > 0 else None
        next_item = all_items[current_idx + 1] if current_idx < len(all_items) - 1 else None
        
        # Calculate path progress info
        current_module_idx = learning_path.modules.index(module_id) if module_id in learning_path.modules else 0
        path_info = {
            'path_id': path_id,
            'path_title': learning_path.title,
            'path_icon': learning_path.icon,
            'current_module_index': current_module_idx,
            'total_modules': len(learning_path.modules),
            'is_first_in_path': current_idx == 0,
            'is_last_in_path': current_idx == len(all_items) - 1,
            'current_item_index': current_idx,
            'total_items': len(all_items),
        }
        
        return (prev_item, next_item, path_info)
    
    def get_path_content_list(self, path_id: str) -> List[dict]:
        """Get a flat list of all content items in a learning path for progress tracking."""
        learning_path = self.get_learning_path(path_id)
        if not learning_path:
            return []
        
        all_items = []
        for path_module_id in learning_path.modules:
            module = self.get_module(path_module_id)
            if not module:
                continue
            
            sections = sorted(module.sections, key=lambda s: s.order)
            for section in sections:
                for page in sorted(section.pages, key=lambda p: p.order):
                    all_items.append({
                        'id': page.id,
                        'title': page.title,
                        'type': 'page',
                        'module_id': module.id,
                        'section_id': section.id,
                    })
                for lab in sorted(section.labs, key=lambda l: l.order):
                    all_items.append({
                        'id': lab.id,
                        'title': lab.title,
                        'type': 'lab',
                        'module_id': module.id,
                        'section_id': section.id,
                    })
                for quiz in sorted(section.quizzes, key=lambda q: q.order):
                    all_items.append({
                        'id': quiz.id,
                        'title': quiz.title,
                        'type': 'quiz',
                        'module_id': module.id,
                        'section_id': section.id,
                    })
        
        return all_items
    
    def get_first_content_in_path(self, path_id: str) -> Optional[dict]:
        """Get the first content item in a learning path."""
        items = self.get_path_content_list(path_id)
        return items[0] if items else None
    
    def get_all_modules(self) -> List[Module]:
        """Get all modules sorted by order."""
        return sorted(self.modules.values(), key=lambda m: m.order)
    
    def get_all_learning_paths(self) -> List[LearningPath]:
        """Get all learning paths."""
        return list(self.learning_paths.values())
    
    def get_learning_path(self, path_id: str) -> Optional[LearningPath]:
        """Get a learning path by ID."""
        return self.learning_paths.get(path_id)
    
    def get_navigation_data(self) -> Dict[str, Any]:
        """Get navigation structure for sidebar."""
        return {
            "modules": [m.to_dict() for m in self.get_all_modules()],
            "learning_paths": [p.to_dict() for p in self.get_all_learning_paths()],
        }


def get_registry() -> ContentRegistry:
    """Get the content registry singleton."""
    return ContentRegistry.get_instance()
