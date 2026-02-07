# Production RAG Architecture

Design and deploy RAG systems that scale reliably in production environments.

## Production Architecture Overview

```yaml
key_concerns:
  scalability: "Handle growing document corpus and traffic"
  reliability: "Consistent performance and availability"
  maintainability: "Easy updates and debugging"
  cost_efficiency: "Optimize API and infrastructure costs"
  observability: "Monitor and trace system behavior"

architecture_layers:
  ingestion: "Document processing pipeline"
  storage: "Vector database and document store"
  retrieval: "Search and ranking services"
  generation: "LLM integration and prompting"
  api: "Client-facing interface"
```

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Applications                       │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                         API Gateway                              │
│  (Rate Limiting, Auth, Load Balancing)                          │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG Service Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Retrieval  │  │  Generation  │  │    Cache     │          │
│  │   Service    │  │   Service    │  │   Service    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
          │                  │                   │
          ▼                  ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Vector Store   │ │   LLM APIs      │ │  Redis Cache    │
│  (Pinecone/     │ │  (OpenAI/       │ │                 │
│   Weaviate)     │ │   Anthropic)    │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Document Ingestion Pipeline                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Document   │  │   Chunking   │  │  Embedding   │          │
│  │   Loader     │→ │   Service    │→ │   Service    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Core Service Components

```python
from dataclasses import dataclass
from typing import Optional
import asyncio
from abc import ABC, abstractmethod
import hashlib
import json
import time


@dataclass
class RAGConfig:
    """Production RAG configuration."""
    
    # Vector store
    vector_store_type: str = "pinecone"  # pinecone, weaviate, qdrant
    vector_store_url: str = ""
    vector_store_api_key: str = ""
    
    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    
    # LLM
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0
    max_context_tokens: int = 4000
    
    # Retrieval
    retrieval_k: int = 5
    rerank_enabled: bool = True
    min_relevance_score: float = 0.5
    
    # Cache
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    
    # Performance
    embedding_batch_size: int = 100
    concurrent_requests: int = 10


class BaseService(ABC):
    """Base class for RAG services."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.metrics = ServiceMetrics()
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check service health."""
        pass


class ServiceMetrics:
    """Track service metrics."""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.latency_sum = 0.0
    
    def record_request(self, latency: float, success: bool):
        self.request_count += 1
        self.latency_sum += latency
        if not success:
            self.error_count += 1
    
    def get_stats(self) -> dict:
        return {
            "requests": self.request_count,
            "errors": self.error_count,
            "avg_latency_ms": (self.latency_sum / self.request_count * 1000) if self.request_count > 0 else 0,
            "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0
        }
```

## Retrieval Service

```python
from openai import OpenAI
import pinecone


class RetrievalService(BaseService):
    """Production retrieval service."""
    
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self.client = OpenAI()
        self._init_vector_store()
    
    def _init_vector_store(self):
        """Initialize vector store connection."""
        
        if self.config.vector_store_type == "pinecone":
            pinecone.init(
                api_key=self.config.vector_store_api_key,
                environment="gcp-starter"
            )
            self.index = pinecone.Index("rag-production")
    
    async def retrieve(
        self,
        query: str,
        filters: dict = None,
        k: int = None
    ) -> list[dict]:
        """Retrieve relevant documents."""
        
        start_time = time.time()
        success = True
        
        try:
            k = k or self.config.retrieval_k
            
            # Embed query
            query_embedding = await self._embed(query)
            
            # Search vector store
            results = self.index.query(
                vector=query_embedding,
                top_k=k * 2 if self.config.rerank_enabled else k,
                include_metadata=True,
                filter=filters
            )
            
            documents = [
                {
                    "id": match.id,
                    "content": match.metadata.get("content", ""),
                    "score": match.score,
                    "metadata": match.metadata
                }
                for match in results.matches
            ]
            
            # Optional reranking
            if self.config.rerank_enabled and documents:
                documents = await self._rerank(query, documents, k)
            
            # Filter by minimum score
            documents = [
                d for d in documents
                if d["score"] >= self.config.min_relevance_score
            ]
            
            return documents[:k]
        
        except Exception as e:
            success = False
            raise
        
        finally:
            latency = time.time() - start_time
            self.metrics.record_request(latency, success)
    
    async def _embed(self, text: str) -> list[float]:
        """Embed text using OpenAI."""
        
        response = self.client.embeddings.create(
            input=text,
            model=self.config.embedding_model
        )
        return response.data[0].embedding
    
    async def _rerank(
        self,
        query: str,
        documents: list[dict],
        k: int
    ) -> list[dict]:
        """Rerank documents using cross-encoder or Cohere."""
        
        # Simplified reranking using LLM scoring
        # In production, use Cohere Rerank or cross-encoder
        
        doc_texts = [d["content"][:500] for d in documents]
        
        prompt = f"""Rate relevance of each document to the query (0-10).
Query: {query}

Documents:
{chr(10).join(f'[{i+1}] {t}' for i, t in enumerate(doc_texts))}

Scores (comma-separated):"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0
        )
        
        try:
            scores = [
                float(s.strip()) / 10
                for s in response.choices[0].message.content.split(",")
            ]
            
            for i, doc in enumerate(documents):
                doc["rerank_score"] = scores[i] if i < len(scores) else 0.5
            
            documents.sort(key=lambda x: x["rerank_score"], reverse=True)
        except:
            pass  # Keep original order on failure
        
        return documents[:k]
    
    async def health_check(self) -> bool:
        try:
            self.index.describe_index_stats()
            return True
        except:
            return False
```

## Generation Service

```python
class GenerationService(BaseService):
    """Production generation service."""
    
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
- Base answers ONLY on the provided context
- Cite sources using [N] notation
- If context is insufficient, say so
- Be concise and accurate"""
    
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self.client = OpenAI()
    
    async def generate(
        self,
        query: str,
        documents: list[dict],
        conversation_history: list[dict] = None,
        stream: bool = False
    ):
        """Generate response from documents."""
        
        start_time = time.time()
        success = True
        
        try:
            # Build messages
            messages = self._build_messages(
                query, documents, conversation_history
            )
            
            if stream:
                return self._generate_stream(messages)
            else:
                return await self._generate_sync(messages)
        
        except Exception as e:
            success = False
            raise
        
        finally:
            latency = time.time() - start_time
            self.metrics.record_request(latency, success)
    
    def _build_messages(
        self,
        query: str,
        documents: list[dict],
        history: list[dict] = None
    ) -> list[dict]:
        """Build messages for API call."""
        
        context = "\n\n".join([
            f"[Source {i+1}] {doc['content']}"
            for i, doc in enumerate(documents)
        ])
        
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        
        # Add history if present
        if history:
            messages.extend(history[-6:])  # Last 3 exchanges
        
        messages.append({
            "role": "user",
            "content": f"""Context:
{context}

Question: {query}"""
        })
        
        return messages
    
    async def _generate_sync(self, messages: list[dict]) -> dict:
        """Synchronous generation."""
        
        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=messages,
            temperature=self.config.llm_temperature
        )
        
        return {
            "answer": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    
    async def _generate_stream(self, messages: list[dict]):
        """Streaming generation."""
        
        stream = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=messages,
            temperature=self.config.llm_temperature,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield {
                    "type": "content",
                    "text": chunk.choices[0].delta.content
                }
        
        yield {"type": "done"}
    
    async def health_check(self) -> bool:
        try:
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5
            )
            return True
        except:
            return False
```

## Caching Layer

```python
import redis
from datetime import timedelta


class CacheService:
    """Redis-based caching for RAG."""
    
    def __init__(self, config: RAGConfig, redis_url: str = "redis://localhost"):
        self.config = config
        self.redis = redis.from_url(redis_url) if config.cache_enabled else None
        self.ttl = timedelta(seconds=config.cache_ttl_seconds)
    
    def _make_key(self, prefix: str, query: str, **kwargs) -> str:
        """Create cache key from query and params."""
        
        key_data = {"query": query, **kwargs}
        key_hash = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        return f"rag:{prefix}:{key_hash}"
    
    async def get_retrieval(
        self,
        query: str,
        filters: dict = None
    ) -> Optional[list[dict]]:
        """Get cached retrieval results."""
        
        if not self.redis:
            return None
        
        key = self._make_key("retrieval", query, filters=filters)
        cached = self.redis.get(key)
        
        if cached:
            return json.loads(cached)
        return None
    
    async def set_retrieval(
        self,
        query: str,
        documents: list[dict],
        filters: dict = None
    ):
        """Cache retrieval results."""
        
        if not self.redis:
            return
        
        key = self._make_key("retrieval", query, filters=filters)
        self.redis.setex(key, self.ttl, json.dumps(documents))
    
    async def get_embedding(self, text: str) -> Optional[list[float]]:
        """Get cached embedding."""
        
        if not self.redis:
            return None
        
        key = self._make_key("embedding", text)
        cached = self.redis.get(key)
        
        if cached:
            return json.loads(cached)
        return None
    
    async def set_embedding(self, text: str, embedding: list[float]):
        """Cache embedding."""
        
        if not self.redis:
            return
        
        key = self._make_key("embedding", text)
        # Longer TTL for embeddings (they don't change)
        self.redis.setex(key, timedelta(days=7), json.dumps(embedding))
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate cached items matching pattern."""
        
        if not self.redis:
            return
        
        for key in self.redis.scan_iter(f"rag:{pattern}:*"):
            self.redis.delete(key)
```

## FastAPI Application

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging

app = FastAPI(title="Production RAG API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services
config = RAGConfig()
retrieval_service = RetrievalService(config)
generation_service = GenerationService(config)
cache_service = CacheService(config)


class QueryRequest(BaseModel):
    query: str
    filters: dict = None
    stream: bool = False
    conversation_id: str = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    cached: bool
    latency_ms: float


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    
    retrieval_ok = await retrieval_service.health_check()
    generation_ok = await generation_service.health_check()
    
    return {
        "status": "healthy" if retrieval_ok and generation_ok else "degraded",
        "services": {
            "retrieval": retrieval_ok,
            "generation": generation_ok
        }
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system."""
    
    start_time = time.time()
    
    try:
        # Check cache
        cached_docs = await cache_service.get_retrieval(
            request.query, request.filters
        )
        
        if cached_docs:
            documents = cached_docs
            from_cache = True
        else:
            # Retrieve
            documents = await retrieval_service.retrieve(
                request.query,
                filters=request.filters
            )
            
            # Cache results
            await cache_service.set_retrieval(
                request.query, documents, request.filters
            )
            from_cache = False
        
        # Generate
        result = await generation_service.generate(
            request.query,
            documents,
            stream=False
        )
        
        latency = (time.time() - start_time) * 1000
        
        return QueryResponse(
            answer=result["answer"],
            sources=[
                {"id": d["id"], "content": d["content"][:200], "score": d["score"]}
                for d in documents
            ],
            cached=from_cache,
            latency_ms=latency
        )
    
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Stream query response."""
    
    async def generate():
        # Retrieve
        documents = await retrieval_service.retrieve(
            request.query,
            filters=request.filters
        )
        
        # Send sources first
        yield f"data: {json.dumps({'type': 'sources', 'data': [{'id': d['id'], 'score': d['score']} for d in documents]})}\n\n"
        
        # Stream generation
        async for chunk in generation_service.generate(
            request.query,
            documents,
            stream=True
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@app.get("/metrics")
async def get_metrics():
    """Get service metrics."""
    
    return {
        "retrieval": retrieval_service.metrics.get_stats(),
        "generation": generation_service.metrics.get_stats()
    }
```

## Deployment Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:alpine
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - rag-api

volumes:
  redis-data:
```

## Monitoring and Observability

```python
# monitoring.py
import time
from functools import wraps
from opentelemetry import trace
from prometheus_client import Counter, Histogram, generate_latest


# Prometheus metrics
REQUEST_COUNT = Counter(
    'rag_requests_total',
    'Total RAG requests',
    ['operation', 'status']
)

REQUEST_LATENCY = Histogram(
    'rag_request_latency_seconds',
    'Request latency',
    ['operation']
)

RETRIEVAL_SCORE = Histogram(
    'rag_retrieval_score',
    'Retrieval relevance scores'
)


def track_operation(operation: str):
    """Decorator to track operation metrics."""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                latency = time.time() - start_time
                REQUEST_COUNT.labels(operation=operation, status=status).inc()
                REQUEST_LATENCY.labels(operation=operation).observe(latency)
        
        return wrapper
    return decorator


# Usage
class MonitoredRetrievalService(RetrievalService):
    
    @track_operation("retrieval")
    async def retrieve(self, query: str, filters: dict = None, k: int = None):
        documents = await super().retrieve(query, filters, k)
        
        # Track retrieval scores
        for doc in documents:
            RETRIEVAL_SCORE.observe(doc["score"])
        
        return documents
```

## Summary

```yaml
production_checklist:
  infrastructure:
    - "Use managed vector database (Pinecone, Weaviate)"
    - "Implement caching layer (Redis)"
    - "Set up load balancing"
    - "Configure auto-scaling"
  
  reliability:
    - "Health check endpoints"
    - "Graceful degradation"
    - "Retry with exponential backoff"
    - "Circuit breakers"
  
  observability:
    - "Request logging"
    - "Performance metrics"
    - "Error tracking"
    - "Distributed tracing"
  
  security:
    - "API authentication"
    - "Rate limiting"
    - "Input validation"
    - "Secrets management"
```

## Next Steps

1. **Evaluation** - Measure RAG quality
2. **Optimization** - Tune for performance
3. **Advanced Patterns** - HyDE, Self-RAG
