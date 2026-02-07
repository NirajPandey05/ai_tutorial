# Integration Patterns for Self-Hosted LLMs

Learn how to integrate self-hosted models into your applications with production-ready patterns and best practices.

## Overview

```yaml
integration_goals:
  reliability:
    - "Handle model failures gracefully"
    - "Implement retries and timeouts"
    - "Queue requests during overload"
  
  performance:
    - "Minimize latency"
    - "Efficient batching"
    - "Connection pooling"
  
  maintainability:
    - "Abstract model specifics"
    - "Easy model swapping"
    - "Configuration-driven"
```

## Pattern 1: Provider Abstraction

Create a unified interface for any LLM backend:

```python
from abc import ABC, abstractmethod
from typing import Iterator, Optional
from dataclasses import dataclass

@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: str

@dataclass
class CompletionResponse:
    content: str
    model: str
    tokens_used: int
    finish_reason: str

class LLMProvider(ABC):
    """Abstract base for LLM providers."""
    
    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        **kwargs
    ) -> CompletionResponse:
        """Generate a completion."""
        pass
    
    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        **kwargs
    ) -> Iterator[str]:
        """Stream a completion."""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if provider is available."""
        pass


class OllamaProvider(LLMProvider):
    """Ollama implementation."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b"
    ):
        self.base_url = base_url
        self.model = model
        self._session = None
    
    @property
    def session(self):
        if self._session is None:
            import httpx
            self._session = httpx.Client(
                base_url=self.base_url,
                timeout=120.0
            )
        return self._session
    
    def complete(
        self,
        messages: list[Message],
        **kwargs
    ) -> CompletionResponse:
        response = self.session.post(
            "/api/chat",
            json={
                "model": kwargs.get("model", self.model),
                "messages": [
                    {"role": m.role, "content": m.content}
                    for m in messages
                ],
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "num_predict": kwargs.get("max_tokens", 1024)
                }
            }
        )
        response.raise_for_status()
        data = response.json()
        
        return CompletionResponse(
            content=data["message"]["content"],
            model=data["model"],
            tokens_used=data.get("eval_count", 0),
            finish_reason=data.get("done_reason", "stop")
        )
    
    def stream(
        self,
        messages: list[Message],
        **kwargs
    ) -> Iterator[str]:
        with self.session.stream(
            "POST",
            "/api/chat",
            json={
                "model": kwargs.get("model", self.model),
                "messages": [
                    {"role": m.role, "content": m.content}
                    for m in messages
                ],
                "stream": True
            }
        ) as response:
            import json
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data:
                        yield data["message"].get("content", "")
    
    def health_check(self) -> bool:
        try:
            response = self.session.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False


class LlamaCppProvider(LLMProvider):
    """llama.cpp server implementation."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model: str = "local"
    ):
        self.base_url = base_url
        self.model = model
        self._session = None
    
    @property
    def session(self):
        if self._session is None:
            import httpx
            self._session = httpx.Client(
                base_url=self.base_url,
                timeout=120.0
            )
        return self._session
    
    def complete(
        self,
        messages: list[Message],
        **kwargs
    ) -> CompletionResponse:
        # llama.cpp uses OpenAI-compatible endpoint
        response = self.session.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": m.role, "content": m.content}
                    for m in messages
                ],
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1024)
            }
        )
        response.raise_for_status()
        data = response.json()
        
        return CompletionResponse(
            content=data["choices"][0]["message"]["content"],
            model=data["model"],
            tokens_used=data["usage"]["total_tokens"],
            finish_reason=data["choices"][0]["finish_reason"]
        )
    
    def stream(
        self,
        messages: list[Message],
        **kwargs
    ) -> Iterator[str]:
        with self.session.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": m.role, "content": m.content}
                    for m in messages
                ],
                "stream": True
            }
        ) as response:
            import json
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str != "[DONE]":
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
    
    def health_check(self) -> bool:
        try:
            response = self.session.get("/health")
            return response.status_code == 200
        except Exception:
            return False
```

## Pattern 2: Connection Pool Manager

Manage connections efficiently:

```python
from contextlib import contextmanager
from queue import Queue, Empty
from threading import Lock
import time

class ConnectionPool:
    """Connection pool for LLM providers."""
    
    def __init__(
        self,
        provider_factory,
        pool_size: int = 5,
        timeout: float = 30.0
    ):
        self.provider_factory = provider_factory
        self.pool_size = pool_size
        self.timeout = timeout
        
        self._pool: Queue = Queue(maxsize=pool_size)
        self._lock = Lock()
        self._created = 0
        
        # Pre-populate pool
        for _ in range(pool_size):
            self._pool.put(provider_factory())
            self._created += 1
    
    @contextmanager
    def get_provider(self):
        """Get a provider from the pool."""
        provider = None
        try:
            provider = self._pool.get(timeout=self.timeout)
            yield provider
        finally:
            if provider:
                self._pool.put(provider)
    
    def health_check(self) -> dict:
        """Check pool health."""
        healthy = 0
        unhealthy = 0
        
        providers = []
        while not self._pool.empty():
            try:
                providers.append(self._pool.get_nowait())
            except Empty:
                break
        
        for p in providers:
            if p.health_check():
                healthy += 1
            else:
                unhealthy += 1
            self._pool.put(p)
        
        return {
            "total": len(providers),
            "healthy": healthy,
            "unhealthy": unhealthy,
            "available": self._pool.qsize()
        }

# Usage
pool = ConnectionPool(
    provider_factory=lambda: OllamaProvider(),
    pool_size=5
)

with pool.get_provider() as provider:
    response = provider.complete([Message("user", "Hello!")])
```

## Pattern 3: Retry with Backoff

Handle transient failures:

```python
import time
import random
from functools import wraps
from typing import Type

class RetryConfig:
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: tuple = (Exception,)
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions

def with_retry(config: RetryConfig = None):
    """Decorator for retry logic."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_retries:
                        break
                    
                    # Calculate delay
                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    
                    # Add jitter
                    if config.jitter:
                        delay *= (0.5 + random.random())
                    
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator

# Usage
class RobustLLMClient:
    def __init__(self, provider: LLMProvider):
        self.provider = provider
    
    @with_retry(RetryConfig(max_retries=3, base_delay=1.0))
    def complete(self, messages: list[Message]) -> CompletionResponse:
        return self.provider.complete(messages)
```

## Pattern 4: Request Queue

Handle high load with queuing:

```python
import asyncio
from dataclasses import dataclass
from typing import Callable, Any
from datetime import datetime

@dataclass
class QueuedRequest:
    id: str
    messages: list[Message]
    callback: Callable
    priority: int = 0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class RequestQueue:
    """Priority queue for LLM requests."""
    
    def __init__(
        self,
        provider: LLMProvider,
        max_concurrent: int = 5,
        max_queue_size: int = 100
    ):
        self.provider = provider
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        
        self._queue: asyncio.PriorityQueue = None
        self._active = 0
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Start the queue processor."""
        self._queue = asyncio.PriorityQueue(maxsize=self.max_queue_size)
        
        # Start workers
        workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.max_concurrent)
        ]
        
        return workers
    
    async def _worker(self, worker_id: int):
        """Process requests from queue."""
        while True:
            try:
                priority, request = await self._queue.get()
                
                async with self._lock:
                    self._active += 1
                
                try:
                    # Process request
                    response = await asyncio.to_thread(
                        self.provider.complete,
                        request.messages
                    )
                    
                    # Call callback with result
                    await request.callback(request.id, response, None)
                    
                except Exception as e:
                    await request.callback(request.id, None, e)
                
                finally:
                    async with self._lock:
                        self._active -= 1
                    self._queue.task_done()
                    
            except asyncio.CancelledError:
                break
    
    async def submit(
        self,
        request: QueuedRequest
    ) -> bool:
        """Submit a request to the queue."""
        try:
            # Priority queue uses (priority, item) tuples
            # Lower number = higher priority
            self._queue.put_nowait((-request.priority, request))
            return True
        except asyncio.QueueFull:
            return False
    
    @property
    def stats(self) -> dict:
        return {
            "queue_size": self._queue.qsize() if self._queue else 0,
            "active_requests": self._active,
            "max_concurrent": self.max_concurrent
        }
```

## Pattern 5: Caching Layer

Cache responses for repeated queries:

```python
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional

class ResponseCache:
    """Cache for LLM responses."""
    
    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_size: int = 1000
    ):
        self.ttl = timedelta(seconds=ttl_seconds)
        self.max_size = max_size
        self._cache: dict = {}
        self._access_times: dict = {}
    
    def _make_key(self, messages: list[Message], **kwargs) -> str:
        """Create cache key from messages and kwargs."""
        data = {
            "messages": [(m.role, m.content) for m in messages],
            "kwargs": kwargs
        }
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def get(
        self,
        messages: list[Message],
        **kwargs
    ) -> Optional[CompletionResponse]:
        """Get cached response if exists and valid."""
        key = self._make_key(messages, **kwargs)
        
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        if datetime.now() - entry["created"] > self.ttl:
            # Expired
            del self._cache[key]
            del self._access_times[key]
            return None
        
        # Update access time
        self._access_times[key] = datetime.now()
        return entry["response"]
    
    def set(
        self,
        messages: list[Message],
        response: CompletionResponse,
        **kwargs
    ):
        """Cache a response."""
        # Evict if at capacity
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        key = self._make_key(messages, **kwargs)
        self._cache[key] = {
            "response": response,
            "created": datetime.now()
        }
        self._access_times[key] = datetime.now()
    
    def _evict_oldest(self):
        """Remove least recently accessed entry."""
        if not self._access_times:
            return
        
        oldest_key = min(
            self._access_times.keys(),
            key=lambda k: self._access_times[k]
        )
        del self._cache[oldest_key]
        del self._access_times[oldest_key]


class CachedLLMClient:
    """LLM client with caching."""
    
    def __init__(
        self,
        provider: LLMProvider,
        cache_ttl: int = 3600
    ):
        self.provider = provider
        self.cache = ResponseCache(ttl_seconds=cache_ttl)
        self._hits = 0
        self._misses = 0
    
    def complete(
        self,
        messages: list[Message],
        use_cache: bool = True,
        **kwargs
    ) -> CompletionResponse:
        """Complete with optional caching."""
        
        if use_cache:
            cached = self.cache.get(messages, **kwargs)
            if cached:
                self._hits += 1
                return cached
        
        self._misses += 1
        response = self.provider.complete(messages, **kwargs)
        
        if use_cache:
            self.cache.set(messages, response, **kwargs)
        
        return response
    
    @property
    def cache_stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
            "cached_entries": len(self.cache._cache)
        }
```

## Pattern 6: Complete Application Integration

```python
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid

app = FastAPI()

# Initialize components
provider = OllamaProvider(model="llama3.2:3b")
pool = ConnectionPool(lambda: OllamaProvider(), pool_size=5)
cache = ResponseCache(ttl_seconds=1800)

class ChatRequest(BaseModel):
    messages: list[dict]
    use_cache: bool = True
    priority: int = 0

class ChatResponse(BaseModel):
    id: str
    content: str
    model: str
    cached: bool
    tokens_used: int

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with caching and pooling."""
    
    request_id = str(uuid.uuid4())
    messages = [
        Message(m["role"], m["content"]) 
        for m in request.messages
    ]
    
    # Check cache first
    if request.use_cache:
        cached = cache.get(messages)
        if cached:
            return ChatResponse(
                id=request_id,
                content=cached.content,
                model=cached.model,
                cached=True,
                tokens_used=cached.tokens_used
            )
    
    # Get from pool and generate
    try:
        with pool.get_provider() as provider:
            response = provider.complete(messages)
            
            # Cache the response
            if request.use_cache:
                cache.set(messages, response)
            
            return ChatResponse(
                id=request_id,
                content=response.content,
                model=response.model,
                cached=False,
                tokens_used=response.tokens_used
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    pool_health = pool.health_check()
    
    return {
        "status": "healthy" if pool_health["healthy"] > 0 else "unhealthy",
        "pool": pool_health,
        "cache_size": len(cache._cache)
    }

@app.get("/stats")
async def stats():
    """Statistics endpoint."""
    return {
        "pool": pool.health_check(),
        "cache": {
            "entries": len(cache._cache),
            "max_size": cache.max_size
        }
    }
```

## Best Practices Summary

```yaml
integration_best_practices:
  abstraction:
    - "Always use provider interfaces"
    - "Never hardcode model names in business logic"
    - "Use configuration for all settings"
  
  reliability:
    - "Implement retries with exponential backoff"
    - "Set appropriate timeouts"
    - "Have fallback options"
    - "Use circuit breakers for failing services"
  
  performance:
    - "Use connection pools"
    - "Cache deterministic responses"
    - "Batch requests when possible"
    - "Stream for long responses"
  
  monitoring:
    - "Log all requests and responses"
    - "Track latency percentiles"
    - "Monitor cache hit rates"
    - "Alert on error rates"
```

## Next Steps

1. **Add Observability** - Integrate with Prometheus/Grafana
2. **Load Testing** - Benchmark your integration
3. **Model Switching** - Implement A/B testing
4. **Fine-tuning** - Customize models for your use case
