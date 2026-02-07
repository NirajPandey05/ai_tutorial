# Semantic Caching for LLM Applications

Learn how to dramatically reduce LLM costs by caching responses based on semantic similarity rather than exact matches.

---

## Why Semantic Caching?

Traditional caching relies on exact key matches, but LLM queries rarely repeat exactly:
- "What's the weather?" vs "Tell me the weather"
- "Explain Python decorators" vs "How do Python decorators work?"

Semantic caching matches queries by meaning, enabling much higher cache hit rates.

### Cost Impact

| Scenario | Without Cache | With Semantic Cache | Savings |
|----------|---------------|---------------------|---------|
| Support Bot | $500/day | $50/day | 90% |
| FAQ System | $200/day | $20/day | 90% |
| Code Assistant | $1000/day | $300/day | 70% |

---

## How Semantic Caching Works

```
┌──────────────────────────────────────────────────────┐
│                    User Query                        │
└───────────────────────┬──────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│                Generate Embedding                     │
└───────────────────────┬──────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│          Search Cache (Vector Similarity)            │
└───────────────────────┬──────────────────────────────┘
                        │
           ┌────────────┴────────────┐
           │                         │
    ▼ Hit (similarity > threshold)   ▼ Miss
┌──────────┐                   ┌──────────────┐
│ Return   │                   │ Call LLM     │
│ Cached   │                   │ Store Result │
│ Response │                   │ Return       │
└──────────┘                   └──────────────┘
```

---

## Basic Implementation

### Using Redis with Vector Search

```python
import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import numpy as np
from openai import OpenAI
import hashlib
import json
from typing import Optional, Tuple

class SemanticCache:
    """Semantic cache using Redis vector search"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        index_name: str = "semantic_cache",
        embedding_dimension: int = 1536,
        similarity_threshold: float = 0.92
    ):
        self.client = redis.from_url(redis_url)
        self.openai = OpenAI()
        self.index_name = index_name
        self.embedding_dim = embedding_dimension
        self.threshold = similarity_threshold
        self._create_index()
    
    def _create_index(self):
        """Create Redis vector search index"""
        try:
            self.client.ft(self.index_name).info()
        except redis.ResponseError:
            # Create index
            schema = [
                TextField("query"),
                TextField("response"),
                TextField("model"),
                VectorField(
                    "embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.embedding_dim,
                        "DISTANCE_METRIC": "COSINE"
                    }
                )
            ]
            
            definition = IndexDefinition(
                prefix=["cache:"],
                index_type=IndexType.HASH
            )
            
            self.client.ft(self.index_name).create_index(
                schema,
                definition=definition
            )
    
    def _get_embedding(self, text: str) -> list:
        """Generate embedding for text"""
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    def _search_similar(self, embedding: list, top_k: int = 1) -> list:
        """Search for similar cached queries"""
        query_vector = np.array(embedding, dtype=np.float32).tobytes()
        
        query = (
            Query(f"*=>[KNN {top_k} @embedding $vec AS score]")
            .return_fields("query", "response", "model", "score")
            .sort_by("score")
            .dialect(2)
        )
        
        results = self.client.ft(self.index_name).search(
            query,
            query_params={"vec": query_vector}
        )
        
        return results.docs
    
    def get(self, query: str, model: str = None) -> Optional[str]:
        """Get cached response for semantically similar query"""
        embedding = self._get_embedding(query)
        results = self._search_similar(embedding)
        
        if results:
            score = 1 - float(results[0].score)  # Convert distance to similarity
            
            if score >= self.threshold:
                # Optional: Check model match
                if model and results[0].model != model:
                    return None
                
                return results[0].response
        
        return None
    
    def set(
        self,
        query: str,
        response: str,
        model: str,
        ttl: int = 86400  # 24 hours default
    ):
        """Cache a query-response pair"""
        embedding = self._get_embedding(query)
        
        # Create unique key
        key = f"cache:{hashlib.md5(query.encode()).hexdigest()}"
        
        # Store with embedding
        self.client.hset(
            key,
            mapping={
                "query": query,
                "response": response,
                "model": model,
                "embedding": np.array(embedding, dtype=np.float32).tobytes()
            }
        )
        
        # Set TTL
        self.client.expire(key, ttl)
    
    def stats(self) -> dict:
        """Get cache statistics"""
        info = self.client.ft(self.index_name).info()
        return {
            "total_entries": int(info["num_docs"]),
            "index_size": info.get("inverted_sz_mb", 0)
        }
```

### Cached LLM Client

```python
from openai import OpenAI
from typing import Optional
import time

class CachedLLMClient:
    """LLM client with semantic caching"""
    
    def __init__(
        self,
        cache: SemanticCache,
        track_metrics: bool = True
    ):
        self.client = OpenAI()
        self.cache = cache
        self.track_metrics = track_metrics
        
        # Metrics
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0
        self.cost_saved = 0
    
    def chat(
        self,
        messages: list,
        model: str = "gpt-4o-mini",
        use_cache: bool = True,
        cache_ttl: int = 86400,
        **kwargs
    ) -> str:
        """Chat with semantic caching"""
        
        # Build cache key from user message
        user_message = next(
            (m["content"] for m in messages if m["role"] == "user"),
            None
        )
        
        if not user_message:
            # No user message, can't cache
            return self._call_llm(messages, model, **kwargs)
        
        # Try cache
        if use_cache:
            cached = self.cache.get(user_message, model)
            
            if cached:
                if self.track_metrics:
                    self.hits += 1
                    # Estimate tokens saved (rough approximation)
                    self.tokens_saved += len(cached.split()) * 1.3
                
                return cached
        
        # Cache miss - call LLM
        if self.track_metrics:
            self.misses += 1
        
        response = self._call_llm(messages, model, **kwargs)
        
        # Store in cache
        if use_cache:
            self.cache.set(
                query=user_message,
                response=response,
                model=model,
                ttl=cache_ttl
            )
        
        return response
    
    def _call_llm(self, messages: list, model: str, **kwargs) -> str:
        """Make actual LLM API call"""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content
    
    def get_metrics(self) -> dict:
        """Get cache performance metrics"""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0,
            "tokens_saved": int(self.tokens_saved),
            "estimated_cost_saved": self.tokens_saved * 0.00001  # Rough estimate
        }

# Usage
cache = SemanticCache()
client = CachedLLMClient(cache)

# First call - cache miss
response1 = client.chat([
    {"role": "user", "content": "What is machine learning?"}
])

# Similar query - cache hit!
response2 = client.chat([
    {"role": "user", "content": "Explain what machine learning is"}
])

print(client.get_metrics())
# {'hits': 1, 'misses': 1, 'hit_rate': 0.5, ...}
```

---

## Advanced Caching Strategies

### Multi-Level Cache

```python
from abc import ABC, abstractmethod
from typing import Optional
import time

class CacheLayer(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        pass
    
    @abstractmethod
    def set(self, key: str, value: str, ttl: int = None):
        pass

class InMemoryCache(CacheLayer):
    """Fast in-memory cache (L1)"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            entry = self.cache[key]
            if entry['expires'] is None or entry['expires'] > time.time():
                self.access_times[key] = time.time()
                return entry['value']
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: str, ttl: int = None):
        # Evict if needed
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = {
            'value': value,
            'expires': time.time() + ttl if ttl else None
        }
        self.access_times[key] = time.time()
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if self.access_times:
            oldest = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest]
            del self.access_times[oldest]

class MultiLevelCache:
    """Multi-level cache with L1 (memory) and L2 (semantic)"""
    
    def __init__(
        self,
        semantic_cache: SemanticCache,
        memory_size: int = 1000
    ):
        self.l1 = InMemoryCache(max_size=memory_size)
        self.l2 = semantic_cache
        
        self.l1_hits = 0
        self.l2_hits = 0
        self.misses = 0
    
    def get(self, query: str, model: str = None) -> tuple[Optional[str], str]:
        """
        Get from cache, trying L1 first then L2.
        Returns (response, cache_level) where cache_level is 'l1', 'l2', or 'miss'
        """
        # Try L1 (exact match, fast)
        l1_key = f"{model}:{hash(query)}"
        l1_result = self.l1.get(l1_key)
        
        if l1_result:
            self.l1_hits += 1
            return l1_result, 'l1'
        
        # Try L2 (semantic match, slower)
        l2_result = self.l2.get(query, model)
        
        if l2_result:
            self.l2_hits += 1
            # Promote to L1
            self.l1.set(l1_key, l2_result, ttl=3600)
            return l2_result, 'l2'
        
        self.misses += 1
        return None, 'miss'
    
    def set(
        self,
        query: str,
        response: str,
        model: str,
        ttl: int = 86400
    ):
        """Store in both cache levels"""
        # L1 (exact match)
        l1_key = f"{model}:{hash(query)}"
        self.l1.set(l1_key, response, ttl=3600)  # Shorter TTL for L1
        
        # L2 (semantic)
        self.l2.set(query, response, model, ttl)
    
    def get_stats(self) -> dict:
        total = self.l1_hits + self.l2_hits + self.misses
        return {
            "l1_hits": self.l1_hits,
            "l2_hits": self.l2_hits,
            "misses": self.misses,
            "l1_hit_rate": self.l1_hits / total if total > 0 else 0,
            "l2_hit_rate": self.l2_hits / total if total > 0 else 0,
            "total_hit_rate": (self.l1_hits + self.l2_hits) / total if total > 0 else 0
        }
```

### Context-Aware Caching

```python
from typing import Optional
import hashlib

class ContextAwareCache:
    """Cache that considers conversation context"""
    
    def __init__(self, semantic_cache: SemanticCache):
        self.cache = semantic_cache
    
    def _build_context_key(
        self,
        query: str,
        conversation_history: list = None,
        system_prompt: str = None
    ) -> str:
        """Build cache key that includes context"""
        components = [query]
        
        # Include recent history (last 2 turns)
        if conversation_history:
            recent = conversation_history[-4:]  # Last 2 exchanges
            for msg in recent:
                components.append(f"{msg['role']}:{msg['content'][:100]}")
        
        # Include system prompt hash
        if system_prompt:
            components.append(f"sys:{hashlib.md5(system_prompt.encode()).hexdigest()[:8]}")
        
        return " | ".join(components)
    
    def get(
        self,
        query: str,
        conversation_history: list = None,
        system_prompt: str = None,
        model: str = None
    ) -> Optional[str]:
        """Get cached response considering context"""
        context_key = self._build_context_key(
            query, conversation_history, system_prompt
        )
        return self.cache.get(context_key, model)
    
    def set(
        self,
        query: str,
        response: str,
        model: str,
        conversation_history: list = None,
        system_prompt: str = None,
        ttl: int = 86400
    ):
        """Cache with context"""
        context_key = self._build_context_key(
            query, conversation_history, system_prompt
        )
        self.cache.set(context_key, response, model, ttl)
```

---

## GPTCache Integration

GPTCache is a dedicated library for LLM caching:

```python
from gptcache import cache, Config
from gptcache.adapter import openai
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

def init_gptcache():
    """Initialize GPTCache with semantic similarity"""
    
    # Embedding function
    onnx = Onnx()
    
    # Cache storage
    cache_base = CacheBase("sqlite")
    vector_base = VectorBase(
        "faiss",
        dimension=onnx.dimension
    )
    
    # Data manager
    data_manager = get_data_manager(cache_base, vector_base)
    
    # Initialize cache
    cache.init(
        embedding_func=onnx.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
        config=Config(
            similarity_threshold=0.8
        )
    )

# Initialize
init_gptcache()

# Use with OpenAI (automatic caching!)
from gptcache.adapter.openai import ChatCompletion

response = ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "What is Python?"}
    ]
)

# Second similar query - cached!
response2 = ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Tell me about Python programming"}
    ]
)
```

---

## Cache Invalidation Strategies

### Time-Based Expiration

```python
class ExpiringCache:
    """Cache with TTL based on content type"""
    
    TTL_CONFIGS = {
        "factual": 86400 * 7,    # 7 days for facts
        "opinion": 3600,          # 1 hour for opinions
        "real_time": 300,         # 5 mins for time-sensitive
        "default": 86400          # 1 day default
    }
    
    def __init__(self, cache: SemanticCache):
        self.cache = cache
    
    def classify_query(self, query: str) -> str:
        """Classify query to determine TTL"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["weather", "stock", "news", "today"]):
            return "real_time"
        elif any(word in query_lower for word in ["opinion", "think", "feel", "should"]):
            return "opinion"
        elif any(word in query_lower for word in ["what is", "define", "explain", "how does"]):
            return "factual"
        
        return "default"
    
    def set(self, query: str, response: str, model: str):
        """Cache with automatic TTL"""
        query_type = self.classify_query(query)
        ttl = self.TTL_CONFIGS[query_type]
        self.cache.set(query, response, model, ttl)
```

### Version-Based Invalidation

```python
class VersionedCache:
    """Cache with version-based invalidation"""
    
    def __init__(self, cache: SemanticCache, version: str = "v1"):
        self.cache = cache
        self.version = version
    
    def set(self, query: str, response: str, model: str, ttl: int = 86400):
        """Store with version prefix"""
        versioned_query = f"[{self.version}] {query}"
        self.cache.set(versioned_query, response, model, ttl)
    
    def get(self, query: str, model: str = None) -> Optional[str]:
        """Get from current version only"""
        versioned_query = f"[{self.version}] {query}"
        return self.cache.get(versioned_query, model)
    
    def invalidate_version(self, old_version: str):
        """
        Bump version to effectively invalidate old cache.
        Old entries will naturally expire via TTL.
        """
        pass  # Old entries won't match new version prefix
```

---

## Measuring Cache Effectiveness

```python
import time
from dataclasses import dataclass, field
from typing import List

@dataclass
class CacheMetrics:
    """Track cache performance"""
    
    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0
    latency_saved_ms: float = 0
    embedding_costs: float = 0
    llm_costs_saved: float = 0
    
    # Track per-request metrics
    request_latencies: List[float] = field(default_factory=list)
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0
    
    @property
    def net_savings(self) -> float:
        return self.llm_costs_saved - self.embedding_costs
    
    def record_hit(self, cached_response: str, avg_llm_latency_ms: float = 1000):
        self.hits += 1
        self.tokens_saved += len(cached_response.split()) * 1.3
        self.latency_saved_ms += avg_llm_latency_ms
        self.llm_costs_saved += len(cached_response.split()) * 0.00003
    
    def record_miss(self, embedding_cost: float = 0.00002):
        self.misses += 1
        self.embedding_costs += embedding_cost
    
    def summary(self) -> dict:
        return {
            "hit_rate": f"{self.hit_rate:.1%}",
            "total_requests": self.hits + self.misses,
            "tokens_saved": int(self.tokens_saved),
            "latency_saved": f"{self.latency_saved_ms/1000:.1f}s",
            "net_savings": f"${self.net_savings:.4f}",
            "embedding_overhead": f"${self.embedding_costs:.4f}"
        }

# Usage
metrics = CacheMetrics()

# On cache hit
metrics.record_hit(cached_response)

# On cache miss
metrics.record_miss()

print(metrics.summary())
```

---

## Best Practices

### 1. Tune Similarity Threshold

```python
# Too low (0.7) - false positives, wrong answers returned
# Too high (0.99) - few cache hits

# Recommended starting points:
THRESHOLDS = {
    "factual_qa": 0.92,      # High precision needed
    "general_chat": 0.85,    # More flexibility OK
    "code_generation": 0.95, # Very precise matching
    "summarization": 0.88    # Moderate precision
}
```

### 2. Cache Warming

```python
def warm_cache(cache: SemanticCache, common_queries: list, model: str):
    """Pre-populate cache with common queries"""
    client = OpenAI()
    
    for query in common_queries:
        # Check if already cached
        if cache.get(query, model):
            continue
        
        # Generate and cache
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}]
        )
        
        cache.set(
            query=query,
            response=response.choices[0].message.content,
            model=model,
            ttl=86400 * 7  # Longer TTL for warmed cache
        )

# Common FAQ queries
common_queries = [
    "How do I reset my password?",
    "What are your business hours?",
    "How do I contact support?",
    # ... more common queries
]

warm_cache(cache, common_queries, "gpt-4o-mini")
```

### 3. Selective Caching

```python
def should_cache(query: str, response: str) -> bool:
    """Determine if response should be cached"""
    
    # Don't cache very short responses
    if len(response) < 50:
        return False
    
    # Don't cache personalized responses
    personal_indicators = ["your account", "your order", "your name"]
    if any(ind in response.lower() for ind in personal_indicators):
        return False
    
    # Don't cache time-sensitive content
    time_words = ["today", "now", "currently", "latest"]
    if any(word in query.lower() for word in time_words):
        return False
    
    return True
```

---

## Key Takeaways

1. **Semantic matching** enables 80-95% hit rates vs 5-10% with exact matching
2. **Multi-level caching** optimizes for both speed and coverage
3. **Context awareness** prevents serving wrong cached answers
4. **Proper invalidation** ensures freshness without losing benefits
5. **Measure ROI** - embedding costs vs LLM cost savings

---

## Next Steps

- [Prompt Compression](/learn/advanced-topics/optimization/prompt-compression) - Reduce tokens before LLM calls
- [Model Routing](/learn/advanced-topics/optimization/model-routing) - Route to cheap/expensive models
