# Embedding Models Comparison

Choose the right embedding model for your use case by understanding the tradeoffs between quality, cost, and performance.

## Overview of Embedding Models

```yaml
model_landscape:
  cloud_providers:
    - "OpenAI (text-embedding-3-small/large)"
    - "Cohere (embed-english-v3.0)"
    - "Google (text-embedding-004)"
    - "Voyage AI (voyage-3)"
    - "Anthropic (via Voyage)"
  
  open_source:
    - "sentence-transformers"
    - "BGE (BAAI)"
    - "E5 (Microsoft)"
    - "GTE (Alibaba)"
    - "Nomic Embed"
```

## Provider Comparison

### OpenAI Embeddings

```python
from openai import OpenAI

client = OpenAI()

def openai_embed(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Generate embeddings using OpenAI."""
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    return [item.embedding for item in response.data]

# Models available:
# - text-embedding-3-small: 1536 dims, $0.02/1M tokens
# - text-embedding-3-large: 3072 dims, $0.13/1M tokens
# - text-embedding-ada-002: 1536 dims (legacy)

embeddings = openai_embed(["Hello world", "How are you?"])
print(f"Dimensions: {len(embeddings[0])}")
```

### Cohere Embeddings

```python
import cohere

co = cohere.Client()

def cohere_embed(
    texts: list[str],
    input_type: str = "search_document"
) -> list[list[float]]:
    """
    Generate embeddings using Cohere.
    
    input_type options:
    - "search_document": For documents to be searched
    - "search_query": For search queries
    - "classification": For classification tasks
    - "clustering": For clustering tasks
    """
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type=input_type
    )
    return response.embeddings

# Cohere recommends different input_type for queries vs documents
doc_embeddings = cohere_embed(["Document text here"], input_type="search_document")
query_embedding = cohere_embed(["search query"], input_type="search_query")
```

### Google Embeddings

```python
import google.generativeai as genai

genai.configure(api_key="your-api-key")

def google_embed(texts: list[str]) -> list[list[float]]:
    """Generate embeddings using Google."""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=texts,
        task_type="retrieval_document"
    )
    return result['embedding']

# Task types:
# - retrieval_document: For documents
# - retrieval_query: For queries
# - semantic_similarity: For similarity comparisons
# - classification: For classification
# - clustering: For clustering
```

### Voyage AI Embeddings

```python
import voyageai

vo = voyageai.Client()

def voyage_embed(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Generate embeddings using Voyage AI."""
    result = vo.embed(
        texts=texts,
        model="voyage-3",
        input_type=input_type  # "document" or "query"
    )
    return result.embeddings

# Voyage models:
# - voyage-3: Latest, best quality
# - voyage-3-lite: Faster, cheaper
# - voyage-code-3: Optimized for code
# - voyage-finance-2: Optimized for finance
# - voyage-law-2: Optimized for legal
```

## Open Source Models

### Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

# Load model (downloads automatically)
model = SentenceTransformer('all-MiniLM-L6-v2')

def local_embed(texts: list[str]) -> list[list[float]]:
    """Generate embeddings locally."""
    embeddings = model.encode(texts)
    return embeddings.tolist()

# Popular models:
# - all-MiniLM-L6-v2: Fast, 384 dims
# - all-mpnet-base-v2: Better quality, 768 dims
# - multi-qa-mpnet-base-dot-v1: Optimized for QA
```

### BGE Models (BAAI)

```python
from sentence_transformers import SentenceTransformer

# BGE models from Beijing Academy of AI
model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# BGE recommends adding instruction prefix for queries
def bge_embed(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """Generate BGE embeddings with proper prefixes."""
    if is_query:
        texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
    return model.encode(texts).tolist()

# BGE models:
# - bge-small-en-v1.5: 384 dims, fast
# - bge-base-en-v1.5: 768 dims, balanced
# - bge-large-en-v1.5: 1024 dims, best quality
# - bge-m3: Multilingual, multiple granularities
```

### E5 Models (Microsoft)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/e5-small-v2')

def e5_embed(texts: list[str], prefix: str = "passage: ") -> list[list[float]]:
    """
    Generate E5 embeddings.
    E5 requires prefixes:
    - "query: " for search queries
    - "passage: " for documents
    """
    texts = [f"{prefix}{t}" for t in texts]
    return model.encode(texts).tolist()

# Usage
doc_emb = e5_embed(["Document text"], prefix="passage: ")
query_emb = e5_embed(["search query"], prefix="query: ")
```

## Comparison Table

```yaml
model_comparison:
  openai_3_small:
    dimensions: 1536
    max_tokens: 8191
    cost_per_1m: "$0.02"
    quality: "Good"
    speed: "Fast"
    multilingual: "Yes"
  
  openai_3_large:
    dimensions: 3072
    max_tokens: 8191
    cost_per_1m: "$0.13"
    quality: "Excellent"
    speed: "Fast"
    multilingual: "Yes"
  
  cohere_v3:
    dimensions: 1024
    max_tokens: 512
    cost_per_1m: "$0.10"
    quality: "Excellent"
    speed: "Fast"
    multilingual: "Yes (100+ languages)"
  
  voyage_3:
    dimensions: 1024
    max_tokens: 32000
    cost_per_1m: "$0.06"
    quality: "Excellent"
    speed: "Fast"
    multilingual: "Yes"
  
  bge_large:
    dimensions: 1024
    max_tokens: 512
    cost_per_1m: "Free (self-hosted)"
    quality: "Very Good"
    speed: "Medium"
    multilingual: "English only (use bge-m3 for multi)"
  
  all_minilm:
    dimensions: 384
    max_tokens: 256
    cost_per_1m: "Free (self-hosted)"
    quality: "Good"
    speed: "Very Fast"
    multilingual: "English only"
```

## Unified Embedding Interface

```python
from abc import ABC, abstractmethod
from typing import Literal

class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        pass
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        pass

class OpenAIEmbeddings(EmbeddingProvider):
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self._dimensions = 1536 if "small" in model else 3072
    
    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]
    
    @property
    def dimensions(self) -> int:
        return self._dimensions

class LocalEmbeddings(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._dimensions = self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        return self.model.encode(texts).tolist()
    
    @property
    def dimensions(self) -> int:
        return self._dimensions

# Factory function
def get_embeddings(provider: str = "openai", **kwargs) -> EmbeddingProvider:
    providers = {
        "openai": OpenAIEmbeddings,
        "local": LocalEmbeddings,
    }
    return providers[provider](**kwargs)

# Usage
embedder = get_embeddings("openai")
vectors = embedder.embed(["Hello world"])
print(f"Dimensions: {embedder.dimensions}")
```

## Benchmarks

### MTEB Leaderboard (Top Models)

```yaml
mteb_benchmarks:  # As of 2024
  retrieval_tasks:
    - model: "voyage-3"
      score: 67.8
    - model: "text-embedding-3-large"
      score: 66.3
    - model: "embed-english-v3.0"
      score: 65.9
    - model: "bge-large-en-v1.5"
      score: 64.2
  
  semantic_similarity:
    - model: "text-embedding-3-large"
      score: 85.2
    - model: "voyage-3"
      score: 84.9
    - model: "all-mpnet-base-v2"
      score: 83.4
```

## Choosing the Right Model

```yaml
decision_guide:
  cost_sensitive:
    recommendation: "all-MiniLM-L6-v2 or bge-small"
    reason: "Free, fast, good enough for many tasks"
  
  quality_focused:
    recommendation: "voyage-3 or text-embedding-3-large"
    reason: "Best retrieval quality"
  
  code_search:
    recommendation: "voyage-code-3"
    reason: "Optimized for code understanding"
  
  multilingual:
    recommendation: "cohere-v3 or bge-m3"
    reason: "Strong multilingual support"
  
  privacy_required:
    recommendation: "Local models (bge, e5)"
    reason: "Data never leaves your infrastructure"
  
  long_documents:
    recommendation: "voyage-3 (32K context)"
    reason: "Handles full documents without chunking"
```

## Practical Tips

```yaml
best_practices:
  batching:
    - "Batch embeddings to reduce API calls"
    - "Most APIs support up to 96-2048 texts per call"
    - "Monitor token limits per batch"
  
  caching:
    - "Cache embeddings - they're deterministic"
    - "Use content hashing for cache keys"
    - "Consider Redis or SQLite for persistence"
  
  normalization:
    - "Most models return normalized vectors"
    - "Verify normalization if mixing models"
    - "Normalize if using dot product similarity"
  
  query_vs_document:
    - "Some models use different modes"
    - "Cohere: input_type='search_query' vs 'search_document'"
    - "E5/BGE: Add 'query:' or 'passage:' prefix"
```

## Cost Comparison Example

```python
def estimate_embedding_cost(
    num_documents: int,
    avg_tokens_per_doc: int,
    model: str
) -> dict:
    """Estimate embedding costs."""
    
    costs_per_1m = {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
        "embed-english-v3.0": 0.10,
        "voyage-3": 0.06,
        "local": 0.0,
    }
    
    total_tokens = num_documents * avg_tokens_per_doc
    cost = (total_tokens / 1_000_000) * costs_per_1m.get(model, 0.10)
    
    return {
        "model": model,
        "documents": num_documents,
        "total_tokens": total_tokens,
        "cost_usd": f"${cost:.4f}",
        "cost_per_1k_docs": f"${(cost / num_documents * 1000):.4f}"
    }

# Example: 100K documents, 500 tokens each
print(estimate_embedding_cost(100_000, 500, "text-embedding-3-small"))
# {'cost_usd': '$1.0000'}

print(estimate_embedding_cost(100_000, 500, "text-embedding-3-large"))
# {'cost_usd': '$6.5000'}

print(estimate_embedding_cost(100_000, 500, "local"))
# {'cost_usd': '$0.0000'}
```

## Summary

```yaml
key_takeaways:
  cloud_models:
    pros: "Easy to use, high quality, no infrastructure"
    cons: "Cost at scale, data leaves your system"
    best_for: "Production apps, highest quality needs"
  
  open_source:
    pros: "Free, private, customizable"
    cons: "Requires GPU for speed, slightly lower quality"
    best_for: "Cost-sensitive, privacy-required, self-hosted"
  
  hybrid_approach:
    suggestion: "Use local for development, cloud for production"
    or: "Use local for bulk processing, cloud for queries"
```

## Next Steps

1. **Vector Databases** - Store embeddings at scale
2. **Semantic Search** - Build search with embeddings
3. **RAG** - Combine embeddings with LLMs
