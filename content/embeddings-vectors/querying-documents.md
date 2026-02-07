# Querying Documents in Vector Databases

Master the art of retrieving relevant documents using semantic similarity, from basic queries to advanced patterns.

## Query Fundamentals

```yaml
query_overview:
  what: "Find documents most similar to a query"
  how: "Compare query embedding to stored embeddings"
  
  components:
    - "Query text or embedding"
    - "Number of results (top_k)"
    - "Optional metadata filters"
    - "Include options (what to return)"
```

## Basic Queries

### Query by Text

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./db")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-key",
    model_name="text-embedding-3-small"
)

collection = client.get_collection("documents", embedding_function=openai_ef)

# Simple text query
results = collection.query(
    query_texts=["What is machine learning?"],
    n_results=5
)

# Access results
for i, (doc_id, document, distance) in enumerate(zip(
    results["ids"][0],
    results["documents"][0],
    results["distances"][0]
)):
    print(f"{i+1}. [{distance:.4f}] {doc_id}: {document[:100]}...")
```

### Query by Embedding

```python
from openai import OpenAI

openai_client = OpenAI()

def get_embedding(text: str) -> list[float]:
    """Get embedding for query."""
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Query with pre-computed embedding
query_embedding = get_embedding("What is machine learning?")

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)
```

### Multiple Queries at Once

```python
# Batch multiple queries
results = collection.query(
    query_texts=[
        "What is machine learning?",
        "How does Python work?",
        "Best practices for web development"
    ],
    n_results=3  # Top 3 for each query
)

# Results structure: results["documents"][query_index][result_index]
for q_idx, query in enumerate(["ML", "Python", "Web"]):
    print(f"\n{query} results:")
    for r_idx, doc in enumerate(results["documents"][q_idx]):
        print(f"  {r_idx+1}. {doc[:50]}...")
```

## Query Options

### Include Parameters

```python
# Control what's returned
results = collection.query(
    query_texts=["machine learning"],
    n_results=5,
    include=[
        "documents",    # Original text
        "metadatas",    # Metadata
        "distances",    # Similarity scores
        "embeddings"    # Vector embeddings (large!)
    ]
)

# Access each type
print("Documents:", results["documents"])
print("Metadatas:", results["metadatas"])
print("Distances:", results["distances"])
print("Embeddings shape:", len(results["embeddings"][0][0]) if results["embeddings"] else "N/A")

# Minimal query (just IDs)
results = collection.query(
    query_texts=["machine learning"],
    n_results=5,
    include=[]  # Only returns IDs
)
```

### Distance Interpretation

```python
# Distance depends on metric:
# - cosine: 0 (identical) to 2 (opposite)
# - l2: 0 (identical) to infinity
# - ip (inner product): larger is more similar

# Convert cosine distance to similarity
def distance_to_similarity(distance: float, metric: str = "cosine") -> float:
    """Convert distance to 0-1 similarity score."""
    if metric == "cosine":
        return 1 - (distance / 2)  # Range 0-2 → 0-1
    elif metric == "l2":
        return 1 / (1 + distance)  # Decay function
    elif metric == "ip":
        return distance  # Already similarity-like
    return distance

# Usage
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    similarity = distance_to_similarity(dist)
    print(f"[{similarity:.4f}] {doc[:50]}...")
```

## Advanced Query Patterns

### Threshold-Based Retrieval

```python
def query_with_threshold(
    collection,
    query: str,
    max_distance: float = 0.5,
    max_results: int = 100
) -> list[dict]:
    """Only return results below distance threshold."""
    
    results = collection.query(
        query_texts=[query],
        n_results=max_results,
        include=["documents", "metadatas", "distances"]
    )
    
    filtered = []
    for i, distance in enumerate(results["distances"][0]):
        if distance <= max_distance:
            filtered.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": distance
            })
    
    return filtered

# Only get highly relevant results
relevant = query_with_threshold(collection, "machine learning", max_distance=0.3)
print(f"Found {len(relevant)} highly relevant documents")
```

### Reranking Results

```python
def rerank_results(
    query: str,
    results: list[dict],
    reranker_model: str = "cross-encoder"
) -> list[dict]:
    """Rerank results using a cross-encoder."""
    
    # Using sentence-transformers cross-encoder
    from sentence_transformers import CrossEncoder
    
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Score each result
    pairs = [(query, r["document"]) for r in results]
    scores = model.predict(pairs)
    
    # Add scores and sort
    for result, score in zip(results, scores):
        result["rerank_score"] = float(score)
    
    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)

# Initial retrieval
initial = query_with_threshold(collection, "machine learning", max_results=20)

# Rerank for better precision
reranked = rerank_results("machine learning", initial)
for r in reranked[:5]:
    print(f"[{r['rerank_score']:.4f}] {r['document'][:50]}...")
```

### Diversified Results

```python
import numpy as np

def maximal_marginal_relevance(
    query_embedding: list[float],
    doc_embeddings: list[list[float]],
    documents: list[str],
    k: int = 5,
    lambda_param: float = 0.5
) -> list[int]:
    """
    MMR: Balance relevance and diversity.
    
    lambda_param: 1.0 = pure relevance, 0.0 = pure diversity
    """
    query = np.array(query_embedding)
    docs = np.array(doc_embeddings)
    
    # Similarity to query
    query_sims = np.dot(docs, query) / (
        np.linalg.norm(docs, axis=1) * np.linalg.norm(query)
    )
    
    selected = []
    remaining = list(range(len(documents)))
    
    for _ in range(min(k, len(documents))):
        if not remaining:
            break
        
        mmr_scores = []
        for idx in remaining:
            # Relevance to query
            relevance = query_sims[idx]
            
            # Max similarity to already selected
            if selected:
                selected_sims = [
                    np.dot(docs[idx], docs[s]) / (
                        np.linalg.norm(docs[idx]) * np.linalg.norm(docs[s])
                    )
                    for s in selected
                ]
                diversity = max(selected_sims)
            else:
                diversity = 0
            
            # MMR score
            mmr = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append((idx, mmr))
        
        # Select highest MMR
        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    return selected

# Usage
results = collection.query(
    query_texts=["programming languages"],
    n_results=20,
    include=["documents", "embeddings"]
)

diverse_indices = maximal_marginal_relevance(
    query_embedding=get_embedding("programming languages"),
    doc_embeddings=results["embeddings"][0],
    documents=results["documents"][0],
    k=5,
    lambda_param=0.7
)

print("Diverse results:")
for idx in diverse_indices:
    print(f"  - {results['documents'][0][idx][:50]}...")
```

### Hybrid Search

```python
def hybrid_search(
    collection,
    query: str,
    keyword: str,
    semantic_weight: float = 0.7,
    n_results: int = 10
) -> list[dict]:
    """Combine semantic search with keyword matching."""
    
    # Semantic search
    semantic_results = collection.query(
        query_texts=[query],
        n_results=n_results * 2,  # Get more for fusion
        include=["documents", "metadatas", "distances"]
    )
    
    # Score semantic results
    semantic_scores = {}
    for i, doc_id in enumerate(semantic_results["ids"][0]):
        # Convert distance to score (0-1)
        semantic_scores[doc_id] = 1 - semantic_results["distances"][0][i]
    
    # Keyword search (simple containment check)
    all_docs = collection.get(include=["documents"])
    keyword_scores = {}
    keyword_lower = keyword.lower()
    
    for doc_id, doc in zip(all_docs["ids"], all_docs["documents"]):
        if keyword_lower in doc.lower():
            # Simple TF-based score
            count = doc.lower().count(keyword_lower)
            keyword_scores[doc_id] = min(count * 0.2, 1.0)
    
    # Combine scores
    all_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
    combined = []
    
    for doc_id in all_ids:
        sem_score = semantic_scores.get(doc_id, 0)
        kw_score = keyword_scores.get(doc_id, 0)
        
        final_score = (semantic_weight * sem_score + 
                      (1 - semantic_weight) * kw_score)
        
        combined.append({
            "id": doc_id,
            "score": final_score,
            "semantic_score": sem_score,
            "keyword_score": kw_score
        })
    
    # Sort and return top results
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:n_results]

# Usage
results = hybrid_search(
    collection,
    query="machine learning algorithms",
    keyword="neural",
    semantic_weight=0.7
)

for r in results:
    print(f"[{r['score']:.3f}] sem:{r['semantic_score']:.3f} kw:{r['keyword_score']:.3f}")
```

## Query Optimization

### Efficient Batch Queries

```python
def batch_query(
    collection,
    queries: list[str],
    n_results: int = 5,
    batch_size: int = 10
) -> list[list[dict]]:
    """Process many queries efficiently in batches."""
    
    all_results = []
    
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        
        results = collection.query(
            query_texts=batch,
            n_results=n_results,
            include=["documents", "distances"]
        )
        
        # Process each query's results
        for q_idx in range(len(batch)):
            query_results = []
            for r_idx in range(len(results["ids"][q_idx])):
                query_results.append({
                    "id": results["ids"][q_idx][r_idx],
                    "document": results["documents"][q_idx][r_idx],
                    "distance": results["distances"][q_idx][r_idx]
                })
            all_results.append(query_results)
    
    return all_results

# Process many queries
queries = [f"Query about topic {i}" for i in range(100)]
all_results = batch_query(collection, queries, n_results=3)
```

### Query Caching

```python
from functools import lru_cache
import hashlib

class CachedQuerier:
    """Query with result caching."""
    
    def __init__(self, collection, cache_size: int = 1000):
        self.collection = collection
        self.cache_size = cache_size
        self._cache = {}
    
    def _cache_key(self, query: str, n_results: int, filters: dict) -> str:
        """Generate cache key from query parameters."""
        data = f"{query}:{n_results}:{sorted(filters.items()) if filters else ''}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def query(
        self, 
        query: str, 
        n_results: int = 5,
        filters: dict = None,
        use_cache: bool = True
    ) -> dict:
        """Query with caching."""
        
        cache_key = self._cache_key(query, n_results, filters)
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filters,
            include=["documents", "metadatas", "distances"]
        )
        
        if use_cache:
            if len(self._cache) >= self.cache_size:
                # Remove oldest entry
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = results
        
        return results

# Usage
querier = CachedQuerier(collection)
results1 = querier.query("machine learning")  # Cache miss
results2 = querier.query("machine learning")  # Cache hit!
```

## Summary

```yaml
query_best_practices:
  basics:
    - "Start with text queries for simplicity"
    - "Use appropriate n_results for your use case"
    - "Only include what you need"
  
  optimization:
    - "Batch queries when possible"
    - "Cache frequent queries"
    - "Use filters to reduce search space"
  
  advanced:
    - "Rerank for better precision"
    - "Use MMR for diversity"
    - "Combine semantic + keyword for hybrid search"
  
  quality:
    - "Set distance thresholds"
    - "Monitor result relevance"
    - "A/B test query strategies"
```

## Next Steps

1. **Metadata Filtering** - Filter by attributes
2. **Semantic Search Lab** - Build search applications
3. **RAG Integration** - Use queries for retrieval
