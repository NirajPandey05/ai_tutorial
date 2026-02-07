# Similarity Search Algorithms

Understanding how vector databases measure similarity is crucial for optimizing your search quality and performance.

## Distance Metrics Overview

```yaml
core_metrics:
  cosine_similarity:
    measures: "Angle between vectors"
    range: "-1 to 1"
    best_for: "Text embeddings, normalized vectors"
  
  euclidean_distance:
    measures: "Straight-line distance"
    range: "0 to ∞"
    best_for: "Spatial data, when magnitude matters"
  
  dot_product:
    measures: "Projection of one vector onto another"
    range: "-∞ to ∞"
    best_for: "Normalized vectors, maximum inner product search"
```

## Cosine Similarity

The most common metric for text embeddings.

```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Measures the angle between two vectors.
    
    Properties:
    - Ignores magnitude (length)
    - Perfect for comparing semantic meaning
    - 1.0 = identical direction
    - 0.0 = orthogonal (unrelated)
    - -1.0 = opposite direction
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Example
vec_a = np.array([1, 2, 3])
vec_b = np.array([2, 4, 6])  # Same direction, different magnitude
vec_c = np.array([3, 2, 1])  # Different direction

print(f"Same direction: {cosine_similarity(vec_a, vec_b):.4f}")  # 1.0
print(f"Different: {cosine_similarity(vec_a, vec_c):.4f}")       # 0.7143
```

### Why Cosine Works for Embeddings

```
Cosine similarity ignores vector length:

Vector A: ─────────────────────→  (long document)
Vector B: ────────→               (short document)
                 θ (small angle = similar meaning)

Both documents about "machine learning" will have similar 
directions regardless of length, making cosine ideal.
```

## Euclidean Distance (L2)

The straight-line distance between two points.

```python
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    L2 distance - straight line between points.
    
    Properties:
    - Considers both direction AND magnitude
    - 0.0 = identical vectors
    - No upper bound
    - Sensitive to outliers
    """
    return np.linalg.norm(a - b)

# Or equivalently:
# np.sqrt(np.sum((a - b) ** 2))

# Example
vec_a = np.array([0, 0])
vec_b = np.array([3, 4])

print(f"L2 distance: {euclidean_distance(vec_a, vec_b):.4f}")  # 5.0
```

### Euclidean vs Cosine for Normalized Vectors

```python
def show_relationship(a: np.ndarray, b: np.ndarray):
    """For normalized vectors, L2 and cosine are related."""
    # Normalize
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    
    cos_sim = cosine_similarity(a_norm, b_norm)
    l2_dist = euclidean_distance(a_norm, b_norm)
    
    # Mathematical relationship:
    # L2² = 2 * (1 - cosine_similarity)
    expected_l2 = np.sqrt(2 * (1 - cos_sim))
    
    print(f"Cosine similarity: {cos_sim:.4f}")
    print(f"L2 distance: {l2_dist:.4f}")
    print(f"Expected L2 from cosine: {expected_l2:.4f}")

show_relationship(np.array([1, 2, 3]), np.array([4, 5, 6]))
```

## Dot Product (Inner Product)

Fast similarity for normalized vectors.

```python
def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Inner product of two vectors.
    
    Properties:
    - For normalized vectors: equals cosine similarity
    - Very fast to compute
    - Affected by magnitude
    """
    return np.dot(a, b)

# For normalized vectors
a_norm = np.array([0.6, 0.8])  # Normalized (length = 1)
b_norm = np.array([0.8, 0.6])  # Normalized

print(f"Dot product: {dot_product(a_norm, b_norm):.4f}")           # 0.96
print(f"Cosine sim:  {cosine_similarity(a_norm, b_norm):.4f}")   # 0.96 (same!)
```

## Manhattan Distance (L1)

Sum of absolute differences.

```python
def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    L1 distance - sum of absolute differences.
    
    Properties:
    - Less sensitive to outliers than L2
    - Used in some specialized applications
    """
    return np.sum(np.abs(a - b))

a = np.array([0, 0])
b = np.array([3, 4])

print(f"Manhattan (L1): {manhattan_distance(a, b):.4f}")  # 7.0
print(f"Euclidean (L2): {euclidean_distance(a, b):.4f}")  # 5.0
```

## Choosing the Right Metric

```yaml
decision_guide:
  cosine_similarity:
    use_when:
      - "Text embeddings"
      - "Document similarity"
      - "When direction matters more than magnitude"
    common_in: "Most embedding models (OpenAI, Cohere, etc.)"
  
  euclidean_distance:
    use_when:
      - "Spatial data"
      - "Image features"
      - "When absolute position matters"
    common_in: "Image search, geographical queries"
  
  dot_product:
    use_when:
      - "Vectors are already normalized"
      - "Need maximum speed"
      - "Maximum inner product search (MIPS)"
    common_in: "Recommendation systems, large-scale search"
  
  manhattan_distance:
    use_when:
      - "High-dimensional sparse data"
      - "Robustness to outliers needed"
    common_in: "Some specialized NLP tasks"
```

## Performance Comparison

```python
import time
import numpy as np

def benchmark_metrics(n_vectors: int, n_dims: int, n_queries: int):
    """Benchmark different distance metrics."""
    
    # Generate random data
    vectors = np.random.randn(n_vectors, n_dims).astype(np.float32)
    queries = np.random.randn(n_queries, n_dims).astype(np.float32)
    
    # Normalize for fair comparison
    vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    queries_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    results = {}
    
    # Cosine similarity
    start = time.time()
    for q in queries_norm:
        sims = np.dot(vectors_norm, q)
    results['cosine'] = time.time() - start
    
    # Euclidean distance
    start = time.time()
    for q in queries:
        dists = np.linalg.norm(vectors - q, axis=1)
    results['euclidean'] = time.time() - start
    
    # Dot product
    start = time.time()
    for q in queries_norm:
        scores = np.dot(vectors_norm, q)
    results['dot_product'] = time.time() - start
    
    return results

# Run benchmark
results = benchmark_metrics(n_vectors=10000, n_dims=1536, n_queries=100)
print("Time for 100 queries against 10K vectors:")
for metric, time_taken in results.items():
    print(f"  {metric}: {time_taken*1000:.2f}ms")
```

## Practical Examples

### Example 1: Semantic Search with Cosine

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def semantic_search(query: str, documents: list[str], top_k: int = 5):
    """Search using cosine similarity."""
    
    # Get embeddings
    all_texts = [query] + documents
    response = client.embeddings.create(
        input=all_texts,
        model="text-embedding-3-small"
    )
    
    embeddings = [np.array(item.embedding) for item in response.data]
    query_emb = embeddings[0]
    doc_embs = embeddings[1:]
    
    # Calculate cosine similarities
    similarities = []
    for i, doc_emb in enumerate(doc_embs):
        sim = np.dot(query_emb, doc_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
        )
        similarities.append((documents[i], sim))
    
    # Sort and return top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Test
docs = [
    "Python is a programming language",
    "Machine learning uses neural networks",
    "The weather is sunny today",
    "Deep learning is a subset of ML"
]

results = semantic_search("What is artificial intelligence?", docs)
for doc, score in results:
    print(f"[{score:.4f}] {doc}")
```

### Example 2: Threshold-Based Filtering

```python
def filter_by_similarity(
    query_embedding: np.ndarray,
    document_embeddings: list[np.ndarray],
    threshold: float = 0.7,
    metric: str = "cosine"
) -> list[int]:
    """Return indices of documents above similarity threshold."""
    
    matching_indices = []
    
    for i, doc_emb in enumerate(document_embeddings):
        if metric == "cosine":
            score = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
            )
            if score >= threshold:
                matching_indices.append((i, score))
        
        elif metric == "euclidean":
            dist = np.linalg.norm(query_embedding - doc_emb)
            if dist <= threshold:  # Note: lower is better for distance
                matching_indices.append((i, dist))
    
    return matching_indices

# Use case: Only show highly relevant results
highly_relevant = filter_by_similarity(query_emb, doc_embs, threshold=0.8)
```

### Example 3: Combined Scoring

```python
def hybrid_score(
    semantic_sim: float,
    keyword_score: float,
    recency_score: float,
    weights: tuple = (0.5, 0.3, 0.2)
) -> float:
    """Combine multiple signals with weights."""
    return (
        weights[0] * semantic_sim +
        weights[1] * keyword_score +
        weights[2] * recency_score
    )

# Example: Rank documents by combined score
def rank_documents(
    query: str,
    documents: list[dict],  # [{text, keywords, date}, ...]
    query_embedding: np.ndarray,
    doc_embeddings: list[np.ndarray]
) -> list[dict]:
    """Rank with hybrid scoring."""
    
    scored = []
    for i, doc in enumerate(documents):
        # Semantic similarity
        sem_sim = cosine_similarity(query_embedding, doc_embeddings[i])
        
        # Keyword overlap
        query_words = set(query.lower().split())
        doc_words = set(doc['keywords'])
        keyword_sim = len(query_words & doc_words) / max(len(query_words), 1)
        
        # Recency (normalize to 0-1)
        days_old = (datetime.now() - doc['date']).days
        recency = max(0, 1 - days_old / 365)  # Decay over a year
        
        final_score = hybrid_score(sem_sim, keyword_sim, recency)
        scored.append({**doc, 'score': final_score})
    
    return sorted(scored, key=lambda x: x['score'], reverse=True)
```

## Vector Database Metric Support

```yaml
database_support:
  chromadb:
    default: "L2 (squared euclidean)"
    supported: ["l2", "ip (inner product)", "cosine"]
  
  pinecone:
    default: "cosine"
    supported: ["cosine", "euclidean", "dotproduct"]
  
  weaviate:
    default: "cosine"
    supported: ["cosine", "dot", "l2-squared", "manhattan", "hamming"]
  
  qdrant:
    default: "cosine"
    supported: ["cosine", "euclid", "dot"]
  
  milvus:
    default: "varies by index"
    supported: ["L2", "IP", "COSINE", "JACCARD", "HAMMING"]
```

## Summary

```yaml
key_takeaways:
  cosine:
    best_for: "Semantic similarity"
    note: "Direction-only, ignore magnitude"
  
  euclidean:
    best_for: "Absolute positioning"
    note: "Consider both direction and magnitude"
  
  dot_product:
    best_for: "Speed with normalized vectors"
    note: "Same as cosine when normalized"
  
  practical_advice:
    - "Start with cosine for text"
    - "Normalize vectors for consistent results"
    - "Use dot product for speed when possible"
    - "Benchmark on your specific data"
```

## Next Steps

1. **Indexing Strategies** - HNSW, IVF, and efficient search
2. **Vector DB Options** - Choose the right database
3. **Practical Labs** - Build search applications
