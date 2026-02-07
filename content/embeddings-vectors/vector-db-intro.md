# Introduction to Vector Databases

Vector databases are specialized storage systems designed to efficiently store, index, and query high-dimensional vectors (embeddings).

## Why Vector Databases?

Traditional databases are built for exact matching and structured queries. Vector databases are built for similarity and semantic search.

```yaml
the_problem:
  scenario: "You have 1 million documents with embeddings"
  challenge: "Find the 10 most similar to a query in real-time"
  
  naive_approach:
    method: "Compare query to all 1M embeddings (linear scan)"
    time_complexity: "O(n × d)" # n=docs, d=dimensions
    for_1m_docs: "~1.5B floating-point operations per query"
    response_time: "5-10 seconds (too slow!)"
    result: "Not viable for production"
  
  vector_db_approach:
    method: "Approximate nearest neighbor (ANN) with indexing"
    indexing: "HNSW, IVF, LSH algorithms"
    time_complexity: "O(log n) to O(√n)"
    for_1m_docs: "~10K-50K operations per query"
    response_time: "10-100 milliseconds (fast!)"
    result: "Production-ready performance"
    tradeoff: "Trade exact accuracy for speed (≈99.5% quality)"
```

### Performance Comparison

| Operation | Traditional DB | Vector DB | Speed Gain |
|-----------|---|---|---|
| Exact match (1M docs) | 1-10 ms | N/A | - |
| Find top 10 similar | 5-10 sec | 50-200 ms | 25-200x faster |
| Range filter | 100-500 ms | 50-200 ms | 1-10x faster |
| Complex semantic query | Not possible | 50-300 ms | N/A |

## How Vector Databases Work

```
Traditional Database (Exact Match):
┌─────────────────────────────────────────────────────────┐
│  Query: SELECT * FROM docs WHERE title = 'Python'       │
│         ↓                                               │
│  [B-Tree Index on title]                                │
│    └─► Hash lookup → Direct match → O(log n)           │
│         Returns: Docs with exact title match            │
└─────────────────────────────────────────────────────────┘

Vector Database (Similarity Search):
┌─────────────────────────────────────────────────────────┐
│  Query: Find 10 nearest vectors to [0.1, 0.3, -0.2..]  │
│         ↓                                               │
│  [ANN Index (HNSW/IVF/FAISS)]                           │
│    └─► Graph traversal → k-nearest neighbors → O(log n) │
│         Returns: Top k most similar vectors + distances │
└─────────────────────────────────────────────────────────┘
```

### ANN Algorithms Explained

```yaml
HNSW (Hierarchical Navigable Small Worlds):
  pros: ["Fast (10-100ms)", "Best for most use cases"]
  cons: ["Higher memory", "Slower updates"]
  best_for: "Production systems with moderate-sized datasets"
  example_dbs: ["Weaviate", "Qdrant", "Milvus"]

IVF (Inverted File):
  pros: ["Memory efficient", "Good for large datasets"]
  cons: ["Slower than HNSW", "Requires careful tuning"]
  best_for: "Very large datasets (100M+), memory-constrained"
  example_dbs: ["Faiss (Meta)", "Vespa", "Elasticsearch"]

LSH (Locality-Sensitive Hashing):
  pros: ["Simple", "Fast hashing-based lookup"]
  cons: ["Lower accuracy", "High false positive rate"]
  best_for: "Quick prototypes, deduplication"
  example_dbs: ["Older implementations"]
```

## Core Concepts

### 1. Collections

```yaml
collection:
  definition: "A named group of vectors with metadata"
  analogous_to: "Table in SQL, Collection in MongoDB"
  
  components:
    - "Vectors (embeddings)"
    - "Documents/IDs"
    - "Metadata (filterable attributes)"
```

### 2. Documents and Embeddings

```python
# A typical document in a vector database
document = {
    "id": "doc_001",
    "text": "Python is a great programming language",
    "embedding": [0.1, 0.3, -0.2, ...],  # 1536 dimensions
    "metadata": {
        "category": "programming",
        "author": "Jane Doe",
        "date": "2024-01-15",
        "language": "en"
    }
}
```

### 3. Similarity Search

```python
# Pseudocode for vector search
results = vector_db.query(
    collection="documents",
    query_vector=[0.15, 0.28, -0.18, ...],  # Query embedding
    top_k=10,                                # Return top 10
    filter={"category": "programming"},      # Optional filter
    include_metadata=True
)
```

## Vector Database Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Vector Database                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Ingestion  │    │    Index     │    │    Query     │  │
│  │   Pipeline   │ → │   Builder    │ → │   Engine     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         ↓                  ↓                    ↓           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Embedding   │    │  ANN Index   │    │  Similarity  │  │
│  │  Generation  │    │  (HNSW/IVF)  │    │  Scoring     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Persistent Storage Layer                   │ │
│  │   [Vectors] [Metadata] [Indices] [Transaction Log]      │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### Approximate Nearest Neighbors (ANN)

```yaml
ann_tradeoff:
  accuracy_vs_speed:
    exact_search:
      recall: "100%"
      speed: "Slow (O(n))"
    
    approximate_search:
      recall: "95-99%"
      speed: "Fast (O(log n))"
  
  why_approximate:
    - "99% recall is often good enough"
    - "100x-1000x faster"
    - "Scales to billions of vectors"
```

### Metadata Filtering

```python
# Combine vector search with metadata filters
results = db.query(
    query_vector=embedding,
    filter={
        "category": {"$in": ["tech", "science"]},
        "date": {"$gte": "2024-01-01"},
        "rating": {"$gt": 4.0}
    },
    top_k=10
)
```

### Hybrid Search

```python
# Combine semantic search with keyword search
results = db.hybrid_search(
    query_vector=embedding,        # Semantic
    query_text="machine learning",  # Keyword
    alpha=0.7,                     # 70% semantic, 30% keyword
    top_k=10
)
```

## Common Operations

### CRUD Operations

```python
# Create: Add vectors
db.add(
    ids=["doc1", "doc2"],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    documents=["Text 1", "Text 2"],
    metadatas=[{"type": "a"}, {"type": "b"}]
)

# Read: Query vectors
results = db.query(
    query_embeddings=[[0.1, 0.2, ...]],
    n_results=10
)

# Update: Modify existing
db.update(
    ids=["doc1"],
    embeddings=[[0.15, 0.25, ...]],
    metadatas=[{"type": "updated"}]
)

# Delete: Remove vectors
db.delete(ids=["doc1", "doc2"])
```

### Batch Operations

```python
# Efficient batch insert
batch_size = 1000
for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    db.add(
        ids=[d["id"] for d in batch],
        embeddings=[d["embedding"] for d in batch],
        documents=[d["text"] for d in batch],
        metadatas=[d["metadata"] for d in batch]
    )
```

## Vector Database vs Traditional Database

```yaml
comparison:
  traditional_db:
    query_type: "Exact match, range, full-text"
    index_type: "B-tree, hash, inverted index"
    use_case: "Structured data, transactions"
    example_query: "WHERE name = 'John' AND age > 30"
  
  vector_db:
    query_type: "Similarity search, nearest neighbor"
    index_type: "HNSW, IVF, LSH, product quantization"
    use_case: "Semantic search, recommendations, RAG"
    example_query: "Find 10 most similar to vector X"
  
  when_to_use_both:
    - "RAG systems (vector for retrieval, SQL for structured data)"
    - "E-commerce (vector for recommendations, SQL for inventory)"
    - "Hybrid search (semantic + keyword)"
```

## Performance Characteristics

```yaml
scaling_considerations:
  memory:
    formula: "vectors × dimensions × 4 bytes (float32)"
    example: "1M vectors × 1536 dims × 4 = ~6 GB"
    tip: "Consider quantization for memory reduction"
  
  query_latency:
    small_dataset: "<10ms (< 100K vectors)"
    medium_dataset: "10-50ms (100K-10M vectors)"
    large_dataset: "50-200ms (10M-1B vectors)"
  
  indexing_time:
    hnsw: "Slower build, faster queries"
    ivf: "Faster build, slightly slower queries"
```

## Common Patterns

### Pattern 1: Document Search

```python
# Index documents for semantic search
def index_document(doc_id: str, content: str, metadata: dict):
    embedding = embed(content)
    db.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[content],
        metadatas=[metadata]
    )

def search_documents(query: str, filters: dict = None):
    query_embedding = embed(query)
    return db.query(
        query_embeddings=[query_embedding],
        where=filters,
        n_results=10
    )
```

### Pattern 2: RAG Retrieval

```python
# Retrieve context for RAG
def get_context(question: str, top_k: int = 5) -> str:
    results = db.query(
        query_embeddings=[embed(question)],
        n_results=top_k
    )
    
    context = "\n\n".join(results["documents"][0])
    return context
```

### Pattern 3: Recommendation System

```python
# Find similar items
def get_recommendations(item_id: str, n: int = 10):
    # Get item's embedding
    item = db.get(ids=[item_id])
    item_embedding = item["embeddings"][0]
    
    # Find similar items (excluding itself)
    results = db.query(
        query_embeddings=[item_embedding],
        n_results=n + 1
    )
    
    # Remove the query item itself
    return [r for r in results["ids"][0] if r != item_id][:n]
```

## Summary

```yaml
key_takeaways:
  what: "Specialized databases for vector similarity search"
  
  why:
    - "Efficient ANN search at scale"
    - "Combines semantic + metadata filtering"
    - "Foundation for RAG, search, recommendations"
  
  how:
    - "Store embeddings with metadata"
    - "Build ANN indices (HNSW, IVF)"
    - "Query by similarity"
  
  when_to_use:
    - "Semantic search"
    - "RAG applications"
    - "Recommendation systems"
    - "Image/audio similarity"
```

## Next Steps

1. **Similarity Algorithms** - Deep dive into cosine, euclidean, dot product
2. **Indexing Strategies** - HNSW, IVF, and when to use each
3. **Vector DB Options** - Compare ChromaDB, Pinecone, Weaviate, Qdrant
