# Indexing Strategies for Vector Databases

Efficient indexing is what makes vector search fast at scale. Understanding these algorithms helps you optimize for your specific use case.

## Why Indexing Matters

```yaml
the_problem:
  brute_force:
    method: "Compare query to every vector"
    complexity: "O(n × d)"  # n=vectors, d=dimensions
    
  at_scale:
    1M_vectors: "~1 million comparisons per query"
    latency: "100ms - 1s per query"
    result: "Too slow for production"
  
  with_indexing:
    method: "Approximate nearest neighbor (ANN)"
    complexity: "O(log n) to O(√n)"
    
  at_scale:
    1M_vectors: "~1000 comparisons per query"
    latency: "1ms - 10ms per query"
    result: "Production ready!"
```

## Common Index Types

```
Index Types Overview:

┌─────────────────────────────────────────────────────────┐
│  FLAT (Brute Force)                                     │
│  ─────────────────                                      │
│  • Exact search, no approximation                       │
│  • Best for small datasets (< 10K)                      │
│  • 100% recall, slow at scale                           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  HNSW (Hierarchical Navigable Small World)              │
│  ──────────────────────────────────────────             │
│  • Graph-based navigation                               │
│  • Best balance of speed and accuracy                   │
│  • Higher memory usage                                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  IVF (Inverted File Index)                              │
│  ─────────────────────────                              │
│  • Cluster-based search                                 │
│  • Good for very large datasets                         │
│  • Requires training phase                              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Product Quantization (PQ)                              │
│  ─────────────────────────                              │
│  • Compresses vectors                                   │
│  • Reduces memory 4-64x                                 │
│  • Some accuracy loss                                   │
└─────────────────────────────────────────────────────────┘
```

## HNSW (Hierarchical Navigable Small World)

The most popular index for vector databases.

### How It Works

```
HNSW Graph Structure:

Layer 3 (sparse):    A ─────────────────────── Z
                      ↓                         ↓
Layer 2:             A ─── E ─────── M ─────── Z
                      ↓     ↓         ↓         ↓
Layer 1:             A ─ C ─ E ─ G ─ M ─ Q ─ T ─ Z
                      ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
Layer 0 (all nodes): A B C D E F G H M N Q R T U Z ...

Search: Start at top layer, navigate down, find neighbors
```

### HNSW Parameters

```python
# HNSW key parameters

hnsw_params = {
    "M": 16,                    # Number of connections per node
    "ef_construction": 200,     # Build-time search depth
    "ef_search": 50             # Query-time search depth
}

# M (edges per node):
#   Higher = better recall, more memory
#   Typical: 8-64, default: 16

# ef_construction:
#   Higher = better graph quality, slower build
#   Typical: 100-500

# ef_search:
#   Higher = better recall, slower search
#   Can be tuned at query time
```

### HNSW Tradeoffs

```yaml
hnsw_characteristics:
  strengths:
    - "Fast query time (1-10ms)"
    - "High recall (95-99%+)"
    - "No training required"
    - "Good for dynamic data (inserts/deletes)"
  
  weaknesses:
    - "High memory usage (vectors + graph)"
    - "Slow initial build"
    - "Memory: ~1.5-2x raw vector size"
  
  best_for:
    - "Production search systems"
    - "Real-time applications"
    - "Datasets up to ~100M vectors"
```

## IVF (Inverted File Index)

Cluster-based approximate search.

### How It Works

```
IVF Structure:

1. Training: Cluster all vectors into k centroids

   Centroid 1 ──→ [Vectors in cluster 1]
   Centroid 2 ──→ [Vectors in cluster 2]
   Centroid 3 ──→ [Vectors in cluster 3]
        ...
   Centroid k ──→ [Vectors in cluster k]

2. Search: 
   a. Find nearest nprobe centroids to query
   b. Search only within those clusters

Query ─→ [Find closest centroids] ─→ [Search within clusters]
```

### IVF Parameters

```python
# IVF key parameters

ivf_params = {
    "nlist": 1024,    # Number of clusters
    "nprobe": 10      # Clusters to search at query time
}

# nlist (number of clusters):
#   Rule of thumb: sqrt(n) to 4*sqrt(n) where n = num vectors
#   100K vectors → nlist = 316 to 1264
#   1M vectors → nlist = 1000 to 4000

# nprobe (clusters to search):
#   Higher = better recall, slower search
#   Typical: 1% to 10% of nlist
```

### IVF Variants

```yaml
ivf_variants:
  IVF_FLAT:
    description: "IVF + exact search within clusters"
    compression: "None"
    accuracy: "High"
  
  IVF_PQ:
    description: "IVF + product quantization"
    compression: "High (8-64x)"
    accuracy: "Medium"
  
  IVF_SQ8:
    description: "IVF + scalar quantization"
    compression: "4x"
    accuracy: "High"
```

## Product Quantization (PQ)

Compress vectors to reduce memory.

### How It Works

```
Original Vector (128 dims):
[v1, v2, v3, v4, ..., v127, v128]

Split into subvectors (8 subspaces):
[v1-v16] [v17-v32] [v33-v48] ... [v113-v128]
    ↓         ↓         ↓            ↓
 Codebook1  Codebook2  Codebook3   Codebook8
    ↓         ↓         ↓            ↓
  Code1     Code2     Code3       Code8

Compressed: [c1, c2, c3, c4, c5, c6, c7, c8]  (8 bytes!)

Original: 128 × 4 bytes = 512 bytes
Compressed: 8 bytes (64x reduction!)
```

### PQ Parameters

```python
pq_params = {
    "m": 8,           # Number of subspaces
    "nbits": 8        # Bits per subspace (256 centroids)
}

# Memory calculation:
# Original: dimensions × 4 bytes (float32)
# PQ: m × nbits/8 bytes

# Example: 1536 dims → 1536 × 4 = 6144 bytes
# With PQ (m=96, nbits=8): 96 bytes (64x reduction)
```

## Comparison Table

```yaml
index_comparison:
  FLAT:
    recall: "100%"
    query_speed: "Slow"
    build_speed: "Fast"
    memory: "1x"
    best_for: "< 10K vectors"
  
  HNSW:
    recall: "95-99%"
    query_speed: "Very Fast"
    build_speed: "Slow"
    memory: "1.5-2x"
    best_for: "< 100M vectors, real-time"
  
  IVF_FLAT:
    recall: "90-99%"
    query_speed: "Fast"
    build_speed: "Medium"
    memory: "1x + centroids"
    best_for: "> 1M vectors"
  
  IVF_PQ:
    recall: "70-95%"
    query_speed: "Fast"
    build_speed: "Medium"
    memory: "0.1-0.5x"
    best_for: "Billions of vectors, memory limited"
```

## Practical Index Selection

```python
def recommend_index(
    num_vectors: int,
    dimensions: int,
    memory_budget_gb: float,
    target_recall: float,
    latency_ms: float
) -> dict:
    """Recommend an index based on requirements."""
    
    vector_size_bytes = dimensions * 4
    total_size_gb = (num_vectors * vector_size_bytes) / 1e9
    
    recommendation = {
        "vectors": num_vectors,
        "raw_size_gb": total_size_gb
    }
    
    # Small dataset
    if num_vectors < 10_000:
        recommendation["index"] = "FLAT"
        recommendation["reason"] = "Small enough for brute force"
    
    # Memory constrained
    elif total_size_gb > memory_budget_gb * 0.5:
        recommendation["index"] = "IVF_PQ"
        recommendation["reason"] = "Need compression to fit in memory"
        recommendation["params"] = {
            "nlist": int(4 * (num_vectors ** 0.5)),
            "m": 96,
            "nbits": 8
        }
    
    # High recall requirement
    elif target_recall > 0.95:
        recommendation["index"] = "HNSW"
        recommendation["reason"] = "Best recall/speed tradeoff"
        recommendation["params"] = {
            "M": 16,
            "ef_construction": 200,
            "ef_search": 100
        }
    
    # Large scale, balanced
    else:
        recommendation["index"] = "IVF_FLAT"
        recommendation["reason"] = "Good balance for large datasets"
        recommendation["params"] = {
            "nlist": int(4 * (num_vectors ** 0.5)),
            "nprobe": max(10, int(0.05 * (num_vectors ** 0.5)))
        }
    
    return recommendation

# Example usage
print(recommend_index(
    num_vectors=1_000_000,
    dimensions=1536,
    memory_budget_gb=16,
    target_recall=0.95,
    latency_ms=10
))
```

## Index Configuration in Vector DBs

### ChromaDB

```python
import chromadb

# ChromaDB uses HNSW by default
client = chromadb.Client()

collection = client.create_collection(
    name="my_collection",
    metadata={
        "hnsw:space": "cosine",  # Distance metric
        "hnsw:construction_ef": 200,
        "hnsw:search_ef": 100,
        "hnsw:M": 16
    }
)
```

### Pinecone

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="your-key")

# Pinecone manages indexing automatically
pc.create_index(
    name="my-index",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)
```

### Qdrant

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff

client = QdrantClient(host="localhost", port=6333)

# Configure HNSW parameters
client.create_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE
    ),
    hnsw_config=HnswConfigDiff(
        m=16,
        ef_construct=200,
        full_scan_threshold=10000
    )
)
```

### Weaviate

```python
import weaviate

client = weaviate.Client("http://localhost:8080")

# Configure index in schema
schema = {
    "class": "Document",
    "vectorIndexType": "hnsw",
    "vectorIndexConfig": {
        "distance": "cosine",
        "ef": 100,
        "efConstruction": 128,
        "maxConnections": 64
    },
    "properties": [...]
}

client.schema.create_class(schema)
```

## Tuning for Your Use Case

```yaml
tuning_guidelines:
  increase_recall:
    hnsw:
      - "Increase ef_search (query time)"
      - "Increase M (build time, memory)"
      - "Increase ef_construction (build time)"
    ivf:
      - "Increase nprobe (query time)"
      - "Increase nlist with more training data"
  
  reduce_latency:
    hnsw:
      - "Decrease ef_search"
      - "Use smaller M"
    ivf:
      - "Decrease nprobe"
      - "Use PQ for faster distance computation"
  
  reduce_memory:
    options:
      - "Use PQ compression"
      - "Use scalar quantization"
      - "Reduce vector dimensions"
```

## Summary

```yaml
key_takeaways:
  flat:
    when: "< 10K vectors"
    tradeoff: "100% accurate, slow at scale"
  
  hnsw:
    when: "Most production use cases"
    tradeoff: "High recall, fast queries, more memory"
  
  ivf:
    when: "Very large datasets"
    tradeoff: "Needs training, tunable recall/speed"
  
  pq:
    when: "Memory constrained"
    tradeoff: "Big memory savings, some accuracy loss"
  
  practical_advice:
    - "Start with HNSW for most cases"
    - "Tune ef_search at query time for recall/speed"
    - "Use IVF+PQ for billions of vectors"
    - "Benchmark on your actual data"
```

## Next Steps

1. **Vector DB Options** - Compare ChromaDB, Pinecone, etc.
2. **ChromaDB Lab** - Hands-on with local vector DB
3. **Semantic Search** - Build search applications
