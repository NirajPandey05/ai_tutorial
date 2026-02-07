# Vector Database Options

A comprehensive comparison of popular vector databases to help you choose the right one for your project.

## Overview

```yaml
vector_db_landscape:
  cloud_managed:
    - "Pinecone - Purpose-built, fully managed"
    - "Weaviate Cloud - Open source with cloud option"
    - "Qdrant Cloud - Open source with cloud option"
    - "Zilliz (Milvus) - Enterprise Milvus"
  
  open_source:
    - "ChromaDB - Simple, embedded"
    - "Qdrant - Rust-based, feature-rich"
    - "Weaviate - GraphQL, multimodal"
    - "Milvus - Large-scale, distributed"
    - "LanceDB - Embedded, serverless"
  
  add_on:
    - "pgvector - PostgreSQL extension"
    - "Elasticsearch - Vector search added"
    - "Redis - Vector search module"
```

## Quick Comparison

```yaml
comparison_matrix:
  chromadb:
    type: "Embedded/Client-Server"
    language: "Python"
    best_for: "Prototyping, small projects"
    max_scale: "~1M vectors"
    pricing: "Free (open source)"
  
  pinecone:
    type: "Fully managed cloud"
    language: "Any (REST API)"
    best_for: "Production, zero ops"
    max_scale: "Billions"
    pricing: "Free tier, then pay-per-use"
  
  weaviate:
    type: "Self-hosted or cloud"
    language: "Any (REST/GraphQL)"
    best_for: "Multimodal, complex queries"
    max_scale: "Billions"
    pricing: "Open source or cloud pricing"
  
  qdrant:
    type: "Self-hosted or cloud"
    language: "Any (REST/gRPC)"
    best_for: "Performance-critical apps"
    max_scale: "Billions"
    pricing: "Open source or cloud pricing"
  
  milvus:
    type: "Self-hosted or Zilliz cloud"
    language: "Any (SDK)"
    best_for: "Large-scale enterprise"
    max_scale: "Billions+"
    pricing: "Open source or Zilliz pricing"
  
  pgvector:
    type: "PostgreSQL extension"
    language: "SQL"
    best_for: "Adding vectors to existing Postgres"
    max_scale: "~10M vectors"
    pricing: "Free (open source)"
```

## ChromaDB

Simple, lightweight, perfect for getting started.

```python
import chromadb
from chromadb.utils import embedding_functions

# Create client (embedded mode)
client = chromadb.Client()

# Or persistent storage
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection with OpenAI embeddings
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-key",
    model_name="text-embedding-3-small"
)

collection = client.create_collection(
    name="documents",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"}
)

# Add documents (embeddings generated automatically)
collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=[
        "Python is great for AI",
        "JavaScript powers the web",
        "Machine learning is fascinating"
    ],
    metadatas=[
        {"category": "programming"},
        {"category": "programming"},
        {"category": "ai"}
    ]
)

# Query
results = collection.query(
    query_texts=["What programming languages are popular?"],
    n_results=2,
    where={"category": "programming"}
)

print(results["documents"])
```

### ChromaDB Pros/Cons

```yaml
chromadb:
  pros:
    - "Zero configuration setup"
    - "Runs in-process (embedded)"
    - "Built-in embedding functions"
    - "Great for development"
  
  cons:
    - "Limited scalability"
    - "Basic feature set"
    - "Single-machine only"
  
  best_for:
    - "Learning and prototyping"
    - "Small applications"
    - "Jupyter notebooks"
```

## Pinecone

Fully managed, production-ready vector database.

```python
from pinecone import Pinecone, ServerlessSpec

# Initialize
pc = Pinecone(api_key="your-api-key")

# Create serverless index
pc.create_index(
    name="my-index",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Connect to index
index = pc.Index("my-index")

# Upsert vectors
index.upsert(vectors=[
    {
        "id": "doc1",
        "values": [0.1, 0.2, ...],  # 1536 dimensions
        "metadata": {"category": "tech", "source": "blog"}
    },
    {
        "id": "doc2",
        "values": [0.3, 0.4, ...],
        "metadata": {"category": "news", "source": "article"}
    }
])

# Query with metadata filtering
results = index.query(
    vector=[0.15, 0.25, ...],
    top_k=10,
    filter={"category": {"$eq": "tech"}},
    include_metadata=True
)

for match in results["matches"]:
    print(f"ID: {match['id']}, Score: {match['score']}")
```

### Pinecone Pros/Cons

```yaml
pinecone:
  pros:
    - "Zero infrastructure management"
    - "Auto-scaling"
    - "High availability"
    - "Great documentation"
    - "Serverless option"
  
  cons:
    - "Vendor lock-in"
    - "Cost at scale"
    - "Data leaves your infrastructure"
  
  best_for:
    - "Production applications"
    - "Teams without DevOps"
    - "Rapid deployment"
```

## Qdrant

High-performance, feature-rich vector database.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)

# Connect (local or cloud)
client = QdrantClient(host="localhost", port=6333)
# Or: client = QdrantClient(url="https://xxx.qdrant.tech", api_key="...")

# Create collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE
    )
)

# Upsert points
client.upsert(
    collection_name="documents",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],
            payload={"category": "tech", "author": "Alice"}
        ),
        PointStruct(
            id=2,
            vector=[0.3, 0.4, ...],
            payload={"category": "news", "author": "Bob"}
        )
    ]
)

# Search with filtering
results = client.search(
    collection_name="documents",
    query_vector=[0.15, 0.25, ...],
    query_filter=Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value="tech")
            )
        ]
    ),
    limit=10
)

for result in results:
    print(f"ID: {result.id}, Score: {result.score}")
```

### Qdrant Pros/Cons

```yaml
qdrant:
  pros:
    - "Excellent performance (Rust)"
    - "Rich filtering capabilities"
    - "On-disk and in-memory modes"
    - "Multi-vector support"
    - "Good documentation"
  
  cons:
    - "Smaller community than others"
    - "Self-hosting complexity"
  
  best_for:
    - "Performance-critical applications"
    - "Complex filtering needs"
    - "Hybrid on-prem/cloud"
```

## Weaviate

Feature-rich with GraphQL and multimodal support.

```python
import weaviate
from weaviate.classes.query import MetadataQuery

# Connect
client = weaviate.connect_to_local()
# Or: client = weaviate.connect_to_wcs(cluster_url="...", auth_credentials=...)

# Create collection (schema)
client.collections.create(
    name="Document",
    vectorizer_config=weaviate.Configure.Vectorizer.text2vec_openai(),
    properties=[
        weaviate.Property(name="content", data_type=weaviate.DataType.TEXT),
        weaviate.Property(name="category", data_type=weaviate.DataType.TEXT),
    ]
)

# Get collection
documents = client.collections.get("Document")

# Add objects (auto-vectorized)
documents.data.insert_many([
    {"content": "Python is great for AI", "category": "programming"},
    {"content": "The stock market rose today", "category": "finance"},
])

# Semantic search
response = documents.query.near_text(
    query="artificial intelligence programming",
    limit=5,
    filters=weaviate.Filter.by_property("category").equal("programming"),
    return_metadata=MetadataQuery(distance=True)
)

for obj in response.objects:
    print(f"Content: {obj.properties['content']}")
    print(f"Distance: {obj.metadata.distance}")
```

### Weaviate Pros/Cons

```yaml
weaviate:
  pros:
    - "GraphQL interface"
    - "Built-in vectorization"
    - "Multimodal support"
    - "Generative search"
    - "Active community"
  
  cons:
    - "Higher resource usage"
    - "Complex configuration"
    - "Steeper learning curve"
  
  best_for:
    - "Multimodal applications"
    - "Complex query requirements"
    - "GraphQL-native teams"
```

## Milvus

Enterprise-grade, distributed vector database.

```python
from pymilvus import (
    connections, Collection, FieldSchema, 
    CollectionSchema, DataType, utility
)

# Connect
connections.connect(host="localhost", port="19530")

# Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=256),
]
schema = CollectionSchema(fields=fields)

# Create collection
collection = Collection(name="documents", schema=schema)

# Create index
collection.create_index(
    field_name="embedding",
    index_params={
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 256}
    }
)

# Insert data
collection.insert([
    [[0.1, 0.2, ...]],  # embeddings
    ["Python is great for AI"],  # content
    ["programming"]  # category
])

# Load for search
collection.load()

# Search
results = collection.search(
    data=[[0.15, 0.25, ...]],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 100}},
    limit=10,
    expr='category == "programming"',
    output_fields=["content", "category"]
)
```

### Milvus Pros/Cons

```yaml
milvus:
  pros:
    - "Massive scale (billions)"
    - "Multiple index types"
    - "GPU acceleration"
    - "Distributed architecture"
  
  cons:
    - "Complex deployment"
    - "Higher resource requirements"
    - "Steeper learning curve"
  
  best_for:
    - "Very large datasets"
    - "Enterprise deployments"
    - "GPU-accelerated search"
```

## pgvector

Add vector search to PostgreSQL.

```python
import psycopg2
from pgvector.psycopg2 import register_vector

# Connect
conn = psycopg2.connect("postgresql://localhost/mydb")
register_vector(conn)

# Create extension and table
with conn.cursor() as cur:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT,
            category TEXT,
            embedding vector(1536)
        )
    """)
    
    # Create index
    cur.execute("""
        CREATE INDEX ON documents 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 200)
    """)

# Insert
with conn.cursor() as cur:
    cur.execute(
        "INSERT INTO documents (content, category, embedding) VALUES (%s, %s, %s)",
        ("Python is great for AI", "programming", [0.1, 0.2, ...])
    )

# Search
with conn.cursor() as cur:
    cur.execute("""
        SELECT content, category, 1 - (embedding <=> %s) as similarity
        FROM documents
        WHERE category = %s
        ORDER BY embedding <=> %s
        LIMIT 10
    """, ([0.15, 0.25, ...], "programming", [0.15, 0.25, ...]))
    
    for row in cur.fetchall():
        print(f"Content: {row[0]}, Similarity: {row[2]:.4f}")
```

### pgvector Pros/Cons

```yaml
pgvector:
  pros:
    - "Use existing PostgreSQL"
    - "SQL interface"
    - "ACID transactions"
    - "Join with regular tables"
  
  cons:
    - "Limited scalability"
    - "Fewer vector-specific features"
    - "Performance below dedicated solutions"
  
  best_for:
    - "Adding vectors to existing apps"
    - "Small to medium datasets"
    - "SQL-first teams"
```

## Decision Guide

```yaml
decision_matrix:
  just_learning:
    choice: "ChromaDB"
    reason: "Simplest setup, great for notebooks"
  
  small_production:
    choice: "Qdrant or ChromaDB"
    reason: "Easy to deploy, good performance"
  
  enterprise_managed:
    choice: "Pinecone"
    reason: "Zero ops, SLAs, support"
  
  self_hosted_production:
    choice: "Qdrant or Weaviate"
    reason: "Feature-rich, good community"
  
  massive_scale:
    choice: "Milvus or Pinecone"
    reason: "Built for billions of vectors"
  
  existing_postgresql:
    choice: "pgvector"
    reason: "Minimal architecture change"
  
  multimodal_needs:
    choice: "Weaviate"
    reason: "Native multimodal support"
```

## Cost Comparison (Approximate)

```yaml
cost_examples:
  1M_vectors_1536_dims:
    chromadb: "$0 (self-hosted)"
    pinecone_serverless: "~$10-50/month"
    qdrant_cloud: "~$25/month"
    weaviate_cloud: "~$25/month"
    pgvector: "$0 (on existing Postgres)"
  
  10M_vectors:
    pinecone: "~$100-500/month"
    qdrant_cloud: "~$100-200/month"
    milvus_self_hosted: "Infrastructure costs only"
  
  notes:
    - "Costs vary by usage patterns"
    - "Self-hosted = your infrastructure costs"
    - "Check current pricing, these are estimates"
```

## Summary

```yaml
recommendations:
  start_here: "ChromaDB for learning"
  production_simple: "Pinecone for managed, Qdrant for self-hosted"
  production_complex: "Weaviate for features, Milvus for scale"
  existing_stack: "pgvector for PostgreSQL apps"
```

## Next Steps

1. **ChromaDB Lab** - Hands-on with local vector DB
2. **Semantic Search Lab** - Build a search engine
3. **RAG Module** - Use vectors for retrieval
