# Lab: Set Up ChromaDB

Build a complete vector database application using ChromaDB, from installation to production-ready code.

## Learning Objectives

By completing this lab, you will:
- Install and configure ChromaDB
- Create collections and add documents
- Perform semantic search with filtering
- Build a document retrieval system

## Prerequisites

```yaml
requirements:
  knowledge:
    - "Understanding of embeddings"
    - "Basic Python"
  
  packages:
    - "chromadb"
    - "openai (for embeddings)"
```

---

## Part 1: Installation and Setup

### Exercise 1.1: Install ChromaDB

```bash
# Install ChromaDB
pip install chromadb

# Optional: Install with server support
pip install chromadb[server]

# For OpenAI embeddings
pip install openai
```

### Exercise 1.2: Basic Client Setup

```python
import chromadb

# Option 1: In-memory (for testing)
client = chromadb.Client()

# Option 2: Persistent storage (recommended)
client = chromadb.PersistentClient(path="./chroma_db")

# Option 3: Client-server mode
# First start server: chroma run --path ./chroma_db
# client = chromadb.HttpClient(host="localhost", port=8000)

print(f"ChromaDB version: {chromadb.__version__}")
print(f"Client type: {type(client).__name__}")
```

### Exercise 1.3: Configure Embedding Functions

```python
from chromadb.utils import embedding_functions

# Option 1: OpenAI embeddings (recommended for production)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-openai-api-key",
    model_name="text-embedding-3-small"
)

# Option 2: Default (all-MiniLM-L6-v2, runs locally)
default_ef = embedding_functions.DefaultEmbeddingFunction()

# Option 3: Sentence Transformers (local, customizable)
sentence_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)

# Test embedding function
test_text = "Hello, world!"
embedding = openai_ef([test_text])[0]
print(f"Embedding dimensions: {len(embedding)}")
```

---

## Part 2: Working with Collections

### Exercise 2.1: Create and Manage Collections

```python
# Create a collection
collection = client.create_collection(
    name="my_documents",
    embedding_function=openai_ef,
    metadata={
        "hnsw:space": "cosine",  # Distance metric
        "description": "My document collection"
    }
)

print(f"Created collection: {collection.name}")
print(f"Collection count: {collection.count()}")

# List all collections
collections = client.list_collections()
print(f"All collections: {[c.name for c in collections]}")

# Get existing collection
collection = client.get_collection(
    name="my_documents",
    embedding_function=openai_ef
)

# Get or create (idempotent)
collection = client.get_or_create_collection(
    name="my_documents",
    embedding_function=openai_ef
)

# Delete collection
# client.delete_collection(name="my_documents")
```

### Exercise 2.2: Collection Configuration

```python
# Create collection with custom HNSW settings
collection = client.create_collection(
    name="optimized_docs",
    embedding_function=openai_ef,
    metadata={
        # Distance metric
        "hnsw:space": "cosine",  # Options: "l2", "ip", "cosine"
        
        # HNSW parameters
        "hnsw:construction_ef": 200,  # Build-time quality (default: 100)
        "hnsw:search_ef": 100,        # Query-time quality (default: 10)
        "hnsw:M": 16,                 # Connections per node (default: 16)
        
        # Custom metadata
        "created_by": "lab_exercise",
        "version": "1.0"
    }
)

# Check collection settings
print(f"Collection metadata: {collection.metadata}")
```

---

## Part 3: Adding Documents

### Exercise 3.1: Basic Document Addition

```python
# Add documents with auto-generated embeddings
collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=[
        "Python is a versatile programming language",
        "Machine learning models require large datasets",
        "The weather today is sunny and warm"
    ],
    metadatas=[
        {"category": "programming", "language": "python"},
        {"category": "ai", "topic": "ml"},
        {"category": "weather", "type": "report"}
    ]
)

print(f"Documents in collection: {collection.count()}")
```

### Exercise 3.2: Add Documents with Pre-computed Embeddings

```python
from openai import OpenAI

openai_client = OpenAI()

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings from OpenAI."""
    response = openai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]

# Documents to add
documents = [
    "Neural networks are inspired by biological brains",
    "Deep learning uses multiple layers of neurons",
    "Transformers revolutionized natural language processing"
]

# Pre-compute embeddings
embeddings = get_embeddings(documents)

# Add with embeddings
collection.add(
    ids=["nn1", "nn2", "nn3"],
    embeddings=embeddings,
    documents=documents,
    metadatas=[
        {"topic": "neural_networks", "difficulty": "intermediate"},
        {"topic": "deep_learning", "difficulty": "advanced"},
        {"topic": "nlp", "difficulty": "advanced"}
    ]
)
```

### Exercise 3.3: Batch Adding Large Datasets

```python
def batch_add_documents(
    collection,
    documents: list[dict],
    batch_size: int = 100
):
    """Add documents in batches for better performance."""
    
    total = len(documents)
    added = 0
    
    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        
        collection.add(
            ids=[doc["id"] for doc in batch],
            documents=[doc["text"] for doc in batch],
            metadatas=[doc.get("metadata", {}) for doc in batch]
        )
        
        added += len(batch)
        print(f"Added {added}/{total} documents")
    
    return added

# Example usage
sample_docs = [
    {"id": f"batch_{i}", "text": f"Sample document number {i}", 
     "metadata": {"index": i}}
    for i in range(250)
]

batch_add_documents(collection, sample_docs, batch_size=50)
```

---

## Part 4: Querying Documents

### Exercise 4.1: Basic Semantic Search

```python
# Query by text (embedding generated automatically)
results = collection.query(
    query_texts=["What programming languages are popular?"],
    n_results=5
)

print("Query Results:")
print("-" * 50)
for i, (doc, distance, metadata) in enumerate(zip(
    results["documents"][0],
    results["distances"][0],
    results["metadatas"][0]
)):
    print(f"{i+1}. [{distance:.4f}] {doc[:60]}...")
    print(f"   Metadata: {metadata}")
```

### Exercise 4.2: Query with Metadata Filtering

```python
# Filter by exact match
results = collection.query(
    query_texts=["artificial intelligence"],
    n_results=5,
    where={"category": "ai"}
)

# Filter with operators
results = collection.query(
    query_texts=["programming"],
    n_results=5,
    where={
        "$and": [
            {"category": {"$eq": "programming"}},
            {"difficulty": {"$in": ["beginner", "intermediate"]}}
        ]
    }
)

# Filter on document content
results = collection.query(
    query_texts=["learning"],
    n_results=5,
    where_document={"$contains": "neural"}
)

# Available operators:
# $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
# $and, $or (for combining conditions)
# $contains (for document text)
```

### Exercise 4.3: Query with Embeddings

```python
# Query using pre-computed embedding
query_text = "How do computers learn?"
query_embedding = get_embeddings([query_text])[0]

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    include=["documents", "metadatas", "distances", "embeddings"]
)

# Access all returned data
print("Query embedding dimensions:", len(query_embedding))
print("Result embeddings shape:", len(results["embeddings"][0]))
```

---

## Part 5: Updating and Deleting

### Exercise 5.1: Update Documents

```python
# Update existing documents
collection.update(
    ids=["doc1"],
    documents=["Python is an amazing programming language for AI and web development"],
    metadatas=[{"category": "programming", "language": "python", "updated": True}]
)

# Upsert (update or insert)
collection.upsert(
    ids=["doc1", "new_doc"],
    documents=[
        "Python is the best language for data science",
        "This is a completely new document"
    ],
    metadatas=[
        {"category": "programming", "updated": True},
        {"category": "general", "new": True}
    ]
)
```

### Exercise 5.2: Delete Documents

```python
# Delete by IDs
collection.delete(ids=["doc1", "doc2"])

# Delete by filter
collection.delete(
    where={"category": "weather"}
)

# Delete by document content
collection.delete(
    where_document={"$contains": "temporary"}
)

print(f"Remaining documents: {collection.count()}")
```

---

## Part 6: Building a Complete Application

### Exercise 6.1: Document Search Engine

```python
class DocumentSearchEngine:
    """A complete document search engine using ChromaDB."""
    
    def __init__(self, persist_path: str = "./search_db"):
        self.client = chromadb.PersistentClient(path=persist_path)
        
        # Use OpenAI embeddings
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key="your-api-key",
            model_name="text-embedding-3-small"
        )
        
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_document(
        self, 
        doc_id: str, 
        content: str, 
        metadata: dict = None
    ):
        """Add a single document."""
        self.collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[metadata or {}]
        )
    
    def add_documents(self, documents: list[dict]):
        """Add multiple documents."""
        self.collection.add(
            ids=[doc["id"] for doc in documents],
            documents=[doc["content"] for doc in documents],
            metadatas=[doc.get("metadata", {}) for doc in documents]
        )
    
    def search(
        self, 
        query: str, 
        n_results: int = 5,
        filter: dict = None
    ) -> list[dict]:
        """Search for documents."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity
            })
        
        return formatted
    
    def delete_document(self, doc_id: str):
        """Delete a document by ID."""
        self.collection.delete(ids=[doc_id])
    
    def get_stats(self) -> dict:
        """Get collection statistics."""
        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection.name
        }

# Usage
engine = DocumentSearchEngine()

# Add sample documents
engine.add_documents([
    {
        "id": "python_intro",
        "content": "Python is a high-level programming language known for its simplicity",
        "metadata": {"category": "programming", "language": "python"}
    },
    {
        "id": "ml_basics",
        "content": "Machine learning is a subset of AI that learns from data",
        "metadata": {"category": "ai", "topic": "machine_learning"}
    },
    {
        "id": "web_dev",
        "content": "Web development involves creating websites and web applications",
        "metadata": {"category": "programming", "topic": "web"}
    }
])

# Search
results = engine.search("How can I learn artificial intelligence?")
for result in results:
    print(f"[{result['similarity']:.4f}] {result['content'][:50]}...")
```

### Exercise 6.2: RAG-Ready Retriever

```python
class RAGRetriever:
    """Retriever component for RAG systems."""
    
    def __init__(self, collection_name: str = "knowledge_base"):
        self.client = chromadb.PersistentClient(path="./rag_db")
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key="your-api-key",
            model_name="text-embedding-3-small"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )
    
    def index_documents(
        self, 
        documents: list[str],
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """Index documents with chunking."""
        chunks = []
        chunk_id = 0
        
        for doc_idx, doc in enumerate(documents):
            # Simple chunking by character count
            for i in range(0, len(doc), chunk_size - chunk_overlap):
                chunk_text = doc[i:i + chunk_size]
                if len(chunk_text) > 50:  # Skip tiny chunks
                    chunks.append({
                        "id": f"chunk_{chunk_id}",
                        "text": chunk_text,
                        "metadata": {
                            "doc_index": doc_idx,
                            "chunk_index": chunk_id,
                            "start_char": i
                        }
                    })
                    chunk_id += 1
        
        # Add to collection
        self.collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks]
        )
        
        return len(chunks)
    
    def retrieve(
        self, 
        query: str, 
        n_results: int = 5,
        min_similarity: float = 0.5
    ) -> str:
        """Retrieve relevant context for RAG."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "distances"]
        )
        
        # Filter by similarity and join
        context_parts = []
        for doc, distance in zip(
            results["documents"][0], 
            results["distances"][0]
        ):
            similarity = 1 - distance
            if similarity >= min_similarity:
                context_parts.append(doc)
        
        return "\n\n".join(context_parts)

# Usage for RAG
retriever = RAGRetriever()

# Index some documents
sample_docs = [
    """Python was created by Guido van Rossum and first released in 1991.
    It emphasizes code readability and allows programmers to express concepts
    in fewer lines of code than languages like C++ or Java.""",
    
    """Machine learning is a type of artificial intelligence that allows
    computers to learn from data without being explicitly programmed.
    Common algorithms include decision trees, neural networks, and SVMs."""
]

num_chunks = retriever.index_documents(sample_docs)
print(f"Indexed {num_chunks} chunks")

# Retrieve context for a question
context = retriever.retrieve("Who created Python?")
print("Retrieved context:")
print(context)
```

---

## Part 7: Best Practices

### Exercise 7.1: Error Handling

```python
import chromadb
from chromadb.errors import InvalidCollectionException

def safe_get_collection(client, name: str, embedding_fn):
    """Safely get or create a collection."""
    try:
        return client.get_collection(name=name, embedding_function=embedding_fn)
    except InvalidCollectionException:
        return client.create_collection(name=name, embedding_function=embedding_fn)

def safe_add_documents(collection, docs: list[dict]):
    """Add documents with error handling."""
    try:
        # Check for duplicate IDs
        existing = collection.get(ids=[d["id"] for d in docs])
        existing_ids = set(existing["ids"])
        
        # Filter out existing
        new_docs = [d for d in docs if d["id"] not in existing_ids]
        
        if new_docs:
            collection.add(
                ids=[d["id"] for d in new_docs],
                documents=[d["text"] for d in new_docs],
                metadatas=[d.get("metadata", {}) for d in new_docs]
            )
        
        return {
            "added": len(new_docs),
            "skipped": len(docs) - len(new_docs)
        }
    
    except Exception as e:
        print(f"Error adding documents: {e}")
        return {"error": str(e)}
```

### Exercise 7.2: Performance Optimization

```python
# Tip 1: Batch operations
def optimized_add(collection, documents: list[dict], batch_size: int = 100):
    """Add documents in optimal batch sizes."""
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        collection.add(
            ids=[d["id"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[d.get("metadata", {}) for d in batch]
        )

# Tip 2: Only include what you need
results = collection.query(
    query_texts=["query"],
    n_results=10,
    include=["documents"]  # Don't include embeddings if not needed
)

# Tip 3: Use filters to reduce search space
results = collection.query(
    query_texts=["query"],
    n_results=10,
    where={"category": "specific_category"}  # Narrows search
)

# Tip 4: Tune HNSW parameters for your use case
collection = client.create_collection(
    name="high_recall",
    metadata={
        "hnsw:search_ef": 200,  # Higher for better recall
        "hnsw:M": 32            # More connections
    }
)
```

---

## Summary

You've learned to:

✅ Install and configure ChromaDB
✅ Create and manage collections
✅ Add documents with metadata
✅ Perform semantic search with filtering
✅ Build production-ready applications
✅ Implement RAG retrieval patterns

## Next Steps

1. **Semantic Search Lab** - Build a full search application
2. **RAG Module** - Use ChromaDB for retrieval-augmented generation
3. **Compare Vector DBs** - Try Qdrant, Pinecone, etc.
