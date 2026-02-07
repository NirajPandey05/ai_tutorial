# Working with Collections

Collections are the fundamental organizational unit in vector databases. Learn how to create, configure, and manage them effectively.

## What is a Collection?

```yaml
collection_concept:
  definition: "A named container for related vectors and metadata"
  analogy: "Like a table in SQL or a collection in MongoDB"
  
  contains:
    - "Vectors (embeddings)"
    - "Documents (original text)"
    - "Metadata (filterable attributes)"
    - "Index configuration"
```

## Creating Collections

### Basic Creation

```python
import chromadb

client = chromadb.PersistentClient(path="./my_db")

# Simple creation
collection = client.create_collection(name="documents")

# With embedding function
from chromadb.utils import embedding_functions

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-key",
    model_name="text-embedding-3-small"
)

collection = client.create_collection(
    name="documents",
    embedding_function=openai_ef
)
```

### With Configuration

```python
# Full configuration
collection = client.create_collection(
    name="optimized_collection",
    embedding_function=openai_ef,
    metadata={
        # Distance metric
        "hnsw:space": "cosine",  # "l2", "ip", or "cosine"
        
        # HNSW index parameters
        "hnsw:construction_ef": 200,  # Build quality
        "hnsw:search_ef": 100,        # Query quality
        "hnsw:M": 16,                 # Graph connectivity
        
        # Custom metadata
        "description": "Production document store",
        "version": "1.0",
        "created_by": "admin"
    }
)
```

## Collection Operations

### CRUD Operations

```python
# Create
collection = client.create_collection(name="new_collection")

# Read/Get
collection = client.get_collection(
    name="new_collection",
    embedding_function=openai_ef  # Must match creation
)

# Get or Create (idempotent)
collection = client.get_or_create_collection(
    name="my_collection",
    embedding_function=openai_ef
)

# List all collections
collections = client.list_collections()
for coll in collections:
    print(f"Collection: {coll.name}")

# Delete
client.delete_collection(name="old_collection")

# Count documents
count = collection.count()
print(f"Documents in collection: {count}")
```

### Collection Properties

```python
# Get collection info
print(f"Name: {collection.name}")
print(f"Metadata: {collection.metadata}")
print(f"Document count: {collection.count()}")

# Peek at sample documents
sample = collection.peek(limit=5)
print(f"Sample IDs: {sample['ids']}")
print(f"Sample documents: {sample['documents']}")
```

## Multi-Collection Patterns

### By Data Type

```python
# Separate collections for different content types
articles_collection = client.get_or_create_collection(
    name="articles",
    metadata={"content_type": "article", "hnsw:space": "cosine"}
)

faqs_collection = client.get_or_create_collection(
    name="faqs",
    metadata={"content_type": "faq", "hnsw:space": "cosine"}
)

code_collection = client.get_or_create_collection(
    name="code_snippets",
    metadata={"content_type": "code", "hnsw:space": "cosine"}
)
```

### By Tenant/User

```python
def get_user_collection(user_id: str):
    """Get or create a user-specific collection."""
    return client.get_or_create_collection(
        name=f"user_{user_id}_documents",
        embedding_function=openai_ef,
        metadata={"user_id": user_id, "type": "personal"}
    )

# Each user has isolated data
alice_docs = get_user_collection("alice")
bob_docs = get_user_collection("bob")
```

### By Version/Environment

```python
# Production vs development
environment = "production"  # or "development"

collection = client.get_or_create_collection(
    name=f"documents_{environment}",
    metadata={"environment": environment}
)
```

## Collection Schemas

While ChromaDB doesn't enforce schemas, you can design consistent patterns:

```python
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class DocumentSchema:
    """Define expected structure for documents."""
    id: str
    content: str
    title: str
    category: str
    author: Optional[str] = None
    created_at: Optional[str] = None
    tags: Optional[list[str]] = None
    
    def to_metadata(self) -> dict:
        """Convert to ChromaDB metadata format."""
        metadata = {
            "title": self.title,
            "category": self.category,
        }
        if self.author:
            metadata["author"] = self.author
        if self.created_at:
            metadata["created_at"] = self.created_at
        if self.tags:
            metadata["tags"] = ",".join(self.tags)  # Lists as comma-separated
        return metadata

# Usage
doc = DocumentSchema(
    id="doc_001",
    content="Python is great for data science",
    title="Python for Data Science",
    category="programming",
    author="Alice",
    tags=["python", "data-science"]
)

collection.add(
    ids=[doc.id],
    documents=[doc.content],
    metadatas=[doc.to_metadata()]
)
```

## Managing Collection Lifecycle

### Migration Pattern

```python
def migrate_collection(
    client,
    old_name: str,
    new_name: str,
    embedding_function,
    transform_fn=None
):
    """Migrate data from old to new collection."""
    
    # Get old collection
    old_collection = client.get_collection(old_name)
    
    # Create new collection
    new_collection = client.create_collection(
        name=new_name,
        embedding_function=embedding_function
    )
    
    # Get all data
    all_data = old_collection.get(include=["documents", "metadatas", "embeddings"])
    
    # Transform if needed
    if transform_fn:
        all_data = transform_fn(all_data)
    
    # Add to new collection
    if all_data["ids"]:
        new_collection.add(
            ids=all_data["ids"],
            documents=all_data["documents"],
            metadatas=all_data["metadatas"],
            embeddings=all_data["embeddings"]
        )
    
    print(f"Migrated {len(all_data['ids'])} documents")
    return new_collection

# Example: Migrate with transformation
def add_version_metadata(data):
    """Add version to all metadata."""
    for metadata in data["metadatas"]:
        metadata["schema_version"] = "2.0"
    return data

new_collection = migrate_collection(
    client,
    "documents_v1",
    "documents_v2",
    openai_ef,
    transform_fn=add_version_metadata
)
```

### Backup and Restore

```python
import json
from pathlib import Path

def backup_collection(collection, backup_path: str):
    """Backup collection to JSON file."""
    # Get all data
    data = collection.get(
        include=["documents", "metadatas", "embeddings"]
    )
    
    # Save to file
    backup_data = {
        "name": collection.name,
        "metadata": collection.metadata,
        "data": data
    }
    
    Path(backup_path).write_text(json.dumps(backup_data, indent=2))
    print(f"Backed up {len(data['ids'])} documents to {backup_path}")

def restore_collection(client, backup_path: str, embedding_function):
    """Restore collection from JSON backup."""
    backup_data = json.loads(Path(backup_path).read_text())
    
    # Create collection
    collection = client.create_collection(
        name=backup_data["name"],
        embedding_function=embedding_function,
        metadata=backup_data.get("metadata", {})
    )
    
    # Restore data
    data = backup_data["data"]
    if data["ids"]:
        collection.add(
            ids=data["ids"],
            documents=data["documents"],
            metadatas=data["metadatas"],
            embeddings=data["embeddings"]
        )
    
    print(f"Restored {len(data['ids'])} documents")
    return collection
```

## Best Practices

```yaml
collection_best_practices:
  naming:
    - "Use descriptive, lowercase names"
    - "Include version or environment if needed"
    - "Avoid special characters"
    examples:
      good: ["documents", "user_uploads", "faq_v2"]
      bad: ["My Documents!", "test 123", "🚀data"]
  
  organization:
    - "Group related data in same collection"
    - "Separate by access patterns"
    - "Consider multi-tenancy needs"
  
  configuration:
    - "Choose distance metric based on embedding model"
    - "Tune HNSW for your recall/speed tradeoff"
    - "Document custom metadata"
  
  lifecycle:
    - "Plan for migrations early"
    - "Implement backup/restore"
    - "Monitor collection size"
```

## Collection Manager Class

```python
class CollectionManager:
    """Centralized collection management."""
    
    def __init__(self, persist_path: str = "./vector_db"):
        self.client = chromadb.PersistentClient(path=persist_path)
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key="your-key",
            model_name="text-embedding-3-small"
        )
        self._collections = {}
    
    def get_collection(self, name: str) -> chromadb.Collection:
        """Get or create a collection with caching."""
        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                embedding_function=self.embedding_fn
            )
        return self._collections[name]
    
    def list_all(self) -> list[str]:
        """List all collection names."""
        return [c.name for c in self.client.list_collections()]
    
    def get_stats(self) -> dict:
        """Get statistics for all collections."""
        stats = {}
        for coll in self.client.list_collections():
            stats[coll.name] = {
                "count": coll.count(),
                "metadata": coll.metadata
            }
        return stats
    
    def delete_collection(self, name: str):
        """Delete a collection and clear cache."""
        self.client.delete_collection(name=name)
        self._collections.pop(name, None)

# Usage
manager = CollectionManager()
docs = manager.get_collection("documents")
faqs = manager.get_collection("faqs")
print(manager.get_stats())
```

## Summary

```yaml
key_takeaways:
  collections_are:
    - "Containers for vectors and metadata"
    - "Configurable with distance metrics and HNSW params"
    - "Isolated - great for multi-tenancy"
  
  best_practices:
    - "Use descriptive names"
    - "Configure based on use case"
    - "Plan for migrations"
    - "Implement backup/restore"
```

## Next Steps

1. **Querying Documents** - Search within collections
2. **Metadata Filtering** - Filter by attributes
3. **Semantic Search Lab** - Build search applications
