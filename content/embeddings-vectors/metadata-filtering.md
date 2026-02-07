# Metadata Filtering in Vector Databases

Combine semantic similarity with attribute filtering for precise, targeted search results.

## Why Metadata Filtering?

```yaml
the_problem:
  scenario: "Search for Python tutorials"
  semantic_only: "Returns all programming content"
  what_you_want: "Only Python content, beginner level, last 6 months"
  
  solution: "Combine semantic search with metadata filters"
```

## Metadata Structure

```python
# Documents with rich metadata
documents = [
    {
        "id": "doc1",
        "content": "Introduction to Python programming",
        "metadata": {
            "category": "programming",
            "language": "python",
            "difficulty": "beginner",
            "author": "Alice",
            "date": "2024-01-15",
            "views": 1500,
            "rating": 4.5
        }
    },
    {
        "id": "doc2",
        "content": "Advanced machine learning with TensorFlow",
        "metadata": {
            "category": "ai",
            "language": "python",
            "difficulty": "advanced",
            "author": "Bob",
            "date": "2024-02-20",
            "views": 3200,
            "rating": 4.8
        }
    }
]

# Add to collection
collection.add(
    ids=[d["id"] for d in documents],
    documents=[d["content"] for d in documents],
    metadatas=[d["metadata"] for d in documents]
)
```

## Basic Filtering

### Equality Filters

```python
# Filter by exact match
results = collection.query(
    query_texts=["programming tutorials"],
    n_results=10,
    where={"language": "python"}
)

# Explicit equality operator
results = collection.query(
    query_texts=["programming tutorials"],
    n_results=10,
    where={"language": {"$eq": "python"}}
)
```

### Comparison Operators

```python
# Greater than
results = collection.query(
    query_texts=["tutorials"],
    n_results=10,
    where={"rating": {"$gt": 4.0}}  # rating > 4.0
)

# Greater than or equal
results = collection.query(
    query_texts=["tutorials"],
    n_results=10,
    where={"views": {"$gte": 1000}}  # views >= 1000
)

# Less than
results = collection.query(
    query_texts=["tutorials"],
    n_results=10,
    where={"difficulty_level": {"$lt": 3}}
)

# Less than or equal
results = collection.query(
    query_texts=["tutorials"],
    n_results=10,
    where={"rating": {"$lte": 4.5}}
)

# Not equal
results = collection.query(
    query_texts=["tutorials"],
    n_results=10,
    where={"status": {"$ne": "draft"}}
)
```

### In/Not In Lists

```python
# Match any in list
results = collection.query(
    query_texts=["tutorials"],
    n_results=10,
    where={"category": {"$in": ["programming", "ai", "data-science"]}}
)

# Exclude from list
results = collection.query(
    query_texts=["tutorials"],
    n_results=10,
    where={"difficulty": {"$nin": ["expert", "advanced"]}}
)
```

## Combining Filters

### AND Logic

```python
# Multiple conditions (all must match)
results = collection.query(
    query_texts=["python tutorials"],
    n_results=10,
    where={
        "$and": [
            {"language": {"$eq": "python"}},
            {"difficulty": {"$eq": "beginner"}},
            {"rating": {"$gte": 4.0}}
        ]
    }
)
```

### OR Logic

```python
# Any condition can match
results = collection.query(
    query_texts=["programming"],
    n_results=10,
    where={
        "$or": [
            {"language": "python"},
            {"language": "javascript"},
            {"language": "typescript"}
        ]
    }
)
```

### Complex Nested Logic

```python
# Complex: (category = 'ai' OR category = 'ml') AND rating > 4
results = collection.query(
    query_texts=["machine learning"],
    n_results=10,
    where={
        "$and": [
            {
                "$or": [
                    {"category": "ai"},
                    {"category": "ml"}
                ]
            },
            {"rating": {"$gt": 4.0}}
        ]
    }
)

# Alternative: High rated Python OR any JavaScript
results = collection.query(
    query_texts=["tutorials"],
    n_results=10,
    where={
        "$or": [
            {
                "$and": [
                    {"language": "python"},
                    {"rating": {"$gte": 4.5}}
                ]
            },
            {"language": "javascript"}
        ]
    }
)
```

## Document Content Filters

```python
# Filter by document text content
results = collection.query(
    query_texts=["programming"],
    n_results=10,
    where_document={"$contains": "tutorial"}
)

# Combine with metadata
results = collection.query(
    query_texts=["programming"],
    n_results=10,
    where={"category": "programming"},
    where_document={"$contains": "beginner"}
)
```

## Date Filtering

```python
# ChromaDB stores dates as strings, compare lexicographically
# Format: YYYY-MM-DD for proper sorting

# After a date
results = collection.query(
    query_texts=["tutorials"],
    n_results=10,
    where={"date": {"$gte": "2024-01-01"}}
)

# Date range
results = collection.query(
    query_texts=["tutorials"],
    n_results=10,
    where={
        "$and": [
            {"date": {"$gte": "2024-01-01"}},
            {"date": {"$lte": "2024-06-30"}}
        ]
    }
)

# Last N days (compute dynamically)
from datetime import datetime, timedelta

cutoff_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
results = collection.query(
    query_texts=["tutorials"],
    n_results=10,
    where={"date": {"$gte": cutoff_date}}
)
```

## Practical Filter Patterns

### Faceted Search

```python
class FacetedSearch:
    """Search with multiple facet filters."""
    
    def __init__(self, collection):
        self.collection = collection
    
    def search(
        self,
        query: str,
        categories: list[str] = None,
        difficulty: list[str] = None,
        min_rating: float = None,
        date_from: str = None,
        date_to: str = None,
        n_results: int = 10
    ) -> dict:
        """Search with optional facet filters."""
        
        conditions = []
        
        if categories:
            conditions.append({"category": {"$in": categories}})
        
        if difficulty:
            conditions.append({"difficulty": {"$in": difficulty}})
        
        if min_rating is not None:
            conditions.append({"rating": {"$gte": min_rating}})
        
        if date_from:
            conditions.append({"date": {"$gte": date_from}})
        
        if date_to:
            conditions.append({"date": {"$lte": date_to}})
        
        # Build where clause
        where = None
        if conditions:
            if len(conditions) == 1:
                where = conditions[0]
            else:
                where = {"$and": conditions}
        
        return self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

# Usage
search = FacetedSearch(collection)

results = search.search(
    query="machine learning tutorials",
    categories=["ai", "ml"],
    difficulty=["beginner", "intermediate"],
    min_rating=4.0,
    date_from="2024-01-01"
)
```

### Access Control Filters

```python
def search_with_access_control(
    collection,
    query: str,
    user_roles: list[str],
    user_id: str,
    n_results: int = 10
) -> dict:
    """Filter results based on user permissions."""
    
    # Documents can be:
    # - public (visible to all)
    # - role-restricted (visible to certain roles)
    # - user-owned (visible to owner)
    
    access_filter = {
        "$or": [
            {"visibility": "public"},
            {"required_role": {"$in": user_roles}},
            {"owner_id": user_id}
        ]
    }
    
    return collection.query(
        query_texts=[query],
        n_results=n_results,
        where=access_filter,
        include=["documents", "metadatas", "distances"]
    )

# Usage
results = search_with_access_control(
    collection,
    query="internal documentation",
    user_roles=["developer", "team-lead"],
    user_id="user_123"
)
```

### Multi-Tenant Filtering

```python
def tenant_search(
    collection,
    query: str,
    tenant_id: str,
    additional_filters: dict = None,
    n_results: int = 10
) -> dict:
    """Search within a specific tenant's data."""
    
    # Always filter by tenant
    tenant_filter = {"tenant_id": tenant_id}
    
    if additional_filters:
        where = {
            "$and": [
                tenant_filter,
                additional_filters
            ]
        }
    else:
        where = tenant_filter
    
    return collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

# Usage - each tenant only sees their data
results = tenant_search(
    collection,
    query="product documentation",
    tenant_id="acme_corp",
    additional_filters={"category": "docs"}
)
```

## Filter Performance Tips

```yaml
filter_optimization:
  index_design:
    - "Store frequently filtered fields in metadata"
    - "Use consistent data types"
    - "Prefer enumerated values over free text"
  
  query_patterns:
    - "Apply selective filters first"
    - "Avoid $or with many branches"
    - "Use $in instead of multiple $eq with $or"
  
  data_modeling:
    good:
      - "status: 'active' (enumerated)"
      - "category: 'ai' (standardized)"
      - "date: '2024-01-15' (ISO format)"
    
    avoid:
      - "status: 'Currently Active!' (free text)"
      - "category: 'Artificial Intelligence' (inconsistent)"
      - "date: 'Jan 15, 2024' (non-standard)"
```

## Metadata Best Practices

```python
# Good metadata design
good_metadata = {
    # Enumerated values for filtering
    "status": "published",  # draft, published, archived
    "category": "ai",       # standardized categories
    "difficulty": "intermediate",  # beginner, intermediate, advanced
    
    # Numeric for comparisons
    "rating": 4.5,
    "view_count": 1234,
    "word_count": 500,
    
    # Dates in ISO format
    "created_at": "2024-01-15",
    "updated_at": "2024-02-20",
    
    # IDs for relationships
    "author_id": "user_123",
    "tenant_id": "acme_corp",
    
    # Tags as comma-separated (ChromaDB limitation)
    "tags": "python,machine-learning,tutorial"
}

# Querying tags
results = collection.query(
    query_texts=["machine learning"],
    where_document={"$contains": "python"}  # Search in tags via document
)
```

## Summary

```yaml
filtering_capabilities:
  operators:
    comparison: "$eq, $ne, $gt, $gte, $lt, $lte"
    set: "$in, $nin"
    logical: "$and, $or"
    document: "$contains"
  
  best_practices:
    - "Use consistent data types"
    - "Design metadata for filterability"
    - "Combine semantic search with precise filters"
    - "Test filter performance with real data"
  
  patterns:
    - "Faceted search for user interfaces"
    - "Access control for security"
    - "Multi-tenancy for SaaS apps"
    - "Date ranges for time-based queries"
```

## Next Steps

1. **Semantic Search Lab** - Build a complete search application
2. **RAG Integration** - Use filtered retrieval for generation
3. **Production Patterns** - Scale your vector search
