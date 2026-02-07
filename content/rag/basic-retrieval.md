# Basic Retrieval Strategies

Master the fundamentals of retrieving relevant documents from your vector database.

## The Retrieval Challenge

```yaml
goals:
  - "Find the most relevant documents for a query"
  - "Balance precision (quality) and recall (coverage)"
  - "Handle diverse query types effectively"
  - "Minimize irrelevant results"

key_metrics:
  precision: "How many retrieved docs are relevant?"
  recall: "How many relevant docs were retrieved?"
  mrr: "Where does the first relevant result appear?"
```

## Simple Similarity Search

```python
from openai import OpenAI
import chromadb


class BasicRetriever:
    """Basic similarity search retriever."""
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./retriever_db"
    ):
        self.client = OpenAI()
        self.chroma = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def retrieve(
        self,
        query: str,
        n_results: int = 5
    ) -> list[dict]:
        """Retrieve similar documents."""
        
        # Embed query
        response = self.client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        return [
            {
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]  # Convert to similarity
            }
            for i in range(len(results["ids"][0]))
        ]


# Usage
retriever = BasicRetriever()
results = retriever.retrieve("How do I configure authentication?", n_results=5)

for r in results:
    print(f"Score: {r['score']:.3f}")
    print(f"Content: {r['content'][:100]}...")
```

## Filtered Retrieval

```python
class FilteredRetriever(BasicRetriever):
    """Retriever with metadata filtering."""
    
    def retrieve_filtered(
        self,
        query: str,
        n_results: int = 5,
        filters: dict = None
    ) -> list[dict]:
        """Retrieve with metadata filters."""
        
        # Embed query
        response = self.client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        
        # Search with filters
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filters,
            include=["documents", "metadatas", "distances"]
        )
        
        return [
            {
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]
            }
            for i in range(len(results["ids"][0]))
        ]


# Examples
retriever = FilteredRetriever()

# Filter by category
results = retriever.retrieve_filtered(
    "setup guide",
    filters={"category": "documentation"}
)

# Filter by date range
results = retriever.retrieve_filtered(
    "recent updates",
    filters={
        "$and": [
            {"date": {"$gte": "2024-01-01"}},
            {"date": {"$lte": "2024-12-31"}}
        ]
    }
)

# Multiple conditions
results = retriever.retrieve_filtered(
    "API reference",
    filters={
        "$and": [
            {"category": {"$in": ["api", "reference"]}},
            {"status": "published"}
        ]
    }
)
```

## Score Thresholding

```python
class ThresholdedRetriever(BasicRetriever):
    """Retriever with minimum score threshold."""
    
    def retrieve_with_threshold(
        self,
        query: str,
        n_results: int = 10,
        min_score: float = 0.5
    ) -> list[dict]:
        """Only return results above threshold."""
        
        # Get more results than needed
        results = self.retrieve(query, n_results=n_results * 2)
        
        # Filter by threshold
        return [r for r in results if r["score"] >= min_score][:n_results]


# Usage
retriever = ThresholdedRetriever()

# Only high-confidence results
results = retriever.retrieve_with_threshold(
    "exact phrase match needed",
    min_score=0.7
)

# More lenient threshold
results = retriever.retrieve_with_threshold(
    "general topic search",
    min_score=0.3
)
```

## Dynamic K Selection

```python
class DynamicKRetriever(BasicRetriever):
    """Dynamically adjust number of results based on scores."""
    
    def retrieve_dynamic(
        self,
        query: str,
        max_results: int = 10,
        score_drop_threshold: float = 0.2
    ) -> list[dict]:
        """Return results until score drops significantly."""
        
        results = self.retrieve(query, n_results=max_results)
        
        if not results:
            return []
        
        # Keep results until score drops significantly
        selected = [results[0]]
        
        for i in range(1, len(results)):
            score_drop = selected[-1]["score"] - results[i]["score"]
            
            # Stop if score drops too much
            if score_drop > score_drop_threshold:
                break
            
            selected.append(results[i])
        
        return selected


# Usage
retriever = DynamicKRetriever()

# Will return fewer results if there's a clear relevance gap
results = retriever.retrieve_dynamic(
    "specific technical question",
    max_results=10,
    score_drop_threshold=0.15
)

print(f"Returned {len(results)} results (stopped at relevance gap)")
```

## Maximal Marginal Relevance (MMR)

MMR balances relevance with diversity to avoid redundant results.

```python
import numpy as np


class MMRRetriever(BasicRetriever):
    """Retriever using Maximal Marginal Relevance."""
    
    def retrieve_mmr(
        self,
        query: str,
        n_results: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> list[dict]:
        """
        Retrieve diverse, relevant results using MMR.
        
        lambda_mult: Balance between relevance (1.0) and diversity (0.0)
        """
        
        # Get query embedding
        response = self.client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = np.array(response.data[0].embedding)
        
        # Get more candidates than needed
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances", "embeddings"]
        )
        
        if not results["ids"][0]:
            return []
        
        # Get embeddings for candidates
        candidate_embeddings = [
            np.array(emb) for emb in results["embeddings"][0]
        ]
        
        # MMR selection
        selected_indices = []
        remaining_indices = list(range(len(candidate_embeddings)))
        
        while len(selected_indices) < n_results and remaining_indices:
            best_score = -float('inf')
            best_idx = -1
            
            for idx in remaining_indices:
                # Relevance to query
                relevance = np.dot(query_embedding, candidate_embeddings[idx])
                
                # Max similarity to already selected
                if selected_indices:
                    similarities = [
                        np.dot(candidate_embeddings[idx], candidate_embeddings[s])
                        for s in selected_indices
                    ]
                    max_sim = max(similarities)
                else:
                    max_sim = 0
                
                # MMR score
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Return selected results
        return [
            {
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]
            }
            for i in selected_indices
        ]


# Usage
retriever = MMRRetriever()

# More relevance focused
results = retriever.retrieve_mmr(
    "machine learning basics",
    n_results=5,
    lambda_mult=0.8  # 80% relevance, 20% diversity
)

# More diversity focused
results = retriever.retrieve_mmr(
    "general programming concepts",
    n_results=5,
    lambda_mult=0.3  # 30% relevance, 70% diversity
)
```

## Parent Document Retrieval

Retrieve small chunks but return larger context.

```python
class ParentDocumentRetriever:
    """Retrieve chunks but return parent documents."""
    
    def __init__(self, persist_directory: str = "./parent_retriever_db"):
        self.client = OpenAI()
        self.chroma = chromadb.PersistentClient(path=persist_directory)
        
        # Two collections: chunks for search, parents for context
        self.chunk_collection = self.chroma.get_or_create_collection(
            name="chunks",
            metadata={"hnsw:space": "cosine"}
        )
        self.parent_collection = self.chroma.get_or_create_collection(
            name="parents"
        )
    
    def add_document(
        self,
        doc_id: str,
        content: str,
        chunk_size: int = 200,
        parent_size: int = 1000
    ):
        """Index document with both chunks and parents."""
        
        # Create parent chunks
        parents = []
        for i in range(0, len(content), parent_size):
            parent_id = f"{doc_id}_parent_{i // parent_size}"
            parent_content = content[i:i + parent_size]
            parents.append({"id": parent_id, "content": parent_content})
        
        # Store parents
        self.parent_collection.add(
            ids=[p["id"] for p in parents],
            documents=[p["content"] for p in parents]
        )
        
        # Create and index small chunks
        chunks = []
        for parent in parents:
            parent_content = parent["content"]
            for j in range(0, len(parent_content), chunk_size):
                chunk_content = parent_content[j:j + chunk_size]
                if len(chunk_content) > 50:  # Min size
                    chunks.append({
                        "id": f"{parent['id']}_chunk_{j // chunk_size}",
                        "content": chunk_content,
                        "parent_id": parent["id"]
                    })
        
        # Embed chunks
        embeddings = self._embed_batch([c["content"] for c in chunks])
        
        # Index chunks
        self.chunk_collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["content"] for c in chunks],
            embeddings=embeddings,
            metadatas=[{"parent_id": c["parent_id"]} for c in chunks]
        )
    
    def retrieve(
        self,
        query: str,
        n_results: int = 3
    ) -> list[dict]:
        """Search chunks, return parents."""
        
        # Embed query
        query_embedding = self._embed(query)
        
        # Search chunks
        chunk_results = self.chunk_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results * 2,  # Get more to dedupe parents
            include=["documents", "metadatas", "distances"]
        )
        
        # Get unique parent IDs
        parent_ids = list(set(
            m["parent_id"] for m in chunk_results["metadatas"][0]
        ))[:n_results]
        
        # Fetch parents
        parents = self.parent_collection.get(ids=parent_ids)
        
        return [
            {
                "content": parents["documents"][i],
                "id": parents["ids"][i],
                "matched_chunk": next(
                    (chunk_results["documents"][0][j]
                     for j, m in enumerate(chunk_results["metadatas"][0])
                     if m["parent_id"] == parents["ids"][i]),
                    None
                )
            }
            for i in range(len(parents["ids"]))
        ]
    
    def _embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]


# Usage
retriever = ParentDocumentRetriever()

# Add document
retriever.add_document(
    doc_id="handbook",
    content=long_document_text,
    chunk_size=200,  # Small for precise matching
    parent_size=1000  # Large for context
)

# Search
results = retriever.retrieve("vacation policy")
for r in results:
    print(f"Matched: {r['matched_chunk'][:50]}...")
    print(f"Full context: {r['content'][:200]}...")
```

## Choosing a Retrieval Strategy

```yaml
decision_guide:
  basic_similarity:
    when: "Simple use cases, homogeneous content"
    pros: "Fast, simple"
    cons: "May miss nuanced matches"
  
  filtered:
    when: "Multi-category content, user permissions"
    pros: "Precise targeting"
    cons: "Requires good metadata"
  
  thresholded:
    when: "Need to avoid low-quality results"
    pros: "Quality control"
    cons: "May return fewer results"
  
  mmr:
    when: "Results tend to be redundant"
    pros: "Diverse results"
    cons: "Slightly slower"
  
  parent_document:
    when: "Need more context than chunks provide"
    pros: "Full context"
    cons: "More complex setup"
```

## Summary

```yaml
key_takeaways:
  1: "Start with basic similarity search"
  2: "Add filtering for multi-category content"
  3: "Use thresholds to control quality"
  4: "Apply MMR when diversity matters"
  5: "Consider parent retrieval for context"

performance_tips:
  - "Index only what you need to search"
  - "Use appropriate chunk sizes"
  - "Cache embeddings when possible"
  - "Monitor retrieval latency"
```

## Next Steps

1. **Hybrid Search** - Combine semantic and keyword search
2. **Reranking** - Improve result quality with cross-encoders
3. **Multi-Query Retrieval** - Handle complex questions
