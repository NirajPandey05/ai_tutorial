# Hybrid Search: Combining Semantic and Keyword Search

Get the best of both worlds by combining vector similarity with traditional keyword matching.

## Why Hybrid Search?

```yaml
semantic_search_strengths:
  - "Understands meaning and context"
  - "Finds paraphrased content"
  - "Works across languages"
  
semantic_search_weaknesses:
  - "May miss exact terms (product codes, names)"
  - "Can be too fuzzy for technical queries"
  - "Struggles with rare/specialized vocabulary"

keyword_search_strengths:
  - "Exact matching (product IDs, error codes)"
  - "Fast and predictable"
  - "Works well for technical jargon"

keyword_search_weaknesses:
  - "No understanding of meaning"
  - "Misses synonyms and paraphrases"
  - "Requires exact term matches"

hybrid_approach:
  goal: "Combine both for better results"
  method: "Score fusion from multiple search types"
```

## Basic Hybrid Implementation

```python
from openai import OpenAI
import chromadb
from typing import Optional
import re


class HybridRetriever:
    """Combine semantic and keyword search."""
    
    def __init__(
        self,
        collection_name: str = "hybrid_docs",
        persist_directory: str = "./hybrid_db"
    ):
        self.client = OpenAI()
        self.chroma = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: list[dict]):
        """Add documents with embeddings."""
        
        texts = [d["content"] for d in documents]
        embeddings = self._embed_batch(texts)
        
        self.collection.add(
            ids=[d["id"] for d in documents],
            documents=texts,
            embeddings=embeddings,
            metadatas=[d.get("metadata", {}) for d in documents]
        )
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> list[dict]:
        """
        Hybrid search combining semantic and keyword.
        
        weights should sum to 1.0
        """
        
        # Semantic search
        semantic_results = self._semantic_search(query, n_results * 2)
        
        # Keyword search
        keyword_results = self._keyword_search(query, n_results * 2)
        
        # Combine with reciprocal rank fusion
        combined = self._reciprocal_rank_fusion(
            semantic_results,
            keyword_results,
            semantic_weight,
            keyword_weight
        )
        
        return combined[:n_results]
    
    def _semantic_search(self, query: str, n_results: int) -> list[dict]:
        """Pure semantic search."""
        
        query_embedding = self._embed(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        return [
            {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],
                "source": "semantic"
            }
            for i in range(len(results["ids"][0]))
        ]
    
    def _keyword_search(self, query: str, n_results: int) -> list[dict]:
        """Simple keyword search using ChromaDB's where_document."""
        
        # Extract important keywords
        keywords = self._extract_keywords(query)
        
        if not keywords:
            return []
        
        # Search for documents containing keywords
        results = self.collection.get(
            include=["documents", "metadatas"]
        )
        
        # Score by keyword matches
        scored = []
        for i, doc in enumerate(results["documents"]):
            doc_lower = doc.lower()
            
            # Count keyword matches
            matches = sum(
                1 for kw in keywords 
                if kw.lower() in doc_lower
            )
            
            if matches > 0:
                scored.append({
                    "id": results["ids"][i],
                    "content": doc,
                    "metadata": results["metadatas"][i],
                    "score": matches / len(keywords),  # Normalize
                    "source": "keyword"
                })
        
        # Sort by score
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:n_results]
    
    def _extract_keywords(self, query: str) -> list[str]:
        """Extract keywords from query."""
        
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "of", "to", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "just", "and", "but", "if", "or",
            "because", "until", "while", "what", "which", "who", "whom",
            "this", "that", "these", "those", "am", "i", "me", "my"
        }
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def _reciprocal_rank_fusion(
        self,
        semantic_results: list[dict],
        keyword_results: list[dict],
        semantic_weight: float,
        keyword_weight: float,
        k: int = 60
    ) -> list[dict]:
        """Combine results using Reciprocal Rank Fusion."""
        
        scores = {}
        results_map = {}
        
        # Score semantic results
        for rank, result in enumerate(semantic_results, 1):
            doc_id = result["id"]
            rrf_score = semantic_weight / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            results_map[doc_id] = result
        
        # Score keyword results
        for rank, result in enumerate(keyword_results, 1):
            doc_id = result["id"]
            rrf_score = keyword_weight / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            if doc_id not in results_map:
                results_map[doc_id] = result
        
        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        return [
            {**results_map[doc_id], "hybrid_score": scores[doc_id]}
            for doc_id in sorted_ids
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
retriever = HybridRetriever()

# Add documents
retriever.add_documents([
    {"id": "doc1", "content": "Error code E-401 indicates authentication failure"},
    {"id": "doc2", "content": "Login problems often stem from incorrect credentials"},
    {"id": "doc3", "content": "The E-401 error requires re-authentication"},
])

# Semantic-heavy search (understanding)
results = retriever.search(
    "I can't log in to my account",
    semantic_weight=0.8,
    keyword_weight=0.2
)

# Keyword-heavy search (exact match needed)
results = retriever.search(
    "E-401 error",
    semantic_weight=0.3,
    keyword_weight=0.7
)
```

## BM25 + Vector Search

```python
from rank_bm25 import BM25Okapi
import numpy as np


class BM25HybridRetriever:
    """Hybrid search using BM25 for keyword matching."""
    
    def __init__(self):
        self.client = OpenAI()
        self.documents = []
        self.doc_ids = []
        self.doc_metadata = []
        self.embeddings = []
        self.bm25 = None
    
    def add_documents(self, documents: list[dict]):
        """Index documents for both BM25 and vector search."""
        
        for doc in documents:
            self.documents.append(doc["content"])
            self.doc_ids.append(doc["id"])
            self.doc_metadata.append(doc.get("metadata", {}))
        
        # Build BM25 index
        tokenized = [self._tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)
        
        # Build vector embeddings
        self.embeddings = self._embed_batch(self.documents)
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        alpha: float = 0.5  # Balance: 0=BM25 only, 1=vector only
    ) -> list[dict]:
        """
        Hybrid search with configurable balance.
        
        alpha: 0.5 = equal weight, higher = more semantic
        """
        
        # BM25 scores
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores to 0-1
        if max(bm25_scores) > 0:
            bm25_scores = bm25_scores / max(bm25_scores)
        
        # Vector similarity scores
        query_embedding = np.array(self._embed(query))
        vector_scores = np.array([
            np.dot(query_embedding, np.array(emb)) / 
            (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            for emb in self.embeddings
        ])
        
        # Normalize vector scores to 0-1
        vector_scores = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min() + 1e-6)
        
        # Combine scores
        combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores
        
        # Get top results
        top_indices = np.argsort(combined_scores)[::-1][:n_results]
        
        return [
            {
                "id": self.doc_ids[i],
                "content": self.documents[i],
                "metadata": self.doc_metadata[i],
                "score": combined_scores[i],
                "bm25_score": bm25_scores[i],
                "vector_score": vector_scores[i]
            }
            for i in top_indices
        ]
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for BM25."""
        return text.lower().split()
    
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
retriever = BM25HybridRetriever()

retriever.add_documents([
    {"id": "1", "content": "Python is a programming language known for readability"},
    {"id": "2", "content": "JavaScript runs in web browsers and Node.js"},
    {"id": "3", "content": "Programming in Python is easy to learn"},
])

# Balanced search
results = retriever.search("Python programming", alpha=0.5)
for r in results:
    print(f"ID: {r['id']}, Score: {r['score']:.3f}")
    print(f"  BM25: {r['bm25_score']:.3f}, Vector: {r['vector_score']:.3f}")
```

## Query-Adaptive Hybrid Search

```python
class AdaptiveHybridRetriever(HybridRetriever):
    """Automatically adjust weights based on query type."""
    
    def search_adaptive(
        self,
        query: str,
        n_results: int = 5
    ) -> list[dict]:
        """Automatically determine optimal weights."""
        
        # Analyze query
        query_type = self._classify_query(query)
        
        # Set weights based on query type
        weights = {
            "technical": (0.3, 0.7),    # Favor keywords
            "conceptual": (0.8, 0.2),   # Favor semantic
            "mixed": (0.5, 0.5),        # Balanced
            "exact": (0.1, 0.9),        # Strong keyword
        }
        
        semantic_w, keyword_w = weights.get(query_type, (0.5, 0.5))
        
        return self.search(
            query,
            n_results=n_results,
            semantic_weight=semantic_w,
            keyword_weight=keyword_w
        )
    
    def _classify_query(self, query: str) -> str:
        """Classify query type for weight selection."""
        
        query_lower = query.lower()
        
        # Check for exact match indicators
        exact_patterns = [
            r'\b[A-Z]{2,}-\d+\b',      # Error codes like E-401
            r'\b\d{5,}\b',              # Long numbers (order IDs)
            r'"[^"]+"',                 # Quoted phrases
            r'\bexact\b',               # Explicit exact request
        ]
        
        for pattern in exact_patterns:
            if re.search(pattern, query):
                return "exact"
        
        # Check for technical queries
        technical_words = [
            "error", "code", "version", "api", "endpoint",
            "function", "method", "class", "config", "setting"
        ]
        
        tech_count = sum(1 for w in technical_words if w in query_lower)
        if tech_count >= 2:
            return "technical"
        
        # Check for conceptual queries
        conceptual_words = [
            "how", "why", "what", "explain", "describe",
            "difference", "compare", "understand", "learn"
        ]
        
        concept_count = sum(1 for w in conceptual_words if w in query_lower)
        if concept_count >= 1:
            return "conceptual"
        
        return "mixed"


# Usage
retriever = AdaptiveHybridRetriever()

# Will use keyword-heavy weights
results = retriever.search_adaptive("Error E-401 fix")

# Will use semantic-heavy weights  
results = retriever.search_adaptive("How does authentication work?")

# Will use balanced weights
results = retriever.search_adaptive("authentication setup")
```

## Ensemble Hybrid Search

```python
class EnsembleHybridRetriever:
    """Combine multiple retrieval methods."""
    
    def __init__(self):
        self.client = OpenAI()
        self.chroma = chromadb.PersistentClient(path="./ensemble_db")
        self.collection = self.chroma.get_or_create_collection("ensemble_docs")
        self.documents = []
        self.bm25 = None
    
    def add_documents(self, documents: list[dict]):
        """Index documents for all methods."""
        
        self.documents = documents
        texts = [d["content"] for d in documents]
        
        # Vector index
        embeddings = self._embed_batch(texts)
        self.collection.add(
            ids=[d["id"] for d in documents],
            documents=texts,
            embeddings=embeddings,
            metadatas=[d.get("metadata", {}) for d in documents]
        )
        
        # BM25 index
        tokenized = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized)
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        methods: list[str] = None
    ) -> list[dict]:
        """
        Ensemble search using multiple methods.
        
        methods: List of methods to use ['semantic', 'bm25', 'exact']
        """
        
        if methods is None:
            methods = ["semantic", "bm25"]
        
        all_results = {}
        
        # Run each method
        if "semantic" in methods:
            semantic_results = self._semantic_search(query, n_results * 2)
            for rank, r in enumerate(semantic_results, 1):
                if r["id"] not in all_results:
                    all_results[r["id"]] = {"result": r, "scores": {}}
                all_results[r["id"]]["scores"]["semantic"] = 1 / rank
        
        if "bm25" in methods:
            bm25_results = self._bm25_search(query, n_results * 2)
            for rank, r in enumerate(bm25_results, 1):
                if r["id"] not in all_results:
                    all_results[r["id"]] = {"result": r, "scores": {}}
                all_results[r["id"]]["scores"]["bm25"] = 1 / rank
        
        if "exact" in methods:
            exact_results = self._exact_search(query, n_results * 2)
            for rank, r in enumerate(exact_results, 1):
                if r["id"] not in all_results:
                    all_results[r["id"]] = {"result": r, "scores": {}}
                all_results[r["id"]]["scores"]["exact"] = 1 / rank
        
        # Combine scores
        combined = []
        for doc_id, data in all_results.items():
            total_score = sum(data["scores"].values()) / len(methods)
            combined.append({
                **data["result"],
                "ensemble_score": total_score,
                "method_scores": data["scores"]
            })
        
        # Sort by combined score
        combined.sort(key=lambda x: x["ensemble_score"], reverse=True)
        
        return combined[:n_results]
    
    def _semantic_search(self, query: str, n_results: int) -> list[dict]:
        query_embedding = self._embed(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        return [
            {"id": results["ids"][0][i], "content": results["documents"][0][i]}
            for i in range(len(results["ids"][0]))
        ]
    
    def _bm25_search(self, query: str, n_results: int) -> list[dict]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:n_results]
        return [
            {"id": self.documents[i]["id"], "content": self.documents[i]["content"]}
            for i in top_indices if scores[i] > 0
        ]
    
    def _exact_search(self, query: str, n_results: int) -> list[dict]:
        query_lower = query.lower()
        matches = []
        for doc in self.documents:
            if query_lower in doc["content"].lower():
                matches.append(doc)
        return matches[:n_results]
    
    def _embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            input=text, model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            input=texts, model="text-embedding-3-small"
        )
        return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]
```

## Summary

```yaml
hybrid_search_benefits:
  - "Catches both exact matches and semantic similarities"
  - "More robust across query types"
  - "Handles technical and conceptual queries"

implementation_options:
  simple_fusion: "RRF combining semantic + keyword"
  bm25_hybrid: "BM25 + vector with alpha blending"
  adaptive: "Automatic weight adjustment"
  ensemble: "Multiple methods with voting"

tuning_tips:
  - "Start with 0.5/0.5 weights"
  - "Analyze failure cases"
  - "Adjust weights per query type"
  - "Test with real user queries"
```

## Next Steps

1. **Reranking** - Further improve results with cross-encoders
2. **Multi-Query Retrieval** - Handle complex questions
3. **Evaluation** - Measure retrieval quality
