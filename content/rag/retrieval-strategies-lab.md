# Lab: Retrieval Strategies Comparison

Build and compare different retrieval strategies to understand their trade-offs.

## Lab Overview

```yaml
objective: "Implement and compare retrieval strategies"
difficulty: "Intermediate"
time: "45-60 minutes"

what_you_will_learn:
  - "Basic similarity search implementation"
  - "Hybrid search combining keyword and semantic"
  - "Multi-query retrieval for better recall"
  - "Reranking for improved precision"
  - "Evaluating retrieval quality"

prerequisites:
  - "Understanding of embeddings"
  - "Familiarity with vector databases"
  - "Python basics"
```

## Setup

```python
# Install required packages
# pip install openai chromadb rank-bm25 numpy

from openai import OpenAI
import chromadb
from rank_bm25 import BM25Okapi
import numpy as np
from typing import Optional
import re
import json
import time


# Initialize clients
client = OpenAI()
chroma = chromadb.Client()

# Sample knowledge base about a fictional tech company
DOCUMENTS = [
    {
        "id": "doc1",
        "content": "TechCorp's vacation policy allows employees to take up to 20 days of paid time off per year. Unused vacation days can be carried over to the next year, up to a maximum of 5 days.",
        "metadata": {"category": "hr", "topic": "vacation"}
    },
    {
        "id": "doc2", 
        "content": "The company provides comprehensive health insurance including medical, dental, and vision coverage. Employees can add dependents to their plan during open enrollment.",
        "metadata": {"category": "hr", "topic": "benefits"}
    },
    {
        "id": "doc3",
        "content": "Remote work is supported for all engineering positions. Employees must be available during core hours (10am-3pm) and attend weekly team meetings.",
        "metadata": {"category": "hr", "topic": "remote-work"}
    },
    {
        "id": "doc4",
        "content": "The API rate limit is 1000 requests per minute for standard tier users. Enterprise customers can request higher limits by contacting support.",
        "metadata": {"category": "technical", "topic": "api"}
    },
    {
        "id": "doc5",
        "content": "Authentication uses OAuth 2.0 with JWT tokens. Tokens expire after 1 hour and can be refreshed using the refresh token endpoint.",
        "metadata": {"category": "technical", "topic": "auth"}
    },
    {
        "id": "doc6",
        "content": "Error code E-401 indicates an authentication failure. This typically occurs when the token is expired or invalid. Obtain a new token to resolve.",
        "metadata": {"category": "technical", "topic": "errors"}
    },
    {
        "id": "doc7",
        "content": "The onboarding process takes approximately two weeks. New employees receive their equipment on day one and complete compliance training by day five.",
        "metadata": {"category": "hr", "topic": "onboarding"}
    },
    {
        "id": "doc8",
        "content": "PTO requests must be submitted at least one week in advance through the HR portal. Managers approve requests within 48 hours.",
        "metadata": {"category": "hr", "topic": "vacation"}
    },
    {
        "id": "doc9",
        "content": "Database queries should use pagination for large result sets. The maximum page size is 100 records. Use cursor-based pagination for best performance.",
        "metadata": {"category": "technical", "topic": "database"}
    },
    {
        "id": "doc10",
        "content": "Employee referral bonuses are $5000 for engineering positions and $3000 for other roles. Bonuses are paid after the referred employee completes 90 days.",
        "metadata": {"category": "hr", "topic": "referrals"}
    }
]

print(f"Loaded {len(DOCUMENTS)} documents for retrieval testing")
```

## Part 1: Basic Similarity Search

```python
class BasicRetriever:
    """Simple vector similarity retriever."""
    
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.collection = chroma.get_or_create_collection(
            name="basic_retriever",
            metadata={"hnsw:space": "cosine"}
        )
        self._index_documents()
    
    def _index_documents(self):
        """Index documents with embeddings."""
        
        texts = [d["content"] for d in self.documents]
        embeddings = self._embed_batch(texts)
        
        self.collection.add(
            ids=[d["id"] for d in self.documents],
            documents=texts,
            embeddings=embeddings,
            metadatas=[d["metadata"] for d in self.documents]
        )
        print(f"Indexed {len(self.documents)} documents")
    
    def search(self, query: str, k: int = 3) -> list[dict]:
        """Search for similar documents."""
        
        query_embedding = self._embed(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        return [
            {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]
            }
            for i in range(len(results["ids"][0]))
        ]
    
    def _embed(self, text: str) -> list[float]:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]


# Test basic retriever
print("\n=== Testing Basic Retriever ===")
basic_retriever = BasicRetriever(DOCUMENTS)

test_queries = [
    "How many vacation days do I get?",
    "What is error E-401?",
    "How does authentication work?"
]

for query in test_queries:
    print(f"\nQuery: {query}")
    results = basic_retriever.search(query, k=3)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['content'][:60]}...")
```

## Part 2: Hybrid Search with BM25

```python
class HybridRetriever:
    """Combine vector similarity with BM25 keyword search."""
    
    def __init__(self, documents: list[dict]):
        self.documents = documents
        
        # Vector index
        self.collection = chroma.get_or_create_collection(
            name="hybrid_retriever",
            metadata={"hnsw:space": "cosine"}
        )
        
        # BM25 index
        texts = [d["content"] for d in documents]
        tokenized = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        
        # Index vectors
        embeddings = self._embed_batch(texts)
        self.collection.add(
            ids=[d["id"] for d in documents],
            documents=texts,
            embeddings=embeddings,
            metadatas=[d["metadata"] for d in documents]
        )
        print("Built hybrid index (vector + BM25)")
    
    def search(
        self,
        query: str,
        k: int = 3,
        alpha: float = 0.5
    ) -> list[dict]:
        """
        Hybrid search with configurable balance.
        alpha: 0=BM25 only, 1=vector only, 0.5=balanced
        """
        
        # Vector scores
        query_embedding = self._embed(query)
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=len(self.documents),
            include=["distances"]
        )
        
        vector_scores = {}
        for i, doc_id in enumerate(vector_results["ids"][0]):
            vector_scores[doc_id] = 1 - vector_results["distances"][0][i]
        
        # BM25 scores
        tokenized_query = self._tokenize(query)
        bm25_scores_raw = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores
        max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1
        bm25_scores = {}
        for i, doc in enumerate(self.documents):
            bm25_scores[doc["id"]] = bm25_scores_raw[i] / max_bm25
        
        # Combine scores
        combined = []
        for doc in self.documents:
            doc_id = doc["id"]
            v_score = vector_scores.get(doc_id, 0)
            b_score = bm25_scores.get(doc_id, 0)
            combined_score = alpha * v_score + (1 - alpha) * b_score
            
            combined.append({
                "id": doc_id,
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": combined_score,
                "vector_score": v_score,
                "bm25_score": b_score
            })
        
        # Sort and return top k
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:k]
    
    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()
    
    def _embed(self, text: str) -> list[float]:
        response = client.embeddings.create(
            input=text, model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(
            input=texts, model="text-embedding-3-small"
        )
        return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]


# Test hybrid retriever
print("\n=== Testing Hybrid Retriever ===")
hybrid_retriever = HybridRetriever(DOCUMENTS)

# Test with different alpha values
query = "E-401 authentication error"

print(f"\nQuery: {query}")
print("\nVector-only (alpha=1.0):")
for r in hybrid_retriever.search(query, k=3, alpha=1.0):
    print(f"  [{r['score']:.3f}] {r['content'][:60]}...")

print("\nBM25-only (alpha=0.0):")
for r in hybrid_retriever.search(query, k=3, alpha=0.0):
    print(f"  [{r['score']:.3f}] {r['content'][:60]}...")

print("\nHybrid (alpha=0.5):")
for r in hybrid_retriever.search(query, k=3, alpha=0.5):
    print(f"  [{r['score']:.3f}] {r['content'][:60]}...")
```

## Part 3: Multi-Query Retrieval

```python
class MultiQueryRetriever:
    """Generate multiple query variations for better recall."""
    
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.collection = chroma.get_or_create_collection(
            name="multiquery_retriever",
            metadata={"hnsw:space": "cosine"}
        )
        
        texts = [d["content"] for d in documents]
        embeddings = self._embed_batch(texts)
        
        self.collection.add(
            ids=[d["id"] for d in documents],
            documents=texts,
            embeddings=embeddings,
            metadatas=[d["metadata"] for d in documents]
        )
        print("Built multi-query retriever index")
    
    def search(
        self,
        query: str,
        k: int = 3,
        num_queries: int = 3
    ) -> list[dict]:
        """Search with multiple query variations."""
        
        # Generate variations
        queries = self._generate_variations(query, num_queries)
        queries.insert(0, query)  # Include original
        
        print(f"  Generated queries: {queries}")
        
        # Search with each query
        all_results = {}
        
        for q in queries:
            results = self._single_search(q, k * 2)
            for r in results:
                if r["id"] not in all_results:
                    all_results[r["id"]] = {
                        **r,
                        "query_matches": 0,
                        "best_score": 0
                    }
                all_results[r["id"]]["query_matches"] += 1
                all_results[r["id"]]["best_score"] = max(
                    all_results[r["id"]]["best_score"],
                    r["score"]
                )
        
        # Rank by query coverage and score
        ranked = list(all_results.values())
        ranked.sort(
            key=lambda x: (x["query_matches"], x["best_score"]),
            reverse=True
        )
        
        return ranked[:k]
    
    def _generate_variations(self, query: str, num: int) -> list[str]:
        """Generate query variations with LLM."""
        
        prompt = f"""Generate {num} alternative search queries for this question.
Use different wording and terminology. One per line, no numbering.

Question: {query}

Alternatives:"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        
        variations = [
            v.strip()
            for v in response.choices[0].message.content.strip().split("\n")
            if v.strip()
        ]
        
        return variations[:num]
    
    def _single_search(self, query: str, k: int) -> list[dict]:
        query_embedding = self._embed(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        return [
            {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]
            }
            for i in range(len(results["ids"][0]))
        ]
    
    def _embed(self, text: str) -> list[float]:
        response = client.embeddings.create(
            input=text, model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(
            input=texts, model="text-embedding-3-small"
        )
        return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]


# Test multi-query retriever
print("\n=== Testing Multi-Query Retriever ===")
multiquery_retriever = MultiQueryRetriever(DOCUMENTS)

query = "How do I take time off work?"
print(f"\nQuery: {query}")
results = multiquery_retriever.search(query, k=3, num_queries=3)

for r in results:
    print(f"  [matches: {r['query_matches']}, score: {r['best_score']:.3f}] {r['content'][:60]}...")
```

## Part 4: LLM Reranking

```python
class RerankedRetriever:
    """Use LLM to rerank initial results."""
    
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.collection = chroma.get_or_create_collection(
            name="reranked_retriever",
            metadata={"hnsw:space": "cosine"}
        )
        
        texts = [d["content"] for d in documents]
        embeddings = self._embed_batch(texts)
        
        self.collection.add(
            ids=[d["id"] for d in documents],
            documents=texts,
            embeddings=embeddings,
            metadatas=[d["metadata"] for d in documents]
        )
        print("Built reranking retriever index")
    
    def search(
        self,
        query: str,
        k: int = 3,
        fetch_k: int = 6
    ) -> list[dict]:
        """Search with LLM reranking."""
        
        # Initial retrieval
        initial_results = self._initial_search(query, fetch_k)
        
        print(f"  Initial results: {[r['id'] for r in initial_results]}")
        
        # Rerank with LLM
        reranked = self._rerank(query, initial_results)
        
        print(f"  After reranking: {[r['id'] for r in reranked]}")
        
        return reranked[:k]
    
    def _initial_search(self, query: str, k: int) -> list[dict]:
        query_embedding = self._embed(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        return [
            {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "initial_score": 1 - results["distances"][0][i]
            }
            for i in range(len(results["ids"][0]))
        ]
    
    def _rerank(self, query: str, results: list[dict]) -> list[dict]:
        """Rerank results using LLM."""
        
        # Build document list for LLM
        doc_list = "\n".join(
            f"[{i+1}] {r['content']}"
            for i, r in enumerate(results)
        )
        
        prompt = f"""Rate the relevance of each document to the query.
Return comma-separated scores from 0-10 (one per document).

Query: {query}

Documents:
{doc_list}

Relevance scores (comma-separated):"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0
        )
        
        # Parse scores
        try:
            scores = [
                float(s.strip()) / 10
                for s in response.choices[0].message.content.strip().split(",")
            ]
        except:
            scores = [0.5] * len(results)
        
        # Add scores and sort
        for i, result in enumerate(results):
            result["rerank_score"] = scores[i] if i < len(scores) else 0.5
        
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results
    
    def _embed(self, text: str) -> list[float]:
        response = client.embeddings.create(
            input=text, model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(
            input=texts, model="text-embedding-3-small"
        )
        return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]


# Test reranking retriever
print("\n=== Testing Reranking Retriever ===")
reranked_retriever = RerankedRetriever(DOCUMENTS)

query = "token expiration authentication"
print(f"\nQuery: {query}")
results = reranked_retriever.search(query, k=3, fetch_k=6)

for r in results:
    print(f"  [rerank: {r['rerank_score']:.2f}, initial: {r['initial_score']:.3f}] {r['content'][:60]}...")
```

## Part 5: Evaluation Framework

```python
class RetrievalEvaluator:
    """Evaluate and compare retrieval strategies."""
    
    def __init__(self):
        self.results = {}
    
    def evaluate(
        self,
        retriever_name: str,
        retriever,
        test_cases: list[dict],
        k: int = 3
    ) -> dict:
        """Evaluate retriever on test cases."""
        
        metrics = {
            "precision": [],
            "recall": [],
            "mrr": [],  # Mean Reciprocal Rank
            "latency": []
        }
        
        for test in test_cases:
            query = test["query"]
            relevant_ids = test["relevant_ids"]
            
            # Measure latency
            start = time.time()
            results = retriever.search(query, k=k)
            latency = time.time() - start
            
            retrieved_ids = [r["id"] for r in results]
            
            # Calculate metrics
            relevant_retrieved = set(retrieved_ids) & set(relevant_ids)
            
            precision = len(relevant_retrieved) / k if k > 0 else 0
            recall = len(relevant_retrieved) / len(relevant_ids) if relevant_ids else 0
            
            # MRR
            mrr = 0
            for i, rid in enumerate(retrieved_ids):
                if rid in relevant_ids:
                    mrr = 1 / (i + 1)
                    break
            
            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            metrics["mrr"].append(mrr)
            metrics["latency"].append(latency)
        
        # Calculate averages
        summary = {
            metric: sum(values) / len(values) if values else 0
            for metric, values in metrics.items()
        }
        
        self.results[retriever_name] = summary
        return summary
    
    def compare(self):
        """Print comparison of all evaluated retrievers."""
        
        print("\n" + "=" * 60)
        print("RETRIEVAL STRATEGY COMPARISON")
        print("=" * 60)
        print(f"{'Strategy':<20} {'Precision':<12} {'Recall':<12} {'MRR':<12} {'Latency':<12}")
        print("-" * 60)
        
        for name, metrics in self.results.items():
            print(
                f"{name:<20} "
                f"{metrics['precision']:.3f}        "
                f"{metrics['recall']:.3f}        "
                f"{metrics['mrr']:.3f}        "
                f"{metrics['latency']*1000:.1f}ms"
            )


# Test cases with known relevant documents
TEST_CASES = [
    {
        "query": "How many vacation days can I take?",
        "relevant_ids": ["doc1", "doc8"]
    },
    {
        "query": "What is error code E-401?",
        "relevant_ids": ["doc5", "doc6"]
    },
    {
        "query": "How does the API authentication work?",
        "relevant_ids": ["doc5", "doc6"]
    },
    {
        "query": "Can I work from home?",
        "relevant_ids": ["doc3"]
    },
    {
        "query": "What health benefits are available?",
        "relevant_ids": ["doc2"]
    }
]

# Evaluate all retrievers
print("\n=== Evaluating Retrieval Strategies ===")
evaluator = RetrievalEvaluator()

# Note: We need to recreate retrievers due to collection naming
evaluator.evaluate("Basic", BasicRetriever(DOCUMENTS), TEST_CASES)
evaluator.evaluate("Hybrid", HybridRetriever(DOCUMENTS), TEST_CASES)
evaluator.evaluate("Multi-Query", MultiQueryRetriever(DOCUMENTS), TEST_CASES)
evaluator.evaluate("Reranked", RerankedRetriever(DOCUMENTS), TEST_CASES)

evaluator.compare()
```

## Exercises

```yaml
exercise_1:
  title: "Tune Hybrid Alpha"
  task: "Find the optimal alpha value for the hybrid retriever"
  hint: "Try values from 0.0 to 1.0 in 0.1 increments"

exercise_2:
  title: "Combine Strategies"
  task: "Create a retriever that uses multi-query + reranking"
  hint: "Generate queries first, then rerank the combined results"

exercise_3:
  title: "Add Filtering"
  task: "Add category filtering to the basic retriever"
  hint: "Use ChromaDB's where parameter"

exercise_4:
  title: "Custom Evaluation"
  task: "Add NDCG (Normalized Discounted Cumulative Gain) metric"
  hint: "NDCG considers the position of relevant documents"
```

## Summary

```yaml
key_learnings:
  1: "Basic similarity search is fast but may miss relevant docs"
  2: "Hybrid search helps with exact term matching"
  3: "Multi-query improves recall for ambiguous queries"
  4: "Reranking improves precision at cost of latency"

trade_offs:
  basic: "Fast, simple, good baseline"
  hybrid: "Better for technical content with specific terms"
  multi_query: "Higher recall, more API calls"
  reranking: "Best precision, slowest"

recommendations:
  - "Start with basic retrieval"
  - "Add hybrid for technical/code content"
  - "Use multi-query for complex questions"
  - "Apply reranking for high-stakes applications"
```
