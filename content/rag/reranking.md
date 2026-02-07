# Reranking: Improving Retrieval Quality

Reranking uses more powerful models to reorder initial retrieval results for better relevance.

## Why Reranking?

```yaml
the_problem:
  bi_encoders: "Fast but limited - encode query and docs separately"
  embedding_limitations:
    - "Can't compare query-document pairs directly"
    - "Fixed representation loses nuance"
    - "May miss subtle relevance signals"

the_solution:
  cross_encoders: "Slower but more accurate - process query+doc together"
  reranking_benefit:
    - "Deep understanding of query-document relationship"
    - "Catches nuances bi-encoders miss"
    - "Significant quality improvement"

workflow:
  1: "Fast retrieval gets candidates (top 20-50)"
  2: "Cross-encoder reranks candidates"
  3: "Return reordered top-k results"
```

## Reranking with Cohere

```python
import cohere
from openai import OpenAI
import chromadb
from typing import Optional


class CohereReranker:
    """Rerank results using Cohere's reranker."""
    
    def __init__(
        self,
        cohere_api_key: str,
        collection_name: str = "rerank_docs",
        persist_directory: str = "./rerank_db"
    ):
        self.cohere = cohere.Client(cohere_api_key)
        self.openai = OpenAI()
        self.chroma = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: list[dict]):
        """Index documents."""
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
        fetch_k: int = 20
    ) -> list[dict]:
        """Search with reranking."""
        
        # Step 1: Initial retrieval (fast, less accurate)
        initial_results = self._initial_retrieve(query, fetch_k)
        
        if not initial_results:
            return []
        
        # Step 2: Rerank with cross-encoder (slower, more accurate)
        reranked = self._rerank(query, initial_results)
        
        return reranked[:n_results]
    
    def _initial_retrieve(self, query: str, n_results: int) -> list[dict]:
        """Fast initial retrieval."""
        
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
                "initial_score": 1 - results["distances"][0][i]
            }
            for i in range(len(results["ids"][0]))
        ]
    
    def _rerank(self, query: str, results: list[dict]) -> list[dict]:
        """Rerank using Cohere."""
        
        documents = [r["content"] for r in results]
        
        # Call Cohere rerank API
        rerank_response = self.cohere.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=documents,
            top_n=len(documents)
        )
        
        # Reorder results based on reranking
        reranked = []
        for item in rerank_response.results:
            original = results[item.index]
            reranked.append({
                **original,
                "rerank_score": item.relevance_score,
                "rank_change": item.index - len(reranked)
            })
        
        return reranked
    
    def _embed(self, text: str) -> list[float]:
        response = self.openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.openai.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]


# Usage
reranker = CohereReranker(cohere_api_key="your-key")

reranker.add_documents([
    {"id": "1", "content": "Python is excellent for data science and machine learning"},
    {"id": "2", "content": "Machine learning algorithms require large datasets"},
    {"id": "3", "content": "Data science combines statistics and programming"},
])

results = reranker.search(
    "best language for ML",
    n_results=3,
    fetch_k=10
)

for r in results:
    print(f"Rerank score: {r['rerank_score']:.3f}")
    print(f"Initial score: {r['initial_score']:.3f}")
    print(f"Content: {r['content'][:80]}...")
```

## Cross-Encoder Reranking

```python
from sentence_transformers import CrossEncoder
import numpy as np


class CrossEncoderReranker:
    """Rerank using local cross-encoder model."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        collection_name: str = "crossenc_docs",
        persist_directory: str = "./crossenc_db"
    ):
        self.cross_encoder = CrossEncoder(model_name)
        self.openai = OpenAI()
        self.chroma = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name
        )
    
    def add_documents(self, documents: list[dict]):
        """Index documents."""
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
        fetch_k: int = 20
    ) -> list[dict]:
        """Search with cross-encoder reranking."""
        
        # Initial retrieval
        query_embedding = self._embed(query)
        
        initial = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"]
        )
        
        if not initial["ids"][0]:
            return []
        
        # Prepare pairs for cross-encoder
        documents = initial["documents"][0]
        pairs = [[query, doc] for doc in documents]
        
        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)
        
        # Combine with results
        results = []
        for i, score in enumerate(scores):
            results.append({
                "id": initial["ids"][0][i],
                "content": documents[i],
                "metadata": initial["metadatas"][0][i],
                "initial_score": 1 - initial["distances"][0][i],
                "rerank_score": float(score)
            })
        
        # Sort by rerank score
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return results[:n_results]
    
    def _embed(self, text: str) -> list[float]:
        response = self.openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.openai.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]


# Usage
reranker = CrossEncoderReranker()

results = reranker.search(
    "How to implement neural networks?",
    n_results=5,
    fetch_k=25
)
```

## LLM-Based Reranking

```python
class LLMReranker:
    """Rerank using LLM scoring."""
    
    def __init__(self):
        self.client = OpenAI()
        self.chroma = chromadb.PersistentClient(path="./llm_rerank_db")
        self.collection = self.chroma.get_or_create_collection("llm_rerank_docs")
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        fetch_k: int = 15
    ) -> list[dict]:
        """Search with LLM-based reranking."""
        
        # Initial retrieval
        initial = self._initial_retrieve(query, fetch_k)
        
        if not initial:
            return []
        
        # Score each document with LLM
        scored = []
        for doc in initial:
            score = self._llm_score(query, doc["content"])
            scored.append({
                **doc,
                "llm_score": score
            })
        
        # Sort by LLM score
        scored.sort(key=lambda x: x["llm_score"], reverse=True)
        
        return scored[:n_results]
    
    def _llm_score(self, query: str, document: str) -> float:
        """Get relevance score from LLM."""
        
        prompt = f"""Rate the relevance of this document to the query.
Return only a number from 0 to 10, where:
- 0 = completely irrelevant
- 5 = somewhat relevant
- 10 = highly relevant

Query: {query}

Document: {document[:500]}

Relevance score (0-10):"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
            return min(10, max(0, score)) / 10  # Normalize to 0-1
        except:
            return 0.5  # Default if parsing fails
    
    def _initial_retrieve(self, query: str, n_results: int) -> list[dict]:
        """Fast initial retrieval."""
        
        response = self.client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        
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
                "initial_score": 1 - results["distances"][0][i]
            }
            for i in range(len(results["ids"][0]))
        ]


# Batch LLM reranking (more efficient)
class BatchLLMReranker(LLMReranker):
    """Batch documents for efficient LLM reranking."""
    
    def _llm_score_batch(
        self,
        query: str,
        documents: list[str]
    ) -> list[float]:
        """Score multiple documents in one LLM call."""
        
        # Format documents for batch scoring
        doc_list = "\n".join(
            f"[{i+1}] {doc[:200]}..."
            for i, doc in enumerate(documents)
        )
        
        prompt = f"""Rate the relevance of each document to the query.
Return scores as comma-separated numbers from 0-10.

Query: {query}

Documents:
{doc_list}

Scores (comma-separated, one per document):"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0
        )
        
        # Parse scores
        try:
            scores_text = response.choices[0].message.content.strip()
            scores = [float(s.strip()) / 10 for s in scores_text.split(",")]
            
            # Ensure we have the right number of scores
            while len(scores) < len(documents):
                scores.append(0.5)
            
            return scores[:len(documents)]
        except:
            return [0.5] * len(documents)
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        fetch_k: int = 15
    ) -> list[dict]:
        """Search with batch LLM reranking."""
        
        initial = self._initial_retrieve(query, fetch_k)
        
        if not initial:
            return []
        
        # Batch score
        documents = [r["content"] for r in initial]
        scores = self._llm_score_batch(query, documents)
        
        # Combine
        for i, result in enumerate(initial):
            result["llm_score"] = scores[i]
        
        # Sort and return
        initial.sort(key=lambda x: x["llm_score"], reverse=True)
        return initial[:n_results]
```

## Cascading Rerankers

```python
class CascadingReranker:
    """Multiple reranking stages for optimal quality/speed."""
    
    def __init__(self):
        self.client = OpenAI()
        self.chroma = chromadb.PersistentClient(path="./cascade_db")
        self.collection = self.chroma.get_or_create_collection("cascade_docs")
        
        # Fast cross-encoder for stage 1
        self.fast_reranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2")
        # Accurate cross-encoder for stage 2
        self.accurate_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        stage1_k: int = 50,
        stage2_k: int = 20
    ) -> list[dict]:
        """
        Three-stage retrieval:
        1. Fast vector search (100 -> 50)
        2. Fast reranker (50 -> 20)
        3. Accurate reranker (20 -> final)
        """
        
        # Stage 0: Initial vector retrieval
        initial = self._vector_retrieve(query, stage1_k)
        
        if not initial:
            return []
        
        # Stage 1: Fast reranking
        stage1_results = self._rerank(
            query,
            initial,
            self.fast_reranker
        )[:stage2_k]
        
        # Stage 2: Accurate reranking
        final_results = self._rerank(
            query,
            stage1_results,
            self.accurate_reranker
        )[:n_results]
        
        return final_results
    
    def _vector_retrieve(self, query: str, n_results: int) -> list[dict]:
        """Fast vector retrieval."""
        
        response = self.client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        
        results = self.collection.query(
            query_embeddings=[response.data[0].embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        return [
            {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "vector_score": 1 - results["distances"][0][i]
            }
            for i in range(len(results["ids"][0]))
        ]
    
    def _rerank(
        self,
        query: str,
        results: list[dict],
        model: CrossEncoder
    ) -> list[dict]:
        """Rerank with given model."""
        
        pairs = [[query, r["content"]] for r in results]
        scores = model.predict(pairs)
        
        for i, score in enumerate(scores):
            results[i]["rerank_score"] = float(score)
        
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results
```

## Choosing Reranking Strategies

```yaml
cohere_rerank:
  pros: "Easy API, high quality, no GPU needed"
  cons: "API cost, latency, external dependency"
  best_for: "Production systems with budget"

cross_encoder_local:
  pros: "Free after download, low latency, privacy"
  cons: "Needs GPU for speed, less accurate than large models"
  best_for: "Privacy-sensitive, high-volume applications"

llm_reranking:
  pros: "Very accurate, handles nuance"
  cons: "Expensive, slow, overkill for simple queries"
  best_for: "Complex queries, low volume"

cascading:
  pros: "Optimal quality/cost balance"
  cons: "Complex to tune"
  best_for: "Large-scale production systems"
```

## Performance Comparison

```yaml
typical_improvements:
  without_reranking:
    mrr: "0.65-0.75"
    precision_at_5: "0.60-0.70"
  
  with_cross_encoder:
    mrr: "0.75-0.85"
    precision_at_5: "0.70-0.80"
    improvement: "10-15%"
  
  with_cohere:
    mrr: "0.80-0.90"
    precision_at_5: "0.75-0.85"
    improvement: "15-25%"

latency_impact:
  vector_only: "50-100ms"
  plus_cross_encoder: "100-200ms"
  plus_cohere_api: "200-400ms"
  plus_llm: "500-1500ms"
```

## Summary

```yaml
key_takeaways:
  1: "Reranking significantly improves relevance"
  2: "Trade-off between quality and latency"
  3: "Cohere is easy and effective"
  4: "Local cross-encoders for privacy/volume"
  5: "Cascading for optimal balance"

implementation_tips:
  - "Start with 20-50 candidates for reranking"
  - "Monitor latency impact"
  - "A/B test reranking improvements"
  - "Consider caching rerank results"
```

## Next Steps

1. **Multi-Query Retrieval** - Handle complex questions
2. **Contextual Compression** - Reduce context size
3. **RAG Evaluation** - Measure improvements
