# Multi-Query Retrieval

Handle complex questions by generating multiple query variations to improve recall.

## The Multi-Query Challenge

```yaml
problem:
  description: "Single queries often miss relevant documents"
  examples:
    original: "What causes climate change?"
    missed_because:
      - "Document says 'global warming factors' not 'climate change'"
      - "Document discusses 'greenhouse gases' without using query terms"
      - "Document uses technical jargon user didn't know"

solution:
  approach: "Generate multiple query variations"
  benefits:
    - "Broader coverage of relevant documents"
    - "Handles vocabulary mismatch"
    - "Captures different aspects of the question"
```

## Basic Multi-Query Retriever

```python
from openai import OpenAI
import chromadb
from typing import Optional


class MultiQueryRetriever:
    """Generate multiple queries for better retrieval."""
    
    def __init__(
        self,
        collection_name: str = "multiquery_docs",
        persist_directory: str = "./multiquery_db"
    ):
        self.client = OpenAI()
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
        num_queries: int = 3
    ) -> list[dict]:
        """Search using multiple generated queries."""
        
        # Generate query variations
        queries = self._generate_queries(query, num_queries)
        queries.insert(0, query)  # Include original
        
        print(f"Generated queries: {queries}")
        
        # Search with each query
        all_results = {}
        
        for q in queries:
            results = self._single_search(q, n_results * 2)
            
            for r in results:
                doc_id = r["id"]
                if doc_id not in all_results:
                    all_results[doc_id] = r
                    all_results[doc_id]["matching_queries"] = []
                
                all_results[doc_id]["matching_queries"].append(q)
        
        # Score by number of matching queries
        for doc_id, result in all_results.items():
            result["query_coverage"] = len(result["matching_queries"]) / len(queries)
        
        # Sort by coverage, then by score
        ranked = sorted(
            all_results.values(),
            key=lambda x: (x["query_coverage"], x["score"]),
            reverse=True
        )
        
        return ranked[:n_results]
    
    def _generate_queries(self, query: str, num_queries: int) -> list[str]:
        """Generate alternative queries using LLM."""
        
        prompt = f"""Generate {num_queries} alternative search queries for the following question.
Each query should approach the topic from a different angle or use different terminology.
Return only the queries, one per line, no numbering.

Original question: {query}

Alternative queries:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )
        
        # Parse queries
        queries = [
            q.strip() 
            for q in response.choices[0].message.content.strip().split("\n")
            if q.strip()
        ]
        
        return queries[:num_queries]
    
    def _single_search(self, query: str, n_results: int) -> list[dict]:
        """Single query search."""
        
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
                "score": 1 - results["distances"][0][i]
            }
            for i in range(len(results["ids"][0]))
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
retriever = MultiQueryRetriever()

results = retriever.search(
    "How do I fix authentication errors?",
    n_results=5,
    num_queries=3
)

# Output:
# Generated queries: [
#   "How do I fix authentication errors?",  # Original
#   "Troubleshooting login failures",
#   "Resolving credential validation issues",
#   "Authentication error solutions"
# ]
```

## Query Decomposition

Break complex questions into simpler sub-questions.

```python
class DecompositionRetriever:
    """Decompose complex queries into sub-questions."""
    
    def __init__(self):
        self.client = OpenAI()
        self.chroma = chromadb.PersistentClient(path="./decomp_db")
        self.collection = self.chroma.get_or_create_collection("decomp_docs")
    
    def search(
        self,
        query: str,
        n_results: int = 5
    ) -> dict:
        """Search with query decomposition."""
        
        # Decompose into sub-questions
        sub_questions = self._decompose(query)
        
        print(f"Sub-questions: {sub_questions}")
        
        # Search for each sub-question
        sub_results = {}
        all_docs = {}
        
        for sq in sub_questions:
            results = self._single_search(sq, n_results)
            sub_results[sq] = results
            
            # Track all unique documents
            for r in results:
                if r["id"] not in all_docs:
                    all_docs[r["id"]] = r
                    all_docs[r["id"]]["relevant_to"] = []
                all_docs[r["id"]]["relevant_to"].append(sq)
        
        # Return both sub-results and combined
        return {
            "original_query": query,
            "sub_questions": sub_questions,
            "sub_results": sub_results,
            "combined_docs": sorted(
                all_docs.values(),
                key=lambda x: len(x["relevant_to"]),
                reverse=True
            )[:n_results]
        }
    
    def _decompose(self, query: str) -> list[str]:
        """Decompose complex query into sub-questions."""
        
        prompt = f"""Break down this complex question into 2-4 simpler sub-questions that would help answer it.
Return only the sub-questions, one per line.

Complex question: {query}

Sub-questions:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3
        )
        
        questions = [
            q.strip().lstrip("0123456789.-) ")
            for q in response.choices[0].message.content.strip().split("\n")
            if q.strip()
        ]
        
        return questions
    
    def _single_search(self, query: str, n_results: int) -> list[dict]:
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
                "score": 1 - results["distances"][0][i]
            }
            for i in range(len(results["ids"][0]))
        ]


# Usage
retriever = DecompositionRetriever()

result = retriever.search(
    "Compare the advantages and disadvantages of SQL vs NoSQL databases for a high-traffic e-commerce application"
)

# Output:
# Sub-questions:
# - What are the advantages of SQL databases?
# - What are the disadvantages of SQL databases?
# - What are the advantages of NoSQL databases?
# - What are the disadvantages of NoSQL databases?
# - What database requirements does a high-traffic e-commerce application have?
```

## Step-Back Prompting

Generate broader queries to find relevant background context.

```python
class StepBackRetriever:
    """Generate step-back queries for broader context."""
    
    def __init__(self):
        self.client = OpenAI()
        self.chroma = chromadb.PersistentClient(path="./stepback_db")
        self.collection = self.chroma.get_or_create_collection("stepback_docs")
    
    def search(
        self,
        query: str,
        n_results: int = 5
    ) -> dict:
        """Search with step-back queries."""
        
        # Generate step-back query
        step_back_query = self._generate_step_back(query)
        
        print(f"Original: {query}")
        print(f"Step-back: {step_back_query}")
        
        # Search both
        specific_results = self._single_search(query, n_results)
        general_results = self._single_search(step_back_query, n_results)
        
        # Combine with deduplication
        all_docs = {}
        
        for r in specific_results:
            all_docs[r["id"]] = r
            r["source"] = "specific"
        
        for r in general_results:
            if r["id"] not in all_docs:
                all_docs[r["id"]] = r
                r["source"] = "general"
        
        return {
            "original_query": query,
            "step_back_query": step_back_query,
            "specific_results": specific_results,
            "general_results": general_results,
            "combined": list(all_docs.values())[:n_results]
        }
    
    def _generate_step_back(self, query: str) -> str:
        """Generate a more general step-back query."""
        
        prompt = f"""Given a specific question, generate a more general "step-back" question that would provide useful background context.

Examples:
Specific: "Why did my Python script fail with ImportError for pandas?"
Step-back: "How does Python's import system and package management work?"

Specific: "What's the best way to optimize a PostgreSQL query with multiple JOINs?"
Step-back: "What are the fundamentals of SQL query optimization and indexing?"

Now generate a step-back question for:
Specific: {query}
Step-back:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    
    def _single_search(self, query: str, n_results: int) -> list[dict]:
        response = self.client.embeddings.create(
            input=query, model="text-embedding-3-small"
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
                "score": 1 - results["distances"][0][i]
            }
            for i in range(len(results["ids"][0]))
        ]
```

## HyDE (Hypothetical Document Embeddings)

Generate a hypothetical answer to query with.

```python
class HyDERetriever:
    """Use hypothetical documents for better retrieval."""
    
    def __init__(self):
        self.client = OpenAI()
        self.chroma = chromadb.PersistentClient(path="./hyde_db")
        self.collection = self.chroma.get_or_create_collection("hyde_docs")
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        num_hypothetical: int = 3
    ) -> list[dict]:
        """Search using hypothetical document embeddings."""
        
        # Generate hypothetical documents
        hypotheticals = self._generate_hypothetical(query, num_hypothetical)
        
        print(f"Generated {len(hypotheticals)} hypothetical docs")
        
        # Embed all hypotheticals
        all_embeddings = self._embed_batch(hypotheticals)
        
        # Average the embeddings
        import numpy as np
        avg_embedding = np.mean(all_embeddings, axis=0).tolist()
        
        # Search with averaged embedding
        results = self.collection.query(
            query_embeddings=[avg_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        return [
            {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],
                "hypotheticals_used": hypotheticals
            }
            for i in range(len(results["ids"][0]))
        ]
    
    def _generate_hypothetical(self, query: str, num_docs: int) -> list[str]:
        """Generate hypothetical document that would answer the query."""
        
        prompt = f"""Generate {num_docs} different hypothetical passages that would answer this question.
Each passage should be written as if it were from a real document, not as a direct answer.
Make each passage approach the topic differently.
Separate passages with ---

Question: {query}

Hypothetical passages:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.8
        )
        
        passages = [
            p.strip()
            for p in response.choices[0].message.content.split("---")
            if p.strip()
        ]
        
        return passages[:num_docs]
    
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]


# Usage
retriever = HyDERetriever()

results = retriever.search(
    "How does garbage collection work in Python?",
    n_results=5,
    num_hypothetical=3
)

# The retriever generates hypothetical documents like:
# "Python uses automatic garbage collection through reference counting..."
# Then searches for real documents similar to these hypotheticals
```

## RAG Fusion

Combine multiple retrieval strategies with reciprocal rank fusion.

```python
class RAGFusionRetriever:
    """Fuse results from multiple query variations."""
    
    def __init__(self):
        self.client = OpenAI()
        self.chroma = chromadb.PersistentClient(path="./fusion_db")
        self.collection = self.chroma.get_or_create_collection("fusion_docs")
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        num_queries: int = 4
    ) -> list[dict]:
        """RAG Fusion search."""
        
        # Generate diverse queries
        queries = self._generate_diverse_queries(query, num_queries)
        queries.insert(0, query)
        
        print(f"Queries for fusion: {queries}")
        
        # Search with each query
        all_rankings = []
        
        for q in queries:
            results = self._single_search(q, n_results * 3)
            all_rankings.append(results)
        
        # Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(all_rankings)
        
        return fused[:n_results]
    
    def _generate_diverse_queries(self, query: str, num_queries: int) -> list[str]:
        """Generate diverse query variations."""
        
        prompt = f"""Generate {num_queries} diverse search queries based on the original question.
Include:
- A rephrased version
- A more specific version
- A broader version
- A version using different terminology

Original: {query}

Variations (one per line):"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )
        
        queries = [
            q.strip().lstrip("0123456789.-) ")
            for q in response.choices[0].message.content.strip().split("\n")
            if q.strip()
        ]
        
        return queries[:num_queries]
    
    def _reciprocal_rank_fusion(
        self,
        rankings: list[list[dict]],
        k: int = 60
    ) -> list[dict]:
        """Combine rankings using RRF."""
        
        fused_scores = {}
        doc_data = {}
        
        for ranking in rankings:
            for rank, doc in enumerate(ranking, 1):
                doc_id = doc["id"]
                
                # RRF score
                rrf_score = 1 / (k + rank)
                fused_scores[doc_id] = fused_scores.get(doc_id, 0) + rrf_score
                
                if doc_id not in doc_data:
                    doc_data[doc_id] = doc
        
        # Sort by fused score
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        
        return [
            {**doc_data[doc_id], "fusion_score": fused_scores[doc_id]}
            for doc_id in sorted_ids
        ]
    
    def _single_search(self, query: str, n_results: int) -> list[dict]:
        response = self.client.embeddings.create(
            input=query, model="text-embedding-3-small"
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
                "score": 1 - results["distances"][0][i]
            }
            for i in range(len(results["ids"][0]))
        ]
```

## Choosing a Strategy

```yaml
multi_query_basic:
  when: "Simple questions with vocabulary mismatch"
  latency: "Low (one LLM call + N searches)"
  complexity: "Simple"

decomposition:
  when: "Complex multi-part questions"
  latency: "Medium (one LLM call + N searches)"
  complexity: "Medium"

step_back:
  when: "Questions needing background context"
  latency: "Low (one LLM call + 2 searches)"
  complexity: "Simple"

hyde:
  when: "Short queries, technical content"
  latency: "Medium (one LLM call + embedding + search)"
  complexity: "Medium"

rag_fusion:
  when: "Maximum recall needed"
  latency: "Higher (one LLM call + N searches + fusion)"
  complexity: "Medium"
```

## Summary

```yaml
key_takeaways:
  1: "Multi-query improves recall significantly"
  2: "Different strategies for different query types"
  3: "HyDE works well for technical content"
  4: "RAG Fusion combines benefits of all approaches"

best_practices:
  - "Monitor LLM costs for query generation"
  - "Cache generated queries for common questions"
  - "Combine with reranking for best results"
  - "A/B test different strategies"
```

## Next Steps

1. **Contextual Compression** - Reduce context size
2. **Generation with Context** - Use retrieved docs in prompts
3. **RAG Evaluation** - Measure retrieval quality
