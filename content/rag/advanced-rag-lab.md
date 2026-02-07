# Advanced RAG Lab: Building a Production-Ready System

Build a complete RAG system with advanced patterns and evaluation.

## Lab Overview

```yaml
objectives:
  - "Implement HyDE and RAG Fusion"
  - "Build production-grade retrieval"
  - "Add comprehensive evaluation"
  - "Create end-to-end pipeline"

difficulty: Advanced
time: 90 minutes
prerequisites:
  - "Basic RAG implementation"
  - "Vector databases"
  - "Python async/await"
```

## Part 1: Setup and Data Preparation

```python
"""
Advanced RAG Lab
Build a production-ready RAG system with advanced patterns
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator
from openai import OpenAI
import chromadb
import numpy as np


# Configuration
@dataclass
class LabConfig:
    """Lab configuration."""
    
    # Models
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    
    # Retrieval
    retrieval_k: int = 5
    similarity_threshold: float = 0.5
    
    # Advanced patterns
    hyde_enabled: bool = True
    hyde_num_docs: int = 3
    fusion_enabled: bool = True
    fusion_num_queries: int = 4
    
    # Generation
    temperature: float = 0
    max_tokens: int = 500


# Initialize
client = OpenAI()
chroma = chromadb.PersistentClient(path="./advanced_rag_lab")
config = LabConfig()


# Sample knowledge base (technical documentation)
SAMPLE_DOCS = [
    {
        "id": "auth_001",
        "title": "Authentication Overview",
        "content": """Authentication in our API uses JWT (JSON Web Tokens). 
        To authenticate, send a POST request to /api/auth/login with email and password.
        The response includes an access_token (valid for 1 hour) and refresh_token (valid for 7 days).
        Include the access token in the Authorization header: Bearer <token>.""",
        "category": "authentication"
    },
    {
        "id": "auth_002",
        "title": "Token Refresh",
        "content": """When your access token expires, use the refresh token to get a new one.
        POST to /api/auth/refresh with the refresh_token in the request body.
        You'll receive new access and refresh tokens. Store them securely.
        If the refresh token is expired, the user must log in again.""",
        "category": "authentication"
    },
    {
        "id": "rate_001",
        "title": "Rate Limiting",
        "content": """API rate limits are enforced per API key. Free tier: 100 requests/minute.
        Pro tier: 1000 requests/minute. Enterprise: Custom limits.
        Rate limit headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset.
        When rate limited, you'll receive a 429 status code. Implement exponential backoff.""",
        "category": "rate_limits"
    },
    {
        "id": "error_001",
        "title": "Error Handling",
        "content": """API errors follow a consistent format: {"error": {"code": "ERROR_CODE", "message": "Description"}}.
        Common error codes: INVALID_INPUT (400), UNAUTHORIZED (401), NOT_FOUND (404), RATE_LIMITED (429).
        Always check the error code and message. Implement retry logic for transient errors (5xx).""",
        "category": "errors"
    },
    {
        "id": "webhook_001",
        "title": "Webhooks",
        "content": """Webhooks notify your server of events in real-time. 
        Configure webhooks in the dashboard under Settings > Webhooks.
        Events include: user.created, order.completed, payment.failed.
        Webhook payloads are signed with HMAC-SHA256. Verify signatures to ensure authenticity.""",
        "category": "webhooks"
    },
    {
        "id": "pagination_001",
        "title": "Pagination",
        "content": """List endpoints support cursor-based pagination. 
        Parameters: limit (default 20, max 100), cursor (from previous response).
        Response includes: data array, has_more boolean, next_cursor string.
        For reliable pagination, always use cursors instead of offset-based pagination.""",
        "category": "pagination"
    },
    {
        "id": "search_001",
        "title": "Search API",
        "content": """The Search API supports full-text search across your data.
        POST to /api/search with query string and optional filters.
        Filters support: eq (equals), gt/lt (greater/less than), in (array match).
        Results are ranked by relevance. Use highlight=true to get matched snippets.""",
        "category": "search"
    },
    {
        "id": "upload_001",
        "title": "File Uploads",
        "content": """Upload files using multipart/form-data to /api/files/upload.
        Maximum file size: 10MB (Free), 100MB (Pro), 1GB (Enterprise).
        Supported formats: images (jpg, png, gif), documents (pdf, docx), data (csv, json).
        Uploads return a file_id. Use this ID to reference the file in other API calls.""",
        "category": "files"
    }
]


# Evaluation test cases
TEST_CASES = [
    {
        "query": "How do I authenticate with the API?",
        "relevant_docs": ["auth_001", "auth_002"],
        "ground_truth": "Use JWT authentication by POSTing to /api/auth/login with email and password. Include the returned access_token in the Authorization header."
    },
    {
        "query": "What happens when I hit the rate limit?",
        "relevant_docs": ["rate_001"],
        "ground_truth": "You receive a 429 status code. Check rate limit headers and implement exponential backoff."
    },
    {
        "query": "How do I paginate through results?",
        "relevant_docs": ["pagination_001"],
        "ground_truth": "Use cursor-based pagination with limit and cursor parameters. Check has_more and use next_cursor for subsequent requests."
    },
    {
        "query": "How do I search with filters?",
        "relevant_docs": ["search_001"],
        "ground_truth": "POST to /api/search with query and filters. Filters support eq, gt/lt, and in operators."
    },
    {
        "query": "What's the max file size I can upload?",
        "relevant_docs": ["upload_001"],
        "ground_truth": "10MB for Free tier, 100MB for Pro, and 1GB for Enterprise."
    }
]


def setup_collection():
    """Create and populate the collection."""
    
    # Delete if exists
    try:
        chroma.delete_collection("advanced_rag_docs")
    except:
        pass
    
    collection = chroma.create_collection(
        name="advanced_rag_docs",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Embed and add documents
    for doc in SAMPLE_DOCS:
        response = client.embeddings.create(
            input=doc["content"],
            model=config.embedding_model
        )
        
        collection.add(
            ids=[doc["id"]],
            documents=[doc["content"]],
            metadatas=[{
                "title": doc["title"],
                "category": doc["category"]
            }],
            embeddings=[response.data[0].embedding]
        )
    
    print(f"✓ Added {len(SAMPLE_DOCS)} documents to collection")
    return collection


# Run setup
collection = setup_collection()
```

## Part 2: Basic Retrieval (Baseline)

```python
class BasicRetriever:
    """Basic semantic search retriever (baseline)."""
    
    def __init__(self, collection, config: LabConfig):
        self.collection = collection
        self.config = config
        self.client = OpenAI()
    
    def retrieve(self, query: str, k: int = None) -> list[dict]:
        """Simple semantic search."""
        
        k = k or self.config.retrieval_k
        
        # Embed query
        response = self.client.embeddings.create(
            input=query,
            model=self.config.embedding_model
        )
        
        # Search
        results = self.collection.query(
            query_embeddings=[response.data[0].embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        return [
            {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "score": 1 - results["distances"][0][i],
                "metadata": results["metadatas"][0][i]
            }
            for i in range(len(results["ids"][0]))
        ]


# Test baseline
print("=== Baseline Retrieval ===\n")

baseline = BasicRetriever(collection, config)

test_query = "How do I authenticate?"
results = baseline.retrieve(test_query)

print(f"Query: {test_query}\n")
for i, doc in enumerate(results[:3]):
    print(f"{i+1}. [{doc['id']}] Score: {doc['score']:.3f}")
    print(f"   {doc['content'][:100]}...\n")
```

## Part 3: HyDE Retriever

```python
class HyDERetriever:
    """Hypothetical Document Embeddings retriever."""
    
    def __init__(self, collection, config: LabConfig):
        self.collection = collection
        self.config = config
        self.client = OpenAI()
    
    def retrieve(self, query: str, k: int = None) -> list[dict]:
        """Search using hypothetical document embeddings."""
        
        k = k or self.config.retrieval_k
        
        # Generate hypothetical documents
        hypotheticals = self._generate_hypotheticals(query)
        
        print(f"  Generated {len(hypotheticals)} hypothetical docs")
        
        # Embed hypotheticals
        embeddings = []
        for hypo in hypotheticals:
            response = self.client.embeddings.create(
                input=hypo,
                model=self.config.embedding_model
            )
            embeddings.append(response.data[0].embedding)
        
        # Average embeddings
        avg_embedding = np.mean(embeddings, axis=0).tolist()
        
        # Search with averaged embedding
        results = self.collection.query(
            query_embeddings=[avg_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        return [
            {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "score": 1 - results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
                "method": "hyde"
            }
            for i in range(len(results["ids"][0]))
        ]
    
    def _generate_hypotheticals(self, query: str) -> list[str]:
        """Generate hypothetical answer documents."""
        
        prompt = f"""Generate {self.config.hyde_num_docs} hypothetical documentation excerpts that would answer this question.
Write as if excerpts from API documentation. Keep each under 100 words. Separate with ---

Question: {query}

Hypothetical documentation excerpts:"""
        
        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400
        )
        
        passages = [
            p.strip()
            for p in response.choices[0].message.content.split("---")
            if p.strip()
        ]
        
        return passages[:self.config.hyde_num_docs]


# Test HyDE
print("\n=== HyDE Retrieval ===\n")

hyde = HyDERetriever(collection, config)

results = hyde.retrieve(test_query)

print(f"\nQuery: {test_query}\n")
for i, doc in enumerate(results[:3]):
    print(f"{i+1}. [{doc['id']}] Score: {doc['score']:.3f}")
    print(f"   {doc['content'][:100]}...\n")
```

## Part 4: RAG Fusion Retriever

```python
class RAGFusionRetriever:
    """RAG Fusion with multi-query and RRF."""
    
    def __init__(self, collection, config: LabConfig):
        self.collection = collection
        self.config = config
        self.client = OpenAI()
    
    def retrieve(self, query: str, k: int = None) -> list[dict]:
        """Multi-query retrieval with fusion."""
        
        k = k or self.config.retrieval_k
        
        # Generate query variations
        queries = self._generate_queries(query)
        queries.insert(0, query)  # Include original
        
        print(f"  Queries: {queries}")
        
        # Search with each query
        all_rankings = []
        
        for q in queries:
            response = self.client.embeddings.create(
                input=q,
                model=self.config.embedding_model
            )
            
            results = self.collection.query(
                query_embeddings=[response.data[0].embedding],
                n_results=k * 2,
                include=["documents", "metadatas", "distances"]
            )
            
            ranking = [
                {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "score": 1 - results["distances"][0][i],
                    "metadata": results["metadatas"][0][i]
                }
                for i in range(len(results["ids"][0]))
            ]
            
            all_rankings.append(ranking)
        
        # Reciprocal Rank Fusion
        return self._rrf_fusion(all_rankings, k)
    
    def _generate_queries(self, query: str) -> list[str]:
        """Generate query variations."""
        
        prompt = f"""Generate {self.config.fusion_num_queries} different search queries for finding documentation about this topic.
Include: rephrased version, more specific version, related concept.
One query per line, no numbering.

Original: {query}

Variations:"""
        
        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        
        queries = [
            q.strip().lstrip("0123456789.-) ")
            for q in response.choices[0].message.content.strip().split("\n")
            if q.strip()
        ]
        
        return queries[:self.config.fusion_num_queries]
    
    def _rrf_fusion(
        self,
        rankings: list[list[dict]],
        k: int,
        rrf_k: int = 60
    ) -> list[dict]:
        """Reciprocal Rank Fusion."""
        
        fused_scores = {}
        doc_data = {}
        
        for ranking in rankings:
            for rank, doc in enumerate(ranking, 1):
                doc_id = doc["id"]
                
                # RRF formula
                rrf_score = 1 / (rrf_k + rank)
                fused_scores[doc_id] = fused_scores.get(doc_id, 0) + rrf_score
                
                if doc_id not in doc_data:
                    doc_data[doc_id] = doc
        
        # Sort by fused score
        sorted_ids = sorted(
            fused_scores.keys(),
            key=lambda x: fused_scores[x],
            reverse=True
        )
        
        return [
            {
                **doc_data[doc_id],
                "fusion_score": fused_scores[doc_id],
                "method": "fusion"
            }
            for doc_id in sorted_ids[:k]
        ]


# Test RAG Fusion
print("\n=== RAG Fusion Retrieval ===\n")

fusion = RAGFusionRetriever(collection, config)

results = fusion.retrieve(test_query)

print(f"\nQuery: {test_query}\n")
for i, doc in enumerate(results[:3]):
    print(f"{i+1}. [{doc['id']}] Fusion Score: {doc['fusion_score']:.4f}")
    print(f"   {doc['content'][:100]}...\n")
```

## Part 5: Adaptive Retriever (Combines All)

```python
class AdaptiveRetriever:
    """Combines multiple retrieval strategies adaptively."""
    
    def __init__(self, collection, config: LabConfig):
        self.collection = collection
        self.config = config
        self.client = OpenAI()
        
        self.basic = BasicRetriever(collection, config)
        self.hyde = HyDERetriever(collection, config)
        self.fusion = RAGFusionRetriever(collection, config)
    
    def retrieve(self, query: str, k: int = None) -> list[dict]:
        """Adaptive retrieval based on query analysis."""
        
        k = k or self.config.retrieval_k
        
        # Analyze query
        query_type = self._classify_query(query)
        
        print(f"  Query type: {query_type}")
        
        if query_type == "simple":
            return self.basic.retrieve(query, k)
        elif query_type == "vague":
            return self.hyde.retrieve(query, k)
        else:  # complex
            return self.fusion.retrieve(query, k)
    
    def _classify_query(self, query: str) -> str:
        """Classify query type."""
        
        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": """Classify this query:
- SIMPLE: Clear, specific question with obvious keywords
- VAGUE: Short or unclear, needs context
- COMPLEX: Multi-part or nuanced question

Respond with only: SIMPLE, VAGUE, or COMPLEX"""
                },
                {"role": "user", "content": query}
            ],
            max_tokens=10,
            temperature=0
        )
        
        result = response.choices[0].message.content.upper()
        
        if "SIMPLE" in result:
            return "simple"
        elif "VAGUE" in result:
            return "vague"
        else:
            return "complex"


# Test adaptive
print("\n=== Adaptive Retrieval ===\n")

adaptive = AdaptiveRetriever(collection, config)

test_queries = [
    "What is the rate limit?",  # Simple
    "auth",  # Vague  
    "How do I handle errors and retry when rate limited?"  # Complex
]

for query in test_queries:
    print(f"\nQuery: {query}")
    results = adaptive.retrieve(query, k=3)
    print(f"Top result: [{results[0]['id']}] - {results[0]['metadata']['title']}")
```

## Part 6: Generation with Citations

```python
class RAGGenerator:
    """Generate answers with citations."""
    
    SYSTEM_PROMPT = """You are a technical documentation assistant.
Answer questions using ONLY the provided context.
- Cite sources with [N] notation
- If context is insufficient, say "I don't have enough information"
- Be concise and accurate"""
    
    def __init__(self, config: LabConfig):
        self.config = config
        self.client = OpenAI()
    
    def generate(
        self,
        query: str,
        documents: list[dict]
    ) -> dict:
        """Generate answer with citations."""
        
        # Build context
        context = "\n\n".join([
            f"[Source {i+1}] {doc['content']}"
            for i, doc in enumerate(documents)
        ])
        
        # Generate
        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        answer = response.choices[0].message.content
        
        # Extract citations
        import re
        citations = list(set(re.findall(r'\[(\d+)\]', answer)))
        
        return {
            "answer": answer,
            "sources": [
                {
                    "id": documents[int(c)-1]["id"],
                    "title": documents[int(c)-1]["metadata"]["title"]
                }
                for c in citations
                if 0 < int(c) <= len(documents)
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        }


# Test generation
print("\n=== RAG Generation ===\n")

generator = RAGGenerator(config)

query = "How do I authenticate with the API?"
docs = adaptive.retrieve(query)

result = generator.generate(query, docs)

print(f"Q: {query}\n")
print(f"A: {result['answer']}\n")
print(f"Sources: {result['sources']}")
```

## Part 7: Evaluation Framework

```python
class RAGEvaluator:
    """Evaluate RAG system performance."""
    
    def __init__(self, config: LabConfig):
        self.config = config
        self.client = OpenAI()
    
    def evaluate_retrieval(
        self,
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int = 5
    ) -> dict:
        """Calculate retrieval metrics."""
        
        retrieved_set = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        # Precision@k
        relevant_retrieved = len(retrieved_set & relevant_set)
        precision = relevant_retrieved / k if k > 0 else 0
        
        # Recall@k
        recall = relevant_retrieved / len(relevant_set) if relevant_set else 1.0
        
        # MRR
        mrr = 0.0
        for rank, doc_id in enumerate(retrieved_ids[:k], 1):
            if doc_id in relevant_set:
                mrr = 1.0 / rank
                break
        
        # Hit rate
        hit = 1 if relevant_retrieved > 0 else 0
        
        return {
            "precision@k": precision,
            "recall@k": recall,
            "mrr": mrr,
            "hit_rate": hit
        }
    
    def evaluate_generation(
        self,
        query: str,
        context: list[str],
        answer: str,
        ground_truth: str = None
    ) -> dict:
        """Evaluate generation quality."""
        
        metrics = {}
        
        # Faithfulness
        metrics["faithfulness"] = self._evaluate_faithfulness(context, answer)
        
        # Relevance
        metrics["relevance"] = self._evaluate_relevance(query, answer)
        
        # Correctness (if ground truth available)
        if ground_truth:
            metrics["correctness"] = self._evaluate_correctness(
                answer, ground_truth
            )
        
        return metrics
    
    def _evaluate_faithfulness(
        self,
        context: list[str],
        answer: str
    ) -> float:
        """Check if answer is supported by context."""
        
        context_text = "\n".join(context)
        
        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "Rate 0-10 how well the answer is supported by context. Return only the number."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nAnswer:\n{answer}"
                }
            ],
            max_tokens=5,
            temperature=0
        )
        
        try:
            return float(response.choices[0].message.content.strip()) / 10
        except:
            return 0.5
    
    def _evaluate_relevance(self, query: str, answer: str) -> float:
        """Check if answer addresses the query."""
        
        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "Rate 0-10 how well the answer addresses the question. Return only the number."
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nAnswer: {answer}"
                }
            ],
            max_tokens=5,
            temperature=0
        )
        
        try:
            return float(response.choices[0].message.content.strip()) / 10
        except:
            return 0.5
    
    def _evaluate_correctness(self, answer: str, ground_truth: str) -> float:
        """Compare answer to ground truth."""
        
        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "Rate 0-10 the factual correctness of the answer compared to ground truth. Return only the number."
                },
                {
                    "role": "user",
                    "content": f"Answer: {answer}\n\nGround Truth: {ground_truth}"
                }
            ],
            max_tokens=5,
            temperature=0
        )
        
        try:
            return float(response.choices[0].message.content.strip()) / 10
        except:
            return 0.5


# Run evaluation
print("\n=== Evaluation ===\n")

evaluator = RAGEvaluator(config)

# Compare retrievers
retrievers = {
    "Baseline": BasicRetriever(collection, config),
    "HyDE": HyDERetriever(collection, config),
    "RAG Fusion": RAGFusionRetriever(collection, config)
}

print("Retrieval Comparison:\n")
print(f"{'Retriever':<15} {'Precision@5':<12} {'Recall@5':<12} {'MRR':<12} {'Hit Rate':<12}")
print("-" * 60)

for name, retriever in retrievers.items():
    metrics_sum = {"precision@k": 0, "recall@k": 0, "mrr": 0, "hit_rate": 0}
    
    for test in TEST_CASES:
        results = retriever.retrieve(test["query"])
        retrieved_ids = [r["id"] for r in results]
        
        metrics = evaluator.evaluate_retrieval(
            retrieved_ids,
            test["relevant_docs"]
        )
        
        for k, v in metrics.items():
            metrics_sum[k] += v
    
    n = len(TEST_CASES)
    print(f"{name:<15} {metrics_sum['precision@k']/n:<12.3f} {metrics_sum['recall@k']/n:<12.3f} {metrics_sum['mrr']/n:<12.3f} {metrics_sum['hit_rate']/n:<12.3f}")
```

## Part 8: Full Pipeline

```python
class AdvancedRAGPipeline:
    """Complete advanced RAG pipeline."""
    
    def __init__(self, collection, config: LabConfig):
        self.collection = collection
        self.config = config
        
        self.retriever = AdaptiveRetriever(collection, config)
        self.generator = RAGGenerator(config)
        self.evaluator = RAGEvaluator(config)
    
    def query(
        self,
        question: str,
        evaluate: bool = False,
        ground_truth: str = None
    ) -> dict:
        """Run full RAG pipeline."""
        
        start_time = time.time()
        
        # Retrieve
        documents = self.retriever.retrieve(question)
        
        retrieval_time = time.time() - start_time
        
        # Generate
        gen_start = time.time()
        result = self.generator.generate(question, documents)
        generation_time = time.time() - gen_start
        
        # Build response
        response = {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "timing": {
                "retrieval_ms": retrieval_time * 1000,
                "generation_ms": generation_time * 1000,
                "total_ms": (retrieval_time + generation_time) * 1000
            }
        }
        
        # Optional evaluation
        if evaluate:
            context = [d["content"] for d in documents]
            eval_metrics = self.evaluator.evaluate_generation(
                question,
                context,
                result["answer"],
                ground_truth
            )
            response["evaluation"] = eval_metrics
        
        return response


# Demo
print("\n=== Full Pipeline Demo ===\n")

pipeline = AdvancedRAGPipeline(collection, config)

# Run on test cases
for test in TEST_CASES[:3]:
    result = pipeline.query(
        test["query"],
        evaluate=True,
        ground_truth=test["ground_truth"]
    )
    
    print(f"Q: {result['question']}")
    print(f"A: {result['answer'][:200]}...")
    print(f"Sources: {[s['id'] for s in result['sources']]}")
    print(f"Faithfulness: {result['evaluation']['faithfulness']:.2f}")
    print(f"Relevance: {result['evaluation']['relevance']:.2f}")
    print(f"Correctness: {result['evaluation']['correctness']:.2f}")
    print(f"Time: {result['timing']['total_ms']:.0f}ms")
    print("-" * 50)
```

## Challenges

### Challenge 1: Add Reranking
Implement a reranker that rescores retrieved documents using a cross-encoder approach.

### Challenge 2: Streaming Output
Modify the generator to stream responses token by token.

### Challenge 3: Query Routing
Add logic to route different query types to specialized indexes.

### Challenge 4: Caching Layer
Implement caching for embeddings and retrieval results.

## Summary

```yaml
completed:
  - "Basic retrieval baseline"
  - "HyDE implementation"
  - "RAG Fusion with RRF"
  - "Adaptive retriever"
  - "Generation with citations"
  - "Comprehensive evaluation"

key_learnings:
  - "HyDE helps with vague queries"
  - "RAG Fusion improves recall"
  - "Evaluation is essential for improvement"
  - "Different queries benefit from different strategies"

next_steps:
  - "Implement caching for production"
  - "Add observability and monitoring"
  - "Build continuous evaluation pipeline"
```
