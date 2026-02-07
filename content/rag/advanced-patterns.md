# Advanced RAG Patterns

Implement sophisticated retrieval techniques for improved accuracy and relevance.

## Advanced Pattern Overview

```yaml
patterns_covered:
  hyde: "Hypothetical Document Embeddings"
  rag_fusion: "Multiple query fusion"
  self_rag: "Self-reflective retrieval"
  corrective_rag: "Quality-based retrieval correction"
  adaptive_rag: "Dynamic strategy selection"

when_to_use:
  hyde: "Short or vague queries"
  rag_fusion: "Complex multi-faceted questions"
  self_rag: "High-stakes applications"
  corrective_rag: "When retrieval quality varies"
  adaptive_rag: "Diverse query types"
```

## HyDE (Hypothetical Document Embeddings)

```python
from openai import OpenAI
import chromadb
import numpy as np


class HyDERetriever:
    """
    HyDE: Generate hypothetical documents, then search.
    
    Instead of embedding the query directly, generate what an ideal
    answer document might look like, then search for similar real documents.
    """
    
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
        hypotheticals = self._generate_hypothetical_docs(query, num_hypothetical)
        
        # Embed all hypotheticals
        hypo_embeddings = self._embed_batch(hypotheticals)
        
        # Average embeddings
        avg_embedding = np.mean(hypo_embeddings, axis=0).tolist()
        
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
                "score": 1 - results["distances"][0][i],
                "metadata": results["metadatas"][0][i]
            }
            for i in range(len(results["ids"][0]))
        ]
    
    def _generate_hypothetical_docs(
        self,
        query: str,
        num_docs: int
    ) -> list[str]:
        """Generate hypothetical answer documents."""
        
        prompt = f"""Generate {num_docs} different hypothetical passages that would answer this question.
Write as if these are excerpts from real documents, not direct answers.
Make each passage distinct and approach the topic differently.
Separate with ---

Question: {query}

Hypothetical document excerpts:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
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
hyde = HyDERetriever()

# For a vague query like "best practices"
results = hyde.search(
    "best practices for database design",
    n_results=5,
    num_hypothetical=3
)

# HyDE generates hypothetical docs like:
# "Database normalization ensures data integrity by organizing tables..."
# "Indexing strategies significantly impact query performance..."
# Then finds real documents similar to these hypotheticals
```

## RAG Fusion

```python
class RAGFusionRetriever:
    """
    RAG Fusion: Generate multiple queries, search each, fuse results.
    
    Combines diverse perspectives on the question to improve recall.
    """
    
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
        queries = self._generate_queries(query, num_queries)
        queries.insert(0, query)  # Include original
        
        print(f"Queries: {queries}")
        
        # Search with each query
        all_rankings = []
        
        for q in queries:
            results = self._single_search(q, n_results * 2)
            all_rankings.append(results)
        
        # Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(all_rankings)
        
        return fused[:n_results]
    
    def _generate_queries(self, query: str, num_queries: int) -> list[str]:
        """Generate diverse search queries."""
        
        prompt = f"""Generate {num_queries} different search queries based on this question.
Include variations like:
- Rephrased with synonyms
- More specific version
- Broader version
- Different angle/perspective

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
        """Fuse multiple rankings using RRF."""
        
        fused_scores = {}
        doc_data = {}
        
        for ranking in rankings:
            for rank, doc in enumerate(ranking, 1):
                doc_id = doc["id"]
                
                # RRF score formula
                rrf_score = 1 / (k + rank)
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
            {**doc_data[doc_id], "fusion_score": fused_scores[doc_id]}
            for doc_id in sorted_ids
        ]
    
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
```

## Self-RAG

```python
class SelfRAGRetriever:
    """
    Self-RAG: Self-reflective retrieval augmented generation.
    
    The model decides when to retrieve, evaluates retrieval quality,
    and critiques its own generation.
    """
    
    def __init__(self):
        self.client = OpenAI()
        self.chroma = chromadb.PersistentClient(path="./selfrag_db")
        self.collection = self.chroma.get_or_create_collection("selfrag_docs")
    
    def generate(
        self,
        query: str,
        n_results: int = 5
    ) -> dict:
        """Self-RAG generation with reflection."""
        
        # Step 1: Decide if retrieval is needed
        needs_retrieval = self._assess_retrieval_need(query)
        
        if not needs_retrieval:
            # Generate without retrieval
            return self._generate_without_retrieval(query)
        
        # Step 2: Retrieve documents
        documents = self._retrieve(query, n_results)
        
        # Step 3: Evaluate document relevance
        relevant_docs = self._filter_relevant(query, documents)
        
        if not relevant_docs:
            # No relevant docs, try to answer without
            return self._generate_without_retrieval(query)
        
        # Step 4: Generate with relevant context
        initial_answer = self._generate_with_context(query, relevant_docs)
        
        # Step 5: Self-critique and refine
        final_answer = self._critique_and_refine(
            query, relevant_docs, initial_answer
        )
        
        return {
            "answer": final_answer,
            "retrieval_used": True,
            "documents_retrieved": len(documents),
            "documents_used": len(relevant_docs),
            "sources": relevant_docs
        }
    
    def _assess_retrieval_need(self, query: str) -> bool:
        """Decide if retrieval is needed."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Determine if answering this question requires looking up external information. Answer YES or NO."
                },
                {"role": "user", "content": query}
            ],
            max_tokens=5,
            temperature=0
        )
        
        return "YES" in response.choices[0].message.content.upper()
    
    def _filter_relevant(
        self,
        query: str,
        documents: list[dict]
    ) -> list[dict]:
        """Filter documents by relevance."""
        
        relevant = []
        
        for doc in documents:
            # Check relevance
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Is this document relevant to answering the question? Answer RELEVANT or NOT_RELEVANT."
                    },
                    {
                        "role": "user",
                        "content": f"Question: {query}\n\nDocument: {doc['content']}"
                    }
                ],
                max_tokens=10,
                temperature=0
            )
            
            if "RELEVANT" in response.choices[0].message.content.upper():
                relevant.append(doc)
        
        return relevant
    
    def _generate_with_context(
        self,
        query: str,
        documents: list[dict]
    ) -> str:
        """Generate answer using context."""
        
        context = "\n\n".join([
            f"[{i+1}] {doc['content']}"
            for i, doc in enumerate(documents)
        ])
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question using the provided context. Cite sources with [N]."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ],
            temperature=0
        )
        
        return response.choices[0].message.content
    
    def _critique_and_refine(
        self,
        query: str,
        documents: list[dict],
        answer: str
    ) -> str:
        """Self-critique and refine the answer."""
        
        context = "\n".join([doc['content'] for doc in documents])
        
        # Get critique
        critique_response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Evaluate this answer for:
1. Accuracy - Is it supported by the context?
2. Completeness - Does it fully answer the question?
3. Clarity - Is it well-written?
Provide brief critique and suggest improvements if needed."""
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nContext: {context}\n\nAnswer: {answer}"
                }
            ],
            max_tokens=200,
            temperature=0
        )
        
        critique = critique_response.choices[0].message.content
        
        # Refine based on critique
        if "improvement" in critique.lower() or "could" in critique.lower():
            refine_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Improve the answer based on the critique. Keep citations."
                    },
                    {
                        "role": "user",
                        "content": f"Original answer: {answer}\n\nCritique: {critique}\n\nImproved answer:"
                    }
                ],
                temperature=0
            )
            return refine_response.choices[0].message.content
        
        return answer
    
    def _generate_without_retrieval(self, query: str) -> dict:
        """Generate answer without retrieval."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question. If you're not sure, say so."
                },
                {"role": "user", "content": query}
            ],
            temperature=0
        )
        
        return {
            "answer": response.choices[0].message.content,
            "retrieval_used": False,
            "documents_retrieved": 0,
            "documents_used": 0,
            "sources": []
        }
    
    def _retrieve(self, query: str, n_results: int) -> list[dict]:
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

## Corrective RAG (CRAG)

```python
class CorrectiveRAG:
    """
    CRAG: Corrective Retrieval Augmented Generation.
    
    Evaluates retrieval quality and takes corrective action
    if initial retrieval is poor.
    """
    
    def __init__(self):
        self.client = OpenAI()
        self.chroma = chromadb.PersistentClient(path="./crag_db")
        self.collection = self.chroma.get_or_create_collection("crag_docs")
    
    def generate(
        self,
        query: str,
        n_results: int = 5
    ) -> dict:
        """CRAG generation with corrective actions."""
        
        # Initial retrieval
        documents = self._retrieve(query, n_results)
        
        # Evaluate retrieval quality
        evaluation = self._evaluate_retrieval(query, documents)
        
        # Take action based on evaluation
        if evaluation["quality"] == "CORRECT":
            # Good retrieval, use as-is
            return self._generate_from_docs(query, documents, "direct")
        
        elif evaluation["quality"] == "AMBIGUOUS":
            # Mixed quality, refine and supplement
            refined_docs = self._refine_documents(query, documents)
            supplemented = self._supplement_with_web(query, refined_docs)
            return self._generate_from_docs(query, supplemented, "refined")
        
        else:  # INCORRECT
            # Poor retrieval, try alternative approaches
            return self._corrective_generation(query, documents)
    
    def _evaluate_retrieval(
        self,
        query: str,
        documents: list[dict]
    ) -> dict:
        """Evaluate retrieval quality."""
        
        if not documents:
            return {"quality": "INCORRECT", "reason": "No documents retrieved"}
        
        # Score each document
        scores = []
        for doc in documents:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Rate document relevance: CORRECT (highly relevant), AMBIGUOUS (partially relevant), or INCORRECT (not relevant)."
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\nDocument: {doc['content'][:500]}"
                    }
                ],
                max_tokens=20,
                temperature=0
            )
            
            result = response.choices[0].message.content.upper()
            if "CORRECT" in result:
                scores.append(2)
            elif "AMBIGUOUS" in result:
                scores.append(1)
            else:
                scores.append(0)
        
        avg_score = sum(scores) / len(scores)
        
        if avg_score >= 1.5:
            quality = "CORRECT"
        elif avg_score >= 0.5:
            quality = "AMBIGUOUS"
        else:
            quality = "INCORRECT"
        
        return {
            "quality": quality,
            "avg_score": avg_score,
            "individual_scores": scores
        }
    
    def _refine_documents(
        self,
        query: str,
        documents: list[dict]
    ) -> list[dict]:
        """Refine documents by extracting relevant parts."""
        
        refined = []
        
        for doc in documents:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract only the parts of this document relevant to the query. If nothing is relevant, respond with 'NONE'."
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nDocument: {doc['content']}"
                    }
                ],
                max_tokens=300,
                temperature=0
            )
            
            extracted = response.choices[0].message.content
            
            if "NONE" not in extracted.upper():
                refined.append({**doc, "content": extracted})
        
        return refined
    
    def _supplement_with_web(
        self,
        query: str,
        documents: list[dict]
    ) -> list[dict]:
        """Supplement with web search (simulated)."""
        
        # In production, this would call a web search API
        # Here we generate supplementary content
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Generate a brief, factual passage that would help answer this query."
                },
                {"role": "user", "content": query}
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        supplementary = {
            "id": "web_supplement",
            "content": response.choices[0].message.content,
            "metadata": {"source": "web_search"}
        }
        
        return documents + [supplementary]
    
    def _corrective_generation(
        self,
        query: str,
        failed_docs: list[dict]
    ) -> dict:
        """Handle failed retrieval with corrective actions."""
        
        # Try query reformulation
        reformulated = self._reformulate_query(query)
        new_docs = self._retrieve(reformulated, 5)
        
        new_eval = self._evaluate_retrieval(query, new_docs)
        
        if new_eval["quality"] != "INCORRECT":
            return self._generate_from_docs(query, new_docs, "reformulated")
        
        # Fall back to generation without retrieval
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Answer to the best of your knowledge. Note if you're uncertain."
                },
                {"role": "user", "content": query}
            ],
            temperature=0
        )
        
        return {
            "answer": response.choices[0].message.content,
            "method": "fallback",
            "retrieval_failed": True
        }
    
    def _reformulate_query(self, query: str) -> str:
        """Reformulate query for better retrieval."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Reformulate this query to be more specific and searchable."
                },
                {"role": "user", "content": query}
            ],
            max_tokens=100,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    
    def _generate_from_docs(
        self,
        query: str,
        documents: list[dict],
        method: str
    ) -> dict:
        """Generate answer from documents."""
        
        context = "\n\n".join([
            f"[{i+1}] {doc['content']}"
            for i, doc in enumerate(documents)
        ])
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Answer using the context. Cite with [N]."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ],
            temperature=0
        )
        
        return {
            "answer": response.choices[0].message.content,
            "method": method,
            "documents_used": len(documents),
            "sources": documents
        }
    
    def _retrieve(self, query: str, n_results: int) -> list[dict]:
        response = self.client.embeddings.create(
            input=query, model="text-embedding-3-small"
        )
        
        results = self.collection.query(
            query_embeddings=[response.data[0].embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["ids"][0]:
            return []
        
        return [
            {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "score": 1 - results["distances"][0][i]
            }
            for i in range(len(results["ids"][0]))
        ]
```

## Choosing the Right Pattern

```yaml
decision_guide:
  hyde:
    best_for: "Vague queries, short questions, exploratory search"
    overhead: "Low (1 LLM call for generation)"
    quality_gain: "10-20% on ambiguous queries"
  
  rag_fusion:
    best_for: "Complex questions, multi-faceted topics"
    overhead: "Medium (1 LLM call + N searches)"
    quality_gain: "15-25% on complex queries"
  
  self_rag:
    best_for: "High-stakes applications, accuracy-critical"
    overhead: "High (multiple LLM calls)"
    quality_gain: "20-30% with reduced hallucination"
  
  crag:
    best_for: "Variable document quality, production systems"
    overhead: "Variable (depends on retrieval quality)"
    quality_gain: "Robust improvement across query types"
```

## Summary

```yaml
key_takeaways:
  1: "HyDE helps with vague queries"
  2: "RAG Fusion improves recall on complex questions"
  3: "Self-RAG reduces hallucination through reflection"
  4: "CRAG handles poor retrieval gracefully"

implementation_priority:
  start_with: "Basic RAG + HyDE for quick wins"
  add_next: "RAG Fusion for complex use cases"
  production: "Self-RAG or CRAG for reliability"
```

## Next Steps

1. **Production Architecture** - Scale advanced RAG
2. **Evaluation** - Measure pattern effectiveness
3. **Optimization** - Tune for your use case
