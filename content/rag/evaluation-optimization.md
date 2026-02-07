# RAG Evaluation & Optimization

Measure and improve your RAG system's performance with systematic evaluation.

## Why Evaluation Matters

```yaml
evaluation_goals:
  measure_quality: "Know how well your RAG system performs"
  identify_issues: "Find retrieval and generation problems"
  guide_improvement: "Make data-driven optimizations"
  track_progress: "Monitor quality over time"

key_dimensions:
  retrieval: "Are we finding the right documents?"
  generation: "Are answers accurate and helpful?"
  end_to_end: "Does the system solve user problems?"
```

## Retrieval Evaluation Metrics

```python
from openai import OpenAI
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality."""
    
    precision_at_k: float
    recall_at_k: float
    mrr: float  # Mean Reciprocal Rank
    ndcg: float  # Normalized Discounted Cumulative Gain
    hit_rate: float


class RetrievalEvaluator:
    """Evaluate retrieval quality."""
    
    def evaluate(
        self,
        queries: list[str],
        retrieved: list[list[str]],  # Document IDs per query
        relevant: list[list[str]],   # Ground truth relevant IDs
        k: int = 5
    ) -> RetrievalMetrics:
        """Calculate retrieval metrics."""
        
        precisions = []
        recalls = []
        mrrs = []
        ndcgs = []
        hits = []
        
        for query_idx in range(len(queries)):
            query_retrieved = retrieved[query_idx][:k]
            query_relevant = set(relevant[query_idx])
            
            # Precision@k
            relevant_retrieved = len(
                set(query_retrieved) & query_relevant
            )
            precision = relevant_retrieved / k if k > 0 else 0
            precisions.append(precision)
            
            # Recall@k
            recall = (
                relevant_retrieved / len(query_relevant)
                if query_relevant else 1.0
            )
            recalls.append(recall)
            
            # MRR (Mean Reciprocal Rank)
            mrr = 0.0
            for rank, doc_id in enumerate(query_retrieved, 1):
                if doc_id in query_relevant:
                    mrr = 1.0 / rank
                    break
            mrrs.append(mrr)
            
            # NDCG
            dcg = sum(
                1.0 / np.log2(rank + 2)
                for rank, doc_id in enumerate(query_retrieved)
                if doc_id in query_relevant
            )
            
            ideal_dcg = sum(
                1.0 / np.log2(rank + 2)
                for rank in range(min(len(query_relevant), k))
            )
            
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
            ndcgs.append(ndcg)
            
            # Hit rate (at least one relevant doc)
            hits.append(1 if relevant_retrieved > 0 else 0)
        
        return RetrievalMetrics(
            precision_at_k=np.mean(precisions),
            recall_at_k=np.mean(recalls),
            mrr=np.mean(mrrs),
            ndcg=np.mean(ndcgs),
            hit_rate=np.mean(hits)
        )


# Usage
evaluator = RetrievalEvaluator()

# Example evaluation data
queries = [
    "What is the return policy?",
    "How do I reset my password?",
    "What are the API rate limits?"
]

# What your retriever returned
retrieved = [
    ["doc_returns_1", "doc_shipping_1", "doc_returns_2"],
    ["doc_password_1", "doc_account_1", "doc_security_1"],
    ["doc_api_1", "doc_api_2", "doc_pricing_1"]
]

# Ground truth relevant documents
relevant = [
    ["doc_returns_1", "doc_returns_2", "doc_returns_3"],
    ["doc_password_1", "doc_password_2"],
    ["doc_api_1", "doc_api_2"]
]

metrics = evaluator.evaluate(queries, retrieved, relevant, k=3)

print(f"Precision@3: {metrics.precision_at_k:.3f}")
print(f"Recall@3: {metrics.recall_at_k:.3f}")
print(f"MRR: {metrics.mrr:.3f}")
print(f"NDCG: {metrics.ndcg:.3f}")
print(f"Hit Rate: {metrics.hit_rate:.3f}")
```

## Generation Evaluation

```python
@dataclass
class GenerationMetrics:
    """Metrics for generation quality."""
    
    faithfulness: float      # Is answer supported by context?
    relevance: float         # Does answer address the question?
    completeness: float      # Are all aspects of question addressed?
    coherence: float         # Is answer well-structured?
    citation_accuracy: float # Are citations correct?


class GenerationEvaluator:
    """Evaluate generation quality using LLM-as-judge."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def evaluate(
        self,
        query: str,
        context: list[str],
        answer: str
    ) -> GenerationMetrics:
        """Evaluate a single generation."""
        
        return GenerationMetrics(
            faithfulness=self._evaluate_faithfulness(context, answer),
            relevance=self._evaluate_relevance(query, answer),
            completeness=self._evaluate_completeness(query, answer),
            coherence=self._evaluate_coherence(answer),
            citation_accuracy=self._evaluate_citations(context, answer)
        )
    
    def _evaluate_faithfulness(
        self,
        context: list[str],
        answer: str
    ) -> float:
        """Check if answer is supported by context."""
        
        context_text = "\n".join(context)
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Evaluate if the answer is fully supported by the context.
Score 0-10 where:
- 0: Answer contains information not in context (hallucination)
- 5: Answer partially supported
- 10: Answer fully supported by context

Return only the number."""
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
        """Check if answer addresses the question."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Evaluate if the answer directly addresses the question.
Score 0-10 where:
- 0: Answer is completely off-topic
- 5: Answer partially addresses question
- 10: Answer directly and fully addresses question

Return only the number."""
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
    
    def _evaluate_completeness(self, query: str, answer: str) -> float:
        """Check if all aspects of question are addressed."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Evaluate if the answer is complete.
Score 0-10 where:
- 0: Major information missing
- 5: Some aspects not addressed
- 10: Thoroughly complete answer

Return only the number."""
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
    
    def _evaluate_coherence(self, answer: str) -> float:
        """Check if answer is well-structured and clear."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Evaluate the coherence and clarity of this answer.
Score 0-10 where:
- 0: Incoherent, hard to understand
- 5: Somewhat clear
- 10: Very clear and well-structured

Return only the number."""
                },
                {
                    "role": "user",
                    "content": answer
                }
            ],
            max_tokens=5,
            temperature=0
        )
        
        try:
            return float(response.choices[0].message.content.strip()) / 10
        except:
            return 0.5
    
    def _evaluate_citations(
        self,
        context: list[str],
        answer: str
    ) -> float:
        """Check if citations are accurate."""
        
        import re
        
        # Extract citations
        citations = re.findall(r'\[(\d+)\]', answer)
        
        if not citations:
            return 1.0  # No citations to evaluate
        
        # Check each citation
        correct = 0
        
        for citation in set(citations):
            num = int(citation)
            if 0 < num <= len(context):
                # Verify citation is used correctly
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "Does the answer correctly cite source for the claims it attributes to it? Answer YES or NO."
                        },
                        {
                            "role": "user",
                            "content": f"Source [{num}]: {context[num-1][:500]}\n\nAnswer: {answer}"
                        }
                    ],
                    max_tokens=5,
                    temperature=0
                )
                
                if "YES" in response.choices[0].message.content.upper():
                    correct += 1
        
        return correct / len(set(citations)) if citations else 1.0


# Usage
gen_evaluator = GenerationEvaluator()

context = [
    "Returns are accepted within 30 days of purchase.",
    "Original receipt required for all returns."
]

answer = "You can return items within 30 days [1] as long as you have the original receipt [2]."

metrics = gen_evaluator.evaluate(
    query="What is the return policy?",
    context=context,
    answer=answer
)

print(f"Faithfulness: {metrics.faithfulness:.2f}")
print(f"Relevance: {metrics.relevance:.2f}")
print(f"Completeness: {metrics.completeness:.2f}")
print(f"Coherence: {metrics.coherence:.2f}")
print(f"Citation Accuracy: {metrics.citation_accuracy:.2f}")
```

## RAGAS-Style Evaluation

```python
class RAGASEvaluator:
    """Evaluation inspired by RAGAS framework."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def evaluate_end_to_end(
        self,
        query: str,
        context: list[str],
        answer: str,
        ground_truth: str = None
    ) -> dict:
        """Comprehensive RAG evaluation."""
        
        metrics = {}
        
        # Context Relevance: Is retrieved context relevant to query?
        metrics["context_relevance"] = self._context_relevance(query, context)
        
        # Context Precision: Are relevant sentences at the top?
        metrics["context_precision"] = self._context_precision(query, context)
        
        # Faithfulness: Is answer grounded in context?
        metrics["faithfulness"] = self._faithfulness(context, answer)
        
        # Answer Relevance: Does answer address the query?
        metrics["answer_relevance"] = self._answer_relevance(query, answer)
        
        # If ground truth available, compute answer similarity
        if ground_truth:
            metrics["answer_similarity"] = self._answer_similarity(
                answer, ground_truth
            )
            metrics["answer_correctness"] = self._answer_correctness(
                query, answer, ground_truth
            )
        
        # Overall score
        core_metrics = [
            metrics["context_relevance"],
            metrics["faithfulness"],
            metrics["answer_relevance"]
        ]
        metrics["overall_score"] = np.mean(core_metrics)
        
        return metrics
    
    def _context_relevance(self, query: str, context: list[str]) -> float:
        """Evaluate context relevance to query."""
        
        scores = []
        
        for doc in context:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Rate how relevant this context is to the query (0-10). Return only the number."
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nContext: {doc}"
                    }
                ],
                max_tokens=5,
                temperature=0
            )
            
            try:
                scores.append(float(response.choices[0].message.content.strip()) / 10)
            except:
                scores.append(0.5)
        
        return np.mean(scores) if scores else 0
    
    def _context_precision(self, query: str, context: list[str]) -> float:
        """Evaluate if relevant context is ranked higher."""
        
        # Get relevance scores
        relevance_scores = []
        
        for doc in context:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Is this context relevant to the query? Answer 1 for yes, 0 for no."
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nContext: {doc}"
                    }
                ],
                max_tokens=5,
                temperature=0
            )
            
            try:
                relevance_scores.append(int(response.choices[0].message.content.strip()))
            except:
                relevance_scores.append(0)
        
        # Calculate precision at each position
        if not any(relevance_scores):
            return 0
        
        precisions = []
        relevant_count = 0
        
        for i, is_relevant in enumerate(relevance_scores):
            relevant_count += is_relevant
            precisions.append(relevant_count / (i + 1))
        
        # Weight by relevance
        weighted_precision = sum(
            p * r for p, r in zip(precisions, relevance_scores)
        ) / sum(relevance_scores) if sum(relevance_scores) > 0 else 0
        
        return weighted_precision
    
    def _faithfulness(self, context: list[str], answer: str) -> float:
        """Check answer faithfulness to context."""
        
        # Extract claims from answer
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Extract individual factual claims from this answer. List each claim on a new line."
                },
                {
                    "role": "user",
                    "content": answer
                }
            ],
            max_tokens=300,
            temperature=0
        )
        
        claims = [
            c.strip() for c in response.choices[0].message.content.split("\n")
            if c.strip()
        ]
        
        if not claims:
            return 1.0
        
        # Verify each claim
        context_text = "\n".join(context)
        supported_claims = 0
        
        for claim in claims:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Is this claim supported by the context? Answer YES or NO."
                    },
                    {
                        "role": "user",
                        "content": f"Context: {context_text}\n\nClaim: {claim}"
                    }
                ],
                max_tokens=5,
                temperature=0
            )
            
            if "YES" in response.choices[0].message.content.upper():
                supported_claims += 1
        
        return supported_claims / len(claims)
    
    def _answer_relevance(self, query: str, answer: str) -> float:
        """Evaluate answer relevance to query."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Rate how well this answer addresses the question (0-10). Return only the number."
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
    
    def _answer_similarity(self, answer: str, ground_truth: str) -> float:
        """Compute semantic similarity between answer and ground truth."""
        
        response = self.client.embeddings.create(
            input=[answer, ground_truth],
            model="text-embedding-3-small"
        )
        
        emb1 = np.array(response.data[0].embedding)
        emb2 = np.array(response.data[1].embedding)
        
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        return float(similarity)
    
    def _answer_correctness(
        self,
        query: str,
        answer: str,
        ground_truth: str
    ) -> float:
        """Evaluate correctness against ground truth."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Compare the answer to the ground truth.
Rate factual correctness 0-10 where:
- 0: Completely wrong
- 5: Partially correct
- 10: Fully correct
Return only the number."""
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nAnswer: {answer}\n\nGround Truth: {ground_truth}"
                }
            ],
            max_tokens=5,
            temperature=0
        )
        
        try:
            return float(response.choices[0].message.content.strip()) / 10
        except:
            return 0.5


# Usage
ragas = RAGASEvaluator()

results = ragas.evaluate_end_to_end(
    query="What is the return policy?",
    context=[
        "Returns are accepted within 30 days of purchase.",
        "Original receipt required for all returns.",
        "Refunds processed within 5-7 business days."
    ],
    answer="You can return items within 30 days [1] with the original receipt [2]. Refunds take 5-7 days [3].",
    ground_truth="Items can be returned within 30 days with receipt. Refunds are processed in 5-7 business days."
)

print("RAGAS Evaluation:")
for metric, score in results.items():
    print(f"  {metric}: {score:.3f}")
```

## Optimization Strategies

```python
class RAGOptimizer:
    """Optimize RAG system based on evaluation results."""
    
    OPTIMIZATION_STRATEGIES = {
        "low_context_relevance": [
            "Improve chunking strategy",
            "Use hybrid search (semantic + keyword)",
            "Try query expansion/rewriting",
            "Increase number of retrieved documents"
        ],
        "low_faithfulness": [
            "Use more restrictive prompts",
            "Add citation requirements",
            "Reduce temperature",
            "Filter retrieved documents by relevance threshold"
        ],
        "low_answer_relevance": [
            "Improve system prompt",
            "Add query classification",
            "Use chain-of-thought prompting",
            "Filter context to most relevant"
        ],
        "low_completeness": [
            "Retrieve more documents",
            "Use multi-query retrieval",
            "Improve document coverage",
            "Add follow-up generation"
        ]
    }
    
    def diagnose(self, metrics: dict) -> list[str]:
        """Diagnose issues and suggest optimizations."""
        
        suggestions = []
        threshold = 0.7
        
        if metrics.get("context_relevance", 1) < threshold:
            suggestions.extend(self.OPTIMIZATION_STRATEGIES["low_context_relevance"])
        
        if metrics.get("faithfulness", 1) < threshold:
            suggestions.extend(self.OPTIMIZATION_STRATEGIES["low_faithfulness"])
        
        if metrics.get("answer_relevance", 1) < threshold:
            suggestions.extend(self.OPTIMIZATION_STRATEGIES["low_answer_relevance"])
        
        if metrics.get("completeness", 1) < threshold:
            suggestions.extend(self.OPTIMIZATION_STRATEGIES["low_completeness"])
        
        return list(set(suggestions))
    
    def run_ablation_study(
        self,
        rag_system,
        test_cases: list[dict],
        parameters_to_test: dict
    ) -> dict:
        """Run ablation study to find optimal parameters."""
        
        results = {}
        
        for param_name, param_values in parameters_to_test.items():
            results[param_name] = {}
            
            for value in param_values:
                # Set parameter
                setattr(rag_system, param_name, value)
                
                # Evaluate
                scores = []
                for test in test_cases:
                    # Run RAG and evaluate
                    # ... (implementation depends on your RAG system)
                    pass
                
                results[param_name][value] = np.mean(scores)
        
        return results


# Example ablation study configuration
ablation_params = {
    "retrieval_k": [3, 5, 7, 10],
    "chunk_size": [256, 512, 1024],
    "temperature": [0, 0.3, 0.7],
    "rerank_enabled": [True, False]
}
```

## Summary

```yaml
evaluation_framework:
  retrieval_metrics:
    - "Precision@k: Relevant docs in top k"
    - "Recall@k: Coverage of relevant docs"
    - "MRR: Position of first relevant doc"
    - "NDCG: Ranking quality"
  
  generation_metrics:
    - "Faithfulness: Grounded in context"
    - "Relevance: Addresses the question"
    - "Completeness: Comprehensive answer"
    - "Coherence: Well-structured"
  
  end_to_end:
    - "Context relevance + precision"
    - "Answer correctness vs ground truth"
    - "Overall quality score"

optimization_process:
  1: "Establish baseline metrics"
  2: "Identify weak points"
  3: "Apply targeted optimizations"
  4: "Measure improvement"
  5: "Iterate"

best_practices:
  - "Build a diverse test set"
  - "Include edge cases"
  - "Evaluate regularly"
  - "Track metrics over time"
  - "A/B test changes"
```

## Next Steps

1. **Build Test Dataset** - Create comprehensive evaluation set
2. **Implement Monitoring** - Track metrics in production
3. **Continuous Improvement** - Regular optimization cycles
