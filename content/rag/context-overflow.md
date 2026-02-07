# Context Overflow: Handling Large Retrieved Context

Strategies for managing context when retrieved documents exceed token limits.

## The Context Overflow Problem

```yaml
the_challenge:
  scenario: "Retrieved 10 documents but only 5 fit in context"
  issues:
    - "Token limits vary by model (4K to 128K+)"
    - "More context isn't always better"
    - "Cost increases with context size"
    - "Quality can degrade with too much context"

strategies:
  1: "Smart truncation"
  2: "Iterative refinement"
  3: "Map-reduce summarization"
  4: "Hierarchical context"
  5: "Dynamic context selection"
```

## Token Counting and Budget Management

```python
from openai import OpenAI
import tiktoken


class TokenBudgetManager:
    """Manage token budgets for RAG context."""
    
    MODEL_LIMITS = {
        "gpt-4o-mini": 128000,
        "gpt-4o": 128000,
        "gpt-4-turbo": 128000,
        "gpt-3.5-turbo": 16385,
        "claude-3-5-sonnet-20241022": 200000,
    }
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_context_ratio: float = 0.7
    ):
        self.model = model
        self.max_tokens = self.MODEL_LIMITS.get(model, 16000)
        self.max_context_tokens = int(self.max_tokens * max_context_ratio)
        
        # Use tiktoken for accurate counting
        try:
            self.encoder = tiktoken.encoding_for_model(model)
        except:
            self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoder.encode(text))
    
    def fit_documents(
        self,
        documents: list[dict],
        query: str,
        system_prompt: str = ""
    ) -> list[dict]:
        """Select documents that fit in budget."""
        
        # Calculate overhead (query, system prompt, formatting)
        overhead = self.count_tokens(query) + self.count_tokens(system_prompt) + 200
        available = self.max_context_tokens - overhead
        
        # Fit documents
        selected = []
        used_tokens = 0
        
        for doc in documents:
            doc_tokens = self.count_tokens(doc["content"])
            
            if used_tokens + doc_tokens <= available:
                selected.append(doc)
                used_tokens += doc_tokens
            else:
                # Try to include truncated version
                remaining = available - used_tokens
                if remaining > 100:  # Worth including partial
                    truncated = self._truncate_to_tokens(doc["content"], remaining)
                    selected.append({**doc, "content": truncated, "truncated": True})
                break
        
        return selected
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit token limit."""
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens - 10]  # Leave room for "..."
        return self.encoder.decode(truncated_tokens) + "..."
    
    def get_budget_report(
        self,
        documents: list[dict],
        query: str
    ) -> dict:
        """Report on token budget usage."""
        
        doc_tokens = sum(self.count_tokens(d["content"]) for d in documents)
        query_tokens = self.count_tokens(query)
        
        return {
            "model": self.model,
            "max_context_tokens": self.max_context_tokens,
            "document_tokens": doc_tokens,
            "query_tokens": query_tokens,
            "total_used": doc_tokens + query_tokens,
            "remaining": self.max_context_tokens - doc_tokens - query_tokens,
            "fits": doc_tokens + query_tokens <= self.max_context_tokens
        }


# Usage
manager = TokenBudgetManager(model="gpt-4o-mini")

documents = [
    {"content": "Long document 1..." * 100, "id": "doc1"},
    {"content": "Long document 2..." * 100, "id": "doc2"},
    {"content": "Long document 3..." * 100, "id": "doc3"},
]

# Check budget
report = manager.get_budget_report(documents, "What is the return policy?")
print(f"Total tokens: {report['total_used']}")
print(f"Fits in budget: {report['fits']}")

# Fit to budget
fitted = manager.fit_documents(documents, "What is the return policy?")
print(f"Documents that fit: {len(fitted)}")
```

## Smart Truncation Strategies

```python
class SmartTruncation:
    """Intelligent document truncation strategies."""
    
    def __init__(self):
        self.client = OpenAI()
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def truncate_keep_relevant(
        self,
        document: str,
        query: str,
        max_tokens: int
    ) -> str:
        """Truncate keeping most relevant parts."""
        
        # Split into paragraphs
        paragraphs = document.split("\n\n")
        
        if len(paragraphs) <= 1:
            return self._simple_truncate(document, max_tokens)
        
        # Score paragraphs by relevance
        scored = self._score_paragraphs(paragraphs, query)
        
        # Select paragraphs that fit
        selected = []
        used_tokens = 0
        
        for para, score in sorted(scored, key=lambda x: x[1], reverse=True):
            para_tokens = len(self.encoder.encode(para))
            
            if used_tokens + para_tokens <= max_tokens:
                selected.append((para, score))
                used_tokens += para_tokens
        
        # Re-order by original position
        original_order = {para: i for i, para in enumerate(paragraphs)}
        selected.sort(key=lambda x: original_order.get(x[0], 999))
        
        return "\n\n".join(para for para, _ in selected)
    
    def _score_paragraphs(
        self,
        paragraphs: list[str],
        query: str
    ) -> list[tuple[str, float]]:
        """Score paragraphs by relevance to query."""
        
        # Embed query and paragraphs
        texts = [query] + paragraphs
        
        response = self.client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        
        embeddings = [d.embedding for d in sorted(response.data, key=lambda x: x.index)]
        query_emb = embeddings[0]
        para_embs = embeddings[1:]
        
        # Calculate similarities
        import numpy as np
        query_vec = np.array(query_emb)
        
        scored = []
        for para, emb in zip(paragraphs, para_embs):
            para_vec = np.array(emb)
            similarity = np.dot(query_vec, para_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(para_vec)
            )
            scored.append((para, similarity))
        
        return scored
    
    def _simple_truncate(self, text: str, max_tokens: int) -> str:
        """Simple truncation."""
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoder.decode(tokens[:max_tokens]) + "..."
    
    def truncate_preserve_structure(
        self,
        document: str,
        max_tokens: int
    ) -> str:
        """Truncate while preserving document structure."""
        
        # Keep introduction and conclusion
        paragraphs = document.split("\n\n")
        
        if len(paragraphs) <= 3:
            return self._simple_truncate(document, max_tokens)
        
        # Always keep first and last paragraphs
        intro = paragraphs[0]
        conclusion = paragraphs[-1]
        middle = paragraphs[1:-1]
        
        intro_tokens = len(self.encoder.encode(intro))
        conclusion_tokens = len(self.encoder.encode(conclusion))
        remaining = max_tokens - intro_tokens - conclusion_tokens - 50
        
        if remaining <= 0:
            return self._simple_truncate(document, max_tokens)
        
        # Fill middle
        selected_middle = []
        used = 0
        
        for para in middle:
            para_tokens = len(self.encoder.encode(para))
            if used + para_tokens <= remaining:
                selected_middle.append(para)
                used += para_tokens
            else:
                selected_middle.append("[... content omitted ...]")
                break
        
        return "\n\n".join([intro] + selected_middle + [conclusion])


# Usage
truncator = SmartTruncation()

long_doc = """
Introduction to machine learning and its applications in modern software.

Machine learning is a subset of artificial intelligence...
[Many paragraphs about ML basics]

Deep learning uses neural networks with many layers...
[Many paragraphs about deep learning]

Practical applications include image recognition, NLP, and recommendation systems...
[Many paragraphs about applications]

In conclusion, machine learning continues to transform how we build software.
"""

truncated = truncator.truncate_keep_relevant(
    long_doc,
    query="What are ML applications?",
    max_tokens=500
)

print(truncated)
```

## Iterative Refinement

```python
class IterativeContextRefinement:
    """Iteratively refine context based on initial answer."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate_with_refinement(
        self,
        query: str,
        documents: list[dict],
        max_iterations: int = 2
    ) -> dict:
        """Generate answer with iterative context refinement."""
        
        current_context = documents
        history = []
        
        for iteration in range(max_iterations):
            # Generate answer
            answer = self._generate(query, current_context)
            
            # Check if answer is complete
            assessment = self._assess_answer(query, answer)
            
            history.append({
                "iteration": iteration + 1,
                "context_count": len(current_context),
                "answer": answer,
                "assessment": assessment
            })
            
            if assessment["is_complete"]:
                break
            
            # Refine context based on gaps
            if assessment["missing_info"]:
                # Re-rank documents for missing info
                current_context = self._refine_context(
                    documents,
                    query,
                    assessment["missing_info"]
                )
        
        return {
            "final_answer": history[-1]["answer"],
            "iterations": len(history),
            "history": history
        }
    
    def _generate(self, query: str, context: list[dict]) -> str:
        """Generate answer from context."""
        
        context_text = "\n\n".join(d["content"] for d in context)
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question based on the provided context. If the context is insufficient, indicate what information is missing."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nQuestion: {query}"
                }
            ],
            temperature=0
        )
        
        return response.choices[0].message.content
    
    def _assess_answer(self, query: str, answer: str) -> dict:
        """Assess if answer is complete."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Assess if the answer fully addresses the question.
Return JSON with:
- is_complete: boolean
- missing_info: list of missing information (empty if complete)"""
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nAnswer: {answer}"
                }
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        import json
        return json.loads(response.choices[0].message.content)
    
    def _refine_context(
        self,
        documents: list[dict],
        query: str,
        missing_info: list[str]
    ) -> list[dict]:
        """Re-rank documents to fill information gaps."""
        
        # Create enhanced query with missing info
        enhanced_query = f"{query}. Specifically need: {', '.join(missing_info)}"
        
        # Re-score documents
        scored = []
        
        for doc in documents:
            score = self._score_for_query(doc["content"], enhanced_query)
            scored.append((doc, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored[:5]]  # Top 5 refined
    
    def _score_for_query(self, document: str, query: str) -> float:
        """Score document relevance."""
        
        response = self.client.embeddings.create(
            input=[query, document],
            model="text-embedding-3-small"
        )
        
        import numpy as np
        embs = [d.embedding for d in sorted(response.data, key=lambda x: x.index)]
        
        return float(np.dot(embs[0], embs[1]))
```

## Map-Reduce for Large Context

```python
class MapReduceRAG:
    """Handle large context with map-reduce summarization."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate(
        self,
        query: str,
        documents: list[dict],
        chunk_size: int = 5
    ) -> dict:
        """Map-reduce generation for many documents."""
        
        # Map phase: summarize each chunk
        chunks = [
            documents[i:i+chunk_size]
            for i in range(0, len(documents), chunk_size)
        ]
        
        summaries = []
        for chunk in chunks:
            summary = self._map_summarize(query, chunk)
            summaries.append(summary)
        
        # Reduce phase: combine summaries
        if len(summaries) == 1:
            final_answer = summaries[0]
        else:
            final_answer = self._reduce_combine(query, summaries)
        
        return {
            "answer": final_answer,
            "num_chunks": len(chunks),
            "chunk_summaries": summaries
        }
    
    def _map_summarize(self, query: str, documents: list[dict]) -> str:
        """Summarize a chunk of documents for the query."""
        
        context = "\n\n".join(d["content"] for d in documents)
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Extract and summarize information relevant to the question. Be concise but preserve key facts."
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nDocuments:\n{context}"
                }
            ],
            max_tokens=300,
            temperature=0
        )
        
        return response.choices[0].message.content
    
    def _reduce_combine(self, query: str, summaries: list[str]) -> str:
        """Combine summaries into final answer."""
        
        summary_text = "\n\n---\n\n".join([
            f"Summary {i+1}:\n{s}"
            for i, s in enumerate(summaries)
        ])
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Synthesize the summaries into a comprehensive answer. Combine related information and resolve any redundancy."
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nSummaries:\n{summary_text}"
                }
            ],
            temperature=0
        )
        
        return response.choices[0].message.content


# Usage
map_reduce = MapReduceRAG()

# Many documents
documents = [
    {"content": f"Document {i} about the topic..."} 
    for i in range(20)
]

result = map_reduce.generate(
    "What are the key findings?",
    documents,
    chunk_size=5
)

print(f"Processed {result['num_chunks']} chunks")
print(f"Answer: {result['answer']}")
```

## Summary

```yaml
strategies_overview:
  token_budget:
    when: "Fixed model limits"
    approach: "Count and fit"
  
  smart_truncation:
    when: "Documents are long but partially relevant"
    approach: "Keep relevant sections"
  
  iterative_refinement:
    when: "Initial context may be insufficient"
    approach: "Generate, assess, refine"
  
  map_reduce:
    when: "Too many documents to fit"
    approach: "Summarize in chunks, combine"

best_practices:
  - "Always count tokens before sending to API"
  - "Preserve document structure when truncating"
  - "Consider quality vs coverage trade-offs"
  - "Cache intermediate results when possible"
```

## Next Steps

1. **Citation Tracking** - Maintain source attribution
2. **Streaming Sources** - Real-time context display
3. **Production Architecture** - Scale RAG systems
