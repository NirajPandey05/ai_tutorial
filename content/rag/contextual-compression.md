# Contextual Compression

Reduce retrieved context to only the most relevant portions for your query.

## Why Contextual Compression?

```yaml
the_problem:
  retrieved_chunks: "Often contain irrelevant information"
  context_limits: "LLMs have limited context windows"
  cost_concerns: "More tokens = higher API costs"
  noise_impact: "Irrelevant context can confuse the model"

the_solution:
  contextual_compression: "Extract only relevant parts from retrieved docs"
  benefits:
    - "Smaller, focused context"
    - "Better answer quality"
    - "Lower costs"
    - "Faster responses"
```

## Basic LLM Compression

```python
from openai import OpenAI
import chromadb


class LLMCompressor:
    """Compress retrieved context using LLM."""
    
    def __init__(
        self,
        collection_name: str = "compress_docs",
        persist_directory: str = "./compress_db"
    ):
        self.client = OpenAI()
        self.chroma = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def search_and_compress(
        self,
        query: str,
        n_results: int = 5
    ) -> list[dict]:
        """Retrieve and compress context."""
        
        # Retrieve documents
        results = self._retrieve(query, n_results)
        
        # Compress each result
        compressed = []
        for r in results:
            compressed_content = self._compress(query, r["content"])
            
            if compressed_content:  # Only include if relevant content remains
                compressed.append({
                    **r,
                    "original_content": r["content"],
                    "compressed_content": compressed_content,
                    "compression_ratio": len(compressed_content) / len(r["content"])
                })
        
        return compressed
    
    def _compress(self, query: str, document: str) -> str:
        """Extract only relevant portions."""
        
        prompt = f"""Given the question and document below, extract ONLY the sentences or passages that are directly relevant to answering the question.

If no part of the document is relevant, respond with "NOT_RELEVANT".

Do not add any new information or interpretation - only extract verbatim relevant portions.

Question: {query}

Document:
{document}

Relevant portions:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        
        if result == "NOT_RELEVANT":
            return ""
        
        return result
    
    def _retrieve(self, query: str, n_results: int) -> list[dict]:
        """Basic retrieval."""
        
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
                "score": 1 - results["distances"][0][i]
            }
            for i in range(len(results["ids"][0]))
        ]


# Usage
compressor = LLMCompressor()

results = compressor.search_and_compress(
    "What is the return policy?",
    n_results=5
)

for r in results:
    print(f"Original: {len(r['original_content'])} chars")
    print(f"Compressed: {len(r['compressed_content'])} chars")
    print(f"Ratio: {r['compression_ratio']:.2%}")
    print(f"Content: {r['compressed_content'][:200]}...")
```

## Extraction-Based Compression

```python
class ExtractionCompressor:
    """Extract relevant sentences without LLM."""
    
    def __init__(self):
        self.client = OpenAI()
        self.chroma = chromadb.PersistentClient(path="./extract_db")
        self.collection = self.chroma.get_or_create_collection("extract_docs")
    
    def compress(
        self,
        query: str,
        document: str,
        threshold: float = 0.5
    ) -> str:
        """Extract relevant sentences using embedding similarity."""
        
        # Split into sentences
        sentences = self._split_sentences(document)
        
        if not sentences:
            return ""
        
        # Embed query and sentences
        query_embedding = self._embed(query)
        sentence_embeddings = self._embed_batch(sentences)
        
        # Calculate similarity
        import numpy as np
        query_vec = np.array(query_embedding)
        
        relevant = []
        for sent, emb in zip(sentences, sentence_embeddings):
            sent_vec = np.array(emb)
            similarity = np.dot(query_vec, sent_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(sent_vec)
            )
            
            if similarity >= threshold:
                relevant.append((sent, similarity))
        
        # Sort by similarity and join
        relevant.sort(key=lambda x: x[1], reverse=True)
        
        return " ".join(sent for sent, _ in relevant)
    
    def _split_sentences(self, text: str) -> list[str]:
        """Simple sentence splitting."""
        import re
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter very short sentences
        return [s.strip() for s in sentences if len(s.strip()) > 20]
    
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
compressor = ExtractionCompressor()

document = """
Our company was founded in 1995 in Seattle. We offer a 30-day return policy 
on all products. Items must be in original packaging. Our CEO won an award 
last year. Refunds are processed within 5-7 business days. We have offices 
in 12 countries. Gift cards are non-refundable.
"""

compressed = compressor.compress(
    query="What is the return policy?",
    document=document,
    threshold=0.4
)

print(f"Compressed: {compressed}")
# Output: "We offer a 30-day return policy on all products. Items must be in 
# original packaging. Refunds are processed within 5-7 business days."
```

## Context Filter Pipeline

```python
class ContextFilterPipeline:
    """Multi-stage context filtering."""
    
    def __init__(self):
        self.client = OpenAI()
        self.chroma = chromadb.PersistentClient(path="./filter_db")
        self.collection = self.chroma.get_or_create_collection("filter_docs")
    
    def search_and_filter(
        self,
        query: str,
        n_results: int = 5,
        max_tokens: int = 2000
    ) -> dict:
        """Multi-stage retrieval and filtering."""
        
        # Stage 1: Broad retrieval
        candidates = self._retrieve(query, n_results * 3)
        
        # Stage 2: Relevance filter
        relevant = self._filter_relevant(query, candidates)
        
        # Stage 3: Compress
        compressed = self._compress_batch(query, relevant)
        
        # Stage 4: Token budget
        final = self._fit_token_budget(compressed, max_tokens)
        
        return {
            "query": query,
            "stages": {
                "retrieved": len(candidates),
                "after_relevance_filter": len(relevant),
                "after_compression": len(compressed),
                "final": len(final)
            },
            "results": final,
            "total_tokens": sum(len(r["compressed_content"].split()) for r in final)
        }
    
    def _filter_relevant(
        self,
        query: str,
        candidates: list[dict],
        threshold: float = 0.4
    ) -> list[dict]:
        """Filter out clearly irrelevant documents."""
        
        return [c for c in candidates if c["score"] >= threshold]
    
    def _compress_batch(
        self,
        query: str,
        documents: list[dict]
    ) -> list[dict]:
        """Compress multiple documents."""
        
        compressed = []
        
        for doc in documents:
            relevant_content = self._extract_relevant(query, doc["content"])
            
            if relevant_content:
                compressed.append({
                    **doc,
                    "compressed_content": relevant_content
                })
        
        return compressed
    
    def _extract_relevant(self, query: str, document: str) -> str:
        """Quick extraction without LLM."""
        
        # Split into paragraphs
        paragraphs = document.split("\n\n")
        
        # Score each paragraph
        query_embedding = self._embed(query)
        
        scored = []
        for para in paragraphs:
            if len(para.strip()) < 50:
                continue
            
            para_embedding = self._embed(para)
            
            import numpy as np
            similarity = np.dot(query_embedding, para_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(para_embedding)
            )
            
            if similarity > 0.3:
                scored.append((para, similarity))
        
        # Return top paragraphs
        scored.sort(key=lambda x: x[1], reverse=True)
        return "\n\n".join(p for p, _ in scored[:3])
    
    def _fit_token_budget(
        self,
        documents: list[dict],
        max_tokens: int
    ) -> list[dict]:
        """Select documents that fit in token budget."""
        
        selected = []
        current_tokens = 0
        
        for doc in sorted(documents, key=lambda x: x["score"], reverse=True):
            # Rough token estimate
            doc_tokens = len(doc["compressed_content"].split()) * 1.3
            
            if current_tokens + doc_tokens <= max_tokens:
                selected.append(doc)
                current_tokens += doc_tokens
            else:
                break
        
        return selected
    
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
    
    def _embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            input=text, model="text-embedding-3-small"
        )
        return response.data[0].embedding
```

## Document Summary Compression

```python
class SummaryCompressor:
    """Summarize documents instead of extraction."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def compress(
        self,
        query: str,
        document: str,
        max_length: int = 200
    ) -> str:
        """Summarize document focused on query."""
        
        prompt = f"""Summarize the following document, focusing specifically on information relevant to the question.
Keep the summary under {max_length} words. Only include facts from the document.

Question: {query}

Document:
{document}

Focused summary:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_length * 2,
            temperature=0
        )
        
        return response.choices[0].message.content.strip()
    
    def compress_multiple(
        self,
        query: str,
        documents: list[str],
        strategy: str = "individual"
    ) -> str | list[str]:
        """Compress multiple documents."""
        
        if strategy == "individual":
            # Summarize each separately
            return [self.compress(query, doc) for doc in documents]
        
        elif strategy == "combined":
            # Combine then summarize
            combined = "\n\n---\n\n".join(documents)
            return self._combined_summary(query, combined)
        
        elif strategy == "map_reduce":
            # Summarize each, then combine summaries
            individual_summaries = [self.compress(query, doc) for doc in documents]
            combined_summaries = "\n\n".join(individual_summaries)
            return self._combined_summary(query, combined_summaries)
    
    def _combined_summary(self, query: str, text: str) -> str:
        """Create single summary from combined text."""
        
        prompt = f"""Create a comprehensive summary from the following information, focused on answering the question.
Synthesize information from multiple sources. Keep under 300 words.

Question: {query}

Information:
{text}

Comprehensive summary:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0
        )
        
        return response.choices[0].message.content.strip()


# Usage
compressor = SummaryCompressor()

# Individual summaries
summaries = compressor.compress_multiple(
    query="What are the company's environmental initiatives?",
    documents=[annual_report, sustainability_report, press_release],
    strategy="individual"
)

# Combined synthesis
synthesis = compressor.compress_multiple(
    query="What are the company's environmental initiatives?",
    documents=[annual_report, sustainability_report, press_release],
    strategy="map_reduce"
)
```

## Compression with Quality Checks

```python
class QualityCheckedCompressor:
    """Compression with quality validation."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def compress_and_validate(
        self,
        query: str,
        document: str
    ) -> dict:
        """Compress and validate result quality."""
        
        # Compress
        compressed = self._compress(query, document)
        
        # Validate
        validation = self._validate(query, document, compressed)
        
        return {
            "original": document,
            "compressed": compressed,
            "compression_ratio": len(compressed) / len(document) if document else 0,
            "validation": validation,
            "is_valid": validation["score"] >= 0.7
        }
    
    def _compress(self, query: str, document: str) -> str:
        """Extract relevant content."""
        
        prompt = f"""Extract the portions of this document that are relevant to the question.
Preserve exact wording where possible.

Question: {query}

Document:
{document}

Relevant portions:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0
        )
        
        return response.choices[0].message.content.strip()
    
    def _validate(
        self,
        query: str,
        original: str,
        compressed: str
    ) -> dict:
        """Validate compression quality."""
        
        prompt = f"""Evaluate this compression for the given question.

Question: {query}

Original document:
{original}

Compressed version:
{compressed}

Rate on these criteria (0-10 each):
1. Relevance: Does compressed version contain relevant information?
2. Completeness: Are all relevant facts from original preserved?
3. Accuracy: Is compressed version factually consistent with original?

Respond in format:
Relevance: [score]
Completeness: [score]
Accuracy: [score]
Issues: [any problems found]"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0
        )
        
        # Parse response
        result = response.choices[0].message.content
        
        scores = {}
        for line in result.split("\n"):
            for metric in ["Relevance", "Completeness", "Accuracy"]:
                if line.startswith(metric):
                    try:
                        scores[metric.lower()] = float(line.split(":")[1].strip().split()[0]) / 10
                    except:
                        scores[metric.lower()] = 0.5
        
        scores["score"] = sum(scores.values()) / len(scores) if scores else 0
        
        return scores
```

## Choosing Compression Strategy

```yaml
llm_compression:
  when: "Need high-quality extraction, complex documents"
  pros: "Accurate, handles nuance"
  cons: "Slow, costs tokens"

extraction_based:
  when: "Simple documents, need speed"
  pros: "Fast, no LLM cost"
  cons: "May miss context"

summary:
  when: "Very long documents, need synthesis"
  pros: "Significant reduction, readable"
  cons: "May lose specific details"

filter_pipeline:
  when: "Large retrieval sets, token budget"
  pros: "Balanced approach"
  cons: "More complex"
```

## Summary

```yaml
key_takeaways:
  1: "Compression significantly improves RAG quality"
  2: "Less context often means better answers"
  3: "Multiple strategies for different needs"
  4: "Always validate compression quality"

implementation_tips:
  - "Start with simple extraction"
  - "Add LLM compression for complex queries"
  - "Monitor compression ratios"
  - "A/B test with/without compression"
```

## Next Steps

1. **Generation with Context** - Use compressed context in prompts
2. **Citation Tracking** - Maintain source attribution
3. **RAG Evaluation** - Measure end-to-end quality
