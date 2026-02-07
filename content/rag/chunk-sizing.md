# Chunk Sizing and Overlap Guide

Finding the optimal chunk size is crucial for RAG performance. Too small loses context, too large wastes tokens.

## The Chunk Size Problem

```yaml
too_small_chunks:
  problem: "Fragments lose meaning"
  example:
    chunk: "increasing by 23%"
    query: "What was the revenue growth?"
    result: "Found match but no context about WHAT increased"

too_large_chunks:
  problem: "Diluted relevance, wasted context"
  example:
    chunk: "5000 characters covering 10 topics"
    query: "What's the refund policy?"
    result: "Found chunk with 1 relevant sentence and 99 irrelevant ones"

just_right:
  goal: "Self-contained units that answer potential questions"
  result: "High relevance, sufficient context, efficient token usage"
```

## Chunk Size Guidelines

### By Content Type

```python
CHUNK_SIZE_RECOMMENDATIONS = {
    # Technical documentation
    "api_docs": {
        "chunk_size": 400,
        "overlap": 50,
        "reasoning": "API methods are discrete units"
    },
    
    # Conversational content
    "faq": {
        "chunk_size": 300,
        "overlap": 0,
        "reasoning": "Q&A pairs should stay together"
    },
    
    # Long-form articles
    "articles": {
        "chunk_size": 600,
        "overlap": 100,
        "reasoning": "Need paragraph-level context"
    },
    
    # Legal/Policy documents
    "legal": {
        "chunk_size": 800,
        "overlap": 150,
        "reasoning": "Clauses need full context"
    },
    
    # Code documentation
    "code": {
        "chunk_size": 500,
        "overlap": 50,
        "reasoning": "Functions as natural units"
    },
    
    # Chat logs/conversations
    "chat": {
        "chunk_size": 400,
        "overlap": 100,
        "reasoning": "Conversation context matters"
    },
    
    # Research papers
    "academic": {
        "chunk_size": 700,
        "overlap": 100,
        "reasoning": "Dense information, need full paragraphs"
    }
}
```

### By Model Context Window

```python
def optimal_chunk_size(
    model_context_window: int,
    num_chunks_to_retrieve: int = 5,
    prompt_overhead: int = 500,
    answer_tokens: int = 500
) -> int:
    """Calculate optimal chunk size based on model limits."""
    
    available_for_context = model_context_window - prompt_overhead - answer_tokens
    
    # Leave some buffer
    usable = int(available_for_context * 0.8)
    
    # Divide by number of chunks
    optimal_size = usable // num_chunks_to_retrieve
    
    return min(optimal_size, 1000)  # Cap at 1000


# Examples for different models
model_configs = {
    "gpt-4o-mini": {"context": 128000, "optimal": optimal_chunk_size(128000)},  # ~20k
    "gpt-4o": {"context": 128000, "optimal": optimal_chunk_size(128000)},       # ~20k
    "claude-3-haiku": {"context": 200000, "optimal": optimal_chunk_size(200000)}, # ~31k
    "claude-3-sonnet": {"context": 200000, "optimal": optimal_chunk_size(200000)},
    "llama-3-8b": {"context": 8192, "optimal": optimal_chunk_size(8192)},       # ~1100
}

# For most practical purposes, 400-800 characters works well
# Even with large context windows, smaller chunks = better retrieval
```

## Understanding Overlap

```yaml
why_overlap:
  - "Maintains continuity between chunks"
  - "Prevents losing context at boundaries"
  - "Helps with sentences split across chunks"
  - "Improves recall for edge-case queries"

overlap_tradeoffs:
  more_overlap:
    pros: "Better context preservation"
    cons: "More redundancy, higher storage, slower indexing"
  
  less_overlap:
    pros: "Less redundancy, faster processing"
    cons: "May lose context at boundaries"
```

### Overlap Strategies

```python
def calculate_overlap(chunk_size: int, strategy: str = "percentage") -> int:
    """Calculate overlap based on strategy."""
    
    strategies = {
        # 10-20% of chunk size
        "percentage": int(chunk_size * 0.15),
        
        # Fixed overlap regardless of size
        "fixed": 50,
        
        # Sentence-based (roughly 1-2 sentences)
        "sentence": 100,
        
        # No overlap (for discrete units like Q&A)
        "none": 0
    }
    
    return strategies.get(strategy, strategies["percentage"])


# When to use each
overlap_guide = {
    "percentage": "General documents, articles",
    "fixed": "When chunk sizes vary",
    "sentence": "Narrative content, stories",
    "none": "FAQs, structured data, code functions"
}
```

## Measuring Chunk Quality

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class ChunkQualityMetrics:
    avg_size: float
    size_std: float
    min_size: int
    max_size: int
    total_chunks: int
    orphan_sentences: int  # Sentences split across chunks


class ChunkAnalyzer:
    """Analyze chunk quality."""
    
    def analyze(self, chunks: list[str], original_text: str) -> ChunkQualityMetrics:
        """Compute quality metrics for chunks."""
        
        sizes = [len(c) for c in chunks]
        
        # Check for orphan sentences
        orphans = self._count_orphan_sentences(chunks)
        
        return ChunkQualityMetrics(
            avg_size=sum(sizes) / len(sizes),
            size_std=self._std(sizes),
            min_size=min(sizes),
            max_size=max(sizes),
            total_chunks=len(chunks),
            orphan_sentences=orphans
        )
    
    def _count_orphan_sentences(self, chunks: list[str]) -> int:
        """Count sentences that might be split."""
        orphans = 0
        for chunk in chunks:
            # Chunk starts mid-sentence (no capital or starts with lowercase)
            if chunk and chunk[0].islower():
                orphans += 1
            # Chunk ends mid-sentence (no period/question/exclamation)
            if chunk and chunk[-1] not in '.!?':
                orphans += 1
        return orphans
    
    def _std(self, values: list[float]) -> float:
        """Calculate standard deviation."""
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5


# Usage
analyzer = ChunkAnalyzer()
metrics = analyzer.analyze(chunks, original_text)

print(f"Average chunk size: {metrics.avg_size:.0f}")
print(f"Size variation (std): {metrics.size_std:.0f}")
print(f"Size range: {metrics.min_size} - {metrics.max_size}")
print(f"Orphan sentences: {metrics.orphan_sentences}")
```

## Adaptive Chunking

```python
class AdaptiveChunker:
    """Dynamically adjust chunk parameters based on content."""
    
    def __init__(
        self,
        target_size: int = 500,
        min_size: int = 100,
        max_size: int = 1000
    ):
        self.target_size = target_size
        self.min_size = min_size
        self.max_size = max_size
    
    def chunk(self, text: str) -> list[str]:
        """Chunk with adaptive sizing."""
        
        # Analyze content
        content_density = self._analyze_density(text)
        
        # Adjust target size based on density
        adjusted_size = self._adjust_size(content_density)
        
        # Perform chunking
        return self._recursive_chunk(text, adjusted_size)
    
    def _analyze_density(self, text: str) -> float:
        """Analyze information density (0-1 scale)."""
        
        # Simple heuristics
        words = text.split()
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        
        # Technical content has longer words
        # Conversational has shorter words
        
        # Normalize to 0-1
        density = min(avg_word_length / 8, 1.0)  # 8 chars = high density
        
        return density
    
    def _adjust_size(self, density: float) -> int:
        """Adjust chunk size based on density."""
        
        # Dense content → smaller chunks (capture specific info)
        # Sparse content → larger chunks (need more context)
        
        # Inverse relationship
        adjustment = 1 - (density * 0.3)  # Max 30% adjustment
        adjusted = int(self.target_size * adjustment)
        
        return max(self.min_size, min(self.max_size, adjusted))
    
    def _recursive_chunk(self, text: str, target_size: int) -> list[str]:
        """Perform recursive chunking with target size."""
        
        separators = ["\n\n", "\n", ". ", " "]
        return self._split_recursive(text, separators, target_size)
    
    def _split_recursive(
        self,
        text: str,
        separators: list[str],
        target_size: int
    ) -> list[str]:
        """Recursively split text."""
        
        if len(text) <= target_size:
            return [text.strip()] if text.strip() else []
        
        if not separators:
            # Force split at target size
            chunks = []
            for i in range(0, len(text), target_size):
                chunk = text[i:i + target_size].strip()
                if chunk:
                    chunks.append(chunk)
            return chunks
        
        sep = separators[0]
        parts = text.split(sep) if sep else list(text)
        
        chunks = []
        current = ""
        
        for part in parts:
            test = current + (sep if current else "") + part
            
            if len(test) <= target_size:
                current = test
            else:
                if current:
                    chunks.extend(
                        self._split_recursive(current, separators[1:], target_size)
                    )
                current = part
        
        if current:
            chunks.extend(
                self._split_recursive(current, separators[1:], target_size)
            )
        
        return chunks


# Usage
chunker = AdaptiveChunker(
    target_size=500,
    min_size=200,
    max_size=800
)

# Technical doc → smaller chunks
tech_chunks = chunker.chunk(technical_documentation)

# Blog post → larger chunks
blog_chunks = chunker.chunk(blog_content)
```

## Testing Chunk Configurations

```python
from openai import OpenAI
import chromadb


class ChunkConfigTester:
    """Test different chunking configurations."""
    
    def __init__(self):
        self.client = OpenAI()
        self.chroma = chromadb.Client()
    
    def test_config(
        self,
        documents: list[str],
        test_queries: list[dict],  # {"query": str, "expected_content": str}
        chunk_size: int,
        overlap: int
    ) -> dict:
        """Test a chunking configuration."""
        
        # Create temporary collection
        collection = self.chroma.create_collection(
            name=f"test_{chunk_size}_{overlap}",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Chunk and index documents
        all_chunks = []
        for doc in documents:
            chunks = recursive_chunking(doc, chunk_size, overlap)
            all_chunks.extend(chunks)
        
        embeddings = self._embed_batch(all_chunks)
        
        collection.add(
            ids=[f"chunk_{i}" for i in range(len(all_chunks))],
            documents=all_chunks,
            embeddings=embeddings
        )
        
        # Test retrieval
        results = {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "total_chunks": len(all_chunks),
            "retrieval_scores": []
        }
        
        for test in test_queries:
            query_embedding = self._embed(test["query"])
            
            retrieved = collection.query(
                query_embeddings=[query_embedding],
                n_results=5
            )
            
            # Check if expected content is in retrieved chunks
            found = any(
                test["expected_content"].lower() in doc.lower()
                for doc in retrieved["documents"][0]
            )
            
            results["retrieval_scores"].append({
                "query": test["query"],
                "found_expected": found,
                "top_score": 1 - retrieved["distances"][0][0]
            })
        
        # Cleanup
        self.chroma.delete_collection(f"test_{chunk_size}_{overlap}")
        
        # Calculate overall score
        results["success_rate"] = sum(
            1 for r in results["retrieval_scores"] if r["found_expected"]
        ) / len(results["retrieval_scores"])
        
        return results
    
    def compare_configs(
        self,
        documents: list[str],
        test_queries: list[dict],
        configs: list[dict]
    ) -> list[dict]:
        """Compare multiple configurations."""
        
        results = []
        for config in configs:
            result = self.test_config(
                documents,
                test_queries,
                config["chunk_size"],
                config["overlap"]
            )
            results.append(result)
            print(f"Config {config}: Success rate = {result['success_rate']:.2%}")
        
        return sorted(results, key=lambda x: x["success_rate"], reverse=True)
    
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
tester = ChunkConfigTester()

documents = [load_document("handbook.pdf"), load_document("faq.md")]
test_queries = [
    {"query": "How many vacation days?", "expected_content": "20 days"},
    {"query": "What's the remote work policy?", "expected_content": "3 days per week"},
]

configs = [
    {"chunk_size": 300, "overlap": 30},
    {"chunk_size": 500, "overlap": 50},
    {"chunk_size": 700, "overlap": 70},
    {"chunk_size": 1000, "overlap": 100},
]

results = tester.compare_configs(documents, test_queries, configs)
print(f"\nBest config: {results[0]}")
```

## Rules of Thumb

```yaml
general_guidelines:
  chunk_size:
    minimum: "200 characters (avoid fragments)"
    sweet_spot: "400-600 characters (most content)"
    maximum: "1000 characters (preserve context)"
  
  overlap:
    minimum: "0 (discrete units like Q&A)"
    sweet_spot: "10-20% of chunk size"
    maximum: "30% (heavy context dependency)"

content_specific:
  technical_docs:
    size: "400-500"
    overlap: "50"
    reason: "Methods/functions are natural units"
  
  conversational:
    size: "300-400"
    overlap: "50-100"
    reason: "Keep question-answer context"
  
  narrative:
    size: "500-700"
    overlap: "100"
    reason: "Paragraphs tell stories"
  
  legal:
    size: "700-1000"
    overlap: "150"
    reason: "Clauses need full context"

testing_recommendations:
  1: "Start with 500 characters, 50 overlap"
  2: "Test with real queries from users"
  3: "Measure retrieval success rate"
  4: "Adjust based on failure analysis"
  5: "Different sections may need different configs"
```

## Summary

```yaml
key_principles:
  1: "Chunk size affects retrieval precision"
  2: "Overlap preserves cross-chunk context"
  3: "Content type determines optimal parameters"
  4: "Test with actual queries"
  5: "One size doesn't fit all"

optimization_process:
  1: "Analyze your content types"
  2: "Start with recommended defaults"
  3: "Create test queries with expected results"
  4: "Measure and compare configurations"
  5: "Iterate based on failure cases"
```

## Next Steps

1. **Document Parsing** - Handle complex document formats
2. **Document Processor Lab** - Build a complete processor
3. **Retrieval Strategies** - Improve search quality
