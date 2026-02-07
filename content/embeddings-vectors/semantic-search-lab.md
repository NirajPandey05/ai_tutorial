# Lab: Build a Semantic Search Engine

Build a complete semantic search application with embeddings, vector storage, and intelligent retrieval.

## Lab Overview

```yaml
objective: "Create a fully functional semantic search engine"
duration: "45-60 minutes"
difficulty: "intermediate"

what_you_build:
  - Document ingestion pipeline
  - Semantic search with ranking
  - Metadata filtering
  - Search result highlighting
  - Simple web interface

skills_learned:
  - End-to-end search architecture
  - Chunking strategies
  - Relevance ranking
  - Production patterns
```

## Prerequisites

```python
# Install dependencies
# pip install chromadb openai python-dotenv

import os
from dotenv import load_dotenv

load_dotenv()

# Verify API key
assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY in .env"
```

## Part 1: Document Processor

```python
"""Document processing and chunking."""

from dataclasses import dataclass
import hashlib
import re
from typing import Iterator


@dataclass
class Chunk:
    """A document chunk with metadata."""
    id: str
    content: str
    metadata: dict
    

class DocumentProcessor:
    """Process documents into searchable chunks."""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_text(
        self,
        text: str,
        source: str,
        extra_metadata: dict = None
    ) -> list[Chunk]:
        """Process text into chunks."""
        
        chunks = []
        
        # Clean text
        text = self._clean_text(text)
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        # Group sentences into chunks
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunk_text = " ".join(current_chunk)
                    chunk_id = self._generate_id(source, chunk_index)
                    
                    metadata = {
                        "source": source,
                        "chunk_index": chunk_index,
                        "char_count": len(chunk_text),
                        **(extra_metadata or {})
                    }
                    
                    chunks.append(Chunk(
                        id=chunk_id,
                        content=chunk_text,
                        metadata=metadata
                    ))
                    
                    chunk_index += 1
                    
                    # Keep overlap
                    overlap_text = " ".join(current_chunk)
                    if len(overlap_text) > self.chunk_overlap:
                        current_chunk = [current_chunk[-1]]
                        current_length = len(current_chunk[0])
                    else:
                        current_chunk = []
                        current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_id = self._generate_id(source, chunk_index)
            
            metadata = {
                "source": source,
                "chunk_index": chunk_index,
                "char_count": len(chunk_text),
                **(extra_metadata or {})
            }
            
            chunks.append(Chunk(
                id=chunk_id,
                content=chunk_text,
                metadata=metadata
            ))
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters
        text = re.sub(r'[^\w\s.,!?;:\-\'\"()]', '', text)
        return text.strip()
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _generate_id(self, source: str, index: int) -> str:
        """Generate unique chunk ID."""
        content = f"{source}_{index}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


# Test the processor
processor = DocumentProcessor(chunk_size=300, chunk_overlap=30)

sample_text = """
Machine learning is a subset of artificial intelligence that enables computers 
to learn from data. Unlike traditional programming where rules are explicitly 
coded, machine learning algorithms discover patterns automatically.

Deep learning is a specialized form of machine learning using neural networks 
with many layers. These networks can learn hierarchical representations of data, 
making them powerful for tasks like image recognition and natural language 
processing.

The field has seen remarkable progress in recent years. Models like GPT and 
BERT have revolutionized how we process and generate text. Computer vision 
systems now rival human accuracy in many tasks.
"""

chunks = processor.process_text(
    sample_text,
    source="ml_intro.txt",
    extra_metadata={"category": "ai", "difficulty": "beginner"}
)

print(f"Created {len(chunks)} chunks:")
for chunk in chunks:
    print(f"\n[{chunk.id}] ({chunk.metadata['char_count']} chars)")
    print(f"  {chunk.content[:100]}...")
```

## Part 2: Vector Store Manager

```python
"""Vector store management with ChromaDB."""

import chromadb
from chromadb.utils import embedding_functions
from typing import Optional


class VectorStore:
    """Manage vector storage and search."""
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./search_db",
        embedding_model: str = "text-embedding-3-small"
    ):
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Use OpenAI embeddings
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=embedding_model
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add chunks to the vector store."""
        
        if not chunks:
            return 0
        
        self.collection.add(
            ids=[c.id for c in chunks],
            documents=[c.content for c in chunks],
            metadatas=[c.metadata for c in chunks]
        )
        
        return len(chunks)
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict = None,
        where_document: dict = None
    ) -> list[dict]:
        """Search for similar documents."""
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return formatted
    
    def get_stats(self) -> dict:
        """Get collection statistics."""
        return {
            "name": self.collection.name,
            "count": self.collection.count()
        }
    
    def clear(self):
        """Clear all documents."""
        # Delete and recreate
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )


# Initialize vector store
store = VectorStore(
    collection_name="semantic_search_lab",
    persist_directory="./lab_db"
)

print(f"Collection stats: {store.get_stats()}")
```

## Part 3: Search Engine Class

```python
"""Complete search engine implementation."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class SearchResult:
    """A search result with highlighting."""
    id: str
    content: str
    highlighted_content: str
    metadata: dict
    score: float
    rank: int


class SemanticSearchEngine:
    """Full-featured semantic search engine."""
    
    def __init__(
        self,
        persist_directory: str = "./search_engine_db",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.store = VectorStore(
            collection_name="search_engine",
            persist_directory=persist_directory
        )
    
    def index_document(
        self,
        content: str,
        source: str,
        metadata: dict = None
    ) -> int:
        """Index a document."""
        
        # Add indexing timestamp
        doc_metadata = {
            "indexed_at": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        # Process into chunks
        chunks = self.processor.process_text(
            content,
            source=source,
            extra_metadata=doc_metadata
        )
        
        # Add to vector store
        return self.store.add_chunks(chunks)
    
    def index_documents(
        self,
        documents: list[dict]
    ) -> int:
        """Batch index multiple documents."""
        
        total_chunks = 0
        for doc in documents:
            count = self.index_document(
                content=doc['content'],
                source=doc['source'],
                metadata=doc.get('metadata')
            )
            total_chunks += count
        
        return total_chunks
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        filters: dict = None,
        min_score: float = 0.0
    ) -> list[SearchResult]:
        """Search with optional filtering and ranking."""
        
        # Get raw results
        raw_results = self.store.search(
            query=query,
            n_results=n_results * 2,  # Get extra for filtering
            where=filters
        )
        
        # Filter by minimum score
        filtered = [r for r in raw_results if r['score'] >= min_score]
        
        # Highlight matches
        results = []
        for rank, result in enumerate(filtered[:n_results], 1):
            highlighted = self._highlight_matches(
                result['content'],
                query
            )
            
            results.append(SearchResult(
                id=result['id'],
                content=result['content'],
                highlighted_content=highlighted,
                metadata=result['metadata'],
                score=result['score'],
                rank=rank
            ))
        
        return results
    
    def _highlight_matches(
        self,
        content: str,
        query: str,
        highlight_tag: str = "**"
    ) -> str:
        """Highlight query terms in content."""
        
        # Extract query terms
        query_terms = query.lower().split()
        
        # Highlight each term
        highlighted = content
        for term in query_terms:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted = pattern.sub(
                f"{highlight_tag}{term}{highlight_tag}",
                highlighted
            )
        
        return highlighted
    
    def get_suggestions(
        self,
        partial_query: str,
        n_suggestions: int = 5
    ) -> list[str]:
        """Get search suggestions based on indexed content."""
        
        # Simple suggestion: search and extract key terms
        results = self.store.search(partial_query, n_results=10)
        
        # Extract unique terms from results
        terms = set()
        for result in results:
            words = result['content'].lower().split()
            for word in words:
                if partial_query.lower() in word and len(word) > 3:
                    terms.add(word)
        
        return list(terms)[:n_suggestions]
    
    def get_stats(self) -> dict:
        """Get search engine statistics."""
        return self.store.get_stats()


# Initialize search engine
engine = SemanticSearchEngine(
    persist_directory="./search_lab_db",
    chunk_size=400,
    chunk_overlap=40
)

print(f"Search engine initialized: {engine.get_stats()}")
```

## Part 4: Index Sample Data

```python
"""Index sample documents for testing."""

# Sample document collection
sample_documents = [
    {
        "source": "python_intro.txt",
        "content": """
        Python is a versatile programming language known for its simplicity and 
        readability. Created by Guido van Rossum in 1991, Python has become one 
        of the most popular languages for web development, data science, and 
        artificial intelligence. Its clean syntax uses indentation to define 
        code blocks, making programs easy to read and maintain. Python supports 
        multiple programming paradigms including procedural, object-oriented, 
        and functional programming.
        """,
        "metadata": {"category": "programming", "language": "python", "difficulty": "beginner"}
    },
    {
        "source": "machine_learning_basics.txt",
        "content": """
        Machine learning is a branch of artificial intelligence that enables 
        computers to learn from data without being explicitly programmed. 
        The three main types are supervised learning, unsupervised learning, 
        and reinforcement learning. Supervised learning uses labeled data to 
        train models for classification and regression tasks. Unsupervised 
        learning discovers patterns in unlabeled data through clustering and 
        dimensionality reduction. Reinforcement learning trains agents through 
        trial and error in interactive environments.
        """,
        "metadata": {"category": "ai", "language": "python", "difficulty": "intermediate"}
    },
    {
        "source": "neural_networks.txt",
        "content": """
        Neural networks are computational models inspired by the human brain. 
        They consist of layers of interconnected nodes called neurons. Each 
        connection has a weight that is adjusted during training. Deep learning 
        uses neural networks with many hidden layers to learn complex patterns. 
        Convolutional neural networks excel at image processing while recurrent 
        neural networks handle sequential data like text and time series.
        """,
        "metadata": {"category": "ai", "language": "python", "difficulty": "advanced"}
    },
    {
        "source": "api_design.txt",
        "content": """
        REST APIs are architectural patterns for building web services. They use 
        HTTP methods like GET, POST, PUT, and DELETE to perform operations on 
        resources. Good API design follows principles like statelessness, 
        resource-based URLs, and proper status codes. Documentation is crucial 
        for developer experience. Tools like OpenAPI and Swagger help document 
        and test APIs effectively.
        """,
        "metadata": {"category": "web", "language": "python", "difficulty": "intermediate"}
    },
    {
        "source": "database_fundamentals.txt",
        "content": """
        Databases store and organize data for applications. Relational databases 
        like PostgreSQL use tables with relationships defined by foreign keys. 
        NoSQL databases like MongoDB offer flexibility with document-based 
        storage. Vector databases are specialized for similarity search using 
        embeddings. Choosing the right database depends on data structure, 
        query patterns, and scalability requirements.
        """,
        "metadata": {"category": "databases", "language": "sql", "difficulty": "intermediate"}
    }
]

# Index all documents
total_chunks = engine.index_documents(sample_documents)
print(f"Indexed {total_chunks} chunks from {len(sample_documents)} documents")
print(f"Stats: {engine.get_stats()}")
```

## Part 5: Search Interface

```python
"""Interactive search interface."""

def display_results(results: list[SearchResult]) -> None:
    """Display search results nicely."""
    
    if not results:
        print("No results found.")
        return
    
    print(f"\n{'='*60}")
    print(f"Found {len(results)} results")
    print('='*60)
    
    for result in results:
        print(f"\n#{result.rank} | Score: {result.score:.3f}")
        print(f"Source: {result.metadata.get('source', 'Unknown')}")
        print(f"Category: {result.metadata.get('category', 'N/A')}")
        print(f"Difficulty: {result.metadata.get('difficulty', 'N/A')}")
        print("-" * 40)
        print(result.highlighted_content[:300] + "...")
        print()


def interactive_search():
    """Run interactive search loop."""
    
    print("\n🔍 Semantic Search Engine")
    print("Type 'quit' to exit, 'stats' for statistics")
    print("-" * 40)
    
    while True:
        query = input("\nSearch: ").strip()
        
        if query.lower() == 'quit':
            break
        elif query.lower() == 'stats':
            print(engine.get_stats())
            continue
        elif not query:
            continue
        
        # Perform search
        results = engine.search(
            query=query,
            n_results=5,
            min_score=0.3
        )
        
        display_results(results)


# Run a sample search
print("\n--- Sample Search: 'machine learning neural networks' ---")
results = engine.search(
    query="machine learning neural networks",
    n_results=3,
    min_score=0.3
)
display_results(results)
```

## Part 6: Filtered Search

```python
"""Search with metadata filters."""

# Search only AI-related content
print("\n--- Filtered Search: AI category only ---")
results = engine.search(
    query="learning algorithms",
    n_results=5,
    filters={"category": "ai"}
)
display_results(results)

# Search beginner content only
print("\n--- Filtered Search: Beginner difficulty ---")
results = engine.search(
    query="programming basics",
    n_results=5,
    filters={"difficulty": "beginner"}
)
display_results(results)

# Combined filters
print("\n--- Filtered Search: Python + Intermediate ---")
results = engine.search(
    query="development techniques",
    n_results=5,
    filters={
        "$and": [
            {"language": "python"},
            {"difficulty": "intermediate"}
        ]
    }
)
display_results(results)
```

## Part 7: Advanced Features

```python
"""Advanced search features."""

class AdvancedSearchEngine(SemanticSearchEngine):
    """Extended search engine with advanced features."""
    
    def search_with_reranking(
        self,
        query: str,
        n_results: int = 10,
        filters: dict = None,
        rerank_by: str = None
    ) -> list[SearchResult]:
        """Search with optional reranking."""
        
        # Get initial results
        results = self.search(query, n_results * 2, filters)
        
        if rerank_by == "recency":
            # Sort by indexed_at timestamp
            results.sort(
                key=lambda r: r.metadata.get('indexed_at', ''),
                reverse=True
            )
        elif rerank_by == "difficulty":
            # Sort by difficulty level
            difficulty_order = {"beginner": 1, "intermediate": 2, "advanced": 3}
            results.sort(
                key=lambda r: difficulty_order.get(r.metadata.get('difficulty', ''), 0)
            )
        
        # Update ranks
        for i, result in enumerate(results[:n_results], 1):
            result.rank = i
        
        return results[:n_results]
    
    def find_similar(
        self,
        document_id: str,
        n_results: int = 5
    ) -> list[SearchResult]:
        """Find documents similar to a given document."""
        
        # Get the original document
        doc = self.store.collection.get(
            ids=[document_id],
            include=["documents"]
        )
        
        if not doc['documents']:
            return []
        
        # Search for similar
        return self.search(
            query=doc['documents'][0],
            n_results=n_results + 1  # +1 to exclude self
        )[1:]  # Skip first result (itself)
    
    def aggregate_by_category(
        self,
        query: str,
        n_per_category: int = 2
    ) -> dict[str, list[SearchResult]]:
        """Get top results for each category."""
        
        # Get all results
        all_results = self.search(query, n_results=50)
        
        # Group by category
        by_category = {}
        for result in all_results:
            category = result.metadata.get('category', 'uncategorized')
            if category not in by_category:
                by_category[category] = []
            if len(by_category[category]) < n_per_category:
                by_category[category].append(result)
        
        return by_category


# Test advanced features
advanced_engine = AdvancedSearchEngine(
    persist_directory="./search_lab_db"
)

# Aggregate by category
print("\n--- Results by Category ---")
by_category = advanced_engine.aggregate_by_category(
    "programming and development",
    n_per_category=2
)

for category, results in by_category.items():
    print(f"\n📁 {category.upper()}")
    for r in results:
        print(f"  - {r.metadata.get('source', 'Unknown')} (score: {r.score:.3f})")
```

## Challenge: Build Your Own

Extend the search engine with these features:

```python
"""
Challenge exercises:

1. Add fuzzy matching for typo tolerance
2. Implement search history tracking
3. Add relevance feedback (user clicks improve ranking)
4. Create a "More Like This" feature
5. Add support for different file formats (PDF, HTML)
6. Implement search analytics dashboard

Example starter for fuzzy matching:
"""

from difflib import SequenceMatcher

def fuzzy_match(query: str, text: str, threshold: float = 0.6) -> bool:
    """Check if query fuzzy-matches text."""
    query_words = query.lower().split()
    text_words = text.lower().split()
    
    for q_word in query_words:
        best_match = max(
            SequenceMatcher(None, q_word, t_word).ratio()
            for t_word in text_words
        )
        if best_match >= threshold:
            return True
    return False

# Your implementation here!
```

## Summary

```yaml
what_you_built:
  - Document processor with chunking
  - Vector store manager
  - Full search engine class
  - Filtering and ranking
  - Advanced features
  
key_patterns:
  - Chunking with overlap
  - Similarity scoring
  - Metadata filtering
  - Result highlighting
  - Category aggregation

production_considerations:
  - Add caching for frequent queries
  - Implement rate limiting
  - Monitor search latency
  - Track relevance metrics
  - Regular index updates
```

## Next Steps

1. **RAG Integration** - Use search results for LLM context
2. **Production Deployment** - Scale your search engine
3. **Advanced Retrieval** - Explore hybrid search and reranking
