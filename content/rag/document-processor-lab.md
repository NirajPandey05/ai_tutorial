# Lab: Build a Document Processing Pipeline

Build a complete document processing pipeline that handles multiple formats, chunks intelligently, and prepares content for RAG.

## Lab Overview

```yaml
objective: "Create a production-ready document processor"
duration: "45-60 minutes"
difficulty: "intermediate"

what_you_build:
  - Multi-format document loader
  - Configurable chunking pipeline
  - Embedding and indexing system
  - Quality analysis tools

skills_learned:
  - Document parsing techniques
  - Chunking strategy implementation
  - Pipeline architecture
  - Error handling and logging
```

## Prerequisites

```python
# Install dependencies
# pip install chromadb openai python-dotenv pymupdf beautifulsoup4 python-docx

import os
from dotenv import load_dotenv

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY in .env"
```

## Part 1: Document Loader

```python
"""Multi-format document loader."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator, Optional
from datetime import datetime
import hashlib


@dataclass
class Document:
    """Loaded document with metadata."""
    id: str
    content: str
    source: str
    format: str
    title: str
    metadata: dict = field(default_factory=dict)
    loaded_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def word_count(self) -> int:
        return len(self.content.split())
    
    @property
    def char_count(self) -> int:
        return len(self.content)


class DocumentLoader:
    """Load documents from various formats."""
    
    SUPPORTED_FORMATS = {
        ".pdf": "pdf",
        ".html": "html",
        ".htm": "html",
        ".md": "markdown",
        ".txt": "text",
        ".docx": "docx",
        ".py": "code",
        ".js": "code",
        ".ts": "code",
    }
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._loaded_count = 0
        self._error_count = 0
    
    def load(self, source: str) -> Document:
        """Load a single document."""
        
        path = Path(source)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")
        
        format_type = self.SUPPORTED_FORMATS.get(path.suffix.lower())
        if not format_type:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        # Generate unique ID
        doc_id = self._generate_id(source)
        
        # Load based on format
        loader_method = getattr(self, f"_load_{format_type}")
        content, title, metadata = loader_method(path)
        
        self._loaded_count += 1
        
        if self.verbose:
            print(f"✓ Loaded: {path.name} ({len(content)} chars)")
        
        return Document(
            id=doc_id,
            content=content,
            source=str(path),
            format=format_type,
            title=title,
            metadata=metadata
        )
    
    def load_directory(
        self,
        directory: str,
        recursive: bool = True,
        extensions: Optional[list[str]] = None
    ) -> Iterator[Document]:
        """Load all documents from a directory."""
        
        path = Path(directory)
        pattern = "**/*" if recursive else "*"
        
        for file_path in sorted(path.glob(pattern)):
            if not file_path.is_file():
                continue
            
            suffix = file_path.suffix.lower()
            
            # Filter by extensions
            if extensions and suffix not in extensions:
                continue
            
            # Skip unsupported
            if suffix not in self.SUPPORTED_FORMATS:
                continue
            
            try:
                yield self.load(str(file_path))
            except Exception as e:
                self._error_count += 1
                if self.verbose:
                    print(f"✗ Error loading {file_path}: {e}")
    
    def get_stats(self) -> dict:
        """Get loading statistics."""
        return {
            "loaded": self._loaded_count,
            "errors": self._error_count
        }
    
    def _generate_id(self, source: str) -> str:
        """Generate unique document ID."""
        return hashlib.md5(source.encode()).hexdigest()[:12]
    
    def _load_pdf(self, path: Path) -> tuple[str, str, dict]:
        """Load PDF document."""
        import fitz
        
        doc = fitz.open(path)
        
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text())
        
        content = "\n\n".join(pages_text)
        title = doc.metadata.get("title", path.stem)
        metadata = {
            "pages": len(doc),
            "author": doc.metadata.get("author", ""),
            "created": doc.metadata.get("creationDate", "")
        }
        
        return content, title, metadata
    
    def _load_html(self, path: Path) -> tuple[str, str, dict]:
        """Load HTML document."""
        from bs4 import BeautifulSoup
        
        with open(path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        
        # Remove scripts and styles
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        
        content = soup.get_text(separator="\n", strip=True)
        title = soup.title.string if soup.title else path.stem
        metadata = {}
        
        return content, title, metadata
    
    def _load_markdown(self, path: Path) -> tuple[str, str, dict]:
        """Load Markdown document."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Extract title from first heading
        import re
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else path.stem
        
        return content, title, {}
    
    def _load_text(self, path: Path) -> tuple[str, str, dict]:
        """Load plain text document."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return content, path.stem, {}
    
    def _load_docx(self, path: Path) -> tuple[str, str, dict]:
        """Load Word document."""
        from docx import Document as DocxDocument
        
        doc = DocxDocument(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        content = "\n\n".join(paragraphs)
        
        return content, path.stem, {"paragraphs": len(paragraphs)}
    
    def _load_code(self, path: Path) -> tuple[str, str, dict]:
        """Load source code file."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        metadata = {
            "language": path.suffix[1:],
            "lines": content.count("\n") + 1
        }
        
        return content, path.name, metadata


# Test the loader
loader = DocumentLoader()

# Create sample documents for testing
sample_md = """# Sample Document

This is a sample document for testing the document processor.

## Section 1

This section contains important information about our topic.
The content here is relevant to search queries.

## Section 2

More content follows here with additional details.
"""

# Save sample
Path("./test_docs").mkdir(exist_ok=True)
with open("./test_docs/sample.md", "w") as f:
    f.write(sample_md)

# Load it
doc = loader.load("./test_docs/sample.md")
print(f"\nLoaded: {doc.title}")
print(f"Format: {doc.format}")
print(f"Words: {doc.word_count}")
```

## Part 2: Chunking Pipeline

```python
"""Configurable text chunking pipeline."""

from dataclasses import dataclass
from typing import Callable
import re


@dataclass
class Chunk:
    """A document chunk."""
    id: str
    content: str
    document_id: str
    document_source: str
    chunk_index: int
    metadata: dict
    
    @property
    def char_count(self) -> int:
        return len(self.content)


class ChunkingPipeline:
    """Configurable chunking with multiple strategies."""
    
    def __init__(
        self,
        strategy: str = "recursive",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 50
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        self._strategies = {
            "fixed": self._fixed_chunking,
            "recursive": self._recursive_chunking,
            "sentence": self._sentence_chunking,
            "markdown": self._markdown_chunking,
        }
    
    def chunk_document(self, document: Document) -> list[Chunk]:
        """Chunk a document using configured strategy."""
        
        # Auto-select strategy for certain formats
        if self.strategy == "auto":
            strategy = self._auto_select_strategy(document)
        else:
            strategy = self.strategy
        
        # Get chunking function
        chunk_fn = self._strategies.get(strategy, self._recursive_chunking)
        
        # Perform chunking
        text_chunks = chunk_fn(document.content)
        
        # Filter small chunks
        text_chunks = [c for c in text_chunks if len(c) >= self.min_chunk_size]
        
        # Create Chunk objects
        chunks = []
        for i, text in enumerate(text_chunks):
            chunk_id = f"{document.id}_chunk_{i}"
            chunks.append(Chunk(
                id=chunk_id,
                content=text,
                document_id=document.id,
                document_source=document.source,
                chunk_index=i,
                metadata={
                    **document.metadata,
                    "document_title": document.title,
                    "chunk_strategy": strategy
                }
            ))
        
        return chunks
    
    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks
    
    def _auto_select_strategy(self, document: Document) -> str:
        """Auto-select best strategy for document type."""
        if document.format == "markdown":
            return "markdown"
        elif document.format == "code":
            return "fixed"  # Preserve code structure
        else:
            return "recursive"
    
    def _fixed_chunking(self, text: str) -> list[str]:
        """Simple fixed-size chunking."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks
    
    def _recursive_chunking(self, text: str) -> list[str]:
        """Recursive chunking with natural boundaries."""
        separators = ["\n\n", "\n", ". ", " "]
        return self._split_recursive(text, separators)
    
    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text."""
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        
        if not separators:
            # Force split
            return self._fixed_chunking(text)
        
        sep = separators[0]
        remaining_seps = separators[1:]
        
        splits = text.split(sep) if sep else [text]
        
        chunks = []
        current = ""
        
        for split in splits:
            test = current + (sep if current else "") + split
            
            if len(test) <= self.chunk_size:
                current = test
            else:
                if current:
                    chunks.extend(self._split_recursive(current, remaining_seps))
                current = split
        
        if current:
            chunks.extend(self._split_recursive(current, remaining_seps))
        
        return chunks
    
    def _sentence_chunking(self, text: str) -> list[str]:
        """Chunk by sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            if current_size + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep overlap
                overlap_text = " ".join(current_chunk)
                if len(overlap_text) > self.chunk_overlap:
                    current_chunk = current_chunk[-1:]
                    current_size = len(current_chunk[0]) if current_chunk else 0
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += len(sentence)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _markdown_chunking(self, text: str) -> list[str]:
        """Chunk markdown by headers."""
        # Split by headers
        pattern = r'(?=^#{1,6}\s)'
        sections = re.split(pattern, text, flags=re.MULTILINE)
        
        chunks = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                # Recursively chunk large sections
                chunks.extend(self._recursive_chunking(section))
        
        return chunks


# Test chunking
pipeline = ChunkingPipeline(
    strategy="recursive",
    chunk_size=200,
    chunk_overlap=30
)

chunks = pipeline.chunk_document(doc)
print(f"\nCreated {len(chunks)} chunks:")
for chunk in chunks[:3]:
    print(f"\n[{chunk.id}] ({chunk.char_count} chars)")
    print(f"  {chunk.content[:100]}...")
```

## Part 3: Embedding & Indexing

```python
"""Embedding and vector store integration."""

from openai import OpenAI
import chromadb
from typing import Optional


class EmbeddingService:
    """Generate embeddings for text."""
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 100
    ):
        self.client = OpenAI()
        self.model = model
        self.batch_size = batch_size
        self._cache = {}
    
    def embed(self, text: str) -> list[float]:
        """Embed single text."""
        if text in self._cache:
            return self._cache[text]
        
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        embedding = response.data[0].embedding
        self._cache[text] = embedding
        return embedding
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts efficiently."""
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model
            )
            sorted_data = sorted(response.data, key=lambda x: x.index)
            embeddings.extend([d.embedding for d in sorted_data])
        
        return embeddings
    
    def embed_chunks(self, chunks: list[Chunk]) -> list[dict]:
        """Embed chunks and return with metadata."""
        texts = [c.content for c in chunks]
        embeddings = self.embed_batch(texts)
        
        return [
            {
                "id": chunk.id,
                "embedding": emb,
                "content": chunk.content,
                "metadata": {
                    **chunk.metadata,
                    "document_id": chunk.document_id,
                    "document_source": chunk.document_source,
                    "chunk_index": chunk.chunk_index
                }
            }
            for chunk, emb in zip(chunks, embeddings)
        ]


class VectorIndex:
    """ChromaDB vector store wrapper."""
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./vector_db"
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add(self, embedded_chunks: list[dict]) -> int:
        """Add embedded chunks to index."""
        if not embedded_chunks:
            return 0
        
        self.collection.add(
            ids=[c["id"] for c in embedded_chunks],
            embeddings=[c["embedding"] for c in embedded_chunks],
            documents=[c["content"] for c in embedded_chunks],
            metadatas=[c["metadata"] for c in embedded_chunks]
        )
        
        return len(embedded_chunks)
    
    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: Optional[dict] = None
    ) -> list[dict]:
        """Search for similar chunks."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
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
    
    def count(self) -> int:
        """Get total indexed chunks."""
        return self.collection.count()
    
    def clear(self) -> None:
        """Clear all indexed data."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )


# Test embedding and indexing
embedder = EmbeddingService()
index = VectorIndex(
    collection_name="doc_processor_lab",
    persist_directory="./lab_vector_db"
)

# Embed and index chunks
print("\nEmbedding chunks...")
embedded = embedder.embed_chunks(chunks)
count = index.add(embedded)
print(f"Indexed {count} chunks")
print(f"Total in index: {index.count()}")
```

## Part 4: Complete Pipeline

```python
"""Complete document processing pipeline."""

from dataclasses import dataclass
from typing import Iterator, Optional
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    chunk_strategy: str = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_size: int = 50
    embedding_model: str = "text-embedding-3-small"
    collection_name: str = "documents"
    persist_directory: str = "./pipeline_db"


class DocumentPipeline:
    """Complete document processing pipeline."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        self.loader = DocumentLoader(verbose=False)
        self.chunker = ChunkingPipeline(
            strategy=self.config.chunk_strategy,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            min_chunk_size=self.config.min_chunk_size
        )
        self.embedder = EmbeddingService(model=self.config.embedding_model)
        self.index = VectorIndex(
            collection_name=self.config.collection_name,
            persist_directory=self.config.persist_directory
        )
        
        self._stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "chunks_indexed": 0,
            "errors": []
        }
    
    def process_file(self, path: str) -> int:
        """Process a single file."""
        try:
            # Load
            document = self.loader.load(path)
            logger.info(f"Loaded: {path} ({document.char_count} chars)")
            
            # Chunk
            chunks = self.chunker.chunk_document(document)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Embed
            embedded = self.embedder.embed_chunks(chunks)
            
            # Index
            count = self.index.add(embedded)
            logger.info(f"Indexed {count} chunks")
            
            # Update stats
            self._stats["documents_processed"] += 1
            self._stats["chunks_created"] += len(chunks)
            self._stats["chunks_indexed"] += count
            
            return count
            
        except Exception as e:
            self._stats["errors"].append({"path": path, "error": str(e)})
            logger.error(f"Error processing {path}: {e}")
            return 0
    
    def process_directory(
        self,
        directory: str,
        recursive: bool = True,
        extensions: Optional[list[str]] = None
    ) -> int:
        """Process all files in a directory."""
        total = 0
        
        for document in self.loader.load_directory(directory, recursive, extensions):
            chunks = self.chunker.chunk_document(document)
            embedded = self.embedder.embed_chunks(chunks)
            count = self.index.add(embedded)
            total += count
            
            self._stats["documents_processed"] += 1
            self._stats["chunks_created"] += len(chunks)
            self._stats["chunks_indexed"] += count
            
            logger.info(f"Processed: {document.title} -> {count} chunks")
        
        return total
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filters: Optional[dict] = None
    ) -> list[dict]:
        """Search indexed documents."""
        query_embedding = self.embedder.embed(query)
        return self.index.search(query_embedding, n_results, filters)
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            **self._stats,
            "total_indexed": self.index.count()
        }
    
    def clear(self) -> None:
        """Clear all indexed data."""
        self.index.clear()
        self._stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "chunks_indexed": 0,
            "errors": []
        }


# Create and test pipeline
config = PipelineConfig(
    chunk_strategy="recursive",
    chunk_size=300,
    chunk_overlap=50,
    collection_name="pipeline_lab"
)

pipeline = DocumentPipeline(config)

# Process test documents
pipeline.process_file("./test_docs/sample.md")

# Search
results = pipeline.search("important information")
print("\nSearch Results:")
for r in results:
    print(f"  Score: {r['score']:.3f}")
    print(f"  Content: {r['content'][:100]}...")
    print()

# Stats
print(f"Pipeline Stats: {pipeline.get_stats()}")
```

## Part 5: Quality Analysis

```python
"""Analyze chunk quality and pipeline performance."""

from dataclasses import dataclass
import statistics


@dataclass
class ChunkAnalysis:
    total_chunks: int
    avg_size: float
    min_size: int
    max_size: int
    size_std: float
    orphan_rate: float  # Chunks starting/ending mid-sentence


class QualityAnalyzer:
    """Analyze chunking quality."""
    
    def analyze_chunks(self, chunks: list[Chunk]) -> ChunkAnalysis:
        """Analyze chunk quality metrics."""
        
        if not chunks:
            return ChunkAnalysis(0, 0, 0, 0, 0, 0)
        
        sizes = [c.char_count for c in chunks]
        
        # Count orphans
        orphans = 0
        for chunk in chunks:
            content = chunk.content
            # Starts mid-sentence
            if content and content[0].islower():
                orphans += 1
            # Ends mid-sentence
            if content and content[-1] not in '.!?':
                orphans += 1
        
        orphan_rate = orphans / (len(chunks) * 2)  # 2 checks per chunk
        
        return ChunkAnalysis(
            total_chunks=len(chunks),
            avg_size=statistics.mean(sizes),
            min_size=min(sizes),
            max_size=max(sizes),
            size_std=statistics.stdev(sizes) if len(sizes) > 1 else 0,
            orphan_rate=orphan_rate
        )
    
    def compare_strategies(
        self,
        document: Document,
        strategies: list[str],
        chunk_sizes: list[int]
    ) -> list[dict]:
        """Compare different chunking configurations."""
        
        results = []
        
        for strategy in strategies:
            for size in chunk_sizes:
                chunker = ChunkingPipeline(
                    strategy=strategy,
                    chunk_size=size,
                    chunk_overlap=int(size * 0.1)
                )
                
                chunks = chunker.chunk_document(document)
                analysis = self.analyze_chunks(chunks)
                
                results.append({
                    "strategy": strategy,
                    "chunk_size": size,
                    "total_chunks": analysis.total_chunks,
                    "avg_size": analysis.avg_size,
                    "size_std": analysis.size_std,
                    "orphan_rate": analysis.orphan_rate
                })
        
        return results


# Analyze our chunks
analyzer = QualityAnalyzer()
analysis = analyzer.analyze_chunks(chunks)

print("\nChunk Quality Analysis:")
print(f"  Total chunks: {analysis.total_chunks}")
print(f"  Average size: {analysis.avg_size:.0f} chars")
print(f"  Size range: {analysis.min_size} - {analysis.max_size}")
print(f"  Size std dev: {analysis.size_std:.0f}")
print(f"  Orphan rate: {analysis.orphan_rate:.1%}")

# Compare strategies
comparison = analyzer.compare_strategies(
    doc,
    strategies=["fixed", "recursive", "sentence"],
    chunk_sizes=[200, 400, 600]
)

print("\nStrategy Comparison:")
for r in comparison:
    print(f"  {r['strategy']:12} size={r['chunk_size']:3}: "
          f"{r['total_chunks']:3} chunks, "
          f"avg={r['avg_size']:.0f}, "
          f"orphans={r['orphan_rate']:.1%}")
```

## Part 6: Interactive Testing

```python
"""Interactive pipeline testing."""


def interactive_test(pipeline: DocumentPipeline):
    """Interactive testing interface."""
    
    print("\n🔍 Document Pipeline Tester")
    print("Commands: search <query>, stats, clear, quit")
    print("-" * 40)
    
    while True:
        cmd = input("\n> ").strip()
        
        if not cmd:
            continue
        
        if cmd.lower() == "quit":
            break
        
        if cmd.lower() == "stats":
            stats = pipeline.get_stats()
            print(f"Documents: {stats['documents_processed']}")
            print(f"Chunks: {stats['chunks_indexed']}")
            print(f"Errors: {len(stats['errors'])}")
            continue
        
        if cmd.lower() == "clear":
            pipeline.clear()
            print("Index cleared!")
            continue
        
        if cmd.lower().startswith("search "):
            query = cmd[7:]
            results = pipeline.search(query, n_results=3)
            
            if not results:
                print("No results found.")
                continue
            
            print(f"\nFound {len(results)} results:")
            for i, r in enumerate(results, 1):
                print(f"\n{i}. Score: {r['score']:.3f}")
                print(f"   Source: {r['metadata'].get('document_source', 'Unknown')}")
                print(f"   {r['content'][:150]}...")
            continue
        
        print("Unknown command. Try: search <query>, stats, clear, quit")


# Run interactive test
# interactive_test(pipeline)
```

## Challenge Exercises

```python
"""
Challenge exercises to extend the pipeline:

1. Add support for more file formats (CSV, JSON, XML)
2. Implement parallel processing for large directories
3. Add deduplication to avoid indexing duplicate content
4. Create a web API for the pipeline
5. Add support for incremental updates (only process changed files)

Example starter for CSV support:
"""

import csv

def load_csv(path: str) -> Document:
    """Load CSV as searchable document."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Convert rows to text
    content_parts = []
    for row in rows:
        parts = [f"{k}: {v}" for k, v in row.items() if v]
        content_parts.append(" | ".join(parts))
    
    content = "\n".join(content_parts)
    
    return Document(
        id=hashlib.md5(path.encode()).hexdigest()[:12],
        content=content,
        source=path,
        format="csv",
        title=Path(path).stem,
        metadata={"rows": len(rows), "columns": len(rows[0]) if rows else 0}
    )

# Your implementations here!
```

## Summary

```yaml
what_you_built:
  - Multi-format document loader
  - Configurable chunking pipeline
  - Embedding and indexing system
  - Quality analysis tools
  - Interactive testing interface

key_patterns:
  - Modular component design
  - Configuration objects
  - Error handling and logging
  - Statistics tracking
  - Quality measurement

next_steps:
  - Add more document formats
  - Implement parallel processing
  - Build a web interface
  - Add retrieval strategies
```

## Next Steps

1. **Retrieval Strategies** - Improve search quality
2. **RAG Generation** - Build complete Q&A system
3. **Evaluation** - Measure pipeline quality
