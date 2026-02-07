# RAG Architecture: System Design Deep Dive

Understand the components, data flows, and design decisions in building production RAG systems.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RAG System Architecture                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌─────────────────────────────────────────────┐   │
│  │   Documents  │    │            INDEXING PIPELINE                │   │
│  │  ┌────────┐  │    │  ┌──────┐  ┌────────┐  ┌────────────────┐  │   │
│  │  │  PDF   │──┼───►│  │Loader│─►│Chunker │─►│Embedding Model │  │   │
│  │  │  HTML  │  │    │  └──────┘  └────────┘  └───────┬────────┘  │   │
│  │  │  Code  │  │    │                                │           │   │
│  │  │  etc.  │  │    │                                ▼           │   │
│  │  └────────┘  │    │                     ┌──────────────────┐   │   │
│  └──────────────┘    │                     │  Vector Database │   │   │
│                      │                     └──────────────────┘   │   │
│                      └─────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────┐    ┌─────────────────────────────────────────────┐   │
│  │    User      │    │            QUERY PIPELINE                   │   │
│  │  ┌────────┐  │    │  ┌────────┐  ┌──────────┐  ┌───────────┐   │   │
│  │  │ Query  │──┼───►│  │Embedder│─►│ Retriever│─►│ Reranker  │   │   │
│  │  └────────┘  │    │  └────────┘  └──────────┘  └─────┬─────┘   │   │
│  │              │    │                                   │         │   │
│  │  ┌────────┐  │    │                                   ▼         │   │
│  │  │Response│◄─┼────│                          ┌───────────────┐  │   │
│  │  └────────┘  │    │                          │   Generator   │  │   │
│  └──────────────┘    │                          │     (LLM)     │  │   │
│                      │                          └───────────────┘  │   │
│                      └─────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Deep Dive

### 1. Document Loader

```python
from pathlib import Path
from typing import Iterator
from dataclasses import dataclass
import fitz  # PyMuPDF
from bs4 import BeautifulSoup


@dataclass
class Document:
    """Loaded document with metadata."""
    content: str
    source: str
    metadata: dict


class DocumentLoader:
    """Load documents from various formats."""
    
    def load(self, path: str) -> Document:
        """Load a single document."""
        path = Path(path)
        
        loaders = {
            ".pdf": self._load_pdf,
            ".html": self._load_html,
            ".htm": self._load_html,
            ".md": self._load_markdown,
            ".txt": self._load_text,
            ".py": self._load_code,
            ".js": self._load_code,
        }
        
        loader = loaders.get(path.suffix.lower())
        if not loader:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        return loader(path)
    
    def load_directory(self, directory: str) -> Iterator[Document]:
        """Load all documents from a directory."""
        for path in Path(directory).rglob("*"):
            if path.is_file() and path.suffix in [".pdf", ".html", ".md", ".txt"]:
                try:
                    yield self.load(str(path))
                except Exception as e:
                    print(f"Error loading {path}: {e}")
    
    def _load_pdf(self, path: Path) -> Document:
        """Load PDF document."""
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        
        return Document(
            content=text,
            source=str(path),
            metadata={
                "format": "pdf",
                "pages": len(doc),
                "title": doc.metadata.get("title", "")
            }
        )
    
    def _load_html(self, path: Path) -> Document:
        """Load HTML document."""
        with open(path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        
        # Remove scripts and styles
        for tag in soup(["script", "style"]):
            tag.decompose()
        
        return Document(
            content=soup.get_text(separator="\n"),
            source=str(path),
            metadata={
                "format": "html",
                "title": soup.title.string if soup.title else ""
            }
        )
    
    def _load_markdown(self, path: Path) -> Document:
        """Load Markdown document."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return Document(
            content=content,
            source=str(path),
            metadata={"format": "markdown"}
        )
    
    def _load_text(self, path: Path) -> Document:
        """Load plain text document."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return Document(
            content=content,
            source=str(path),
            metadata={"format": "text"}
        )
    
    def _load_code(self, path: Path) -> Document:
        """Load source code file."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return Document(
            content=content,
            source=str(path),
            metadata={
                "format": "code",
                "language": path.suffix[1:]  # .py -> py
            }
        )
```

### 2. Text Chunker

```python
from typing import Iterator
from dataclasses import dataclass
import re


@dataclass
class Chunk:
    """A text chunk with metadata."""
    content: str
    source: str
    chunk_index: int
    metadata: dict


class TextChunker:
    """Split documents into chunks for embedding."""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        strategy: str = "recursive"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
    
    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into chunks."""
        
        if self.strategy == "fixed":
            return self._fixed_chunking(document)
        elif self.strategy == "recursive":
            return self._recursive_chunking(document)
        elif self.strategy == "semantic":
            return self._semantic_chunking(document)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _fixed_chunking(self, document: Document) -> list[Chunk]:
        """Simple fixed-size chunking."""
        chunks = []
        text = document.content
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            chunks.append(Chunk(
                content=chunk_text.strip(),
                source=document.source,
                chunk_index=chunk_index,
                metadata={**document.metadata, "chunk_strategy": "fixed"}
            ))
            
            start = end - self.chunk_overlap
            chunk_index += 1
        
        return chunks
    
    def _recursive_chunking(self, document: Document) -> list[Chunk]:
        """Recursive chunking with natural boundaries."""
        separators = ["\n\n", "\n", ". ", " ", ""]
        return self._recursive_split(
            document.content,
            separators,
            document
        )
    
    def _recursive_split(
        self,
        text: str,
        separators: list[str],
        document: Document,
        chunk_index: int = 0
    ) -> list[Chunk]:
        """Recursively split text using separators."""
        chunks = []
        
        if not separators:
            # Base case: just split by size
            return self._fixed_chunking(
                Document(text, document.source, document.metadata)
            )
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = len(split) + len(separator)
            
            if current_length + split_length > self.chunk_size:
                if current_chunk:
                    chunk_text = separator.join(current_chunk)
                    
                    if len(chunk_text) > self.chunk_size:
                        # Recursively split with next separator
                        sub_chunks = self._recursive_split(
                            chunk_text,
                            remaining_separators,
                            document,
                            chunk_index
                        )
                        chunks.extend(sub_chunks)
                        chunk_index += len(sub_chunks)
                    else:
                        chunks.append(Chunk(
                            content=chunk_text.strip(),
                            source=document.source,
                            chunk_index=chunk_index,
                            metadata={**document.metadata, "chunk_strategy": "recursive"}
                        ))
                        chunk_index += 1
                    
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(split)
            current_length += split_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            chunks.append(Chunk(
                content=chunk_text.strip(),
                source=document.source,
                chunk_index=chunk_index,
                metadata={**document.metadata, "chunk_strategy": "recursive"}
            ))
        
        return chunks
    
    def _semantic_chunking(self, document: Document) -> list[Chunk]:
        """Chunk by semantic boundaries (headers, paragraphs)."""
        # Split by markdown headers or double newlines
        pattern = r'(?=^#{1,6}\s|\n\n)'
        sections = re.split(pattern, document.content, flags=re.MULTILINE)
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for section in sections:
            if len(current_chunk) + len(section) <= self.chunk_size:
                current_chunk += section
            else:
                if current_chunk:
                    chunks.append(Chunk(
                        content=current_chunk.strip(),
                        source=document.source,
                        chunk_index=chunk_index,
                        metadata={**document.metadata, "chunk_strategy": "semantic"}
                    ))
                    chunk_index += 1
                current_chunk = section
        
        if current_chunk:
            chunks.append(Chunk(
                content=current_chunk.strip(),
                source=document.source,
                chunk_index=chunk_index,
                metadata={**document.metadata, "chunk_strategy": "semantic"}
            ))
        
        return chunks
```

### 3. Embedding Pipeline

```python
from openai import OpenAI
from typing import Optional
import numpy as np


class EmbeddingPipeline:
    """Generate embeddings for text."""
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 100
    ):
        self.client = OpenAI()
        self.model = model
        self.batch_size = batch_size
        self._cache: dict[str, list[float]] = {}
    
    def embed(self, text: str) -> list[float]:
        """Embed a single text."""
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
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            response = self.client.embeddings.create(
                input=batch,
                model=self.model
            )
            
            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [d.embedding for d in sorted_data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def embed_chunks(self, chunks: list[Chunk]) -> list[dict]:
        """Embed chunks and return with metadata."""
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed_batch(texts)
        
        return [
            {
                "id": f"{chunk.source}_{chunk.chunk_index}",
                "content": chunk.content,
                "embedding": embedding,
                "metadata": {
                    **chunk.metadata,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index
                }
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]
```

### 4. Vector Store

```python
import chromadb
from chromadb.config import Settings


class VectorStore:
    """Store and retrieve embeddings."""
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./vector_db"
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add(self, embedded_chunks: list[dict]) -> int:
        """Add embedded chunks to the store."""
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
    
    def delete(self, ids: list[str]):
        """Delete chunks by ID."""
        self.collection.delete(ids=ids)
    
    def count(self) -> int:
        """Get total document count."""
        return self.collection.count()
```

### 5. Retriever

```python
from typing import Optional


class Retriever:
    """Retrieve relevant context for queries."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_pipeline: EmbeddingPipeline,
        default_k: int = 5
    ):
        self.store = vector_store
        self.embedder = embedding_pipeline
        self.default_k = default_k
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[dict] = None,
        min_score: float = 0.0
    ) -> list[dict]:
        """Retrieve relevant documents for a query."""
        
        # Embed query
        query_embedding = self.embedder.embed(query)
        
        # Search vector store
        results = self.store.search(
            query_embedding=query_embedding,
            n_results=k or self.default_k,
            where=filters
        )
        
        # Filter by minimum score
        return [r for r in results if r["score"] >= min_score]
    
    def retrieve_with_mmr(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> list[dict]:
        """Retrieve with Maximal Marginal Relevance for diversity."""
        
        # Get more candidates than needed
        candidates = self.retrieve(query, k=fetch_k)
        
        if not candidates:
            return []
        
        # MMR selection
        query_embedding = np.array(self.embedder.embed(query))
        selected = []
        remaining = list(range(len(candidates)))
        
        for _ in range(min(k, len(candidates))):
            best_score = -float('inf')
            best_idx = -1
            
            for idx in remaining:
                # Relevance to query
                candidate_embedding = np.array(
                    self.embedder.embed(candidates[idx]["content"])
                )
                relevance = np.dot(query_embedding, candidate_embedding)
                
                # Max similarity to already selected
                if selected:
                    similarities = [
                        np.dot(
                            candidate_embedding,
                            np.array(self.embedder.embed(candidates[s]["content"]))
                        )
                        for s in selected
                    ]
                    max_sim = max(similarities)
                else:
                    max_sim = 0
                
                # MMR score
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return [candidates[i] for i in selected]
```

### 6. Generator

```python
from openai import OpenAI
from typing import Optional, Iterator


class Generator:
    """Generate responses using LLM with context."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 1000
    ):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(
        self,
        query: str,
        context: list[dict],
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response with context."""
        
        # Format context
        context_text = self._format_context(context)
        
        # Default system prompt
        if not system_prompt:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Always cite your sources using [1], [2], etc.
If the context doesn't contain enough information, say so clearly."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content
    
    def generate_stream(
        self,
        query: str,
        context: list[dict],
        system_prompt: Optional[str] = None
    ) -> Iterator[str]:
        """Generate streaming response."""
        
        context_text = self._format_context(context)
        
        if not system_prompt:
            system_prompt = "Answer based on the provided context. Cite sources."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
        ]
        
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def _format_context(self, context: list[dict]) -> str:
        """Format context for prompt."""
        formatted = []
        for i, doc in enumerate(context, 1):
            source = doc.get("metadata", {}).get("source", "Unknown")
            formatted.append(f"[{i}] Source: {source}\n{doc['content']}")
        return "\n\n---\n\n".join(formatted)
```

### 7. Complete RAG System

```python
class RAGSystem:
    """Complete RAG system combining all components."""
    
    def __init__(
        self,
        collection_name: str = "rag_docs",
        persist_directory: str = "./rag_db",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        # Initialize components
        self.loader = DocumentLoader()
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embedder = EmbeddingPipeline()
        self.store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        self.retriever = Retriever(self.store, self.embedder)
        self.generator = Generator()
    
    def index_document(self, path: str) -> int:
        """Index a single document."""
        document = self.loader.load(path)
        chunks = self.chunker.chunk(document)
        embedded = self.embedder.embed_chunks(chunks)
        return self.store.add(embedded)
    
    def index_directory(self, directory: str) -> int:
        """Index all documents in a directory."""
        total = 0
        for document in self.loader.load_directory(directory):
            chunks = self.chunker.chunk(document)
            embedded = self.embedder.embed_chunks(chunks)
            total += self.store.add(embedded)
        return total
    
    def query(
        self,
        question: str,
        k: int = 5,
        filters: Optional[dict] = None
    ) -> dict:
        """Answer a question using RAG."""
        
        # Retrieve context
        context = self.retriever.retrieve(
            query=question,
            k=k,
            filters=filters
        )
        
        # Generate answer
        answer = self.generator.generate(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "content": c["content"][:200] + "...",
                    "source": c["metadata"].get("source", "Unknown"),
                    "score": c["score"]
                }
                for c in context
            ]
        }
    
    def query_stream(
        self,
        question: str,
        k: int = 5
    ) -> Iterator[str]:
        """Stream answer generation."""
        context = self.retriever.retrieve(question, k=k)
        yield from self.generator.generate_stream(question, context)


# Usage example
rag = RAGSystem(
    collection_name="my_knowledge_base",
    persist_directory="./my_rag_db",
    chunk_size=400,
    chunk_overlap=50
)

# Index documents
rag.index_directory("./documents")

# Query
result = rag.query("What is the refund policy?")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

## Architecture Patterns

```yaml
basic_rag:
  description: "Simple retrieve-then-generate"
  components: ["Retriever", "Generator"]
  use_case: "Simple Q&A"

advanced_rag:
  description: "Multi-stage retrieval with reranking"
  components: ["Retriever", "Reranker", "Generator"]
  use_case: "High-accuracy applications"

conversational_rag:
  description: "RAG with conversation history"
  components: ["Memory", "Retriever", "Generator"]
  use_case: "Chatbots"

multi_index_rag:
  description: "Different indices for different content"
  components: ["Router", "Multiple Retrievers", "Generator"]
  use_case: "Diverse document types"
```

## Summary

```yaml
key_components:
  1. Document Loader: "Parse various file formats"
  2. Chunker: "Split into manageable pieces"
  3. Embedder: "Convert text to vectors"
  4. Vector Store: "Store and search embeddings"
  5. Retriever: "Find relevant context"
  6. Generator: "Create answers with context"

design_principles:
  - "Modular components for flexibility"
  - "Batch processing for efficiency"
  - "Caching to reduce API calls"
  - "Metadata for filtering and tracking"
```

## Next Steps

1. **Document Processing** - Deep dive into chunking strategies
2. **Retrieval Strategies** - Advanced retrieval techniques
3. **Generation Patterns** - Prompt engineering for RAG
