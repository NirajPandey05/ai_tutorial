# Lab: Build a RAG Chatbot

Build a complete RAG-powered chatbot with retrieval, citations, and conversation memory.

## Lab Overview

```yaml
objective: "Build a production-ready RAG chatbot"
difficulty: "Intermediate-Advanced"
time: "60-90 minutes"

what_you_will_build:
  - "Document ingestion pipeline"
  - "Conversational retrieval"
  - "Citation tracking"
  - "Conversation memory"
  - "Streaming responses"

prerequisites:
  - "Understanding of RAG fundamentals"
  - "Familiarity with vector databases"
  - "Python and async basics"
```

## Part 1: Project Setup

```python
# Install dependencies
# pip install openai chromadb tiktoken

from openai import OpenAI
import chromadb
import tiktoken
from typing import Generator, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import re
import hashlib


# Configuration
@dataclass
class RAGConfig:
    """Configuration for RAG chatbot."""
    
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_k: int = 5
    max_context_tokens: int = 4000
    collection_name: str = "rag_chatbot"


config = RAGConfig()
client = OpenAI()
chroma = chromadb.Client()

print("RAG Chatbot initialized")
print(f"Model: {config.model}")
print(f"Embedding: {config.embedding_model}")
```

## Part 2: Document Processor

```python
class DocumentProcessor:
    """Process and chunk documents for RAG."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def process_document(
        self,
        content: str,
        metadata: dict = None
    ) -> list[dict]:
        """Process a document into chunks."""
        
        # Generate document ID
        doc_id = hashlib.md5(content.encode()).hexdigest()[:12]
        
        # Clean content
        content = self._clean_text(content)
        
        # Chunk
        chunks = self._chunk_text(content)
        
        # Create chunk records
        records = []
        for i, chunk in enumerate(chunks):
            records.append({
                "id": f"{doc_id}_chunk_{i}",
                "content": chunk,
                "metadata": {
                    **(metadata or {}),
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "processed_at": datetime.now().isoformat()
                }
            })
        
        return records
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _chunk_text(self, text: str) -> list[str]:
        """Chunk text with overlap."""
        
        chunks = []
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.config.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Keep overlap
                overlap_text = ' '.join(current_chunk)
                overlap_start = max(0, len(overlap_text) - self.config.chunk_overlap)
                
                current_chunk = [overlap_text[overlap_start:]] if overlap_start > 0 else []
                current_length = len(current_chunk[0]) if current_chunk else 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoder.encode(text))


# Test processor
processor = DocumentProcessor(config)

sample_doc = """
Welcome to TechCorp's employee handbook. This document contains important information about company policies and procedures.

Our vacation policy allows employees to take up to 20 days of paid time off per year. Vacation requests must be submitted at least one week in advance through the HR portal.

The company provides comprehensive health insurance including medical, dental, and vision coverage. Open enrollment occurs annually in November.

Remote work is supported for most positions. Employees working remotely must be available during core hours (10am-3pm) and attend weekly team meetings.
"""

chunks = processor.process_document(
    sample_doc,
    metadata={"title": "Employee Handbook", "category": "HR"}
)

print(f"Created {len(chunks)} chunks")
for chunk in chunks:
    print(f"  [{chunk['id']}] {len(chunk['content'])} chars")
```

## Part 3: Vector Store

```python
class VectorStore:
    """Manage document vectors and retrieval."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.collection = chroma.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: list[dict]) -> int:
        """Add documents to the vector store."""
        
        if not documents:
            return 0
        
        # Generate embeddings
        texts = [d["content"] for d in documents]
        embeddings = self._embed_batch(texts)
        
        # Add to collection
        self.collection.add(
            ids=[d["id"] for d in documents],
            documents=texts,
            embeddings=embeddings,
            metadatas=[d["metadata"] for d in documents]
        )
        
        return len(documents)
    
    def retrieve(
        self,
        query: str,
        k: int = None,
        filter: dict = None
    ) -> list[dict]:
        """Retrieve relevant documents."""
        
        k = k or self.config.retrieval_k
        
        # Embed query
        query_embedding = self._embed(query)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        documents = []
        for i in range(len(results["ids"][0])):
            documents.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]
            })
        
        return documents
    
    def _embed(self, text: str) -> list[float]:
        """Embed single text."""
        response = client.embeddings.create(
            input=text,
            model=self.config.embedding_model
        )
        return response.data[0].embedding
    
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed batch of texts."""
        response = client.embeddings.create(
            input=texts,
            model=self.config.embedding_model
        )
        return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]
    
    def get_stats(self) -> dict:
        """Get collection statistics."""
        return {
            "name": self.config.collection_name,
            "count": self.collection.count()
        }


# Test vector store
store = VectorStore(config)

# Add processed chunks
count = store.add_documents(chunks)
print(f"Added {count} documents to vector store")
print(f"Stats: {store.get_stats()}")

# Test retrieval
results = store.retrieve("vacation policy", k=3)
print(f"\nRetrieved {len(results)} documents for 'vacation policy':")
for r in results:
    print(f"  [{r['score']:.3f}] {r['content'][:80]}...")
```

## Part 4: Conversation Memory

```python
@dataclass
class Message:
    """A single conversation message."""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


class ConversationMemory:
    """Manage conversation history."""
    
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.messages: list[Message] = []
        self.context_used: list[dict] = []
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: dict = None
    ):
        """Add a message to history."""
        
        self.messages.append(Message(
            role=role,
            content=content,
            metadata=metadata or {}
        ))
        
        # Trim if needed
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def add_user_message(self, content: str):
        """Add user message."""
        self.add_message("user", content)
    
    def add_assistant_message(self, content: str, sources: list[dict] = None):
        """Add assistant message with optional sources."""
        self.add_message(
            "assistant",
            content,
            metadata={"sources": sources or []}
        )
    
    def get_messages_for_prompt(self) -> list[dict]:
        """Get messages formatted for API."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages
        ]
    
    def get_recent_context(self, n: int = 3) -> str:
        """Get recent conversation for context."""
        
        recent = self.messages[-n*2:] if len(self.messages) > n*2 else self.messages
        
        lines = []
        for m in recent:
            prefix = "User" if m.role == "user" else "Assistant"
            lines.append(f"{prefix}: {m.content[:200]}")
        
        return "\n".join(lines)
    
    def clear(self):
        """Clear conversation history."""
        self.messages = []
        self.context_used = []


# Test memory
memory = ConversationMemory()
memory.add_user_message("What's the vacation policy?")
memory.add_assistant_message(
    "Employees get 20 days of PTO per year.",
    sources=[{"id": "doc1", "content": "..."}]
)

print(f"Messages in memory: {len(memory.messages)}")
print(f"Recent context:\n{memory.get_recent_context()}")
```

## Part 5: RAG Chatbot Core

```python
class RAGChatbot:
    """Complete RAG chatbot with retrieval and generation."""
    
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Guidelines:
- Base your answers ONLY on the provided context
- Cite sources using [N] notation after each fact
- If the context doesn't contain the answer, say so
- Be conversational and helpful
- Consider the conversation history for context"""
    
    def __init__(
        self,
        config: RAGConfig,
        vector_store: VectorStore,
        processor: DocumentProcessor
    ):
        self.config = config
        self.store = vector_store
        self.processor = processor
        self.memory = ConversationMemory()
    
    def ingest_document(
        self,
        content: str,
        metadata: dict = None
    ) -> int:
        """Ingest a new document."""
        
        chunks = self.processor.process_document(content, metadata)
        return self.store.add_documents(chunks)
    
    def chat(self, user_message: str) -> dict:
        """Process a chat message and generate response."""
        
        # Add user message to memory
        self.memory.add_user_message(user_message)
        
        # Enhance query with conversation context
        enhanced_query = self._enhance_query(user_message)
        
        # Retrieve relevant documents
        documents = self.store.retrieve(enhanced_query, k=self.config.retrieval_k)
        
        # Build prompt
        messages = self._build_messages(user_message, documents)
        
        # Generate response
        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=0
        )
        
        answer = response.choices[0].message.content
        
        # Extract citations
        citations = self._extract_citations(answer, documents)
        
        # Add to memory
        self.memory.add_assistant_message(answer, citations)
        
        return {
            "answer": answer,
            "sources": citations,
            "documents_retrieved": len(documents)
        }
    
    def chat_stream(self, user_message: str) -> Generator[dict, None, None]:
        """Stream chat response with citation events."""
        
        # Add user message to memory
        self.memory.add_user_message(user_message)
        
        # Retrieve
        enhanced_query = self._enhance_query(user_message)
        documents = self.store.retrieve(enhanced_query)
        
        # Emit source event
        yield {
            "type": "sources",
            "sources": [
                {"id": d["id"], "preview": d["content"][:100], "score": d["score"]}
                for d in documents
            ]
        }
        
        # Build messages
        messages = self._build_messages(user_message, documents)
        
        # Stream response
        stream = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=0,
            stream=True
        )
        
        full_response = ""
        seen_citations = set()
        
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if not content:
                continue
            
            full_response += content
            
            # Check for new citations
            new_citations = re.findall(r'\[(\d+)\]', content)
            for citation in new_citations:
                num = int(citation)
                if num not in seen_citations and num <= len(documents):
                    seen_citations.add(num)
                    yield {
                        "type": "citation",
                        "source_id": num,
                        "source": documents[num - 1]
                    }
            
            yield {"type": "content", "text": content}
        
        # Finalize
        citations = self._extract_citations(full_response, documents)
        self.memory.add_assistant_message(full_response, citations)
        
        yield {
            "type": "done",
            "full_response": full_response,
            "citations": citations
        }
    
    def _enhance_query(self, query: str) -> str:
        """Enhance query with conversation context."""
        
        if len(self.memory.messages) <= 1:
            return query
        
        # Get recent context
        recent = self.memory.get_recent_context(n=2)
        
        # Use LLM to create better search query
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Create a search query that captures the user's question considering the conversation context. Return only the search query."
                },
                {
                    "role": "user",
                    "content": f"Conversation:\n{recent}\n\nNew question: {query}"
                }
            ],
            max_tokens=100,
            temperature=0
        )
        
        return response.choices[0].message.content.strip()
    
    def _build_messages(
        self,
        user_message: str,
        documents: list[dict]
    ) -> list[dict]:
        """Build messages for API call."""
        
        # Format context
        context = "\n\n".join([
            f"[Source {i+1}] {doc['content']}"
            for i, doc in enumerate(documents)
        ])
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]
        
        # Add conversation history (limited)
        history = self.memory.get_messages_for_prompt()[-6:]  # Last 3 exchanges
        messages.extend(history[:-1])  # Exclude current message
        
        # Add current message with context
        messages.append({
            "role": "user",
            "content": f"""Context:
{context}

Question: {user_message}"""
        })
        
        return messages
    
    def _extract_citations(
        self,
        answer: str,
        documents: list[dict]
    ) -> list[dict]:
        """Extract cited sources from answer."""
        
        cited_nums = set(int(n) for n in re.findall(r'\[(\d+)\]', answer))
        
        return [
            documents[num - 1]
            for num in sorted(cited_nums)
            if 0 < num <= len(documents)
        ]
    
    def reset_conversation(self):
        """Reset conversation memory."""
        self.memory.clear()


# Create chatbot
chatbot = RAGChatbot(config, store, processor)

print("RAG Chatbot ready!")
print(f"Documents in store: {store.get_stats()['count']}")
```

## Part 6: Interactive Testing

```python
# Add more documents
additional_docs = [
    {
        "content": """
        The API rate limit is 1000 requests per minute for standard accounts.
        Enterprise customers can request higher limits by contacting support.
        Rate limit errors return HTTP 429 status code.
        """,
        "metadata": {"title": "API Documentation", "category": "Technical"}
    },
    {
        "content": """
        Employee referral bonuses are $5000 for engineering positions and $3000 for other roles.
        Bonuses are paid after the referred employee completes 90 days of employment.
        Submit referrals through the HR portal before the candidate applies.
        """,
        "metadata": {"title": "Referral Program", "category": "HR"}
    }
]

for doc in additional_docs:
    chunks_added = chatbot.ingest_document(doc["content"], doc["metadata"])
    print(f"Ingested '{doc['metadata']['title']}': {chunks_added} chunks")

print(f"\nTotal documents: {store.get_stats()['count']}")


# Test conversation
print("\n" + "="*50)
print("CONVERSATION TEST")
print("="*50)

questions = [
    "What's the vacation policy?",
    "How do I request time off?",  # Follow-up
    "What about the referral program?",
    "How much is the bonus for engineers?"  # Follow-up with pronoun
]

for q in questions:
    print(f"\nUser: {q}")
    result = chatbot.chat(q)
    print(f"Assistant: {result['answer']}")
    print(f"[Sources used: {len(result['sources'])}]")


# Test streaming
print("\n" + "="*50)
print("STREAMING TEST")
print("="*50)

chatbot.reset_conversation()

print("\nUser: What are the API rate limits?")
print("Assistant: ", end="")

for event in chatbot.chat_stream("What are the API rate limits?"):
    if event["type"] == "content":
        print(event["text"], end="", flush=True)
    elif event["type"] == "citation":
        pass  # Handle in UI
    elif event["type"] == "done":
        print(f"\n[Citations: {len(event['citations'])}]")
```

## Part 7: Simple CLI Interface

```python
def run_chatbot_cli():
    """Run interactive CLI chatbot."""
    
    print("\n" + "="*50)
    print("RAG CHATBOT CLI")
    print("="*50)
    print("Commands: /reset, /stats, /quit")
    print("="*50 + "\n")
    
    # Create fresh chatbot
    cli_chatbot = RAGChatbot(config, store, processor)
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == "/quit":
                print("Goodbye!")
                break
            
            if user_input.lower() == "/reset":
                cli_chatbot.reset_conversation()
                print("[Conversation reset]")
                continue
            
            if user_input.lower() == "/stats":
                stats = store.get_stats()
                print(f"[Documents: {stats['count']}, Messages: {len(cli_chatbot.memory.messages)}]")
                continue
            
            # Generate response
            print("Assistant: ", end="", flush=True)
            
            for event in cli_chatbot.chat_stream(user_input):
                if event["type"] == "content":
                    print(event["text"], end="", flush=True)
                elif event["type"] == "done":
                    sources = event.get("citations", [])
                    if sources:
                        print(f"\n\n📚 Sources: {', '.join(s['id'] for s in sources)}")
                    print()
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


# Uncomment to run interactive CLI
# run_chatbot_cli()
```

## Exercises

```yaml
exercise_1:
  title: "Add Source Verification"
  task: "Implement citation verification to ensure claims are supported"
  hint: "Compare each cited claim against its source document"

exercise_2:
  title: "Implement Hybrid Search"
  task: "Add BM25 keyword search alongside vector search"
  hint: "Use rank_bm25 library and combine scores"

exercise_3:
  title: "Add Conversation Summarization"
  task: "Summarize long conversations to fit context window"
  hint: "Periodically summarize older messages"

exercise_4:
  title: "Multi-Turn Source Tracking"
  task: "Track which sources are used across conversation turns"
  hint: "Maintain a session-level source usage log"
```

## Summary

```yaml
components_built:
  document_processor: "Chunk and prepare documents"
  vector_store: "Store and retrieve with embeddings"
  conversation_memory: "Track chat history"
  rag_chatbot: "Complete chat with retrieval"

key_features:
  - "Document ingestion pipeline"
  - "Conversation-aware retrieval"
  - "Streaming with citations"
  - "Source attribution"

production_considerations:
  - "Add persistent storage (not in-memory ChromaDB)"
  - "Implement rate limiting"
  - "Add authentication"
  - "Monitor retrieval quality"
  - "Log conversations for improvement"
```

## Next Steps

1. **Advanced Patterns** - HyDE, RAG Fusion, Self-RAG
2. **Production Architecture** - Scale and deploy
3. **Evaluation** - Measure RAG quality
