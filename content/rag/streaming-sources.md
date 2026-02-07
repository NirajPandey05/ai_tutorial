# Streaming RAG with Sources

Implement real-time streaming responses with source attribution.

## Why Streaming for RAG?

```yaml
benefits:
  user_experience: "See responses as they generate"
  perceived_speed: "Feels faster even if same total time"
  source_visibility: "Know where info comes from in real-time"
  interruptibility: "Users can stop if answer is sufficient"

challenges:
  - "Citations appear before full context is clear"
  - "Source mapping during streaming"
  - "Handling partial citations"
  - "UI complexity"
```

## Basic Streaming RAG

```python
from openai import OpenAI
from typing import Generator


class StreamingRAG:
    """Basic streaming RAG implementation."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def stream_with_context(
        self,
        query: str,
        documents: list[dict]
    ) -> Generator[dict, None, None]:
        """Stream response with context."""
        
        # Prepare context
        context = "\n\n".join([
            f"[Source {i+1}] {doc['content']}"
            for i, doc in enumerate(documents)
        ])
        
        # Stream response
        stream = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Answer using the provided sources. Cite with [N] notation."
                },
                {
                    "role": "user",
                    "content": f"Sources:\n{context}\n\nQuestion: {query}"
                }
            ],
            stream=True,
            temperature=0
        )
        
        # Yield chunks
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield {
                    "type": "content",
                    "text": chunk.choices[0].delta.content
                }
        
        yield {"type": "done"}


# Usage
rag = StreamingRAG()

documents = [
    {"content": "Returns accepted within 30 days."},
    {"content": "Original receipt required for refunds."}
]

print("Streaming response:")
for chunk in rag.stream_with_context("What is the return policy?", documents):
    if chunk["type"] == "content":
        print(chunk["text"], end="", flush=True)
    else:
        print("\n[Done]")
```

## Streaming with Citation Events

```python
import re
from typing import Generator


class CitationStreamingRAG:
    """Streaming with citation detection events."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def stream(
        self,
        query: str,
        documents: list[dict]
    ) -> Generator[dict, None, None]:
        """Stream with citation events."""
        
        # Build source map
        sources = {
            i + 1: doc for i, doc in enumerate(documents)
        }
        
        context = "\n\n".join([
            f"[Source {i+1}] {doc['content']}"
            for i, doc in enumerate(documents)
        ])
        
        # First yield sources for UI preparation
        yield {
            "type": "sources",
            "sources": [
                {
                    "id": i + 1,
                    "preview": doc["content"][:100],
                    "metadata": doc.get("metadata", {})
                }
                for i, doc in enumerate(documents)
            ]
        }
        
        # Stream response
        stream = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Answer using provided sources. Cite each fact with [N]."
                },
                {
                    "role": "user",
                    "content": f"Sources:\n{context}\n\nQuestion: {query}"
                }
            ],
            stream=True,
            temperature=0
        )
        
        buffer = ""
        seen_citations = set()
        
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if not content:
                continue
            
            buffer += content
            
            # Check for new citations in buffer
            citations = re.findall(r'\[(\d+)\]', buffer)
            
            for citation in citations:
                num = int(citation)
                if num not in seen_citations and num in sources:
                    seen_citations.add(num)
                    
                    # Emit citation event
                    yield {
                        "type": "citation",
                        "source_id": num,
                        "source": sources[num]
                    }
            
            # Yield content
            yield {
                "type": "content",
                "text": content
            }
        
        # Final summary
        yield {
            "type": "done",
            "citations_used": list(seen_citations),
            "full_text": buffer
        }


# Usage with UI simulation
rag = CitationStreamingRAG()

documents = [
    {"content": "30-day return policy for all items.", "metadata": {"title": "Returns"}},
    {"content": "Original packaging required.", "metadata": {"title": "Requirements"}}
]

print("=== Streaming with Citations ===\n")

full_response = ""
cited_sources = []

for event in rag.stream("What's the return policy?", documents):
    if event["type"] == "sources":
        print(f"[Loaded {len(event['sources'])} sources]\n")
    
    elif event["type"] == "citation":
        cited_sources.append(event["source_id"])
        print(f"\n[📖 Source {event['source_id']}: {event['source']['metadata'].get('title', 'Unknown')}]")
    
    elif event["type"] == "content":
        print(event["text"], end="", flush=True)
        full_response += event["text"]
    
    elif event["type"] == "done":
        print(f"\n\n[Complete - Used sources: {event['citations_used']}]")
```

## Progressive Source Highlighting

```python
class ProgressiveSourceRAG:
    """Highlight sources as they're referenced."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def stream_with_highlighting(
        self,
        query: str,
        documents: list[dict]
    ) -> Generator[dict, None, None]:
        """Stream with progressive source highlighting."""
        
        sources = {i + 1: doc for i, doc in enumerate(documents)}
        
        context = "\n\n".join([
            f"[{i+1}] {doc['content']}"
            for i, doc in enumerate(documents)
        ])
        
        # Initial state: all sources dimmed
        yield {
            "type": "init",
            "sources": [
                {
                    "id": i + 1,
                    "content": doc["content"],
                    "highlighted": False,
                    "relevance": 0.0
                }
                for i, doc in enumerate(documents)
            ]
        }
        
        stream = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer with [N] citations."},
                {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {query}"}
            ],
            stream=True,
            temperature=0
        )
        
        buffer = ""
        citation_counts = {}
        
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if not content:
                continue
            
            buffer += content
            
            # Update citation counts
            new_citations = re.findall(r'\[(\d+)\]', content)
            
            for citation in new_citations:
                num = int(citation)
                citation_counts[num] = citation_counts.get(num, 0) + 1
                
                # Calculate relevance (more citations = more relevant)
                max_count = max(citation_counts.values()) if citation_counts else 1
                
                yield {
                    "type": "highlight",
                    "source_id": num,
                    "relevance": citation_counts[num] / max_count,
                    "citation_count": citation_counts[num]
                }
            
            yield {"type": "content", "text": content}
        
        # Final relevance ranking
        yield {
            "type": "complete",
            "source_ranking": sorted(
                citation_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
        }


# Usage for UI
rag = ProgressiveSourceRAG()

documents = [
    {"content": "Returns within 30 days with receipt."},
    {"content": "Exchanges available for same item."},
    {"content": "Refunds processed in 5-7 days."}
]

source_state = {}

for event in rag.stream_with_highlighting("Return policy?", documents):
    if event["type"] == "init":
        for src in event["sources"]:
            source_state[src["id"]] = {"relevance": 0}
        print(f"Initialized {len(event['sources'])} sources")
    
    elif event["type"] == "highlight":
        source_state[event["source_id"]]["relevance"] = event["relevance"]
        bar = "█" * int(event["relevance"] * 10)
        print(f"\r[Source {event['source_id']}] {bar:<10} ({event['citation_count']}x)", end="")
    
    elif event["type"] == "content":
        pass  # Handle in UI
    
    elif event["type"] == "complete":
        print(f"\n\nMost relevant: Source {event['source_ranking'][0][0]}")
```

## FastAPI Streaming Endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()


class RAGService:
    def __init__(self):
        self.client = OpenAI()
    
    async def stream_response(
        self,
        query: str,
        documents: list[dict]
    ):
        """Async streaming for FastAPI."""
        
        context = "\n\n".join([
            f"[{i+1}] {doc['content']}"
            for i, doc in enumerate(documents)
        ])
        
        # Send sources first
        yield f"data: {json.dumps({'type': 'sources', 'data': documents})}\n\n"
        
        stream = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer with [N] citations."},
                {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {query}"}
            ],
            stream=True
        )
        
        full_text = ""
        
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                full_text += content
                
                # Check for citations
                citations = re.findall(r'\[(\d+)\]', content)
                
                event = {
                    "type": "content",
                    "text": content,
                    "citations": [int(c) for c in citations] if citations else []
                }
                
                yield f"data: {json.dumps(event)}\n\n"
        
        # Final event
        yield f"data: {json.dumps({'type': 'done', 'full_text': full_text})}\n\n"


rag_service = RAGService()


@app.post("/api/rag/stream")
async def stream_rag(request: dict):
    """Stream RAG responses with Server-Sent Events."""
    
    return StreamingResponse(
        rag_service.stream_response(
            request["query"],
            request["documents"]
        ),
        media_type="text/event-stream"
    )


# Frontend JavaScript to consume:
"""
const eventSource = new EventSource('/api/rag/stream');

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'sources') {
        displaySources(data.data);
    } else if (data.type === 'content') {
        appendContent(data.text);
        if (data.citations.length > 0) {
            highlightSources(data.citations);
        }
    } else if (data.type === 'done') {
        finalizeResponse(data.full_text);
    }
};
"""
```

## Buffered Citation Streaming

```python
class BufferedCitationStream:
    """Buffer content to emit complete sentences with citations."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def stream(
        self,
        query: str,
        documents: list[dict]
    ) -> Generator[dict, None, None]:
        """Stream complete sentences with their citations."""
        
        context = "\n\n".join([
            f"[{i+1}] {doc['content']}"
            for i, doc in enumerate(documents)
        ])
        
        yield {"type": "start", "source_count": len(documents)}
        
        stream = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer with [N] citations after each sentence."},
                {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {query}"}
            ],
            stream=True
        )
        
        buffer = ""
        
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if not content:
                continue
            
            buffer += content
            
            # Check for complete sentences
            while True:
                # Find sentence boundary (. ! ? followed by space or end)
                match = re.search(r'([^.!?]*[.!?])(\s*\[\d+\])*(\s|$)', buffer)
                
                if not match:
                    break
                
                sentence_end = match.end()
                sentence = buffer[:sentence_end].strip()
                buffer = buffer[sentence_end:]
                
                # Extract citations from this sentence
                citations = [int(c) for c in re.findall(r'\[(\d+)\]', sentence)]
                
                # Clean sentence for display
                clean_sentence = re.sub(r'\s*\[\d+\]', '', sentence).strip()
                
                yield {
                    "type": "sentence",
                    "text": clean_sentence,
                    "citations": citations,
                    "raw": sentence
                }
        
        # Emit remaining buffer
        if buffer.strip():
            citations = [int(c) for c in re.findall(r'\[(\d+)\]', buffer)]
            yield {
                "type": "sentence",
                "text": re.sub(r'\s*\[\d+\]', '', buffer).strip(),
                "citations": citations,
                "raw": buffer
            }
        
        yield {"type": "done"}


# Usage
rag = BufferedCitationStream()

documents = [
    {"content": "Returns within 30 days."},
    {"content": "Receipt required."},
    {"content": "Refund in 5-7 days."}
]

print("=== Buffered Sentence Streaming ===\n")

for event in rag.stream("Return policy?", documents):
    if event["type"] == "sentence":
        citations = event["citations"]
        citation_str = f" (Sources: {citations})" if citations else ""
        print(f"📝 {event['text']}{citation_str}")
    elif event["type"] == "done":
        print("\n[Complete]")
```

## Summary

```yaml
streaming_patterns:
  basic: "Simple content chunks"
  citation_events: "Emit events when citations detected"
  progressive: "Update source relevance in real-time"
  buffered: "Emit complete sentences with citations"

implementation_tips:
  - "Always send sources first for UI preparation"
  - "Buffer for sentence-level citation mapping"
  - "Use SSE for web streaming"
  - "Handle partial citations in buffer"

ui_considerations:
  - "Show source panel that highlights on citation"
  - "Provide way to click citations for details"
  - "Show loading state during initial retrieval"
  - "Allow users to stop streaming"
```

## Next Steps

1. **RAG Chatbot Lab** - Build complete streaming chatbot
2. **Advanced Patterns** - HyDE, RAG Fusion
3. **Production Architecture** - Scale streaming RAG
