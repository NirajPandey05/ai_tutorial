# Understanding Context Windows

## What is a Context Window?

The **context window** is the maximum amount of text (measured in tokens) that an LLM can process in a single interaction. It includes both the input (your prompt) and the output (model's response).

```
┌─────────────────────────────────────────────────────────────────┐
│                      Context Window                             │
│                    (e.g., 128,000 tokens)                       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    YOUR INPUT                            │   │
│  │  • System prompt                                        │   │
│  │  • Conversation history                                 │   │
│  │  • Retrieved documents (RAG)                            │   │
│  │  • User's current message                               │   │
│  │                                                         │   │
│  │                [Variable size]                          │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                   MODEL OUTPUT                           │   │
│  │  • The response                                         │   │
│  │                                                         │   │
│  │              [Limited by max_tokens]                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│            input_tokens + output_tokens ≤ context_window       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Context Window Sizes (2025)

| Model | Context Window | Equivalent |
|-------|---------------|------------|
| **GPT-5.2** | 256,000 tokens | ~200,000 words |
| **GPT-4 Turbo** | 128,000 tokens | ~100,000 words |
| **Claude 4.5 Sonnet** | 200,000 tokens | ~150,000 words |
| **Claude 3.5** | 200,000 tokens | ~150,000 words |
| **Gemini 3 Pro** | 1,000,000 tokens | ~750,000 words |
| **Gemini 2 Flash** | 1,000,000 tokens | ~750,000 words |
| **Grok 4** | 131,072 tokens | ~100,000 words |
| **Llama 3** | 128,000 tokens | ~100,000 words |

### What Fits in Different Context Windows?

| Context Size | Can Fit |
|--------------|---------|
| 4K tokens | Short conversation, one document page |
| 16K tokens | Several pages, moderate conversation |
| 32K tokens | Long article, extended conversation |
| 128K tokens | Multiple documents, small codebase |
| 200K tokens | Book chapter, large documentation |
| 1M tokens | Multiple books, entire codebase |

```
128K tokens ≈ 
  • 300 pages of text
  • 5-6 average novels
  • 200 Wikipedia articles
  • 50,000 lines of code
```

## The Attention Mechanism and Context

### How Models "See" Context

LLMs use **attention** to weigh the importance of each token relative to others. Every token can attend to every other token in the context.

```
Query: "What is the capital?"

Context: "France is a country in Europe. Paris is its capital..."
                                         ^^^^^
                                    High attention here!
```

### The "Lost in the Middle" Problem

Research has shown that LLMs tend to prioritize information at the **beginning** and **end** of their context, sometimes missing crucial details in the middle.

```
┌─────────────────────────────────────────────────────────────────┐
│                 Attention Distribution                          │
│                                                                 │
│   Strong  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░██████████        │
│   Weak    ░░░░░░░░████████████████████████████░░░░░░░░░         │
│                                                                 │
│           ← Beginning        Middle          End →              │
│                                                                 │
│   "Lost in the Middle": Information here may be overlooked      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Implications for Prompt Design

```python
# BAD: Important info buried in middle
prompt = f"""
Here is document 1: {doc1}
Here is document 2: {doc2}
Here is document 3: {doc3}  # ← Critical info here, might be missed
Here is document 4: {doc4}
Here is document 5: {doc5}

What does document 3 say about X?
"""

# BETTER: Move important content to edges
prompt = f"""
CRITICAL DOCUMENT (about topic X):
{doc3}

Additional context:
{doc1}
{doc2}
{doc4}
{doc5}

Based on the CRITICAL DOCUMENT above, answer: What does it say about X?
"""
```

## Cost Considerations

### Input vs Output Pricing

Most providers charge differently for input and output tokens:

| Provider | Input (per 1M) | Output (per 1M) |
|----------|---------------|-----------------|
| GPT-5.2 | ~$5 | ~$15 |
| GPT-4 Turbo | ~$2.50 | ~$7.50 |
| Claude 4.5 | ~$3 | ~$15 |
| Gemini 3 Pro | ~$1.25 | ~$5 |

### Why Output is More Expensive

- Output requires **generation** (autoregressive, token-by-token)
- Input uses **parallel processing** (all tokens at once)

### Cost Example

```python
# Scenario: Summarizing a 50,000 token document

input_tokens = 50000 + 100  # Document + prompt
output_tokens = 500         # Summary

# Using GPT-4 Turbo pricing
input_cost = (input_tokens / 1_000_000) * 2.50   # $0.125
output_cost = (output_tokens / 1_000_000) * 7.50  # $0.00375

total_cost = input_cost + output_cost  # ~$0.13 per summary

# 1000 documents = ~$130
```

## Context Window Strategies

### Strategy 1: Truncation

Simply cut off content that exceeds the limit.

```python
def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """Truncate text to fit within token limit."""
    import tiktoken
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    truncated = enc.decode(tokens[:max_tokens])
    return truncated + "\n\n[Content truncated...]"
```

**Pros:** Simple, predictable
**Cons:** May lose important information at the end

### Strategy 2: Chunking with Overlap

Split content into chunks and process separately.

```python
def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 200):
    """Split text into overlapping chunks."""
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(text)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        
        if end >= len(tokens):
            break
        start = end - overlap  # Overlap to maintain context
    
    return chunks
```

**Pros:** Handles any length, maintains context with overlap
**Cons:** More API calls, need to aggregate results

### Strategy 3: Summarize and Compress

Replace verbose content with summaries.

```python
async def compress_context(long_context: str, target_tokens: int) -> str:
    """Summarize long context to fit target token count."""
    
    summary_prompt = f"""
    Summarize the following content, preserving key information:
    
    {long_context}
    
    Provide a detailed summary in approximately {target_tokens} words.
    """
    
    summary = await model.generate(summary_prompt)
    return summary
```

**Pros:** Retains key information
**Cons:** May lose nuance, requires extra API call

### Strategy 4: Retrieval-Augmented Generation (RAG)

Only include relevant portions using semantic search.

```python
async def rag_context(query: str, documents: list, top_k: int = 5):
    """Retrieve relevant documents for the query."""
    
    # Embed query
    query_embedding = await embed(query)
    
    # Find most relevant documents
    ranked = []
    for doc in documents:
        doc_embedding = await embed(doc)
        similarity = cosine_similarity(query_embedding, doc_embedding)
        ranked.append((similarity, doc))
    
    # Return top-k most relevant
    ranked.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in ranked[:top_k]]
```

**Pros:** Highly relevant context, efficient use of tokens
**Cons:** May miss related information, requires embedding setup

### Strategy 5: Hierarchical Context

Multi-level summarization for very long documents.

```
┌─────────────────────────────────────────────────────────────────┐
│                  Hierarchical Context                           │
│                                                                 │
│   Level 0: Full documents (stored, not sent)                   │
│        ↓                                                        │
│   Level 1: Section summaries (medium detail)                   │
│        ↓                                                        │
│   Level 2: Document summaries (high level)                     │
│        ↓                                                        │
│   Level 3: Collection summary (overview)                       │
│                                                                 │
│   Query flow:                                                   │
│   1. Match query to Level 2/3 → Identify relevant docs         │
│   2. Expand to Level 1 → Get relevant sections                 │
│   3. Include Level 0 chunks → Specific passages                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Practical Context Budgeting

### Planning Your Context Window

```python
def plan_context_budget(
    context_window: int = 128000,
    system_prompt_tokens: int = 500,
    max_response_tokens: int = 4000,
    safety_buffer: int = 500
) -> dict:
    """Plan how to allocate context window."""
    
    reserved = system_prompt_tokens + max_response_tokens + safety_buffer
    available_for_content = context_window - reserved
    
    return {
        "context_window": context_window,
        "system_prompt": system_prompt_tokens,
        "max_response": max_response_tokens,
        "safety_buffer": safety_buffer,
        "available_for_content": available_for_content,
        "breakdown": {
            "conversation_history": int(available_for_content * 0.3),
            "retrieved_documents": int(available_for_content * 0.6),
            "user_message": int(available_for_content * 0.1),
        }
    }

budget = plan_context_budget()
print(f"Available for content: {budget['available_for_content']} tokens")
```

### Dynamic Context Management

```python
class ContextManager:
    def __init__(self, max_tokens: int = 120000):
        self.max_tokens = max_tokens
        self.system_prompt = ""
        self.messages = []
        self.documents = []
    
    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._trim_if_needed()
    
    def add_document(self, doc: str, priority: int = 1):
        self.documents.append({"content": doc, "priority": priority})
        self._trim_if_needed()
    
    def _trim_if_needed(self):
        """Remove lowest priority items when over limit."""
        while self._count_tokens() > self.max_tokens:
            if self.documents:
                # Remove lowest priority document
                self.documents.sort(key=lambda x: x["priority"])
                self.documents.pop(0)
            elif len(self.messages) > 2:
                # Remove oldest message (keep last 2)
                self.messages.pop(1)  # Keep system message at 0
            else:
                break
    
    def _count_tokens(self) -> int:
        # Simplified counting
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")
        total = len(enc.encode(self.system_prompt))
        for msg in self.messages:
            total += len(enc.encode(msg["content"]))
        for doc in self.documents:
            total += len(enc.encode(doc["content"]))
        return total
```

## When Context Size Matters

### Scenarios Requiring Large Context

| Scenario | Typical Need |
|----------|-------------|
| Legal document analysis | 100K+ tokens |
| Codebase understanding | 50K-200K tokens |
| Book summarization | 100K-500K tokens |
| Multi-document research | 50K-100K tokens |
| Long conversation history | 10K-50K tokens |

### Scenarios Where Smaller is Better

| Scenario | Optimal Size |
|----------|-------------|
| Simple Q&A | 1K-4K tokens |
| Classification | 500-2K tokens |
| Short-form generation | 2K-8K tokens |
| Quick lookups | 1K-2K tokens |

## Key Takeaways

1. **Context = Input + Output** - Budget for both
2. **Position matters** - Important info at beginning and end
3. **Larger isn't always better** - More context = more cost and latency
4. **Use strategies** - Chunking, summarization, RAG for long content
5. **Monitor usage** - Track token counts for cost optimization

## Next Steps

Now let's explore practical strategies for **Managing Context Effectively** in real-world applications.
