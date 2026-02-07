# Token Limits and Max Tokens

## What Are Tokens?

Before understanding limits, let's clarify what tokens are:

A **token** is a unit of text that LLMs process. It's not exactly a word—it's a piece of text determined by the model's tokenizer.

```
"Hello, world!" → ["Hello", ",", " world", "!"]
                   4 tokens

"ChatGPT" → ["Chat", "G", "PT"]
            3 tokens

"🚀" → ["🚀"]
       1 token (usually)
```

### Token Approximations

| Language | Approximation |
|----------|---------------|
| English | ~4 characters ≈ 1 token |
| English | ~0.75 words ≈ 1 token |
| Code | More tokens per line (special chars) |
| Chinese/Japanese | ~1-2 characters ≈ 1 token |

```
┌─────────────────────────────────────────────────────────────────┐
│                   Token Examples                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   "I love programming"                                         │
│   ["I", " love", " programming"] = 3 tokens                    │
│                                                                 │
│   "def calculate_sum(numbers):"                                │
│   ["def", " calculate", "_", "sum", "(", "numbers", "):", ] = 7│
│                                                                 │
│   "GPT-4" vs "GPT4"                                            │
│   ["G", "PT", "-", "4"] = 4  vs  ["G", "PT", "4"] = 3         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Context Window

The **context window** is the maximum number of tokens a model can process in a single interaction (input + output combined).

### Context Windows by Model (2025)

| Model | Context Window | Approx. Words |
|-------|---------------|---------------|
| GPT-5.2 | 256,000 | ~192,000 |
| GPT-4 Turbo | 128,000 | ~96,000 |
| Claude 4.5 | 200,000 | ~150,000 |
| Claude 3.5 | 200,000 | ~150,000 |
| Gemini 3 Pro | 1,000,000 | ~750,000 |
| Gemini 2 Flash | 1,000,000 | ~750,000 |
| Grok 4 | 131,072 | ~98,000 |

### Context Budget

```
┌─────────────────────────────────────────────────────────────────┐
│                    Context Window Budget                        │
│                                                                 │
│   Total Context Window: 128,000 tokens                          │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │          Your Input (Prompt + History)                  │   │
│   │                    Variable                             │   │
│   ├─────────────────────────────────────────────────────────┤   │
│   │          Model's Output (Response)                      │   │
│   │              Limited by max_tokens                      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   input_tokens + output_tokens ≤ context_window                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## The max_tokens Parameter

**max_tokens** sets the maximum length of the model's response.

```python
# Limit response to 500 tokens (~375 words)
response = client.chat(
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_tokens=500
)
```

### Why Set max_tokens?

1. **Cost Control** - You pay per token
2. **Response Time** - Shorter = faster
3. **Format Enforcement** - Force concise answers
4. **Prevent Rambling** - Stop over-explanation

### max_tokens Values

| Value | Use Case |
|-------|----------|
| 50-100 | Short answers, classifications |
| 150-300 | Summaries, brief explanations |
| 500-1000 | Detailed responses, code snippets |
| 2000-4000 | Long-form content, documentation |
| 8000+ | Articles, comprehensive analysis |

### Default Behavior

If you don't set max_tokens:

| Provider | Default Behavior |
|----------|-----------------|
| OpenAI | Model-dependent, often 4096 |
| Anthropic | 4096 or until natural stop |
| Google | Model-dependent |
| xAI | Model-dependent |

## Counting Tokens

### Using tiktoken (OpenAI)

```python
import tiktoken

# For GPT-4 models
encoding = tiktoken.encoding_for_model("gpt-4")

text = "Hello, how are you doing today?"
tokens = encoding.encode(text)

print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")
# Output: Token count: 8
```

### Common Token Counts

| Content Type | Tokens | Example |
|--------------|--------|---------|
| Tweet (280 chars) | ~70 | Social media post |
| Email | 200-400 | Business correspondence |
| Blog paragraph | 100-200 | Web content |
| Full page | 500-700 | Document page |
| Short story | 2000-5000 | Fiction |
| Technical doc | 3000-10000 | Documentation |

## Handling Token Limits

### Problem: Input Too Long

```python
# ❌ This will fail if prompt exceeds context window
long_document = "..." * 200000  # Too long!
response = client.chat(messages=[
    {"role": "user", "content": f"Summarize: {long_document}"}
])
# Error: Maximum context length exceeded
```

### Solution 1: Truncation

```python
def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token limit."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

# Use it
safe_text = truncate_to_tokens(long_text, max_tokens=3000)
```

### Solution 2: Chunking

```python
def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200):
    """Split text into overlapping chunks."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(text)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(encoding.decode(chunk_tokens))
        start = end - overlap  # Overlap for context continuity
    
    return chunks

# Process each chunk
chunks = chunk_text(long_document)
summaries = []

for chunk in chunks:
    summary = summarize(chunk)
    summaries.append(summary)

final_summary = combine_summaries(summaries)
```

### Solution 3: Map-Reduce Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                     Map-Reduce for Long Docs                    │
│                                                                 │
│   Long Document                                                 │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────┬─────────┬─────────┬─────────┐                    │
│   │ Chunk 1 │ Chunk 2 │ Chunk 3 │ Chunk 4 │   MAP              │
│   └────┬────┴────┬────┴────┬────┴────┬────┘                    │
│        │         │         │         │                          │
│        ▼         ▼         ▼         ▼                          │
│   ┌─────────┬─────────┬─────────┬─────────┐                    │
│   │Summary 1│Summary 2│Summary 3│Summary 4│                    │
│   └────┬────┴────┬────┴────┬────┴────┬────┘                    │
│        │         │         │         │                          │
│        └─────────┴────┬────┴─────────┘                          │
│                       │                                         │
│                       ▼                                REDUCE   │
│              ┌────────────────┐                                 │
│              │ Final Summary  │                                 │
│              └────────────────┘                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Token Budgeting

### Calculate Available Space

```python
def calculate_available_tokens(
    context_window: int,
    system_prompt: str,
    conversation_history: list,
    reserved_for_response: int = 1000
) -> int:
    """Calculate how many tokens available for new content."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    
    # Count system prompt tokens
    system_tokens = len(encoding.encode(system_prompt))
    
    # Count conversation history tokens
    history_tokens = sum(
        len(encoding.encode(msg["content"])) 
        for msg in conversation_history
    )
    
    # Calculate available
    used = system_tokens + history_tokens + reserved_for_response
    available = context_window - used
    
    return max(0, available)

# Example usage
available = calculate_available_tokens(
    context_window=128000,
    system_prompt="You are a helpful assistant...",
    conversation_history=messages,
    reserved_for_response=2000
)
print(f"Available tokens for new content: {available}")
```

### Budget Example

```
Context Window: 128,000 tokens
├── System Prompt: 500 tokens
├── Conversation History: 5,000 tokens  
├── Retrieved Context (RAG): 10,000 tokens
├── Current User Message: 200 tokens
├── Reserved for Response: 4,000 tokens
└── Buffer/Safety: 1,000 tokens
    ─────────────────────────────
    Total Used: 20,700 tokens
    Available: 107,300 tokens
```

## Cost Implications

### Pricing Model

Most providers charge separately for input and output tokens:

```
Total Cost = (Input Tokens × Input Price) + (Output Tokens × Output Price)
```

### Example Pricing (2025 Estimates)

| Model | Input (per 1M) | Output (per 1M) |
|-------|---------------|-----------------|
| GPT-5.2 | $5.00 | $15.00 |
| GPT-4 Turbo | $2.50 | $7.50 |
| Claude 4.5 Sonnet | $3.00 | $15.00 |
| Gemini 3 Pro | $1.25 | $5.00 |

### Cost Optimization Tips

```python
# 1. Set appropriate max_tokens
response = client.chat(
    messages=messages,
    max_tokens=500  # Don't over-allocate
)

# 2. Use smaller models for simple tasks
if task_complexity == "low":
    model = "gpt-4-mini"  # Cheaper
else:
    model = "gpt-5.2"

# 3. Cache repeated queries
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_response(prompt_hash):
    return expensive_api_call(prompt)

# 4. Compress conversation history
def summarize_old_messages(messages):
    if len(messages) > 20:
        old = messages[:-10]
        summary = summarize(old)
        return [{"role": "system", "content": f"Previous context: {summary}"}] + messages[-10:]
    return messages
```

## Best Practices

### 1. Always Estimate Before Calling

```python
def safe_api_call(prompt: str, max_input: int = 100000):
    tokens = count_tokens(prompt)
    if tokens > max_input:
        raise ValueError(f"Input too long: {tokens} > {max_input}")
    return client.chat(messages=[{"role": "user", "content": prompt}])
```

### 2. Set max_tokens Appropriately

```python
# Task-based max_tokens
MAX_TOKENS = {
    "classification": 50,
    "summary": 300,
    "explanation": 800,
    "code_generation": 2000,
    "article": 4000,
}

response = client.chat(
    messages=messages,
    max_tokens=MAX_TOKENS[task_type]
)
```

### 3. Monitor Usage

```python
# Track token usage
response = client.chat(messages=messages)

if response.usage:
    print(f"Input tokens: {response.usage['prompt_tokens']}")
    print(f"Output tokens: {response.usage['completion_tokens']}")
    print(f"Total tokens: {response.usage['total_tokens']}")
    
    # Calculate cost
    cost = calculate_cost(response.usage, model="gpt-5.2")
    print(f"Estimated cost: ${cost:.4f}")
```

## Key Takeaways

1. **Tokens ≠ words** - They're chunks determined by the tokenizer
2. **Context window is shared** - Input + output must fit
3. **max_tokens controls output length** - Set it intentionally
4. **Plan for long content** - Use chunking or summarization
5. **Monitor costs** - Token usage directly impacts billing

## Next Steps

Now let's explore **Stop Sequences** to control exactly when the model stops generating.
