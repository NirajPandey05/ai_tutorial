# Deep Dive: Tokenization

## What is Tokenization?

**Tokenization** is the process of breaking text into smaller units called **tokens** that LLMs can process. It's the first step in any LLM operation and fundamentally shapes how models understand language.

```
┌─────────────────────────────────────────────────────────────────┐
│                    The Tokenization Pipeline                    │
│                                                                 │
│   Raw Text → Tokenizer → Token IDs → Embedding → Model         │
│                                                                 │
│   "Hello!"  →  ["Hello", "!"]  →  [15496, 0]  →  [vectors]     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Why Not Just Use Words?

At first glance, splitting by words seems logical. But consider:

### Problem 1: Vocabulary Size

```python
# English has ~170,000 words
# Plus technical terms, names, neologisms...
# Vocabulary would be enormous!

"defragmentation"  # Rare word
"ChatGPT"          # New word  
"🚀"               # Emoji
"東京"             # Other languages
```

### Problem 2: Unknown Words

```python
# What if the model encounters a word it's never seen?
"pneumonoultramicroscopicsilicovolcanoconiosis"  # ???
```

### Problem 3: Morphology

```python
# Related words should be related in the model
"run", "running", "runs", "ran"  # Same base concept
```

## Subword Tokenization

Modern LLMs use **subword tokenization** - breaking words into smaller, meaningful pieces:

```
"unhappiness" → ["un", "happiness"] or ["un", "happ", "iness"]
"ChatGPT"     → ["Chat", "G", "PT"]
"running"     → ["run", "ning"]
```

### Benefits

1. **Fixed vocabulary size** - Typically 30,000-100,000 tokens
2. **No unknown words** - Can represent any text
3. **Morphological awareness** - Related words share tokens
4. **Multilingual support** - Works across languages

## Common Tokenization Algorithms

### 1. Byte Pair Encoding (BPE)

Used by: GPT models, Llama, most modern LLMs

**How it works:**
1. Start with character-level vocabulary
2. Find most frequent pair of tokens
3. Merge them into a new token
4. Repeat until vocabulary size reached

```
Starting text: "low lower lowest"

Iteration 1: Most frequent pair = "l" + "o" → "lo"
  Vocabulary: [..., "lo"]
  Text: "low lower lowest" → "lo" + "w", "lo" + "w" + "er", "lo" + "w" + "est"

Iteration 2: Most frequent pair = "lo" + "w" → "low"
  Vocabulary: [..., "lo", "low"]
  Text: "low", "low" + "er", "low" + "est"

Iteration 3: Most frequent pair = "low" + "er" → "lower"
  ...and so on
```

### 2. WordPiece

Used by: BERT, some Google models

Similar to BPE but uses a different scoring mechanism based on likelihood improvement.

```
"playing" → ["play", "##ing"]

The "##" prefix indicates a continuation token
```

### 3. Unigram

Used by: T5, some multilingual models

Starts with a large vocabulary and progressively removes tokens to optimize likelihood.

### 4. SentencePiece

A library that can implement BPE or Unigram, treating text as raw bytes (no pre-tokenization needed).

Used by: Llama, Mistral, multilingual models

```python
# SentencePiece treats spaces as underscores
"Hello World" → ["▁Hello", "▁World"]
```

## Tokenization in Practice

### Using tiktoken (OpenAI)

```python
import tiktoken

# Get encoder for specific model
enc = tiktoken.encoding_for_model("gpt-4")

# Encode text to tokens
text = "Hello, how are you today?"
tokens = enc.encode(text)
print(f"Tokens: {tokens}")
# Tokens: [9906, 11, 1268, 527, 499, 3432, 30]

print(f"Count: {len(tokens)}")
# Count: 7

# Decode back to text
decoded = enc.decode(tokens)
print(f"Decoded: {decoded}")
# Decoded: Hello, how are you today?
```

### Visualizing Tokens

```python
def visualize_tokens(text: str, model: str = "gpt-4"):
    """Show individual tokens with their IDs."""
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    
    for token_id in tokens:
        token_text = enc.decode([token_id])
        print(f"  {token_id:>6} → '{token_text}'")
    
    print(f"\nTotal: {len(tokens)} tokens")

visualize_tokens("Hello, world!")
#   15496 → 'Hello'
#      11 → ','
#     995 → ' world'
#       0 → '!'
#
# Total: 4 tokens
```

## Tokenization Quirks

### Leading Spaces Matter

```python
enc = tiktoken.encoding_for_model("gpt-4")

# Without leading space
print(enc.encode("hello"))    # [15339]

# With leading space  
print(enc.encode(" hello"))   # [24748]

# Different tokens!
```

### Numbers Are Tricky

```python
# Each digit might be separate
enc.encode("12345")  # Might be: [123, 45] or [12, 345] or [1, 2, 3, 4, 5]

# Large numbers use more tokens
enc.encode("1000000000")  # Multiple tokens vs
enc.encode("1e9")         # Fewer tokens
```

### Code Uses More Tokens

```python
# Natural language
enc.encode("Calculate the sum")  # ~3 tokens

# Code
enc.encode("calculate_the_sum")  # ~4-5 tokens (underscores are separate)
```

### Emojis and Special Characters

```python
enc.encode("🚀")    # Usually 1 token
enc.encode("🧑‍💻")  # Compound emoji = multiple tokens (4+)
enc.encode("→")     # Single token
enc.encode("⟨⟩")   # Multiple tokens
```

## Token Boundaries and Meaning

### Why Token Boundaries Matter

```python
# The model sees tokens, not characters
# This affects tasks like:

# 1. Counting characters (hard!)
prompt = "How many letters are in 'strawberry'?"
# Model must mentally decode tokens to count

# 2. Reversing strings
prompt = "Reverse 'hello'"
# Token boundaries may not align with characters

# 3. Acronyms
enc.encode("NASA")  # Might be [N, AS, A] or [NASA]
# Depends on tokenizer training data
```

### Prompt Engineering Implications

```python
# BAD: Token might split unexpectedly
prompt = "Extract entities like GoogleAI and OpenAI"
# "GoogleAI" might become ["Google", "AI"] or ["Goo", "gle", "AI"]

# GOOD: Use consistent formatting
prompt = "Extract entities like 'Google AI' and 'OpenAI'"
```

## Multilingual Tokenization

### Token Efficiency Varies by Language

```python
# English (efficient - trained on mostly English)
enc.encode("Hello, how are you?")  # ~6 tokens

# Chinese (less efficient)
enc.encode("你好，你怎么样？")  # ~10+ tokens

# Programming (common patterns trained)
enc.encode("def hello():")  # ~4 tokens
```

### Implications

| Language | Relative Token Efficiency |
|----------|--------------------------|
| English | 1.0x (baseline) |
| Spanish | ~1.1x |
| German | ~1.3x |
| Chinese | ~2-3x |
| Japanese | ~2-3x |
| Korean | ~2-3x |

This affects:
- **Cost** - More tokens = higher price
- **Context** - Less room in context window
- **Speed** - More tokens = slower generation

## Comparing Tokenizers

```python
import tiktoken

def compare_tokenizers(text: str):
    """Compare token counts across models."""
    models = ["gpt-4", "gpt-3.5-turbo", "text-davinci-003"]
    
    for model in models:
        enc = tiktoken.encoding_for_model(model)
        tokens = enc.encode(text)
        print(f"{model}: {len(tokens)} tokens")

compare_tokenizers("Artificial intelligence is transforming industries.")
# gpt-4: 6 tokens
# gpt-3.5-turbo: 6 tokens
# text-davinci-003: 7 tokens
```

## Best Practices

### 1. Count Tokens, Not Characters

```python
# BAD: Estimating by characters
estimated_tokens = len(text) / 4  # Unreliable!

# GOOD: Use actual tokenizer
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")
actual_tokens = len(enc.encode(text))
```

### 2. Be Aware of Token Boundaries in Prompts

```python
# If doing text manipulation tasks, be explicit:
prompt = """
Count the CHARACTERS (not words) in 'hello'.
Think step by step:
1. List each character
2. Count them
"""
```

### 3. Optimize for Token Efficiency

```python
# VERBOSE (more tokens):
"Please provide me with a comprehensive list of items"

# CONCISE (fewer tokens):
"List the items"

# JSON keys matter:
{"customer_email_address": "..."}  # More tokens
{"email": "..."}                    # Fewer tokens
```

### 4. Test with Target Tokenizer

```python
def will_fit_in_context(
    prompt: str, 
    max_context: int = 128000,
    reserved_output: int = 4000
) -> bool:
    """Check if prompt fits in context window."""
    enc = tiktoken.encoding_for_model("gpt-4")
    prompt_tokens = len(enc.encode(prompt))
    return prompt_tokens <= (max_context - reserved_output)
```

## Key Takeaways

1. **Tokens ≠ words** - Subword tokenization breaks text into smaller units
2. **Fixed vocabulary** - ~30K-100K tokens handle any text
3. **Language affects efficiency** - English uses fewer tokens
4. **Boundaries matter** - Tasks like counting characters are hard for models
5. **Always count tokens** - Don't estimate by characters

## Next Steps

Now that you understand tokenization, let's explore **Context Windows** and how to manage large amounts of text.
