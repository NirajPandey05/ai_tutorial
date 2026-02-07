# How LLMs Work: Transformers & Attention

Understanding how LLMs work under the hood will make you a more effective AI engineer. This page dives into the Transformer architecture, attention mechanisms, and tokenization.

## The Big Picture

When you send a message to an LLM, here's what happens:

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Processing Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  "Hello, how are you?"                                       │
│           ↓                                                  │
│  ┌─────────────────┐                                        │
│  │  1. Tokenizer   │  "Hello" "," " how" " are" " you" "?"  │
│  └────────┬────────┘                                        │
│           ↓                                                  │
│  ┌─────────────────┐                                        │
│  │  2. Embeddings  │  [0.23, -0.45, ...] for each token     │
│  └────────┬────────┘                                        │
│           ↓                                                  │
│  ┌─────────────────┐                                        │
│  │  3. Transformer │  96+ layers of attention & FFN         │
│  │     Layers      │                                        │
│  └────────┬────────┘                                        │
│           ↓                                                  │
│  ┌─────────────────┐                                        │
│  │  4. Output      │  Probability for each possible token   │
│  │     Logits      │                                        │
│  └────────┬────────┘                                        │
│           ↓                                                  │
│  ┌─────────────────┐                                        │
│  │  5. Sampling    │  Select next token based on probs      │
│  └────────┬────────┘                                        │
│           ↓                                                  │
│  "I'm doing well, thank you!"                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Step 1: Tokenization

LLMs don't see text—they see **tokens**. A tokenizer converts text into numbers.

### What is a Token?

Tokens are pieces of text, typically:
- Common words: "the", "is", "and"
- Word parts: "un" + "believ" + "able"
- Single characters: for rare text
- Special tokens: `<|endoftext|>`, `<|user|>`

```python
# Example tokenization (simplified)
text = "Hello, how are you?"
tokens = ["Hello", ",", " how", " are", " you", "?"]
token_ids = [15496, 11, 703, 527, 499, 30]
```

### Why Not Just Use Words?

| Approach | Problem |
|----------|---------|
| Characters | Too slow—each character needs processing |
| Words | Vocabulary too large, can't handle new words |
| **Subwords** | ✅ Balance of efficiency and coverage |

Modern LLMs use **BPE (Byte Pair Encoding)** or similar algorithms:

```python
# BPE can handle any text
"ChatGPT" → ["Chat", "G", "PT"]
"TensorFlow" → ["Tensor", "Flow"]
"日本語" → ["日", "本", "語"]  # Works for any language!
```

### Token Limits Matter

```python
# GPT-5.2 context window: 256,000 tokens
# Claude 4.5 context window: 200,000 tokens

# Rule of thumb: 1 token ≈ 4 characters in English
# So 256K tokens ≈ 1 million characters ≈ 750 pages of text
```

## Step 2: Embeddings

Each token ID is converted to a **dense vector** (embedding).

```python
# Token ID 15496 ("Hello") becomes:
embedding = [0.023, -0.456, 0.789, ..., 0.012]  # ~4096 dimensions

# Similar words have similar embeddings
cosine_similarity("king", "queen") ≈ 0.85
cosine_similarity("king", "banana") ≈ 0.15
```

### Positional Encoding

Transformers process all tokens in parallel, so they need position information:

```python
# Without position encoding:
"The cat sat on the mat" = "mat the on sat cat The"  # Same!

# With position encoding:
position_1 = [sin(1/10000), cos(1/10000), ...]
position_2 = [sin(2/10000), cos(2/10000), ...]
# Each position gets a unique "fingerprint"
```

Modern LLMs use **RoPE (Rotary Position Embedding)** for better handling of long sequences.

## Step 3: The Transformer Block

The core of an LLM is the Transformer, built from repeated blocks:

```
┌─────────────────────────────────────────┐
│           Transformer Block              │
├─────────────────────────────────────────┤
│                                          │
│  Input Embeddings                        │
│         ↓                                │
│  ┌──────────────────┐                   │
│  │ Layer Norm       │                   │
│  └────────┬─────────┘                   │
│           ↓                              │
│  ┌──────────────────┐                   │
│  │ Self-Attention   │  ← The magic!     │
│  └────────┬─────────┘                   │
│           ↓                              │
│       + Residual Connection              │
│           ↓                              │
│  ┌──────────────────┐                   │
│  │ Layer Norm       │                   │
│  └────────┬─────────┘                   │
│           ↓                              │
│  ┌──────────────────┐                   │
│  │ Feed-Forward     │  (MLP)            │
│  └────────┬─────────┘                   │
│           ↓                              │
│       + Residual Connection              │
│           ↓                              │
│  Output (→ next block)                   │
│                                          │
└─────────────────────────────────────────┘
         × 96+ layers (GPT-5.2)
```

## The Attention Mechanism

Attention is what makes Transformers powerful. It allows each token to "look at" all other tokens.

### Self-Attention Explained

For the sentence: **"The cat sat on the mat because it was tired"**

Q: What does "it" refer to?

```
Attention weights from "it":
  "The"     → 0.02
  "cat"     → 0.45  ← High attention! "it" = "the cat"
  "sat"     → 0.05
  "on"      → 0.01
  "the"     → 0.02
  "mat"     → 0.15
  "because" → 0.08
  "it"      → 0.12
  "was"     → 0.05
  "tired"   → 0.05
```

The model learns that "it" should attend strongly to "cat"!

### Query, Key, Value (QKV)

Attention uses three projections:

```python
# For each token:
Q = query  # "What am I looking for?"
K = key    # "What do I contain?"
V = value  # "What information do I provide?"

# Attention formula:
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V

# In plain English:
# 1. Compare each query with all keys (dot product)
# 2. Normalize scores with softmax
# 3. Use scores to weight the values
# 4. Sum weighted values = output
```

### Multi-Head Attention

LLMs use multiple attention "heads" in parallel:

```
┌─────────────────────────────────────────────────┐
│              Multi-Head Attention                │
├─────────────────────────────────────────────────┤
│                                                  │
│  Head 1: Focuses on syntax (subject-verb)       │
│  Head 2: Focuses on coreference (pronouns)      │
│  Head 3: Focuses on semantics (meaning)         │
│  ...                                             │
│  Head 96: Focuses on [learned pattern]          │
│                                                  │
│  All heads concatenated → Linear projection      │
│                                                  │
└─────────────────────────────────────────────────┘
```

Each head can learn to attend to different types of relationships!

### Causal Masking (Decoder-Only)

For text generation, tokens can only attend to **previous** tokens:

```
Position:    1     2     3     4     5
Token:      "The" "cat" "sat" "on"  ???

Attention mask:
            The   cat   sat   on    ???
The          ✓     ✗     ✗     ✗     ✗
cat          ✓     ✓     ✗     ✗     ✗
sat          ✓     ✓     ✓     ✗     ✗
on           ✓     ✓     ✓     ✓     ✗
???          ✓     ✓     ✓     ✓     ✓
```

This prevents the model from "cheating" by looking at future tokens during training.

## Step 4: Output Generation

After all transformer layers, we get output logits:

```python
# Output: probability for each token in vocabulary
# Vocabulary size: ~100,000 tokens

logits = model(input_tokens)  # Shape: [seq_len, vocab_size]

# For position 5 (predicting next word after "on"):
# "the" → 0.35
# "a"   → 0.15  
# "his" → 0.08
# ...
```

## Step 5: Sampling Strategies

How do we choose the next token from probabilities?

### Greedy Decoding
```python
# Always pick highest probability
next_token = argmax(probabilities)
# Problem: Boring, repetitive text
```

### Temperature Sampling
```python
# Temperature controls randomness
adjusted_probs = softmax(logits / temperature)

# temperature = 0.0: Always pick highest (greedy)
# temperature = 1.0: Sample from original distribution
# temperature = 2.0: More random, creative
```

### Top-P (Nucleus) Sampling
```python
# Only consider tokens in top P% of probability mass
# If top_p = 0.9:
#   Include tokens until cumulative prob ≥ 0.9
#   Ignore the long tail of unlikely tokens
```

### Top-K Sampling
```python
# Only consider the K most likely tokens
# top_k = 50: Choose from 50 most likely tokens
```

## Putting It All Together

```python
# Pseudocode for LLM generation
def generate(prompt, max_tokens=100):
    tokens = tokenize(prompt)
    
    for _ in range(max_tokens):
        # 1. Embed tokens
        embeddings = embed(tokens) + positional_encoding
        
        # 2. Pass through transformer layers
        hidden = embeddings
        for layer in transformer_layers:
            hidden = layer(hidden)  # Attention + FFN
        
        # 3. Get next token probabilities
        logits = output_projection(hidden[-1])
        probs = softmax(logits / temperature)
        
        # 4. Sample next token
        next_token = sample(probs, top_p=0.9)
        
        # 5. Append and continue
        tokens.append(next_token)
        
        if next_token == EOS_TOKEN:
            break
    
    return detokenize(tokens)
```

## Model Sizes

| Component | GPT-3 (175B) | GPT-5.2 (~1.8T) |
|-----------|-------------|-----------------|
| Layers | 96 | 128+ |
| Hidden size | 12,288 | 16,384+ |
| Attention heads | 96 | 128+ |
| Vocabulary | 50,257 | 100,000+ |
| Context length | 4,096 | 256,000 |

## Key Takeaways

1. **Tokenization** converts text to numbers the model can process
2. **Embeddings** represent tokens as dense vectors
3. **Self-attention** lets tokens share information across the sequence
4. **Multi-head attention** captures different relationship types
5. **Causal masking** ensures autoregressive generation
6. **Sampling strategies** control creativity vs. coherence

---

## Interactive Exercise

Try this mental exercise:

For the prompt: **"The Eiffel Tower is located in"**

1. How would you expect attention to work on this prompt?
2. What token would you expect to have highest probability next?
3. Why might temperature = 0 vs temperature = 1 give different results?

In the next section, you'll make your first API call to see this in action!
