# Attention Mechanisms Deep Dive

Attention is the core innovation that makes modern LLMs possible. This lesson explores how attention works, why it matters, and the variations used in state-of-the-art models.

## Learning Objectives

By the end of this lesson, you will:
- Understand how self-attention computes relationships between tokens
- Know the difference between attention variants (MHA, MQA, GQA)
- Understand positional encoding methods
- Recognize attention patterns and their implications

---

## The Attention Intuition

Before diving into the math, let's understand attention intuitively.

### The Problem Attention Solves

Consider this sentence:

> "The **animal** didn't cross the street because **it** was too tired."

What does "it" refer to? A human instantly knows it's "animal" (not "street"). But how?

**Attention allows the model to look at other positions** in the sequence when processing each word:

```
Processing "it":
                    attention weights
"The"      ──────── 0.02 ────────┐
"animal"   ──────── 0.85 ────────┼──→ "it" representation
"didn't"   ──────── 0.01 ────────┤    (informed by animal)
"cross"    ──────── 0.01 ────────┤
"the"      ──────── 0.02 ────────┤
"street"   ──────── 0.05 ────────┤
"because"  ──────── 0.02 ────────┤
"it"       ──────── 0.02 ────────┘
```

---

## Self-Attention Mechanics

### The Query-Key-Value Framework

Self-attention uses three vectors for each token:

| Vector | Role | Analogy |
|--------|------|---------|
| **Query (Q)** | "What am I looking for?" | A search query |
| **Key (K)** | "What do I contain?" | Document metadata |
| **Value (V)** | "What information do I provide?" | Document content |

### Step-by-Step Computation

```python
# Input: sequence of token embeddings
# X shape: (sequence_length, embedding_dim)

# Step 1: Create Q, K, V through learned linear projections
Q = X @ W_Q  # Query matrix
K = X @ W_K  # Key matrix
V = X @ W_V  # Value matrix

# Step 2: Compute attention scores
scores = Q @ K.T  # (seq_len, seq_len) - how much each token attends to others

# Step 3: Scale scores (prevents gradient issues with large dimensions)
scores = scores / sqrt(d_k)  # d_k is the dimension of keys

# Step 4: Apply softmax to get attention weights
attention_weights = softmax(scores, dim=-1)  # Rows sum to 1

# Step 5: Weighted sum of values
output = attention_weights @ V  # Final attended representation
```

### Visual Representation

```
Input:  ["The", "cat", "sat"]
         ↓      ↓      ↓
        E_1    E_2    E_3  (embeddings)
         │      │      │
    ┌────┼──────┼──────┼────┐
    │    ↓      ↓      ↓    │
    │   Q_1    Q_2    Q_3   │ (Queries)
    │   K_1    K_2    K_3   │ (Keys)
    │   V_1    V_2    V_3   │ (Values)
    └───────────────────────┘
              │
              ↓
    ┌─────────────────────────────┐
    │   Attention Matrix          │
    │   ┌─────┬─────┬─────┐      │
    │   │ .3  │ .5  │ .2  │ Q_1  │
    │   ├─────┼─────┼─────┤      │
    │   │ .4  │ .4  │ .2  │ Q_2  │
    │   ├─────┼─────┼─────┤      │
    │   │ .1  │ .3  │ .6  │ Q_3  │
    │   └─────┴─────┴─────┘      │
    │     K_1   K_2   K_3        │
    └─────────────────────────────┘
              │
              ↓
    Output: Weighted combination of V's
```

---

## Multi-Head Attention (MHA)

### Why Multiple Heads?

A single attention head can only focus on one type of relationship at a time. **Multiple heads allow parallel attention patterns**:

- **Head 1**: Might focus on syntactic relationships
- **Head 2**: Might focus on semantic similarity
- **Head 3**: Might focus on positional proximity
- ...

### Multi-Head Architecture

```
Input X
    │
    ├──→ Head 1: Q₁K₁V₁ → output₁
    │
    ├──→ Head 2: Q₂K₂V₂ → output₂
    │
    ├──→ Head 3: Q₃K₃V₃ → output₃
    │
    └──→ Head N: QₙKₙVₙ → outputₙ
              │
              ↓
         Concatenate
              │
              ↓
      Linear Projection
              │
              ↓
        Final Output
```

### Implementation

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Separate projections for each head (or combined)
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)
    
    def forward(self, x):
        batch, seq_len, d_model = x.shape
        
        # Project and reshape for multi-head
        Q = self.W_q(x).view(batch, seq_len, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch, seq_len, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch, seq_len, self.num_heads, self.d_k)
        
        # Transpose for attention: (batch, heads, seq, d_k)
        Q, K, V = [t.transpose(1, 2) for t in (Q, K, V)]
        
        # Compute attention for all heads in parallel
        attn_output = self.attention(Q, K, V)
        
        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, d_model)
        return self.W_o(attn_output)
```

### Typical Configurations

| Model | Heads | d_model | d_k per head |
|-------|-------|---------|--------------|
| GPT-2 Small | 12 | 768 | 64 |
| GPT-3 | 96 | 12288 | 128 |
| LLaMA 7B | 32 | 4096 | 128 |
| LLaMA 70B | 64 | 8192 | 128 |

---

## Attention Variants

### Multi-Query Attention (MQA)

**Problem**: In MHA, we store K and V for every head during inference → huge memory cost for long sequences.

**Solution**: Share K and V across all heads, only Q is per-head:

```
MHA:                    MQA:
Q₁ Q₂ Q₃ Q₄            Q₁ Q₂ Q₃ Q₄
K₁ K₂ K₃ K₄      →     K (shared)
V₁ V₂ V₃ V₄            V (shared)
```

**Trade-off**: Slightly worse quality, but much faster inference (especially for long contexts).

**Used in**: PaLM, Falcon

### Grouped Query Attention (GQA)

**Compromise**: Share K/V within groups of query heads:

```
MHA (8 heads):         GQA (2 groups):        MQA:
Q₁Q₂Q₃Q₄Q₅Q₆Q₇Q₈      Q₁Q₂Q₃Q₄ Q₅Q₆Q₇Q₈      Q₁Q₂Q₃Q₄Q₅Q₆Q₇Q₈
K₁K₂K₃K₄K₅K₆K₇K₈  →   K₁K₁K₁K₁ K₂K₂K₂K₂  →   K K K K K K K K
V₁V₂V₃V₄V₅V₆V₇V₈      V₁V₁V₁V₁ V₂V₂V₂V₂      V V V V V V V V
(8 KV heads)          (2 KV heads)            (1 KV head)
```

**Used in**: LLaMA 2/3, Mistral, Gemma

### Comparison

| Variant | KV Heads | Quality | Memory | Speed |
|---------|----------|---------|--------|-------|
| **MHA** | N heads | Best | High | Slower |
| **GQA** | N/G groups | Very Good | Medium | Faster |
| **MQA** | 1 | Good | Low | Fastest |

---

## Causal (Masked) Attention

For autoregressive generation, each position can only attend to **previous positions**:

### The Causal Mask

```
Attention allowed (✓) vs blocked (✗):

         Position 1  Position 2  Position 3  Position 4
Query 1:     ✓           ✗           ✗           ✗
Query 2:     ✓           ✓           ✗           ✗
Query 3:     ✓           ✓           ✓           ✗
Query 4:     ✓           ✓           ✓           ✓
```

### Implementation

```python
def causal_attention(Q, K, V):
    seq_len = Q.shape[1]
    
    # Create causal mask: lower triangular matrix
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    
    # Compute attention with mask
    scores = Q @ K.T / sqrt(d_k)
    scores = scores + mask  # -inf becomes 0 after softmax
    attention = softmax(scores, dim=-1)
    
    return attention @ V
```

---

## Positional Encodings

Transformers have no inherent sense of position. We must inject it.

### Absolute Position Encodings (Original)

Add fixed sinusoidal patterns:

```python
def sinusoidal_encoding(position, d_model):
    encodings = []
    for i in range(d_model):
        if i % 2 == 0:
            encodings.append(sin(position / 10000**(i/d_model)))
        else:
            encodings.append(cos(position / 10000**(i/d_model)))
    return encodings
```

**Problem**: Fixed maximum length, doesn't generalize well.

### Learned Position Embeddings (GPT-2)

Train position embeddings as parameters:

```python
position_embeddings = nn.Embedding(max_positions, d_model)
# Added to token embeddings
x = token_embeddings + position_embeddings[positions]
```

**Problem**: Still fixed maximum length.

### Rotary Position Embeddings (RoPE)

**Modern solution**: Encode position by rotating Q and K vectors:

```python
def apply_rotary_pos_emb(x, position):
    # Rotate pairs of dimensions
    x1, x2 = x[..., ::2], x[..., 1::2]
    
    # Rotation angles based on position
    cos_pos = cos(position * frequencies)
    sin_pos = sin(position * frequencies)
    
    # Apply rotation
    x_rotated = torch.cat([
        x1 * cos_pos - x2 * sin_pos,
        x1 * sin_pos + x2 * cos_pos
    ], dim=-1)
    
    return x_rotated
```

**Benefits**:
- Relative positions emerge naturally
- Extrapolates to longer sequences
- Computationally efficient

**Used in**: LLaMA, Mistral, Qwen, Phi, Gemma

### ALiBi (Attention with Linear Biases)

Add position-dependent bias directly to attention scores:

```python
# Bias decreases with distance
bias = -m * |query_pos - key_pos|

attention_scores = Q @ K.T + bias
```

**Used in**: BLOOM, MPT

### Position Encoding Comparison

| Method | Extrapolation | Efficiency | Used By |
|--------|---------------|------------|---------|
| Sinusoidal | Poor | Good | Original Transformer |
| Learned | Poor | Good | GPT-2, BERT |
| RoPE | Good | Good | LLaMA, Mistral |
| ALiBi | Excellent | Good | BLOOM |

---

## Attention Patterns in Practice

### What Models Learn to Attend To

Research has revealed interesting attention patterns:

```
┌─────────────────────────────────────────────────────────────┐
│                 Common Attention Patterns                   │
├─────────────────┬───────────────────────────────────────────┤
│ Local Attention │ Adjacent tokens (grammar, syntax)         │
│ Skip-gram       │ Every N tokens (structural patterns)     │
│ Coreference     │ Pronouns → referents (it → animal)       │
│ Delimiter       │ Punctuation, special tokens              │
│ Global Sink     │ First/last tokens (collect information)  │
└─────────────────┴───────────────────────────────────────────┘
```

### The "Attention Sink" Phenomenon

Models often attend heavily to the **first token**, even if it's not semantically important. This creates an "attention sink" that stabilizes training.

---

## Flash Attention

### The Memory Problem

Standard attention requires storing the full attention matrix:
- Sequence length 8K → 64 million entries
- Sequence length 128K → 16 billion entries!

### Flash Attention Solution

Recompute attention in blocks, never storing the full matrix:

```
Standard:                Flash:
1. Compute full Q@K^T    1. For each block of Q:
2. Store in memory       2.   For each block of K,V:
3. Apply softmax         3.     Compute partial attention
4. Multiply by V         4.     Accumulate results
                         5.   (Never store full matrix)
```

### Benefits

| Metric | Standard | Flash Attention |
|--------|----------|-----------------|
| Memory | O(N²) | O(N) |
| Speed | 1x | 2-4x faster |
| Max Context | ~4K | 128K+ |

---

## Practical Implications

### Choosing Attention Mechanisms

| Scenario | Recommendation |
|----------|----------------|
| Training from scratch | MHA with Flash Attention |
| Fine-tuning for speed | Consider GQA models |
| Long context (>32K) | Must use Flash Attention |
| Inference optimization | GQA or MQA models |

### Context Length vs. Quality Trade-off

```
┌─────────────────────────────────────────────────────────────┐
│              Context Length Impact                          │
├───────────────┬─────────────────────────────────────────────┤
│ Short (2-4K)  │ Highest quality attention                   │
│ Medium (8-32K)│ Good quality, needs proper position encoding│
│ Long (64K+)   │ Attention dilution, needs retrieval/RAG     │
└───────────────┴─────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **Attention is learned association** - The model learns which tokens should influence each other

2. **Multi-head attention** enables learning multiple relationship types simultaneously

3. **GQA is the modern standard** - Best balance of quality and efficiency

4. **Position encoding matters** - RoPE enables longer context generalization

5. **Flash Attention is essential** - Enables 128K+ contexts practically

6. **Attention has patterns** - Models learn syntactic, semantic, and structural attention

---

## What's Next?

Now that you understand attention mechanisms, let's explore how **Mixture of Experts** allows models to scale efficiently → [Mixture of Experts Models](/learn/advanced-llm/advanced-architectures/moe)

---

## Quick Reference

| Concept | Key Point |
|---------|-----------|
| **Q, K, V** | Query asks, Key matches, Value provides |
| **Scaling** | Divide by √d_k to stabilize gradients |
| **Causal mask** | Future tokens set to -∞ before softmax |
| **Multi-head** | Parallel attention for different patterns |
| **GQA** | Share K/V within groups of Q heads |
| **RoPE** | Position through rotation, extrapolates well |
| **Flash Attention** | Block-wise computation, O(N) memory |
