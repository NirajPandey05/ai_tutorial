# Understanding Model Architectures

Modern Large Language Models come in various architectural flavors, each with unique strengths and trade-offs. Understanding these architectures helps you choose the right model for your use case and optimize your applications.

## Learning Objectives

By the end of this lesson, you will:
- Understand the key differences between major LLM architectures
- Know when to use decoder-only vs encoder-decoder models
- Recognize the trade-offs between model families
- Make informed decisions about model selection

---

## The Transformer Foundation

All modern LLMs are built on the **Transformer** architecture, introduced in the 2017 paper "Attention Is All You Need." However, different models use different parts of this architecture.

### Original Transformer Structure

```
┌─────────────────────────────────────────────┐
│              Encoder-Decoder                │
├─────────────────┬───────────────────────────┤
│    Encoder      │        Decoder            │
│  (Understands)  │      (Generates)          │
│                 │                           │
│  ┌───────────┐  │  ┌───────────────────┐   │
│  │ Self-     │  │  │ Masked Self-      │   │
│  │ Attention │  │  │ Attention         │   │
│  └───────────┘  │  └───────────────────┘   │
│       ↓         │          ↓               │
│  ┌───────────┐  │  ┌───────────────────┐   │
│  │ FFN       │  │  │ Cross-Attention   │   │
│  └───────────┘  │  └───────────────────┘   │
│                 │          ↓               │
│                 │  ┌───────────────────┐   │
│                 │  │ FFN               │   │
│                 │  └───────────────────┘   │
└─────────────────┴───────────────────────────┘
```

---

## Decoder-Only Models (GPT Family)

Most modern LLMs use a **decoder-only** architecture. This includes GPT-4, GPT-5, Claude, Grok, LLaMA, and Mistral.

### How Decoder-Only Works

```
Input: "The capital of France is"
        ↓
┌─────────────────────────────────┐
│     Decoder Stack (N layers)    │
│  ┌───────────────────────────┐  │
│  │ Masked Self-Attention     │  │ ← Can only look at previous tokens
│  └───────────────────────────┘  │
│  ┌───────────────────────────┐  │
│  │ Feed-Forward Network      │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
        ↓
Output: "Paris"
```

### Causal (Autoregressive) Generation

Decoder-only models generate text **one token at a time**, always looking left:

```
Step 1: "The"           → Predict next token
Step 2: "The capital"   → Predict next token  
Step 3: "The capital of" → Predict next token
...
```

### Key Characteristics

| Aspect | Description |
|--------|-------------|
| **Generation** | Autoregressive (left-to-right) |
| **Training** | Next-token prediction |
| **Strengths** | Text generation, chat, reasoning |
| **Weaknesses** | Can't naturally "look ahead" |

### Notable Models

| Model Family | Organization | Key Features |
|--------------|--------------|--------------|
| **GPT-4/5** | OpenAI | Largest, most capable general models |
| **Claude 3.5/4** | Anthropic | Strong reasoning, safety-focused |
| **Grok 4** | xAI | Real-time knowledge, humor |
| **LLaMA 3** | Meta | Open weights, efficient |
| **Mistral** | Mistral AI | Excellent size-to-performance ratio |
| **Qwen 2.5** | Alibaba | Multilingual, coding strength |
| **Phi-3/4** | Microsoft | Small but capable |

---

## Encoder-Only Models (BERT Family)

**Encoder-only** models like BERT are designed for **understanding** rather than generation.

### Bidirectional Attention

Unlike decoder models, encoders can look at the **entire input** simultaneously:

```
Input: "The bank by the [MASK] was flooded"
        ↓
┌─────────────────────────────────┐
│     Encoder Stack (N layers)    │
│  ┌───────────────────────────┐  │
│  │ Bidirectional Attention   │  │ ← Can see ALL tokens
│  └───────────────────────────┘  │
│  ┌───────────────────────────┐  │
│  │ Feed-Forward Network      │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
        ↓
[MASK] → "river" (understands context from both sides)
```

### Use Cases

| Task | Why BERT-style works |
|------|---------------------|
| **Classification** | Needs full context understanding |
| **Named Entity Recognition** | Context from both sides helps |
| **Semantic Search** | Creates rich embeddings |
| **Question Answering** | Finds spans in context |

### Notable Models

| Model | Key Features |
|-------|--------------|
| **BERT** | Original bidirectional encoder |
| **RoBERTa** | Better training of BERT |
| **DeBERTa** | Disentangled attention |
| **E5/BGE** | Modern embedding models |

---

## Encoder-Decoder Models (T5 Family)

**Encoder-decoder** models use both components for sequence-to-sequence tasks.

### Architecture Flow

```
Input: "Translate to French: Hello world"
        ↓
┌─────────────────────────────────┐
│           Encoder               │
│  (Processes input sequence)     │
└─────────────────────────────────┘
        ↓ (Hidden states)
┌─────────────────────────────────┐
│           Decoder               │
│  (Generates output sequence)    │
│  + Cross-attention to encoder   │
└─────────────────────────────────┘
        ↓
Output: "Bonjour le monde"
```

### Use Cases

| Task | Why Encoder-Decoder works |
|------|--------------------------|
| **Translation** | Natural sequence-to-sequence mapping |
| **Summarization** | Compress then generate |
| **Code Generation** | Understand spec, generate code |

### Notable Models

| Model | Key Features |
|-------|--------------|
| **T5/Flan-T5** | Text-to-text framework |
| **BART** | Denoising autoencoder |
| **mT5** | Multilingual T5 |
| **UL2** | Unified language learner |

---

## Architecture Comparison

### When to Use Each

```
┌──────────────────────────────────────────────────────────────┐
│                    Architecture Selection                     │
├────────────────┬─────────────────────────────────────────────┤
│ Decoder-Only   │ • Chat/conversation                         │
│ (GPT, Claude)  │ • Creative writing                          │
│                │ • Code generation                            │
│                │ • General reasoning                          │
│                │ • Most modern LLM applications               │
├────────────────┼─────────────────────────────────────────────┤
│ Encoder-Only   │ • Text classification                       │
│ (BERT)         │ • Semantic similarity                       │
│                │ • Named entity recognition                   │
│                │ • Embedding generation                       │
│                │ • Search/retrieval                           │
├────────────────┼─────────────────────────────────────────────┤
│ Encoder-Decoder│ • Translation                               │
│ (T5, BART)     │ • Summarization                             │
│                │ • Structured generation                      │
│                │ • When input/output differ significantly     │
└────────────────┴─────────────────────────────────────────────┘
```

### Parameter Efficiency

| Architecture | Typical Sizes | Efficiency Notes |
|--------------|---------------|------------------|
| **Decoder-Only** | 7B - 1T+ params | Can be very large, but efficient inference |
| **Encoder-Only** | 100M - 1B params | Smaller, fast inference |
| **Encoder-Decoder** | 1B - 20B params | More params for same capability |

---

## Modern Architectural Innovations

### 1. Grouped Query Attention (GQA)

Reduces memory usage by sharing key-value heads:

```
Standard MHA:     GQA (4 groups):
Q Q Q Q Q Q Q Q   Q Q Q Q Q Q Q Q
↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓   ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
K K K K K K K K   K K K K
V V V V V V V V   V V V V
(8 KV heads)      (4 KV heads, shared)
```

**Used in**: LLaMA 2/3, Mistral, Gemini

### 2. Sliding Window Attention

Limits attention to a local window for efficiency:

```
Full Attention:        Sliding Window (size=3):
[A B C D E]           [A B C D E]
 ↑↑↑↑↑ (all tokens)    ↑↑↑ (local only)
```

**Used in**: Mistral, Gemma

### 3. RoPE (Rotary Position Embeddings)

Encodes position through rotation in complex space:

```python
# Conceptually:
def apply_rope(x, position):
    angle = position * frequency
    return x * cos(angle) + rotate(x) * sin(angle)
```

**Used in**: LLaMA, Mistral, Qwen, Phi

### 4. Flash Attention

Hardware-aware attention implementation that's 2-4x faster:

```
Standard:              Flash Attention:
Load all Q,K,V        Block-wise computation
Compute attention     Fuse operations
Store all results     Minimize memory access
```

**Benefit**: Enables longer context windows (128K+)

---

## Model Selection Guide

### Decision Framework

```
┌─────────────────────────────────────────────────────────────┐
│                    What's your task?                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Text Generation/Chat ──────────→ Decoder-Only (GPT/Claude) │
│          │                                                   │
│  Understanding/Classification ──→ Encoder (BERT/DeBERTa)    │
│          │                                                   │
│  Translation/Summarization ─────→ Encoder-Decoder (T5)      │
│          │                                                   │
│  Embeddings/Search ─────────────→ Encoder (E5/BGE)          │
│          │                                                   │
│  Multimodal ────────────────────→ Vision-LLM (GPT-4V/Gemini)│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Practical Recommendations

| Use Case | Recommended Architecture | Top Models |
|----------|-------------------------|------------|
| **Chatbot** | Decoder-only | GPT-4, Claude 3.5, Llama 3 |
| **Code Assistant** | Decoder-only | GPT-4, Claude, DeepSeek Coder |
| **Search** | Encoder | E5-large, BGE-base |
| **Translation** | Encoder-Decoder | NLLB, mT5 |
| **Classification** | Encoder | DeBERTa, RoBERTa |

---

## Key Takeaways

1. **Decoder-only models dominate** modern LLM applications due to their flexibility and generation quality

2. **Encoder models excel** at understanding tasks like classification and embedding generation

3. **Architecture choice matters** - pick based on your task requirements, not just model size

4. **Modern innovations** (GQA, Flash Attention, RoPE) enable longer contexts and faster inference

5. **The trend is towards** larger decoder-only models with architectural optimizations

---

## What's Next?

Now that you understand the major architectures, let's dive deeper into the **attention mechanism** that powers all of them → [Attention Mechanisms Deep Dive](/learn/advanced-llm/model-architectures/attention)

---

## Quick Reference

| Architecture | Sees | Best For | Examples |
|--------------|------|----------|----------|
| **Decoder-only** | Past tokens | Generation | GPT, Claude, LLaMA |
| **Encoder-only** | All tokens | Understanding | BERT, RoBERTa |
| **Encoder-Decoder** | Full input → Generate output | Seq2Seq | T5, BART |
