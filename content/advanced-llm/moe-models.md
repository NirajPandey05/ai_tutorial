# Mixture of Experts (MoE) Models

Mixture of Experts is an architectural pattern that allows models to have many parameters while only activating a subset during inference, dramatically improving efficiency.

## Learning Objectives

By the end of this lesson, you will:
- Understand how MoE architecture works
- Know the benefits and trade-offs of MoE models
- Recognize MoE patterns in modern LLMs
- Understand routing mechanisms and load balancing

---

## The MoE Concept

### The Efficiency Problem

Traditional dense models have a fundamental limitation:

```
┌─────────────────────────────────────────────────────────────┐
│                    Dense Model                              │
│                                                             │
│  Input → [All parameters activated] → Output                │
│                                                             │
│  Parameters: 70B  →  Compute: 70B parameters every token    │
└─────────────────────────────────────────────────────────────┘
```

**Problem**: To make models more capable, we add more parameters, but then **every inference uses all parameters**.

### The MoE Solution

```
┌─────────────────────────────────────────────────────────────┐
│                    MoE Model                                │
│                                                             │
│  Input → [Router selects 2 of 8 experts] → Output           │
│                                                             │
│  Total params: 400B  →  Active: ~50B per token              │
│  8x parameters, similar compute cost!                       │
└─────────────────────────────────────────────────────────────┘
```

---

## How MoE Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    MoE Layer                                │
│                                                             │
│                      Input                                  │
│                        │                                    │
│                        ↓                                    │
│               ┌────────────────┐                           │
│               │    Router      │                           │
│               │  (Gating Net)  │                           │
│               └────────────────┘                           │
│                ↙   ↙   ↘   ↘                               │
│           w₁   w₂       w₃   w₄   (routing weights)        │
│            ↓    ↓        ↓    ↓                            │
│      ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                      │
│      │ E₁  │ │ E₂  │ │ E₃  │ │ E₄  │  (Experts - FFNs)    │
│      └─────┘ └─────┘ └─────┘ └─────┘                      │
│         ↓      ↓        ×      ×     (only top-k used)    │
│         └──────┴────────┘                                  │
│                 ↓                                          │
│         Weighted Sum                                       │
│                 ↓                                          │
│              Output                                        │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Role | Details |
|-----------|------|---------|
| **Router/Gate** | Decides which experts to use | Small neural network |
| **Experts** | Specialized sub-networks | Usually feed-forward networks |
| **Top-k Selection** | Limits active experts | Typically k=1 or k=2 |

### The Router (Gating Network)

```python
class Router(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2):
        self.gate = nn.Linear(d_model, num_experts)
        self.top_k = top_k
    
    def forward(self, x):
        # Compute routing scores
        router_logits = self.gate(x)  # (batch, seq, num_experts)
        
        # Get top-k experts
        routing_weights, selected_experts = torch.topk(
            router_logits.softmax(dim=-1), 
            self.top_k, 
            dim=-1
        )
        
        # Normalize weights for selected experts
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        return routing_weights, selected_experts
```

### Expert Networks

Experts are typically standard FFN blocks:

```python
class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.activation = nn.SiLU()
    
    def forward(self, x):
        return self.w2(self.activation(self.w1(x)))
```

---

## MoE in Transformer Layers

MoE typically replaces the FFN in transformer layers:

```
Standard Transformer Layer:        MoE Transformer Layer:
┌─────────────────────────┐       ┌─────────────────────────┐
│    Self-Attention       │       │    Self-Attention       │
└───────────┬─────────────┘       └───────────┬─────────────┘
            ↓                                 ↓
┌─────────────────────────┐       ┌─────────────────────────┐
│    Layer Norm           │       │    Layer Norm           │
└───────────┬─────────────┘       └───────────┬─────────────┘
            ↓                                 ↓
┌─────────────────────────┐       ┌─────────────────────────┐
│    Feed-Forward (FFN)   │       │    MoE Layer            │
│    (Dense)              │       │    (Sparse)             │
└───────────┬─────────────┘       └───────────┬─────────────┘
            ↓                                 ↓
┌─────────────────────────┐       ┌─────────────────────────┐
│    Layer Norm           │       │    Layer Norm           │
└─────────────────────────┘       └─────────────────────────┘
```

---

## Modern MoE Models

### Mixtral 8x7B (Mistral AI)

```
┌─────────────────────────────────────────────────────────────┐
│                    Mixtral 8x7B                             │
├─────────────────────────────────────────────────────────────┤
│  Total Parameters:     46.7B                                │
│  Active Parameters:    12.9B (per token)                    │
│  Number of Experts:    8                                    │
│  Experts per Token:    2                                    │
│  Expert Type:          SMoE (Sparse MoE)                    │
│                                                             │
│  Performance:          Matches LLaMA 70B                    │
│  Inference Cost:       ~13B model equivalent                │
└─────────────────────────────────────────────────────────────┘
```

### Grok-1 (xAI)

```
┌─────────────────────────────────────────────────────────────┐
│                    Grok-1                                   │
├─────────────────────────────────────────────────────────────┤
│  Total Parameters:     314B                                 │
│  Active Parameters:    ~78B (per token)                     │
│  Number of Experts:    8                                    │
│  Experts per Token:    2                                    │
│  Architecture:         MoE variant                          │
└─────────────────────────────────────────────────────────────┘
```

### GPT-4 (Speculated MoE)

While not officially confirmed, GPT-4 is widely believed to use MoE:

```
┌─────────────────────────────────────────────────────────────┐
│                    GPT-4 (Rumored)                          │
├─────────────────────────────────────────────────────────────┤
│  Total Parameters:     ~1.7T (speculated)                   │
│  Number of Experts:    ~16 (speculated)                     │
│  Architecture:         MoE                                  │
│  Note: Official details not confirmed by OpenAI             │
└─────────────────────────────────────────────────────────────┘
```

### DeepSeek MoE

```
┌─────────────────────────────────────────────────────────────┐
│                    DeepSeek V2                              │
├─────────────────────────────────────────────────────────────┤
│  Total Parameters:     236B                                 │
│  Active Parameters:    21B (per token)                      │
│  Number of Experts:    160                                  │
│  Experts per Token:    6                                    │
│  Innovation:           Fine-grained experts + shared experts│
└─────────────────────────────────────────────────────────────┘
```

---

## Routing Mechanisms

### Token-Level Routing

Each token is routed independently:

```
Sentence: "The cat sat on the mat"
           ↓    ↓   ↓   ↓   ↓   ↓
Expert:    1,3  2,5 1,4 3,6 2,3 1,5
```

**Benefit**: Fine-grained specialization
**Challenge**: Load balancing

### Expert-Level Routing

Experts have capacity limits:

```python
def capacity_constrained_routing(router_logits, capacity_factor=1.25):
    num_tokens = router_logits.shape[0]
    num_experts = router_logits.shape[1]
    
    # Calculate capacity per expert
    capacity = int(capacity_factor * num_tokens / num_experts)
    
    # Route tokens with capacity constraints
    # Overflow tokens might be dropped or processed by backup
    ...
```

### Auxiliary Load Balancing Loss

To prevent experts from being under/over-utilized:

```python
def load_balancing_loss(router_logits, expert_indices):
    """
    Encourage uniform expert utilization
    """
    num_experts = router_logits.shape[-1]
    
    # Fraction of tokens routed to each expert
    tokens_per_expert = expert_indices.bincount(minlength=num_experts)
    fraction_tokens = tokens_per_expert / tokens_per_expert.sum()
    
    # Router probability for each expert
    router_probs = router_logits.softmax(dim=-1).mean(dim=0)
    
    # Loss encourages uniform distribution
    aux_loss = num_experts * (fraction_tokens * router_probs).sum()
    
    return aux_loss
```

---

## Benefits and Trade-offs

### Advantages

| Benefit | Explanation |
|---------|-------------|
| **Parameter Efficiency** | More parameters without proportional compute |
| **Specialization** | Experts can specialize in different domains |
| **Scaling** | Easier to scale to very large models |
| **Inference Speed** | Similar to smaller dense model |

### Challenges

| Challenge | Details |
|-----------|---------|
| **Training Instability** | Router can collapse to using few experts |
| **Load Imbalance** | Some experts may be over/under-utilized |
| **Memory Requirements** | All experts must fit in memory |
| **Communication Overhead** | In distributed training, expert placement matters |
| **Batch Efficiency** | Different tokens need different experts |

---

## Sparse vs Dense: When to Choose

### Choose MoE When:

```
✓ Training very large models (100B+ parameters)
✓ Need to match larger model quality with less compute
✓ Have distributed training infrastructure
✓ Can handle increased memory requirements
✓ Tasks benefit from specialization
```

### Choose Dense When:

```
✓ Smaller models (< 10B parameters)
✓ Limited memory availability
✓ Simpler deployment requirements
✓ Need more predictable latency
✓ Fine-tuning existing models
```

### Comparison Table

| Aspect | Dense Model | MoE Model |
|--------|-------------|-----------|
| **Params for Quality** | High | Lower |
| **Inference Compute** | All params | Subset |
| **Memory** | Lower | Higher |
| **Training Complexity** | Lower | Higher |
| **Deployment** | Simpler | More complex |

---

## Expert Specialization

### What Do Experts Learn?

Research shows experts naturally specialize:

```
┌─────────────────────────────────────────────────────────────┐
│              Example Expert Specialization                  │
├───────────────┬─────────────────────────────────────────────┤
│ Expert 1      │ Punctuation, syntax                         │
│ Expert 2      │ Named entities, proper nouns                │
│ Expert 3      │ Mathematical expressions                    │
│ Expert 4      │ Code keywords, programming                  │
│ Expert 5      │ Common words, general text                  │
│ Expert 6      │ Scientific terminology                      │
│ Expert 7      │ Foreign language tokens                     │
│ Expert 8      │ Rare tokens, edge cases                     │
└───────────────┴─────────────────────────────────────────────┘
```

### Visualizing Expert Usage

```python
# Analyze which experts process which token types
def analyze_expert_specialization(model, texts):
    token_expert_map = defaultdict(Counter)
    
    for text in texts:
        tokens, expert_choices = model.route(text)
        for token, experts in zip(tokens, expert_choices):
            token_type = classify_token(token)  # e.g., "noun", "verb", "code"
            for expert in experts:
                token_expert_map[token_type][expert] += 1
    
    return token_expert_map
```

---

## Advanced MoE Patterns

### Shared Expert + Routed Experts

Some architectures combine a shared expert with routed ones:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│        Input                                                │
│          │                                                  │
│    ┌─────┴─────┐                                           │
│    ↓           ↓                                           │
│  Shared    ┌───────┐                                       │
│  Expert    │Router │                                       │
│    ↓       └───┬───┘                                       │
│    │           │                                           │
│    │     ┌─────┴─────┐                                     │
│    │     ↓     ↓     ↓                                     │
│    │   Exp1  Exp2  Exp3  (routed)                          │
│    │     └─────┬─────┘                                     │
│    │           │                                           │
│    └─────┬─────┘                                           │
│          ↓                                                  │
│        Sum                                                  │
│          ↓                                                  │
│       Output                                                │
└─────────────────────────────────────────────────────────────┘
```

**Used in**: DeepSeek V2

### Fine-Grained Experts

More, smaller experts for better specialization:

```
Traditional MoE:          Fine-Grained MoE:
8 large experts           64 small experts
Top-2 selection     →     Top-6 selection
                          More specialization
                          Better load balancing
```

**Used in**: DeepSeek V2 (160 experts!)

---

## Practical Considerations

### Running MoE Models Locally

```python
# Mixtral with transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
    device_map="auto",  # Automatic multi-GPU placement
    load_in_4bit=True,  # Quantization for memory
)

# Memory requirement: ~24GB in 4-bit
# All 8 experts must fit, even though only 2 active
```

### MoE with Ollama

```bash
# Mixtral through Ollama
ollama run mixtral

# Memory: ~24GB VRAM for full precision
# Or use quantized: ~13GB for Q4
```

### Inference Optimization

```python
# Expert parallelism - place experts on different GPUs
# Each GPU handles specific experts

expert_device_map = {
    "expert_0": "cuda:0",
    "expert_1": "cuda:0", 
    "expert_2": "cuda:1",
    "expert_3": "cuda:1",
    "expert_4": "cuda:2",
    "expert_5": "cuda:2",
    "expert_6": "cuda:3",
    "expert_7": "cuda:3",
}
```

---

## Key Takeaways

1. **MoE enables scaling** - More parameters without proportional compute

2. **Sparse activation** - Only top-k experts process each token

3. **Experts specialize** - Different experts learn different patterns

4. **Trade-offs exist** - Higher memory, more complex training

5. **Modern LLMs use MoE** - Mixtral, GPT-4, DeepSeek, Grok

6. **Routing matters** - Load balancing is critical for training

---

## What's Next?

Now that you understand MoE, let's explore **multimodal LLMs** that can process images, audio, and more → [Multimodal LLMs](/learn/advanced-llm/multimodal/multimodal-llms)

---

## Quick Reference

| Concept | Key Point |
|---------|-----------|
| **MoE** | Many experts, few active per token |
| **Router** | Learned network that selects experts |
| **Top-k** | Usually k=2 experts active |
| **Load Balancing** | Auxiliary loss prevents collapse |
| **Specialization** | Experts learn different patterns |
| **Memory** | Need all experts, use only some |
