# Types of Fine-tuning

There are several approaches to fine-tuning, each with different trade-offs in terms of cost, quality, and resource requirements. This lesson covers the major methods.

## Learning Objectives

By the end of this lesson, you will:
- Understand full fine-tuning vs parameter-efficient methods
- Know when to use LoRA, QLoRA, and other techniques
- Compare resource requirements for each approach
- Choose the right method for your use case

---

## Fine-tuning Spectrum

```
┌─────────────────────────────────────────────────────────────┐
│                 Fine-tuning Methods                         │
│                                                             │
│  Full Fine-tuning ◄────────────────────► Prompt Tuning     │
│                                                             │
│  ┌──────────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────────┐ │
│  │   Full   │  │ LoRA │  │QLoRA │  │Prefix│  │  Prompt  │ │
│  │Fine-tune │  │      │  │      │  │Tuning│  │  Tuning  │ │
│  └──────────┘  └──────┘  └──────┘  └──────┘  └──────────┘ │
│                                                             │
│  All params    Low-rank   Quantized Prefix    Soft         │
│  updated       adapters   + LoRA    tokens    prompts      │
│                                                             │
│  Most          ◄─────────────────────────►   Least         │
│  Expensive                                   Expensive     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Full Fine-tuning

### What It Is

Update **all parameters** of the model during training:

```
┌─────────────────────────────────────────────────────────────┐
│                    Full Fine-tuning                         │
│                                                             │
│   Original Model (7B params)                                │
│   ┌─────────────────────────────────────────────────────┐  │
│   │ ████████████████████████████████████████████████████│  │
│   │ All 7 billion parameters updated                    │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   Training: All weights modified                            │
│   Storage: Full model copy required                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Pros and Cons

| Pros | Cons |
|------|------|
| Maximum adaptability | Highest GPU memory |
| Best potential quality | Longest training time |
| Full model control | Risk of catastrophic forgetting |
| | Expensive (multiple GPU needed) |

### Resource Requirements

| Model Size | GPU Memory (Training) | Time (1K examples) |
|------------|----------------------|-------------------|
| 7B params | ~60GB (2x A100) | ~2-4 hours |
| 13B params | ~100GB (4x A100) | ~4-8 hours |
| 70B params | ~500GB (8x A100) | ~24+ hours |

### When to Use

- Maximum customization needed
- Budget for compute is not a concern
- Training on very large datasets (100K+ examples)
- Need to modify model behavior fundamentally

---

## LoRA (Low-Rank Adaptation)

### The Key Insight

Most weight changes during fine-tuning can be approximated by **low-rank matrices**:

```
┌─────────────────────────────────────────────────────────────┐
│                    LoRA Concept                             │
│                                                             │
│   Original Weight Matrix W (d × k)                         │
│   ┌─────────────────────────────────────┐                  │
│   │                                     │                  │
│   │    (e.g., 4096 × 4096 = 16M params) │                  │
│   │                                     │                  │
│   └─────────────────────────────────────┘                  │
│                                                             │
│   LoRA: Learn small matrices A and B where:                │
│   ΔW = B × A  (low-rank approximation)                     │
│                                                             │
│   ┌─────┐      ┌─────────────────────────┐                 │
│   │  A  │  ×   │           B             │  = ΔW           │
│   │(r×k)│      │         (d×r)           │                 │
│   └─────┘      └─────────────────────────┘                 │
│                                                             │
│   Example: r=8, d=k=4096                                   │
│   Original: 16M parameters                                  │
│   LoRA: 8×4096 + 4096×8 = 65K parameters (0.4%)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### How LoRA Works

```python
# Conceptual implementation
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        self.original = original_layer  # Frozen
        self.original.requires_grad = False
        
        # Low-rank matrices
        d, k = original_layer.weight.shape
        self.A = nn.Parameter(torch.randn(rank, k) * 0.01)
        self.B = nn.Parameter(torch.zeros(d, rank))
        
        self.scaling = alpha / rank
    
    def forward(self, x):
        # Original forward (frozen) + LoRA adjustment
        original_output = self.original(x)
        lora_output = (x @ self.A.T @ self.B.T) * self.scaling
        return original_output + lora_output
```

### LoRA Hyperparameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| **rank (r)** | Rank of adaptation matrices | 4, 8, 16, 32, 64 |
| **alpha** | Scaling factor | Usually 2×rank |
| **target_modules** | Which layers to adapt | attention layers |
| **dropout** | LoRA dropout rate | 0.0-0.1 |

### Pros and Cons

| Pros | Cons |
|------|------|
| ~10x less memory | Slightly less flexible |
| ~3x faster training | Rank choice affects quality |
| Small adapter files (~10MB) | May not reach full fine-tune quality |
| Easy to switch/combine | |
| No catastrophic forgetting | |

### Resource Requirements

| Model Size | GPU Memory (LoRA) | Adapter Size |
|------------|------------------|--------------|
| 7B params | ~12GB (1x RTX 3090) | ~20MB |
| 13B params | ~24GB (1x A100) | ~40MB |
| 70B params | ~80GB (with tricks) | ~200MB |

---

## QLoRA (Quantized LoRA)

### The Innovation

Combine **4-bit quantization** with LoRA for extreme memory efficiency:

```
┌─────────────────────────────────────────────────────────────┐
│                    QLoRA Architecture                       │
│                                                             │
│   Base Model                      LoRA Adapters             │
│   ┌─────────────────────┐        ┌─────────────────────┐   │
│   │ 4-bit Quantized     │        │ Full Precision      │   │
│   │ NF4 data type       │   +    │ (bf16/fp16)         │   │
│   │ (Frozen)            │        │ (Trainable)         │   │
│   └─────────────────────┘        └─────────────────────┘   │
│                                                             │
│   Memory: ~4GB for 7B model      Memory: ~20MB              │
│                                                             │
│   Total: Train 7B model on 12GB VRAM!                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Techniques

| Technique | Description |
|-----------|-------------|
| **4-bit NormalFloat (NF4)** | Optimal quantization for normally distributed weights |
| **Double Quantization** | Quantize the quantization constants |
| **Paged Optimizers** | CPU offloading for optimizer states |

### Resource Requirements

| Model Size | QLoRA Memory | Consumer GPU? |
|------------|--------------|---------------|
| 7B params | ~6GB | RTX 3060 ✓ |
| 13B params | ~10GB | RTX 3080 ✓ |
| 33B params | ~20GB | RTX 4090 ✓ |
| 70B params | ~40GB | A100 40GB ✓ |

### QLoRA vs LoRA

| Aspect | LoRA | QLoRA |
|--------|------|-------|
| Base model | Full precision | 4-bit quantized |
| Memory | Moderate | Very low |
| Training speed | Faster | Slower (~25%) |
| Quality | Excellent | Very good |
| Consumer GPU | Maybe | Yes! |

---

## Other Parameter-Efficient Methods

### Prefix Tuning

Add trainable "prefix" tokens to each layer:

```
┌─────────────────────────────────────────────────────────────┐
│                    Prefix Tuning                            │
│                                                             │
│   [PREFIX_1][PREFIX_2]...[PREFIX_k] [Actual Input Tokens]  │
│        ↓        ↓            ↓              ↓               │
│   Trainable                            Frozen model         │
│                                                             │
│   Prefix length: 10-100 tokens per layer                   │
│   Total params: ~0.1% of model                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Prompt Tuning

Learn soft prompts prepended to input:

```python
# Soft prompt: learnable embedding vectors
soft_prompt = nn.Parameter(torch.randn(num_tokens, embed_dim))

# During forward pass:
input_embeds = torch.cat([soft_prompt, input_embeds], dim=1)
```

### Adapters

Insert small trainable modules between layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    Adapter Modules                          │
│                                                             │
│   Transformer Layer (Frozen)                                │
│          ↓                                                  │
│   ┌─────────────────┐                                      │
│   │    Adapter      │  ← Trainable                         │
│   │   (Down → Up)   │                                      │
│   └─────────────────┘                                      │
│          ↓                                                  │
│   Transformer Layer (Frozen)                                │
│                                                             │
│   Adapter: Linear(d → r) → ReLU → Linear(r → d)            │
│   Typical r: 64-256                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Comparison Table

| Method | Trainable Params | Memory | Quality | Flexibility |
|--------|-----------------|--------|---------|-------------|
| **Full Fine-tune** | 100% | Highest | Best | Highest |
| **LoRA** | 0.1-1% | Low | Very Good | High |
| **QLoRA** | 0.1-1% | Lowest | Good | High |
| **Adapters** | 1-5% | Medium | Good | Medium |
| **Prefix Tuning** | 0.1% | Very Low | Moderate | Low |
| **Prompt Tuning** | 0.01% | Minimal | Lower | Low |

---

## Choosing the Right Method

### Decision Guide

```
┌─────────────────────────────────────────────────────────────┐
│                 Method Selection Guide                      │
│                                                             │
│  What's your constraint?                                    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Limited GPU Memory (<24GB)?                         │   │
│  │    → QLoRA                                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Need maximum quality, have resources?               │   │
│  │    → Full Fine-tuning                               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Moderate GPU (24-48GB), good quality?               │   │
│  │    → LoRA                                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Multiple tasks, need to switch adapters?            │   │
│  │    → LoRA (easy adapter swapping)                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Cloud fine-tuning (OpenAI, etc.)?                   │   │
│  │    → Provider handles method (usually full)         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Practical Recommendations

| Scenario | Recommended Method |
|----------|-------------------|
| **Consumer GPU, 12GB** | QLoRA |
| **RTX 4090 (24GB)** | LoRA |
| **A100 40GB** | LoRA or Full (7B) |
| **Cloud provider API** | Full (provider managed) |
| **Multiple domain adapters** | LoRA |
| **Maximum quality, cost no object** | Full |

---

## Combining Adapters

LoRA adapters can be combined:

### Merging Adapters

```python
# Load base model
base_model = AutoModelForCausalLM.from_pretrained("base-model")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "lora-adapter")

# Merge adapter weights into base model
merged_model = model.merge_and_unload()

# Now it's a regular model with adapter weights baked in
```

### Switching Adapters

```python
from peft import PeftModel

# Base model loaded once
base_model = AutoModelForCausalLM.from_pretrained("base-model")

# Switch between adapters without reloading base
model = PeftModel.from_pretrained(base_model, "adapter-legal")
# ... use for legal tasks

model.load_adapter("adapter-medical")
# ... use for medical tasks
```

### Stacking Adapters

```python
# Combine multiple adapters
model.load_adapter("adapter-style", adapter_name="style")
model.load_adapter("adapter-format", adapter_name="format")

# Set active adapters
model.set_adapter(["style", "format"])  # Both active
```

---

## Key Takeaways

1. **Full fine-tuning** is most powerful but most expensive

2. **LoRA** offers 90%+ of quality at 10% of resources

3. **QLoRA** enables fine-tuning on consumer GPUs

4. **Method choice** depends on resources, not just quality needs

5. **Adapters are modular** - train once, combine and switch easily

6. **Start with QLoRA/LoRA** - only go to full fine-tune if needed

---

## What's Next?

Now that you understand the methods, let's learn about **preparing datasets** for fine-tuning → [Dataset Preparation](/learn/fine-tuning/fine-tuning-fundamentals/dataset)

---

## Quick Reference

| Method | Memory (7B) | Quality | Best For |
|--------|-------------|---------|----------|
| **Full** | 60GB | ★★★★★ | Max quality, big budget |
| **LoRA** | 12GB | ★★★★☆ | Good balance |
| **QLoRA** | 6GB | ★★★★☆ | Consumer GPUs |
| **Prefix** | 4GB | ★★★☆☆ | Simple adaptation |
