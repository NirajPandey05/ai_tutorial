# Open Source LLM Landscape

The open-source LLM ecosystem has exploded with capable models. This guide helps you navigate the landscape and choose the right model for your use case.

## The Major Model Families

### Meta Llama Family

```yaml
llama_family:
  llama_3_2:
    sizes: ["1B", "3B", "11B-Vision", "90B-Vision"]
    context: "128K tokens"
    strengths:
      - "Lightweight models (1B, 3B) excellent for edge"
      - "Vision capabilities in larger models"
      - "Strong instruction following"
    use_cases: ["Mobile deployment", "Edge AI", "Multimodal"]
    
  llama_3_1:
    sizes: ["8B", "70B", "405B"]
    context: "128K tokens"
    strengths:
      - "State-of-the-art open weights"
      - "Excellent reasoning"
      - "Strong coding ability"
      - "Multilingual (8 languages)"
    use_cases: ["General purpose", "Coding", "Enterprise"]
    
  code_llama:
    sizes: ["7B", "13B", "34B", "70B"]
    context: "16K-100K tokens"
    strengths:
      - "Code-specialized"
      - "Fill-in-the-middle capability"
      - "Python specialist variant"
    use_cases: ["Code completion", "Code generation"]
```

### Mistral/Mixtral

```yaml
mistral_family:
  mistral_7b:
    size: "7B"
    context: "32K tokens"
    strengths:
      - "Best 7B model quality"
      - "Sliding window attention"
      - "Fast inference"
    use_cases: ["General purpose", "Chat"]
    
  mixtral_8x7b:
    size: "47B total, 13B active (MoE)"
    context: "32K tokens"
    strengths:
      - "8x7B Mixture of Experts"
      - "Only uses 2 experts per token"
      - "Excellent quality-to-compute ratio"
    use_cases: ["High quality at moderate compute"]
    
  mixtral_8x22b:
    size: "176B total, 44B active"
    context: "64K tokens"
    strengths:
      - "Larger MoE model"
      - "Near GPT-4 level on some benchmarks"
    use_cases: ["Enterprise", "Complex reasoning"]
```

### Qwen Family (Alibaba)

```yaml
qwen_family:
  qwen2_5:
    sizes: ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"]
    context: "32K-128K tokens"
    strengths:
      - "Best multilingual support"
      - "Excellent at coding"
      - "Strong math abilities"
      - "Wide size range"
    use_cases: ["Multilingual", "Coding", "Math"]
    
  qwen2_5_coder:
    sizes: ["1.5B", "7B", "32B"]
    context: "128K tokens"
    strengths:
      - "State-of-the-art open code model"
      - "92 programming languages"
      - "Repository-level understanding"
    use_cases: ["Code generation", "Code review"]
    
  qwen_vl:
    sizes: ["2B", "7B", "72B"]
    strengths:
      - "Vision-language model"
      - "Strong OCR capabilities"
      - "Document understanding"
    use_cases: ["Image analysis", "Document processing"]
```

### Microsoft Phi Family

```yaml
phi_family:
  phi_4:
    size: "14B"
    context: "16K tokens"
    strengths:
      - "Excellent reasoning"
      - "Compact but powerful"
      - "Strong on benchmarks"
    use_cases: ["Reasoning", "Education", "Edge deployment"]
    
  phi_3:
    sizes: ["3.8B (mini)", "7B (small)", "14B (medium)"]
    context: "4K-128K tokens"
    strengths:
      - "High quality for size"
      - "Good for mobile/edge"
    use_cases: ["Mobile", "Embedded", "Low resource"]
```

### Google Gemma

```yaml
gemma_family:
  gemma_2:
    sizes: ["2B", "9B", "27B"]
    context: "8K tokens"
    strengths:
      - "Efficient architecture"
      - "Good safety alignment"
      - "Permissive license"
    use_cases: ["Research", "Fine-tuning base"]
    
  codegemma:
    sizes: ["2B", "7B"]
    strengths:
      - "Code-specialized Gemma"
      - "Fill-in-the-middle"
    use_cases: ["Code completion"]
```

### DeepSeek

```yaml
deepseek_family:
  deepseek_v3:
    size: "671B total, ~37B active (MoE)"
    context: "128K tokens"
    strengths:
      - "Massive MoE model"
      - "Near GPT-4 performance"
      - "Efficient inference"
    use_cases: ["Enterprise", "Complex tasks"]
    
  deepseek_coder_v2:
    sizes: ["16B", "236B (MoE)"]
    context: "128K tokens"
    strengths:
      - "Excellent code model"
      - "Mixture of Experts"
      - "Strong reasoning"
    use_cases: ["Code generation", "Agentic coding"]
```

## Model Selection Guide

### By Use Case

```python
def recommend_model(
    use_case: str,
    max_vram_gb: float,
    quality_priority: float = 0.5
) -> list[dict]:
    """Recommend models based on use case and constraints."""
    
    recommendations = {
        "general_chat": [
            {"model": "llama-3.2-3b", "vram": 4, "quality": 0.7},
            {"model": "llama-3.1-8b", "vram": 10, "quality": 0.85},
            {"model": "mistral-7b", "vram": 8, "quality": 0.82},
            {"model": "qwen2.5-14b", "vram": 16, "quality": 0.88},
            {"model": "llama-3.1-70b", "vram": 45, "quality": 0.95},
        ],
        
        "coding": [
            {"model": "qwen2.5-coder-1.5b", "vram": 3, "quality": 0.7},
            {"model": "qwen2.5-coder-7b", "vram": 8, "quality": 0.85},
            {"model": "deepseek-coder-v2-16b", "vram": 18, "quality": 0.9},
            {"model": "qwen2.5-coder-32b", "vram": 38, "quality": 0.93},
        ],
        
        "reasoning": [
            {"model": "phi-4-14b", "vram": 16, "quality": 0.88},
            {"model": "qwen2.5-32b", "vram": 38, "quality": 0.9},
            {"model": "llama-3.1-70b", "vram": 45, "quality": 0.95},
        ],
        
        "multilingual": [
            {"model": "qwen2.5-7b", "vram": 8, "quality": 0.83},
            {"model": "qwen2.5-14b", "vram": 16, "quality": 0.88},
            {"model": "qwen2.5-72b", "vram": 85, "quality": 0.95},
        ],
        
        "vision": [
            {"model": "llava-v1.6-7b", "vram": 10, "quality": 0.8},
            {"model": "llama-3.2-11b-vision", "vram": 14, "quality": 0.85},
            {"model": "qwen-vl-7b", "vram": 10, "quality": 0.82},
        ],
        
        "edge_mobile": [
            {"model": "llama-3.2-1b", "vram": 2, "quality": 0.6},
            {"model": "phi-3-mini-3.8b", "vram": 4, "quality": 0.7},
            {"model": "qwen2.5-0.5b", "vram": 1, "quality": 0.5},
        ]
    }
    
    candidates = recommendations.get(use_case, recommendations["general_chat"])
    
    # Filter by VRAM (Q4 reduces VRAM by ~75%)
    suitable = []
    for model in candidates:
        q4_vram = model["vram"] * 0.3  # Q4 estimation
        if q4_vram <= max_vram_gb:
            suitable.append({
                **model,
                "q4_vram": round(q4_vram, 1),
                "fits_at": "Q4_K_M"
            })
        elif model["vram"] <= max_vram_gb:
            suitable.append({
                **model,
                "fits_at": "FP16"
            })
    
    # Sort by quality
    return sorted(suitable, key=lambda x: x["quality"], reverse=True)

# Examples
print("Coding with 12GB VRAM:")
for m in recommend_model("coding", 12):
    print(f"  {m['model']}: {m.get('fits_at', 'FP16')}")

print("\nGeneral chat with 8GB VRAM:")
for m in recommend_model("general_chat", 8):
    print(f"  {m['model']}: {m.get('fits_at', 'FP16')}")
```

### Decision Flowchart

```
                        START
                          │
           ┌──────────────┴──────────────┐
           │ What's your primary task?   │
           └──────────────┬──────────────┘
                          │
    ┌─────────┬─────────┬─┴───┬─────────┬─────────┐
    ▼         ▼         ▼     ▼         ▼         ▼
  Coding   General   Vision  Math    Multi-    Edge/
           Chat              /Logic  lingual   Mobile
    │         │         │     │         │         │
    ▼         ▼         ▼     ▼         ▼         ▼
  Qwen      Llama    Llama  Qwen     Qwen     Llama
  Coder     3.1/3.2  3.2    2.5      2.5      3.2-1B
  DeepSeek  Mistral  Vision Phi-4            Phi-3
  Coder     Qwen     Qwen                    mini
            2.5      VL
```

## Model Comparison Benchmarks

### Coding (HumanEval Pass@1)

```
Model                    │ Score  │ Size
─────────────────────────┼────────┼──────
GPT-4 Turbo             │  87%   │ ~1.8T
Qwen2.5-Coder-32B       │  83%   │ 32B
DeepSeek-Coder-V2-236B  │  81%   │ 236B MoE
Llama-3.1-70B           │  77%   │ 70B
Qwen2.5-Coder-7B        │  74%   │ 7B
Mistral-7B              │  62%   │ 7B
Llama-3.2-3B            │  55%   │ 3B
```

### General (MMLU)

```
Model                    │ Score  │ Size
─────────────────────────┼────────┼──────
GPT-4                   │  86%   │ ~1.8T
Llama-3.1-405B          │  85%   │ 405B
Qwen2.5-72B             │  82%   │ 72B
Llama-3.1-70B           │  79%   │ 70B
Mixtral-8x22B           │  77%   │ 176B MoE
Qwen2.5-14B             │  70%   │ 14B
Llama-3.1-8B            │  68%   │ 8B
Mistral-7B              │  64%   │ 7B
```

### Math (MATH Benchmark)

```
Model                    │ Score  │ Size
─────────────────────────┼────────┼──────
GPT-4                   │  52%   │ ~1.8T
Qwen2.5-Math-72B        │  58%   │ 72B
Qwen2.5-72B             │  50%   │ 72B
Llama-3.1-70B           │  47%   │ 70B
Phi-4-14B               │  45%   │ 14B
Qwen2.5-14B             │  40%   │ 14B
```

## Practical Selection Tips

### For Tight VRAM Budgets

```python
# Best models for limited VRAM (Q4 quantization)

vram_recommendations = {
    "4GB": {
        "best": "llama-3.2-3b",
        "alt": ["phi-3-mini", "qwen2.5-3b"],
        "note": "Good for basic tasks, fast"
    },
    "6GB": {
        "best": "mistral-7b",
        "alt": ["llama-3.1-8b", "qwen2.5-7b"],
        "note": "Sweet spot for quality/size"
    },
    "8GB": {
        "best": "llama-3.1-8b",
        "alt": ["qwen2.5-7b-coder", "gemma-2-9b"],
        "note": "RTX 3070/4070 friendly"
    },
    "12GB": {
        "best": "qwen2.5-14b",
        "alt": ["phi-4-14b", "mistral-nemo-12b"],
        "note": "Excellent quality at this tier"
    },
    "24GB": {
        "best": "qwen2.5-32b or llama-3.1-70b-q3",
        "alt": ["mixtral-8x7b", "deepseek-coder-33b"],
        "note": "RTX 4090 / A5000 tier"
    }
}
```

### Evaluating for Your Use Case

```python
def evaluate_model_for_task(model_name: str, tasks: list[str]) -> dict:
    """Template for evaluating a model on your specific tasks."""
    
    # Define your test cases
    test_cases = {
        "coding": [
            "Write a Python function to find prime numbers",
            "Debug this code: [insert buggy code]",
            "Explain this algorithm: [insert algorithm]"
        ],
        "reasoning": [
            "A bat and ball cost $1.10 total...",
            "If it takes 5 machines 5 minutes...",
            "Solve step by step: [math problem]"
        ],
        "instruction_following": [
            "Write exactly 3 bullet points about...",
            "Respond in JSON format with fields...",
            "Answer only yes or no: ..."
        ]
    }
    
    results = {}
    for task in tasks:
        if task in test_cases:
            # Run tests and evaluate
            # results[task] = run_evaluation(model_name, test_cases[task])
            pass
    
    return results
```

## License Comparison

| Model | License | Commercial Use | Modifications | Distribution |
|-------|---------|----------------|---------------|--------------|
| Llama 3 | Llama 3 Community | ✅ (<700M MAU) | ✅ | ✅ |
| Mistral | Apache 2.0 | ✅ | ✅ | ✅ |
| Qwen | Qwen License | ✅ | ✅ | ✅ |
| Gemma | Gemma Terms | ✅ | ✅ | ✅ (terms) |
| Phi | MIT | ✅ | ✅ | ✅ |
| DeepSeek | MIT | ✅ | ✅ | ✅ |

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│              OPEN SOURCE MODEL QUICK REFERENCE              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  GENERAL PURPOSE:                                           │
│    Small:  Llama-3.2-3B, Phi-3-mini                        │
│    Medium: Llama-3.1-8B, Mistral-7B, Qwen2.5-7B           │
│    Large:  Llama-3.1-70B, Qwen2.5-72B                      │
│                                                             │
│  CODING:                                                    │
│    Best:   Qwen2.5-Coder (1.5B/7B/32B)                     │
│    Alt:    DeepSeek-Coder-V2, CodeLlama                    │
│                                                             │
│  REASONING/MATH:                                            │
│    Best:   Phi-4-14B, Qwen2.5-Math                         │
│    Alt:    Llama-3.1-70B                                   │
│                                                             │
│  VISION:                                                    │
│    Best:   Llama-3.2-Vision, Qwen-VL                       │
│    Alt:    LLaVA                                           │
│                                                             │
│  EDGE/MOBILE:                                               │
│    Best:   Llama-3.2-1B, Qwen2.5-0.5B                      │
│    Alt:    Phi-3-mini                                      │
│                                                             │
│  MoE (Quality/Efficiency):                                  │
│    Best:   Mixtral-8x7B, DeepSeek-V3                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Next Steps

1. **Ollama Setup** - Run these models locally
2. **Quantization Guide** - Fit larger models
3. **vLLM Production** - Serve at scale
