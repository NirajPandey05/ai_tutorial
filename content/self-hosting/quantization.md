# Model Quantization: Balancing Quality and Performance

Quantization is the process of reducing the precision of model weights to decrease memory usage and increase inference speed. Understanding quantization trade-offs is crucial for self-hosting LLMs effectively.

## What is Quantization?

```
Original Model (FP32)              Quantized Model (INT4)
┌─────────────────────┐            ┌─────────────────────┐
│ Weight: 3.14159265  │            │ Weight: 3           │
│ Precision: 32 bits  │  ───────►  │ Precision: 4 bits   │
│ Size: 4 bytes       │            │ Size: 0.5 bytes     │
└─────────────────────┘            └─────────────────────┘
        ▼                                   ▼
   100% memory                         12.5% memory
   Baseline quality                    ~95% quality
```

### Why Quantize?

```python
# Memory reduction through quantization
def calculate_model_size(
    parameters_billions: float,
    precision: str
) -> dict:
    """Calculate model size at different precisions."""
    
    bits_per_param = {
        "FP32": 32,
        "FP16": 16,
        "BF16": 16,
        "INT8": 8,
        "INT4": 4,
        "INT3": 3,
        "INT2": 2
    }
    
    bits = bits_per_param[precision]
    size_bytes = (parameters_billions * 1e9 * bits) / 8
    size_gb = size_bytes / (1024**3)
    
    return {
        "precision": precision,
        "bits_per_param": bits,
        "size_gb": round(size_gb, 1),
        "relative_to_fp16": round((bits / 16) * 100, 1)
    }

# Compare sizes for a 70B model
for precision in ["FP32", "FP16", "INT8", "INT4"]:
    result = calculate_model_size(70, precision)
    print(f"70B @ {precision}: {result['size_gb']} GB ({result['relative_to_fp16']}% of FP16)")

# Output:
# 70B @ FP32: 260.8 GB (200.0% of FP16)
# 70B @ FP16: 130.4 GB (100.0% of FP16)
# 70B @ INT8: 65.2 GB (50.0% of FP16)
# 70B @ INT4: 32.6 GB (25.0% of FP16)
```

## Quantization Methods

### 1. Post-Training Quantization (PTQ)

Quantizes a pre-trained model without additional training.

```python
# Simple post-training quantization concept
import numpy as np

def naive_quantize_int8(weights: np.ndarray) -> tuple:
    """Simple INT8 quantization (demonstration)."""
    
    # Find scale factor
    max_val = np.max(np.abs(weights))
    scale = max_val / 127  # INT8 range: -128 to 127
    
    # Quantize
    quantized = np.round(weights / scale).astype(np.int8)
    
    return quantized, scale

def dequantize(quantized: np.ndarray, scale: float) -> np.ndarray:
    """Convert back to floating point."""
    return quantized.astype(np.float32) * scale

# Example
original = np.array([0.15, -0.42, 0.89, -0.03, 0.67])
quantized, scale = naive_quantize_int8(original)
restored = dequantize(quantized, scale)

print(f"Original:  {original}")
print(f"Quantized: {quantized} (scale: {scale:.4f})")
print(f"Restored:  {restored}")
print(f"Error:     {np.abs(original - restored)}")
```

### 2. Quantization-Aware Training (QAT)

Trains the model with quantization in mind.

```
QAT Process:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  FP32 Model ──► Simulate Quantization ──► Fine-tune ──►    │
│                 during forward pass       with simulated    │
│                                          quantization error │
│                                                  │          │
│                                                  ▼          │
│                                          Quantized Model    │
│                                          (Higher quality)   │
└─────────────────────────────────────────────────────────────┘
```

### 3. Group Quantization

Divides weights into groups with separate scale factors.

```python
def group_quantize(weights: np.ndarray, group_size: int = 128) -> dict:
    """
    Quantize weights in groups for better accuracy.
    Each group has its own scale factor.
    """
    
    # Reshape into groups
    num_weights = weights.size
    num_groups = (num_weights + group_size - 1) // group_size
    
    # Pad if necessary
    padded_size = num_groups * group_size
    padded = np.zeros(padded_size)
    padded[:num_weights] = weights.flatten()
    
    # Reshape to groups
    grouped = padded.reshape(num_groups, group_size)
    
    # Quantize each group
    scales = []
    quantized_groups = []
    
    for group in grouped:
        max_val = np.max(np.abs(group))
        scale = max_val / 7 if max_val > 0 else 1  # INT4 range: -8 to 7
        scales.append(scale)
        quantized_groups.append(np.round(group / scale).astype(np.int8))
    
    return {
        "quantized": np.array(quantized_groups),
        "scales": np.array(scales),
        "group_size": group_size,
        "original_shape": weights.shape
    }
```

## GGUF Quantization Types Explained

### K-Quants (Knowledge-Distillation Inspired)

```yaml
k_quants_explanation:
  Q4_K_M:
    description: "4-bit with medium group size"
    bits_per_weight: 4.5
    technique: "Mixed precision - important weights get more bits"
    quality: "Excellent balance of size and quality"
    recommended_for: "Most use cases"
    
  Q4_K_S:
    description: "4-bit with small group size"
    bits_per_weight: 4.5
    technique: "Similar to K_M but smaller groups"
    quality: "Slightly better quality, slightly larger"
    
  Q5_K_M:
    description: "5-bit with medium group size"
    bits_per_weight: 5.5
    quality: "Near-FP16 quality"
    recommended_for: "When quality is priority"
    
  Q6_K:
    description: "6-bit quantization"
    bits_per_weight: 6.5
    quality: "Very close to FP16"
    recommended_for: "Quality-critical applications"
```

### I-Quants (Importance-Aware)

```yaml
i_quants_explanation:
  IQ4_XS:
    description: "4-bit importance-aware, extra small"
    technique: "Identifies important weights, preserves them better"
    quality: "Better than Q4_K at same size"
    
  IQ3_XS:
    description: "3-bit importance-aware"
    technique: "Aggressive compression with smart weight selection"
    quality: "Impressive for 3-bit"
    
  IQ2_XXS:
    description: "2-bit importance-aware"
    technique: "Extreme compression"
    quality: "Usable but noticeable degradation"
    recommended_for: "Edge devices, extreme memory constraints"
```

## Quality vs Compression Trade-offs

### Perplexity Comparison

```
Perplexity on WikiText-2 (Lower is better)
Model: Llama 2 7B

FP16:     5.47  ████████████████████████████████████████ (baseline)
Q8_0:     5.48  ████████████████████████████████████████ (+0.2%)
Q6_K:     5.52  ████████████████████████████████████████ (+0.9%)
Q5_K_M:   5.58  ███████████████████████████████████████  (+2.0%)
Q4_K_M:   5.68  ██████████████████████████████████████   (+3.8%)
Q4_0:     5.79  █████████████████████████████████████    (+5.8%)
Q3_K_M:   6.12  ██████████████████████████████████       (+11.9%)
Q2_K:     7.24  ████████████████████████████             (+32.4%)
```

### Benchmark Code

```python
import subprocess
import json

def benchmark_quantization(model_path: str, prompt: str, num_tokens: int = 100) -> dict:
    """Benchmark a GGUF model's performance."""
    
    # Using llama.cpp's llama-bench or similar
    result = subprocess.run([
        "./llama-cli",
        "-m", model_path,
        "-p", prompt,
        "-n", str(num_tokens),
        "--log-disable",
        "-v"  # Verbose for timing info
    ], capture_output=True, text=True)
    
    # Parse output for metrics
    output = result.stderr + result.stdout
    
    # Extract timing (implementation-specific parsing)
    metrics = {
        "model": model_path,
        "prompt_tokens": len(prompt.split()),
        "generated_tokens": num_tokens,
        "output": result.stdout
    }
    
    return metrics

def compare_quantizations(base_model_path: str, quant_types: list[str]):
    """Compare different quantization levels."""
    
    test_prompt = "Explain the theory of relativity in simple terms:"
    results = []
    
    for quant in quant_types:
        model_path = f"{base_model_path}-{quant}.gguf"
        metrics = benchmark_quantization(model_path, test_prompt)
        results.append(metrics)
    
    return results
```

## Choosing the Right Quantization

### Decision Framework

```python
def recommend_quantization(
    available_vram_gb: float,
    model_size_billions: float,
    use_case: str,
    quality_priority: float = 0.5  # 0 = speed priority, 1 = quality priority
) -> str:
    """
    Recommend quantization level based on constraints.
    
    Args:
        available_vram_gb: Available GPU memory
        model_size_billions: Model parameters in billions
        use_case: "chat", "coding", "reasoning", "creative"
        quality_priority: 0-1 scale
    """
    
    # Calculate FP16 size
    fp16_size = model_size_billions * 2
    
    # Quality requirements by use case
    quality_needs = {
        "chat": 0.4,      # Can tolerate some degradation
        "coding": 0.7,    # Needs good quality
        "reasoning": 0.8, # Needs high quality
        "creative": 0.5   # Moderate needs
    }
    
    combined_quality = (quality_priority + quality_needs.get(use_case, 0.5)) / 2
    
    # Define quantization options
    quants = [
        ("Q2_K", 0.08, 0.75),   # (name, size_ratio, quality)
        ("Q3_K_M", 0.11, 0.88),
        ("IQ3_XS", 0.10, 0.89),
        ("Q4_0", 0.125, 0.92),
        ("Q4_K_M", 0.14, 0.95),
        ("IQ4_XS", 0.13, 0.94),
        ("Q5_K_M", 0.17, 0.97),
        ("Q6_K", 0.20, 0.98),
        ("Q8_0", 0.25, 0.99),
        ("F16", 0.50, 1.00),
    ]
    
    # Find best fit
    for name, size_ratio, quality in reversed(quants):
        estimated_size = fp16_size * size_ratio * 1.2  # 20% overhead
        if estimated_size <= available_vram_gb and quality >= combined_quality:
            return f"{name} (Est. size: {estimated_size:.1f}GB, Quality: {quality*100:.0f}%)"
    
    # Nothing fits
    return f"Model too large. Need {fp16_size * 0.08 * 1.2:.1f}GB minimum for Q2_K"

# Examples
print(recommend_quantization(24, 70, "chat"))      # Q4_K_M for 70B on 24GB
print(recommend_quantization(12, 7, "coding"))    # Q6_K or Q8_0 for 7B on 12GB
print(recommend_quantization(8, 13, "reasoning")) # Q3_K_M for 13B on 8GB
```

### Quick Reference Table

| VRAM | 7B Model | 13B Model | 34B Model | 70B Model |
|------|----------|-----------|-----------|-----------|
| 4 GB | Q3_K_M | IQ2_XXS | ❌ | ❌ |
| 6 GB | Q4_K_M | Q2_K | ❌ | ❌ |
| 8 GB | Q6_K | Q3_K_M | IQ2_XXS | ❌ |
| 12 GB | Q8_0 | Q4_K_M | Q2_K | ❌ |
| 16 GB | F16 | Q5_K_M | Q3_K_M | IQ2_XXS |
| 24 GB | F16 | Q8_0/F16 | Q4_K_M | Q2_K |
| 48 GB | F16 | F16 | Q8_0 | Q4_K_M |
| 80 GB | F16 | F16 | F16 | Q6_K |

## Quantization in Practice

### Converting Models with llama.cpp

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build
make -j

# Convert HF model to GGUF (FP16)
python convert_hf_to_gguf.py \
    /path/to/huggingface/model \
    --outfile model-f16.gguf \
    --outtype f16

# Quantize to different levels
./llama-quantize model-f16.gguf model-q8_0.gguf Q8_0
./llama-quantize model-f16.gguf model-q6_k.gguf Q6_K
./llama-quantize model-f16.gguf model-q5_k_m.gguf Q5_K_M
./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
./llama-quantize model-f16.gguf model-q3_k_m.gguf Q3_K_M

# With importance matrix for better quality (optional)
./llama-quantize \
    model-f16.gguf \
    model-q4_k_m.gguf \
    Q4_K_M \
    --imatrix importance_matrix.dat
```

### Creating Importance Matrix

```bash
# Generate importance matrix for better quantization
./llama-imatrix \
    -m model-f16.gguf \
    -f calibration_data.txt \
    -o importance_matrix.dat

# calibration_data.txt should contain diverse text samples
# representing your expected use case
```

## Mixed-Precision Strategies

### Layer-Specific Quantization

Some tools support different quantization per layer:

```yaml
mixed_precision_strategy:
  attention_layers:
    precision: "Q6_K or higher"
    reason: "Critical for quality"
    
  feedforward_layers:
    precision: "Q4_K_M acceptable"
    reason: "More tolerant of compression"
    
  embedding_layers:
    precision: "Q8_0 or higher"
    reason: "Important for token understanding"
    
  output_layers:
    precision: "Q6_K or higher"
    reason: "Affects final output quality"
```

## Quantization Artifacts

### Common Issues

```python
# Signs of over-quantization

quantization_artifacts = {
    "repetition": {
        "symptom": "Model repeats phrases or gets stuck in loops",
        "likely_cause": "Attention weights too compressed",
        "solution": "Use higher quantization (Q5 instead of Q4)"
    },
    
    "gibberish": {
        "symptom": "Model outputs nonsensical text",
        "likely_cause": "Severe quantization error",
        "solution": "Use Q4_K_M minimum, avoid Q2/Q3"
    },
    
    "wrong_facts": {
        "symptom": "Model confuses facts or names",
        "likely_cause": "Knowledge compressed too much",
        "solution": "Higher quantization for knowledge tasks"
    },
    
    "format_issues": {
        "symptom": "Model can't follow format instructions",
        "likely_cause": "Instruction-following degraded",
        "solution": "Use Q5_K_M or higher for structured output"
    }
}
```

## Summary

### Quantization Cheat Sheet

```
┌─────────────────────────────────────────────────────────────┐
│                 QUANTIZATION QUICK GUIDE                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Quality Critical (coding, reasoning):                      │
│    → Q6_K, Q8_0, or FP16                                   │
│                                                             │
│  Balanced (general chat, most uses):                        │
│    → Q4_K_M or Q5_K_M                                      │
│                                                             │
│  Memory Constrained:                                        │
│    → Q4_K_M (best quality-size ratio)                      │
│    → IQ4_XS (slightly better quality)                      │
│                                                             │
│  Extreme Constraints:                                       │
│    → Q3_K_M (last resort before Q2)                        │
│    → Q2_K (significant quality loss)                       │
│                                                             │
│  Rule of Thumb:                                             │
│    Q4_K_M is the sweet spot for most users                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Next Steps

Now that you understand quantization:
1. **Ollama Setup** - Run quantized models easily
2. **llama.cpp** - Direct quantization and inference
3. **Model Selection** - Choose the right base model
