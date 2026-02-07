# Model Formats: GGUF, GPTQ, AWQ, and More

Understanding model formats is essential for self-hosting LLMs. Different formats offer trade-offs between compatibility, performance, and file size.

## The Model Format Landscape

```
                        Model Format Ecosystem
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Training Format          Inference Formats                │
│   ┌─────────────┐          ┌──────────────────────────────┐ │
│   │ PyTorch     │          │ GGUF  - CPU/GPU universal    │ │
│   │ (.pt, .bin) │ ───────► │ GPTQ  - GPU quantized        │ │
│   │             │ Convert  │ AWQ   - GPU activation-aware │ │
│   │ Safetensors │ ───────► │ ONNX  - Cross-platform       │ │
│   │ (.safetensors)         │ TensorRT - NVIDIA optimized  │ │
│   └─────────────┘          └──────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## GGUF (GPT-Generated Unified Format)

The most popular format for local inference, developed by llama.cpp.

### Key Features
```yaml
gguf:
  name: "GPT-Generated Unified Format"
  developed_by: "llama.cpp project"
  file_extension: ".gguf"
  
  advantages:
    - Single file contains model + metadata
    - CPU and GPU inference support
    - Wide tool compatibility (Ollama, LM Studio, llama.cpp)
    - Multiple quantization levels in same format
    - Efficient memory mapping
    - Active development and community
  
  disadvantages:
    - Slightly slower than native CUDA formats
    - Conversion required from original weights
    
  best_for:
    - Local inference on consumer hardware
    - CPU-only deployments
    - Mixed CPU/GPU inference
    - Cross-platform compatibility
```

### GGUF Quantization Levels

```python
# GGUF quantization types and their characteristics
GGUF_QUANTS = {
    # Name: (bits_per_weight, quality, speed, size_ratio)
    "F32": (32, "100%", "1.0x", "100%"),
    "F16": (16, "~100%", "1.5x", "50%"),
    "BF16": (16, "~100%", "1.5x", "50%"),
    "Q8_0": (8, "~99%", "2.0x", "25%"),
    "Q6_K": (6.5, "~98%", "2.5x", "20%"),
    "Q5_K_M": (5.5, "~97%", "2.8x", "17%"),
    "Q5_K_S": (5.5, "~96%", "2.8x", "17%"),
    "Q4_K_M": (4.5, "~95%", "3.2x", "14%"),
    "Q4_K_S": (4.5, "~94%", "3.2x", "14%"),
    "Q4_0": (4, "~93%", "3.5x", "12.5%"),
    "Q3_K_M": (3.4, "~90%", "3.8x", "11%"),
    "Q3_K_S": (3.4, "~88%", "4.0x", "10%"),
    "Q2_K": (2.5, "~80%", "4.5x", "8%"),
    "IQ4_XS": (4.25, "~94%", "3.3x", "13%"),
    "IQ3_XS": (3.3, "~89%", "3.9x", "10%"),
    "IQ2_XXS": (2.1, "~75%", "5.0x", "6.5%"),
}

def recommend_quantization(
    available_vram_gb: float,
    model_params_billions: float,
    quality_priority: str = "balanced"
) -> str:
    """Recommend a quantization level based on constraints."""
    
    # Estimate base model size at FP16
    base_size_gb = model_params_billions * 2
    
    # Calculate what fits
    for quant, (bits, quality, speed, size_ratio) in GGUF_QUANTS.items():
        size_pct = float(size_ratio.rstrip('%')) / 100
        estimated_size = base_size_gb * size_pct * 1.2  # 20% overhead
        
        if estimated_size <= available_vram_gb:
            if quality_priority == "quality" and "Q4" in quant:
                continue  # Skip low quality for quality priority
            return quant
    
    return "Model too large for available VRAM"

# Example
print(recommend_quantization(24, 70))  # "Q4_K_M" for 70B on RTX 4090
print(recommend_quantization(12, 7))   # "Q8_0" for 7B on RTX 4070
```

### GGUF File Naming Convention

```
Model naming: {model}-{size}-{variant}-{quantization}.gguf

Examples:
├── llama-3.2-3b-instruct-q4_k_m.gguf      # 3B instruct, Q4_K_M
├── llama-3.1-70b-instruct-q4_k_m.gguf     # 70B instruct, Q4_K_M
├── mistral-7b-instruct-v0.3-q5_k_m.gguf   # Mistral 7B v0.3
├── qwen2.5-14b-instruct-q6_k.gguf         # Qwen 14B, higher quality
└── phi-4-14b-instruct-iq4_xs.gguf         # Phi-4 with iQuant
```

## GPTQ (Generative Pre-trained Transformer Quantization)

GPU-optimized quantization format using post-training quantization.

### Key Features
```yaml
gptq:
  name: "GPT Quantization"
  file_extensions: [".safetensors", ".pt"]
  
  advantages:
    - Highly optimized for CUDA GPUs
    - Better perplexity than naive quantization
    - Supports 4-bit and 8-bit
    - Fast inference on NVIDIA GPUs
  
  disadvantages:
    - GPU-only (no CPU support)
    - Requires CUDA
    - Larger calibration dataset needed
    - Less flexible than GGUF
    
  best_for:
    - GPU-only deployments
    - Maximum GPU inference speed
    - NVIDIA hardware
```

### GPTQ Usage with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load GPTQ quantized model
model_id = "TheBloke/Llama-2-7B-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True
)

# Generate text
inputs = tokenizer("The future of AI is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### GPTQ Quantization Process

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# Load original model
model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Prepare calibration data
calibration_data = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    # Add more diverse examples...
]

# Configure GPTQ
gptq_config = GPTQConfig(
    bits=4,
    group_size=128,
    dataset=calibration_data,
    tokenizer=tokenizer
)

# Quantize
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=gptq_config,
    device_map="auto"
)

# Save
quantized_model.save_pretrained("./llama-2-7b-gptq-4bit")
```

## AWQ (Activation-aware Weight Quantization)

Advanced quantization that considers activation patterns.

### Key Features
```yaml
awq:
  name: "Activation-aware Weight Quantization"
  file_extensions: [".safetensors", ".pt"]
  
  advantages:
    - Better quality than GPTQ at same bit-width
    - Activation-aware preserves important weights
    - Fast CUDA kernels
    - Good for instruction-tuned models
  
  disadvantages:
    - GPU-only
    - Newer, less tooling support
    - Requires calibration data
    
  best_for:
    - Quality-critical 4-bit deployments
    - Chat/instruction models
    - When GPTQ quality isn't sufficient
```

### AWQ Usage

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Load AWQ quantized model
model_path = "TheBloke/Llama-2-7B-AWQ"

model = AutoAWQForCausalLM.from_quantized(
    model_path,
    fuse_layers=True,  # Enable kernel fusion
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Generate
prompt = "Explain quantum computing in simple terms:"
tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**tokens, max_new_tokens=200)
print(tokenizer.decode(output[0]))
```

## Safetensors

A safe and efficient tensor serialization format.

### Key Features
```yaml
safetensors:
  name: "Safetensors"
  file_extension: ".safetensors"
  developed_by: "Hugging Face"
  
  advantages:
    - Security (no arbitrary code execution)
    - Fast loading (memory mapping)
    - Cross-framework compatibility
    - Industry standard
  
  purpose: "Base format for storing model weights"
  
  note: "Not a quantization format - used WITH other formats"
```

### Working with Safetensors

```python
from safetensors import safe_open
from safetensors.torch import save_file, load_file
import torch

# Load tensors
tensors = {}
with safe_open("model.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

# Save tensors safely
save_file(tensors, "model_copy.safetensors")

# Quick load all tensors
all_tensors = load_file("model.safetensors")
```

## Format Comparison

### Quality vs Size vs Speed

```
                    Quality Retention at 4-bit
                    ┌────────────────────────────────┐
                    │                                │
              100%  │ ▓▓▓▓ FP16 (baseline)          │
                    │                                │
               95%  │ ████ AWQ                      │
                    │ ████ GPTQ                     │
                    │ ████ GGUF Q4_K_M              │
                    │                                │
               90%  │ ████ GGUF Q4_0                │
                    │                                │
               85%  │ ████ GGUF Q3_K_M              │
                    │                                │
               80%  │ ████ GGUF Q2_K                │
                    └────────────────────────────────┘
```

### Decision Matrix

| Criterion | GGUF | GPTQ | AWQ |
|-----------|------|------|-----|
| CPU Support | ✅ Yes | ❌ No | ❌ No |
| GPU Support | ✅ Yes | ✅ Yes | ✅ Yes |
| Quality (4-bit) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Inference Speed | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Tool Support | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Ease of Use | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

### When to Use Each Format

```python
def recommend_format(
    hardware: str,
    use_case: str,
    quality_requirement: str
) -> str:
    """Recommend model format based on requirements."""
    
    if hardware == "cpu_only":
        return "GGUF - Only format with good CPU support"
    
    if hardware == "mixed_cpu_gpu":
        return "GGUF - Supports CPU/GPU offloading"
    
    if hardware == "gpu_only":
        if quality_requirement == "maximum":
            return "AWQ - Best 4-bit quality"
        elif use_case == "maximum_speed":
            return "GPTQ or AWQ with fused kernels"
        else:
            return "GGUF or GPTQ - Both work well"
    
    return "GGUF - Most versatile default"

# Recommendations
print(recommend_format("cpu_only", "inference", "balanced"))
# "GGUF - Only format with good CPU support"

print(recommend_format("gpu_only", "chat_bot", "maximum"))
# "AWQ - Best 4-bit quality"
```

## Converting Between Formats

### PyTorch to GGUF

```bash
# Using llama.cpp conversion script
python convert_hf_to_gguf.py \
    ./original-model \
    --outfile ./model.gguf \
    --outtype f16

# Then quantize
./llama-quantize ./model.gguf ./model-q4_k_m.gguf Q4_K_M
```

### Conversion with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Load original model
model_id = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Configure quantization
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False
)

# Create quantized model
quantized_model = AutoGPTQForCausalLM.from_pretrained(
    model_id,
    quantize_config
)

# Prepare calibration data
examples = [tokenizer(text, return_tensors="pt") for text in calibration_texts]

# Quantize
quantized_model.quantize(examples)

# Save
quantized_model.save_quantized("./llama-2-7b-gptq")
```

## Finding Pre-Quantized Models

### Hugging Face Sources

```yaml
popular_quantizers:
  TheBloke:
    url: "huggingface.co/TheBloke"
    formats: ["GGUF", "GPTQ", "AWQ"]
    models: "2000+"
    naming: "{model}-{size}B-{format}"
    
  bartowski:
    url: "huggingface.co/bartowski"
    formats: ["GGUF"]
    specialty: "iQuant formats, latest models"
    
  Qwen:
    url: "huggingface.co/Qwen"
    formats: ["GPTQ", "AWQ"]
    models: "Official Qwen quantizations"
```

### Model Repositories

```python
# Search for quantized models
from huggingface_hub import HfApi

api = HfApi()

# Find GGUF models
gguf_models = api.list_models(
    filter="gguf",
    sort="downloads",
    direction=-1,
    limit=20
)

for model in gguf_models:
    print(f"{model.id}: {model.downloads:,} downloads")
```

## Summary

| Format | Best For | Tool Support |
|--------|----------|--------------|
| **GGUF** | Local/CPU inference, Ollama, LM Studio | Excellent |
| **GPTQ** | GPU production, HuggingFace ecosystem | Good |
| **AWQ** | Quality-critical GPU deployment | Growing |
| **Safetensors** | Secure model storage (base format) | Excellent |

## Next Steps

1. **Quantization Deep Dive** - Understand the trade-offs
2. **Ollama Setup** - Start using GGUF models
3. **vLLM Deployment** - Production GPU inference
