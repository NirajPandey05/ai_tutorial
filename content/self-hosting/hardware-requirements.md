# Hardware Requirements for Self-Hosting LLMs

Understanding hardware requirements is essential for successful self-hosting. This guide covers GPU, CPU, RAM, and storage needs for different model sizes and use cases.

## The Hardware Equation

LLM inference requirements depend on three main factors:

```
Memory Required = Model Parameters × Bytes per Parameter + Context Memory

Throughput = f(GPU Compute, Memory Bandwidth, Model Optimization)
```

## GPU Requirements

GPUs are the primary workhorse for LLM inference due to their parallel processing capabilities.

### GPU Memory (VRAM) Calculator

```python
def calculate_vram_requirements(
    parameters_billions: float,
    precision: str = "fp16",
    context_length: int = 4096,
    batch_size: int = 1
) -> dict:
    """
    Calculate VRAM requirements for loading an LLM.
    
    Args:
        parameters_billions: Model size in billions (e.g., 7 for 7B)
        precision: fp32, fp16, int8, int4
        context_length: Maximum context window
        batch_size: Concurrent requests
    """
    
    # Bytes per parameter by precision
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5
    }
    
    # Model weights
    model_memory_gb = (
        parameters_billions * 1e9 * bytes_per_param[precision]
    ) / (1024**3)
    
    # KV cache for attention (rough estimate)
    # ~2 bytes per token per layer for fp16
    num_layers = int(parameters_billions * 4)  # Rough estimate
    hidden_dim = int(parameters_billions * 512)  # Rough estimate
    kv_cache_gb = (
        2 * num_layers * context_length * hidden_dim * 2 * batch_size
    ) / (1024**3)
    
    # Overhead for activations (~20%)
    overhead_gb = (model_memory_gb + kv_cache_gb) * 0.2
    
    total_gb = model_memory_gb + kv_cache_gb + overhead_gb
    
    return {
        "model_weights_gb": round(model_memory_gb, 1),
        "kv_cache_gb": round(kv_cache_gb, 1),
        "overhead_gb": round(overhead_gb, 1),
        "total_vram_gb": round(total_gb, 1),
        "recommended_gpu": recommend_gpu(total_gb)
    }

def recommend_gpu(vram_needed: float) -> str:
    """Recommend a GPU based on VRAM requirements."""
    gpus = [
        (8, "RTX 3060 12GB / RTX 4060 Ti 16GB"),
        (12, "RTX 3080 12GB / RTX 4070 12GB"),
        (16, "RTX 4080 16GB / RTX A4000 16GB"),
        (24, "RTX 3090 24GB / RTX 4090 24GB / A5000 24GB"),
        (48, "A6000 48GB / 2x RTX 4090"),
        (80, "A100 80GB / H100 80GB"),
    ]
    
    for vram, gpu_name in gpus:
        if vram_needed <= vram:
            return gpu_name
    return "Multiple H100s or A100s required"

# Examples
for model_size in [7, 13, 34, 70]:
    for precision in ["fp16", "int8", "int4"]:
        req = calculate_vram_requirements(model_size, precision)
        print(f"{model_size}B @ {precision}: {req['total_vram_gb']}GB → {req['recommended_gpu']}")
```

### GPU VRAM Quick Reference

| Model Size | FP16 | INT8 | INT4 (GGUF Q4) | Recommended GPU |
|-----------|------|------|----------------|-----------------|
| 1B | 2 GB | 1.5 GB | 1 GB | Any modern GPU |
| 3B | 6 GB | 3.5 GB | 2.5 GB | RTX 3060 12GB |
| 7B | 14 GB | 8 GB | 5 GB | RTX 4070 12GB |
| 13B | 26 GB | 14 GB | 8 GB | RTX 4090 24GB |
| 34B | 68 GB | 35 GB | 20 GB | A6000 48GB |
| 70B | 140 GB | 70 GB | 40 GB | A100 80GB / 2x RTX 4090 |
| 405B | 810 GB | 405 GB | 230 GB | 8x A100 / 4x H100 |

### Consumer GPUs for Self-Hosting

```yaml
nvidia_consumer_gpus:
  entry_level:
    - name: "RTX 4060 Ti 16GB"
      vram: 16
      price: "$450"
      good_for: "7B models quantized, 3B models full precision"
      tokens_per_second: "~40 tps @ 7B Q4"
  
  mid_range:
    - name: "RTX 4070 Ti Super 16GB"
      vram: 16
      price: "$800"
      good_for: "7B models, fast inference"
      tokens_per_second: "~60 tps @ 7B Q4"
  
  high_end:
    - name: "RTX 4090 24GB"
      vram: 24
      price: "$1,600"
      good_for: "13B quantized, 7B full precision"
      tokens_per_second: "~80 tps @ 7B Q4"

nvidia_professional_gpus:
  workstation:
    - name: "RTX A6000 48GB"
      vram: 48
      price: "$4,500"
      good_for: "34B quantized, 13B full precision"
    
  datacenter:
    - name: "A100 80GB"
      vram: 80
      price: "$10,000-15,000"
      good_for: "70B quantized, large batch processing"
    
    - name: "H100 80GB"
      vram: 80
      price: "$25,000-40,000"
      good_for: "Maximum performance, 70B+ models"
```

### Multi-GPU Configurations

```python
# Tensor parallelism across GPUs
class MultiGPUConfig:
    """Configuration for multi-GPU inference."""
    
    def __init__(self, gpus: list[dict]):
        self.gpus = gpus
        self.total_vram = sum(g["vram"] for g in gpus)
    
    def can_run_model(self, model_vram_required: float) -> bool:
        """Check if model fits across all GPUs."""
        # Need ~10% overhead for tensor parallelism
        usable_vram = self.total_vram * 0.9
        return model_vram_required <= usable_vram
    
    def recommended_parallelism(self) -> str:
        """Recommend parallelism strategy."""
        if len(self.gpus) == 1:
            return "single_gpu"
        elif all(g["vram"] == self.gpus[0]["vram"] for g in self.gpus):
            return "tensor_parallel"  # Balanced GPUs
        else:
            return "pipeline_parallel"  # Mixed GPUs

# Example: 2x RTX 4090
config = MultiGPUConfig([
    {"name": "RTX 4090", "vram": 24},
    {"name": "RTX 4090", "vram": 24}
])
print(f"Total VRAM: {config.total_vram}GB")
print(f"Can run 70B Q4 (40GB): {config.can_run_model(40)}")
```

## CPU Inference

CPU inference is viable for smaller models or when GPU isn't available:

### CPU Requirements

```yaml
cpu_recommendations:
  minimum:
    cores: 4
    ram: 16GB
    good_for: "1-3B models with Q4 quantization"
    speed: "~5 tokens/second"
  
  recommended:
    cores: 8+
    ram: 32GB
    features: "AVX2, AVX-512"
    good_for: "7B models with Q4 quantization"
    speed: "~15 tokens/second"
  
  optimal:
    cores: 16+
    ram: 64GB+
    features: "AVX-512, high memory bandwidth"
    good_for: "13B models with Q4 quantization"
    speed: "~25 tokens/second"
```

### CPU vs GPU Comparison

```python
# Performance comparison for 7B Q4 model
BENCHMARKS = {
    "consumer_cpu": {
        "hardware": "Intel i7-13700K (16 cores)",
        "tokens_per_second": 12,
        "time_to_first_token_ms": 500,
        "cost": "$400"
    },
    "server_cpu": {
        "hardware": "AMD EPYC 7763 (64 cores)",
        "tokens_per_second": 35,
        "time_to_first_token_ms": 200,
        "cost": "$7,000"
    },
    "consumer_gpu": {
        "hardware": "RTX 4090 24GB",
        "tokens_per_second": 80,
        "time_to_first_token_ms": 50,
        "cost": "$1,600"
    },
    "datacenter_gpu": {
        "hardware": "A100 80GB",
        "tokens_per_second": 150,
        "time_to_first_token_ms": 30,
        "cost": "$15,000"
    }
}
```

## RAM Requirements

System RAM is needed even with GPU inference:

```python
def calculate_ram_requirements(
    model_size_gb: float,
    use_gpu: bool = True,
    mmap_loading: bool = True
) -> dict:
    """Calculate system RAM requirements."""
    
    if use_gpu:
        # Need RAM for model loading and CPU operations
        if mmap_loading:
            # Memory-mapped loading is efficient
            base_ram = model_size_gb * 0.5
        else:
            # Full model loaded to RAM first
            base_ram = model_size_gb * 1.2
        
        system_overhead = 4  # GB for OS and services
        application_ram = 2  # GB for your application
        
    else:
        # CPU inference: model in RAM
        base_ram = model_size_gb * 1.1  # Some overhead
        system_overhead = 4
        application_ram = 4  # More for CPU processing
    
    total_ram = base_ram + system_overhead + application_ram
    recommended_ram = max(16, int(total_ram * 1.5))  # 50% headroom
    
    return {
        "minimum_ram_gb": round(total_ram),
        "recommended_ram_gb": recommended_ram,
        "swap_recommended": recommended_ram * 2
    }

# Example: 7B model at Q4 (needs ~5GB)
print(calculate_ram_requirements(5, use_gpu=True))
# {'minimum_ram_gb': 9, 'recommended_ram_gb': 16, 'swap_recommended': 32}
```

### RAM Quick Reference

| Model Size | GPU Inference | CPU Inference | Recommended |
|-----------|---------------|---------------|-------------|
| 3B Q4 | 8 GB | 16 GB | 16 GB |
| 7B Q4 | 12 GB | 24 GB | 32 GB |
| 13B Q4 | 16 GB | 32 GB | 32 GB |
| 34B Q4 | 24 GB | 48 GB | 64 GB |
| 70B Q4 | 32 GB | 96 GB | 128 GB |

## Storage Requirements

### Model Storage

```yaml
storage_requirements:
  model_files:
    7B_Q4: "4-5 GB"
    7B_FP16: "14 GB"
    13B_Q4: "7-8 GB"
    13B_FP16: "26 GB"
    70B_Q4: "35-40 GB"
    70B_FP16: "140 GB"
  
  recommended_storage:
    type: "NVMe SSD"
    reason: "Fast model loading (30-60 seconds vs 2-5 minutes for HDD)"
    minimum: "100 GB"
    recommended: "500 GB (multiple models)"
    
  storage_layout:
    - path: "/models"
      purpose: "Model weights storage"
      size: "Variable based on models"
    - path: "/cache"  
      purpose: "KV cache overflow, temp files"
      size: "50 GB"
    - path: "/logs"
      purpose: "Inference logs, metrics"
      size: "20 GB"
```

### Loading Speed by Storage Type

| Storage Type | 7B Model Load Time | 70B Model Load Time |
|-------------|-------------------|---------------------|
| HDD | 2-3 minutes | 15-20 minutes |
| SATA SSD | 30-60 seconds | 5-8 minutes |
| NVMe SSD | 10-20 seconds | 2-3 minutes |
| NVMe (PCIe 4.0) | 5-10 seconds | 1-2 minutes |

## Complete Build Examples

### Budget Build (~$1,500)
```yaml
budget_build:
  purpose: "7B models for development/testing"
  
  components:
    cpu: "AMD Ryzen 5 7600 ($200)"
    ram: "32GB DDR5 ($100)"
    gpu: "RTX 4060 Ti 16GB ($450)"
    storage: "1TB NVMe SSD ($80)"
    motherboard: "B650 chipset ($150)"
    psu: "650W Gold ($80)"
    case: "Basic ATX ($60)"
  
  total: "~$1,120"
  
  capabilities:
    - "7B models at Q4-Q6 quantization"
    - "~40-50 tokens/second"
    - "Single user inference"
    - "Development and testing"
```

### Performance Build (~$3,500)
```yaml
performance_build:
  purpose: "13B models, small team usage"
  
  components:
    cpu: "AMD Ryzen 9 7950X ($500)"
    ram: "64GB DDR5 ($200)"
    gpu: "RTX 4090 24GB ($1,600)"
    storage: "2TB NVMe SSD ($150)"
    motherboard: "X670E chipset ($350)"
    psu: "1000W Platinum ($200)"
    case: "Good airflow ATX ($150)"
    cooling: "360mm AIO ($150)"
  
  total: "~$3,300"
  
  capabilities:
    - "13B models at Q4-Q5"
    - "7B models at FP16"
    - "~80 tokens/second @ 7B"
    - "5-10 concurrent users"
```

### Production Build (~$20,000)
```yaml
production_build:
  purpose: "70B models, production workloads"
  
  components:
    cpu: "AMD EPYC 7313 or Intel Xeon ($1,500)"
    ram: "256GB ECC DDR5 ($1,500)"
    gpu: "2x RTX 4090 or 1x A6000 ($3,200-4,500)"
    storage: "4TB NVMe RAID ($600)"
    motherboard: "Server/Workstation ($800)"
    psu: "1600W Titanium ($400)"
    case: "Server chassis ($500)"
    
  or_cloud:
    provider: "Lambda Labs / RunPod / Vast.ai"
    instance: "1x A100 80GB"
    cost: "$1-2/hour"
  
  capabilities:
    - "70B models at Q4"
    - "34B models at higher precision"
    - "~100+ tokens/second @ 7B"
    - "50+ concurrent users"
```

## Cloud GPU Pricing (2026)

| Provider | GPU | VRAM | Price/Hour |
|----------|-----|------|------------|
| Lambda Labs | A100 80GB | 80GB | $1.29 |
| Lambda Labs | H100 | 80GB | $2.49 |
| RunPod | A100 80GB | 80GB | $1.44 |
| RunPod | RTX 4090 | 24GB | $0.44 |
| Vast.ai | RTX 4090 | 24GB | $0.30-0.50 |
| AWS | p4d.24xlarge (8x A100) | 640GB | $32.77 |
| GCP | a2-highgpu-1g (1x A100) | 40GB | $3.67 |

## Summary Checklist

```markdown
## Hardware Checklist for Self-Hosting

### Determine Your Needs
- [ ] Target model size (7B, 13B, 70B?)
- [ ] Acceptable quantization level
- [ ] Expected concurrent users
- [ ] Latency requirements

### GPU Selection
- [ ] Calculate VRAM needed
- [ ] Consider multi-GPU if necessary
- [ ] Check CUDA compatibility

### System Requirements
- [ ] Sufficient RAM (2x model size minimum)
- [ ] Fast storage (NVMe recommended)
- [ ] Adequate cooling

### Budget
- [ ] Hardware purchase vs cloud rental
- [ ] Break-even calculation
- [ ] Maintenance costs
```

## Next Steps

Now that you understand hardware requirements:
1. **Model Formats** - Learn about GGUF, GPTQ, AWQ
2. **Quantization** - Understand precision trade-offs
3. **Ollama Setup** - Start running models locally
