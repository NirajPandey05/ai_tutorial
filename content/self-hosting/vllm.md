# vLLM: High-Throughput Production Inference

vLLM is the go-to solution for production LLM deployments. It offers dramatically higher throughput than other inference engines through innovative memory management and batching techniques.

## Why vLLM?

```yaml
vllm_advantages:
  throughput:
    - "2-24x higher than HuggingFace Transformers"
    - "PagedAttention eliminates memory waste"
    - "Continuous batching for concurrent requests"
    
  compatibility:
    - "OpenAI-compatible API"
    - "Supports most popular models"
    - "HuggingFace model hub integration"
    
  production_ready:
    - "Built for serving at scale"
    - "Tensor parallelism for large models"
    - "Quantization support (AWQ, GPTQ, FP8)"
    - "Speculative decoding"
    
  ecosystem:
    - "Ray integration for distributed serving"
    - "Prometheus metrics"
    - "Docker and Kubernetes support"
```

## Installation

### Basic Installation

```bash
# Basic install (requires CUDA)
pip install vllm

# With specific CUDA version
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121

# From source (for latest features)
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

### Docker

```bash
# Official vLLM Docker image
docker run --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.2-3B-Instruct

# With environment variables
docker run --gpus all \
    -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 2
```

## Basic Usage

### Offline Inference

```python
from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(
    model="meta-llama/Llama-3.2-3B-Instruct",
    dtype="auto",  # auto, float16, bfloat16
    gpu_memory_utilization=0.9  # Use 90% of GPU memory
)

# Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
    stop=["<|eot_id|>"]
)

# Generate
prompts = [
    "Explain machine learning in simple terms:",
    "Write a haiku about programming:",
    "What is the capital of France?"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt[:50]}...")
    print(f"Generated: {output.outputs[0].text}")
    print("---")
```

### Chat Interface

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct")

# Format as chat
def format_chat(messages: list) -> str:
    """Format messages for Llama-3 chat."""
    formatted = "<|begin_of_text|>"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return formatted

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"}
]

prompt = format_chat(messages)
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
output = llm.generate([prompt], sampling_params)[0]
print(output.outputs[0].text)
```

## Server Mode

### Starting the Server

```bash
# Basic server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --host 0.0.0.0 \
    --port 8000

# With more options
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --dtype auto \
    --api-key your-secret-key

# For large models with tensor parallelism
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8000
```

### Server Configuration

```yaml
vllm_server_options:
  model:
    --model: "HuggingFace model name or path"
    --tokenizer: "Custom tokenizer (if different)"
    --revision: "Model revision/branch"
    --dtype: "float16, bfloat16, auto"
    
  memory:
    --gpu-memory-utilization: "GPU memory fraction (0-1)"
    --max-model-len: "Maximum context length"
    --kv-cache-dtype: "KV cache precision"
    
  parallelism:
    --tensor-parallel-size: "Number of GPUs for tensor parallelism"
    --pipeline-parallel-size: "Number of GPUs for pipeline parallelism"
    
  serving:
    --host: "Server host"
    --port: "Server port"
    --api-key: "API key for authentication"
    --max-num-seqs: "Maximum concurrent sequences"
    
  quantization:
    --quantization: "awq, gptq, squeezellm, fp8"
    --load-format: "Model loading format"
```

### Using the API

```python
from openai import OpenAI

# Point to vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secret-key"  # If configured
)

# Chat completion
response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## PagedAttention

The key innovation in vLLM is PagedAttention, which manages KV cache efficiently.

```
Traditional Approach (Wasteful):
┌────────────────────────────────────────────────────┐
│ Request 1: [KV Cache ████████████         ]        │
│ Request 2: [KV Cache ████████                 ]    │
│ Request 3: [KV Cache ██████████████████   ]        │
│            ^ Pre-allocated, mostly wasted memory   │
└────────────────────────────────────────────────────┘

PagedAttention (Efficient):
┌────────────────────────────────────────────────────┐
│ Memory Pages: [1][2][3][4][5][6][7][8][9][10]...  │
│                                                    │
│ Request 1: Uses pages [1,2,3]                      │
│ Request 2: Uses pages [4,5]                        │
│ Request 3: Uses pages [6,7,8,9]                    │
│            ^ Dynamic allocation, minimal waste     │
└────────────────────────────────────────────────────┘
```

### Benefits

```python
# PagedAttention enables:

benefits = {
    "memory_efficiency": {
        "description": "Near-zero memory waste",
        "improvement": "2-4x more concurrent requests"
    },
    "copy_on_write": {
        "description": "Shared prompts share memory",
        "use_case": "Same system prompt for many users"
    },
    "prefix_caching": {
        "description": "Cache common prefixes",
        "use_case": "RAG with repeated context"
    }
}
```

## Quantization

### AWQ Models

```python
from vllm import LLM, SamplingParams

# Load AWQ quantized model
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
    dtype="auto"
)

# Use normally
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
output = llm.generate(["Hello, world!"], sampling_params)
```

### GPTQ Models

```python
from vllm import LLM

# Load GPTQ quantized model
llm = LLM(
    model="TheBloke/Llama-2-7B-GPTQ",
    quantization="gptq",
    dtype="float16"
)
```

### FP8 Quantization

```python
from vllm import LLM

# FP8 for H100/Ada GPUs
llm = LLM(
    model="meta-llama/Llama-3.2-3B-Instruct",
    quantization="fp8",
    dtype="auto"
)
```

## Tensor Parallelism

For models that don't fit on a single GPU:

```python
from vllm import LLM

# 70B model across 4 GPUs
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,  # Split across 4 GPUs
    dtype="auto"
)

# Server mode
# python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Llama-3.1-70B-Instruct \
#     --tensor-parallel-size 4
```

### GPU Requirements for Large Models

| Model | FP16 VRAM | Tensor Parallel Config |
|-------|-----------|----------------------|
| 7B | 14GB | 1x RTX 4090 |
| 13B | 26GB | 1x A100 40GB or 2x RTX 4090 |
| 34B | 68GB | 2x A100 40GB or 4x RTX 4090 |
| 70B | 140GB | 2x A100 80GB or 4x A100 40GB |
| 405B | 810GB | 8x A100 80GB or 16x A100 40GB |

## Advanced Features

### Speculative Decoding

Use a smaller model to speed up generation:

```python
from vllm import LLM, SamplingParams

# Main model + draft model for speculation
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_model="meta-llama/Llama-3.2-1B-Instruct",
    num_speculative_tokens=5,
    tensor_parallel_size=4
)

# Can be 1.5-2x faster for certain workloads
```

### Prefix Caching

Cache common prefixes for efficiency:

```python
# Server mode
# python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Llama-3.2-3B-Instruct \
#     --enable-prefix-caching

# Useful when many requests share the same system prompt or context
```

### Structured Output

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct")

# JSON schema constraint
json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "city": {"type": "string"}
    },
    "required": ["name", "age"]
}

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=100,
    guided_decoding=GuidedDecodingParams(json=json_schema)
)

output = llm.generate(
    ["Generate a person's info:"],
    sampling_params
)[0]

print(output.outputs[0].text)  # Valid JSON guaranteed
```

## Production Deployment

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  vllm:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    ports:
      - "8000:8000"
    command: >
      --model meta-llama/Llama-3.2-3B-Instruct
      --host 0.0.0.0
      --port 8000
      --max-model-len 4096
      --gpu-memory-utilization 0.9
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### Kubernetes Deployment

```yaml
# vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        ports:
        - containerPort: 8000
        args:
        - "--model"
        - "meta-llama/Llama-3.2-3B-Instruct"
        - "--host"
        - "0.0.0.0"
        - "--port"
        - "8000"
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: cache
          mountPath: /root/.cache/huggingface
      volumes:
      - name: cache
        persistentVolumeClaim:
          claimName: huggingface-cache
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

### Load Balancing

```python
# Simple round-robin load balancer
from openai import OpenAI
import random

class VLLMLoadBalancer:
    def __init__(self, endpoints: list[str]):
        self.endpoints = endpoints
        self.clients = [
            OpenAI(base_url=f"{ep}/v1", api_key="none")
            for ep in endpoints
        ]
    
    def get_client(self) -> OpenAI:
        """Get a random client (simple load balancing)."""
        return random.choice(self.clients)
    
    def chat_completion(self, messages, **kwargs):
        """Route to available server."""
        client = self.get_client()
        return client.chat.completions.create(
            messages=messages,
            **kwargs
        )

# Usage
balancer = VLLMLoadBalancer([
    "http://gpu-server-1:8000",
    "http://gpu-server-2:8000",
    "http://gpu-server-3:8000"
])

response = balancer.chat_completion(
    messages=[{"role": "user", "content": "Hello!"}],
    model="meta-llama/Llama-3.2-3B-Instruct"
)
```

## Monitoring

### Prometheus Metrics

```bash
# Enable metrics
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --port 8000

# Metrics available at http://localhost:8000/metrics
```

### Key Metrics

```python
# Important vLLM metrics to monitor
metrics = {
    "vllm:num_requests_running": "Current concurrent requests",
    "vllm:num_requests_waiting": "Requests in queue",
    "vllm:gpu_cache_usage_perc": "KV cache utilization",
    "vllm:avg_generation_throughput_toks_per_s": "Token throughput",
    "vllm:avg_prompt_throughput_toks_per_s": "Prompt processing rate"
}
```

## Benchmarking

```python
import time
import asyncio
from openai import AsyncOpenAI

async def benchmark_vllm(
    num_requests: int = 100,
    max_concurrent: int = 10
):
    """Benchmark vLLM throughput."""
    
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="none"
    )
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def make_request():
        async with semaphore:
            start = time.time()
            response = await client.chat.completions.create(
                model="meta-llama/Llama-3.2-3B-Instruct",
                messages=[{"role": "user", "content": "Tell me a joke"}],
                max_tokens=100
            )
            elapsed = time.time() - start
            tokens = response.usage.completion_tokens
            return elapsed, tokens
    
    # Run benchmark
    start_time = time.time()
    results = await asyncio.gather(*[make_request() for _ in range(num_requests)])
    total_time = time.time() - start_time
    
    total_tokens = sum(r[1] for r in results)
    avg_latency = sum(r[0] for r in results) / len(results)
    
    print(f"Total requests: {num_requests}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Requests/second: {num_requests / total_time:.2f}")
    print(f"Tokens/second: {total_tokens / total_time:.2f}")
    print(f"Average latency: {avg_latency:.3f}s")

# Run benchmark
asyncio.run(benchmark_vllm(num_requests=100, max_concurrent=20))
```

## Summary

```
┌─────────────────────────────────────────────────────────────┐
│                      vLLM Cheat Sheet                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Quick Start:                                               │
│    pip install vllm                                         │
│    python -m vllm.entrypoints.openai.api_server \          │
│        --model meta-llama/Llama-3.2-3B-Instruct            │
│                                                             │
│  Key Features:                                              │
│    - PagedAttention for memory efficiency                   │
│    - OpenAI-compatible API                                  │
│    - Tensor parallelism for large models                    │
│    - Continuous batching                                    │
│                                                             │
│  When to Use:                                               │
│    ✅ High-throughput production serving                   │
│    ✅ Multiple concurrent users                            │
│    ✅ Maximum GPU utilization                              │
│    ❌ CPU-only inference (use llama.cpp)                   │
│    ❌ Single-user desktop (use Ollama)                     │
│                                                             │
│  Tensor Parallel Examples:                                  │
│    70B: --tensor-parallel-size 4                           │
│    405B: --tensor-parallel-size 8                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Next Steps

1. **Open Source Models** - Choose the right model
2. **Docker Deployment** - Containerized production
3. **Integration Patterns** - Connect to your app
