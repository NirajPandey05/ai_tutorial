# llama.cpp: The Foundation of Local LLM Inference

llama.cpp is the foundational project that enabled practical local LLM inference. It powers Ollama, LM Studio, and many other tools. Understanding llama.cpp gives you maximum control over inference.

## What is llama.cpp?

```yaml
llama_cpp:
  description: "C/C++ implementation of LLM inference"
  created_by: "Georgi Gerganov"
  key_innovation: "4-bit quantization on consumer hardware"
  
  features:
    - Pure C/C++ with no dependencies
    - CPU and GPU inference
    - Apple Silicon optimization
    - CUDA, ROCm, Vulkan support
    - GGUF model format
    - Highly optimized inference
    
  used_by:
    - Ollama (uses llama.cpp under the hood)
    - LM Studio
    - text-generation-webui
    - Many other projects
```

## Installation

### Building from Source

```bash
# Clone the repository
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build (CPU only)
make -j

# Build with CUDA support
make -j GGML_CUDA=1

# Build with Metal support (Mac)
make -j GGML_METAL=1

# Build with Vulkan support (cross-platform GPU)
make -j GGML_VULKAN=1

# Build with all optimizations
make -j GGML_CUDA=1 GGML_CUDA_F16=true
```

### Using CMake

```bash
# More control with CMake
mkdir build
cd build

# CPU build
cmake ..
cmake --build . --config Release

# CUDA build
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release

# With specific CUDA architecture
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86"
cmake --build . --config Release
```

### Python Bindings

```bash
# Install llama-cpp-python
pip install llama-cpp-python

# With CUDA support
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# With Metal support (Mac)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

## Basic Usage

### Command Line

```bash
# Basic inference
./llama-cli -m ./models/llama-3.2-3b-instruct-q4_k_m.gguf \
    -p "Explain quantum computing in simple terms:"

# With parameters
./llama-cli -m ./models/llama-3.2-3b-instruct-q4_k_m.gguf \
    -p "Write a haiku about programming:" \
    --temp 0.8 \
    --top-p 0.9 \
    --top-k 40 \
    -n 100 \
    --seed 42

# Interactive chat mode
./llama-cli -m ./models/llama-3.2-3b-instruct-q4_k_m.gguf \
    -i \
    --color \
    -c 4096 \
    --temp 0.7 \
    -r "User:" \
    --in-prefix " " \
    -p "You are a helpful AI assistant.\n\nUser: Hello!\nAssistant:"

# Benchmark mode
./llama-bench -m ./models/llama-3.2-3b-instruct-q4_k_m.gguf
```

### Key Parameters

```yaml
llama_cli_parameters:
  model_loading:
    -m, --model: "Path to GGUF model file"
    -c, --ctx-size: "Context size (default 512)"
    --n-gpu-layers: "Layers to offload to GPU"
    --mlock: "Lock model in RAM (prevent swapping)"
    --no-mmap: "Disable memory mapping"
    
  generation:
    -n, --n-predict: "Number of tokens to generate"
    --temp: "Temperature (0.0-2.0)"
    --top-p: "Top-p sampling (0.0-1.0)"
    --top-k: "Top-k sampling"
    --repeat-penalty: "Repetition penalty"
    --seed: "Random seed (-1 for random)"
    
  prompt:
    -p, --prompt: "Initial prompt"
    -f, --file: "Read prompt from file"
    --in-prefix: "Prefix for user input"
    --in-suffix: "Suffix for user input"
    
  output:
    --color: "Colorized output"
    --log-disable: "Disable logging"
    -i: "Interactive mode"
```

## Python Usage

### Basic Generation

```python
from llama_cpp import Llama

# Load model
llm = Llama(
    model_path="./models/llama-3.2-3b-instruct-q4_k_m.gguf",
    n_ctx=4096,           # Context window
    n_threads=8,          # CPU threads
    n_gpu_layers=35,      # Layers on GPU (-1 for all)
    verbose=False
)

# Generate text
output = llm(
    "Explain the theory of relativity in simple terms:",
    max_tokens=256,
    temperature=0.7,
    top_p=0.9,
    stop=["User:", "\n\n"],
    echo=False
)

print(output["choices"][0]["text"])
```

### Chat Interface

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./models/llama-3.2-3b-instruct-q4_k_m.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
    chat_format="llama-3"  # Use appropriate chat format
)

# Chat completion
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"}
]

response = llm.create_chat_completion(
    messages=messages,
    temperature=0.7,
    max_tokens=256
)

print(response["choices"][0]["message"]["content"])
```

### Streaming

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./models/llama-3.2-3b-instruct-q4_k_m.gguf",
    n_ctx=4096,
    n_gpu_layers=-1
)

# Streaming generation
prompt = "Write a short story about a robot:"

for token in llm(
    prompt,
    max_tokens=512,
    temperature=0.8,
    stream=True
):
    print(token["choices"][0]["text"], end="", flush=True)
print()

# Streaming chat
messages = [{"role": "user", "content": "Tell me a joke"}]

for chunk in llm.create_chat_completion(
    messages=messages,
    stream=True
):
    delta = chunk["choices"][0]["delta"]
    if "content" in delta:
        print(delta["content"], end="", flush=True)
```

### Embeddings

```python
from llama_cpp import Llama

# Load model with embedding enabled
llm = Llama(
    model_path="./models/nomic-embed-text-v1.5-Q8_0.gguf",
    embedding=True,
    n_ctx=8192
)

# Generate embeddings
texts = [
    "The quick brown fox",
    "A fast red fox",
    "Machine learning is fascinating"
]

embeddings = [llm.embed(text) for text in texts]

# Calculate similarity
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim_01 = cosine_similarity(embeddings[0], embeddings[1])
sim_02 = cosine_similarity(embeddings[0], embeddings[2])

print(f"'fox' sentences similarity: {sim_01:.3f}")
print(f"Different topics similarity: {sim_02:.3f}")
```

## Model Conversion

### Converting Hugging Face Models

```bash
# Convert HF model to GGUF
python convert_hf_to_gguf.py \
    /path/to/hf-model \
    --outfile model-f16.gguf \
    --outtype f16

# Available output types
# f32  - 32-bit float (largest)
# f16  - 16-bit float
# bf16 - Brain float 16
# q8_0 - 8-bit quantized

# For specific architectures
python convert_hf_to_gguf.py /path/to/model \
    --outtype f16 \
    --vocab-type bpe
```

### Quantizing Models

```bash
# Available quantization types
./llama-quantize --help

# Common quantizations
./llama-quantize model-f16.gguf model-q8_0.gguf Q8_0
./llama-quantize model-f16.gguf model-q6_k.gguf Q6_K
./llama-quantize model-f16.gguf model-q5_k_m.gguf Q5_K_M
./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
./llama-quantize model-f16.gguf model-q3_k_m.gguf Q3_K_M

# With importance matrix for better quality
./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M \
    --imatrix importance.dat

# Generate importance matrix first
./llama-imatrix -m model-f16.gguf -f calibration.txt -o importance.dat
```

## Server Mode

### Running the Server

```bash
# Start server with model
./llama-server \
    -m ./models/llama-3.2-3b-instruct-q4_k_m.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -c 4096 \
    --n-gpu-layers 35

# With multiple slots (concurrent requests)
./llama-server \
    -m ./models/llama-3.2-3b-instruct-q4_k_m.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -c 4096 \
    --n-gpu-layers 35 \
    --parallel 4 \
    --cont-batching

# Enable embeddings endpoint
./llama-server \
    -m ./models/nomic-embed-text.gguf \
    --host 0.0.0.0 \
    --port 8081 \
    --embedding
```

### Server API

```python
import requests

BASE_URL = "http://localhost:8080"

# Completion endpoint
response = requests.post(
    f"{BASE_URL}/completion",
    json={
        "prompt": "The meaning of life is",
        "n_predict": 128,
        "temperature": 0.7,
        "stop": ["\n\n", "User:"]
    }
)
print(response.json()["content"])

# Chat endpoint (OpenAI-compatible)
response = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    json={
        "model": "local-model",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 100
    }
)
print(response.json()["choices"][0]["message"]["content"])

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Model info
response = requests.get(f"{BASE_URL}/props")
print(response.json())
```

## Performance Optimization

### GPU Offloading

```python
from llama_cpp import Llama

# Full GPU offload (if model fits)
llm = Llama(
    model_path="model.gguf",
    n_gpu_layers=-1  # All layers on GPU
)

# Partial offload (for larger models)
llm = Llama(
    model_path="model.gguf",
    n_gpu_layers=20  # First 20 layers on GPU
)

# Check memory usage
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                       capture_output=True, text=True)
print(f"GPU Memory Used: {result.stdout.strip()} MB")
```

### Batch Processing

```python
from llama_cpp import Llama

llm = Llama(
    model_path="model.gguf",
    n_ctx=4096,
    n_batch=512,  # Larger batch for throughput
    n_gpu_layers=-1
)

# Process multiple prompts
prompts = [
    "Summarize machine learning:",
    "Explain neural networks:",
    "What is deep learning:",
]

# Note: llama-cpp-python processes one at a time
# For true batching, use the server with parallel slots
results = []
for prompt in prompts:
    output = llm(prompt, max_tokens=100)
    results.append(output["choices"][0]["text"])
```

### Context Caching

```python
from llama_cpp import Llama

llm = Llama(
    model_path="model.gguf",
    n_ctx=4096,
    n_gpu_layers=-1
)

# The model caches the KV state
# Subsequent prompts that share a prefix are faster

# First call - full processing
system_prompt = "You are a helpful assistant. "
response1 = llm(system_prompt + "What is Python?", max_tokens=100)

# Second call with same prefix - partially cached
response2 = llm(system_prompt + "What is JavaScript?", max_tokens=100)
```

## Benchmarking

### Using llama-bench

```bash
# Basic benchmark
./llama-bench -m model.gguf

# Custom benchmark
./llama-bench -m model.gguf \
    -p 512 \          # Prompt tokens
    -n 128 \          # Generated tokens
    -r 5 \            # Repetitions
    --n-gpu-layers 35

# Compare quantizations
./llama-bench \
    -m model-f16.gguf \
    -m model-q8_0.gguf \
    -m model-q4_k_m.gguf
```

### Python Benchmark

```python
import time
from llama_cpp import Llama

def benchmark_model(model_path: str, n_gpu_layers: int = -1):
    """Benchmark model performance."""
    
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=n_gpu_layers,
        verbose=False
    )
    
    prompt = "Write a detailed explanation of how computers work: "
    
    # Warm up
    llm(prompt, max_tokens=10)
    
    # Benchmark
    start = time.time()
    output = llm(prompt, max_tokens=256)
    elapsed = time.time() - start
    
    tokens_generated = output["usage"]["completion_tokens"]
    tokens_per_second = tokens_generated / elapsed
    
    return {
        "model": model_path,
        "tokens": tokens_generated,
        "time": elapsed,
        "tokens_per_second": tokens_per_second
    }

# Compare models
models = [
    "llama-3.2-3b-q8_0.gguf",
    "llama-3.2-3b-q4_k_m.gguf",
]

for model in models:
    result = benchmark_model(model)
    print(f"{model}: {result['tokens_per_second']:.1f} t/s")
```

## Advanced Features

### Grammar-Constrained Generation

```python
from llama_cpp import Llama, LlamaGrammar

llm = Llama(model_path="model.gguf", n_gpu_layers=-1)

# JSON grammar
json_grammar = LlamaGrammar.from_string('''
root ::= object
object ::= "{" ws pair ("," ws pair)* ws "}"
pair ::= string ":" ws value
string ::= "\"" [a-zA-Z_]+ "\""
value ::= string | number | "true" | "false" | "null"
number ::= [0-9]+
ws ::= " "*
''')

output = llm(
    "Generate a JSON object with name and age fields:",
    max_tokens=50,
    grammar=json_grammar
)

print(output["choices"][0]["text"])
# Output will be valid JSON like: {"name": "John", "age": 30}
```

### Function Calling

```python
from llama_cpp import Llama
import json

llm = Llama(
    model_path="model.gguf",
    n_gpu_layers=-1,
    chat_format="chatml-function-calling"
)

# Define functions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

messages = [
    {"role": "user", "content": "What's the weather in Paris?"}
]

response = llm.create_chat_completion(
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

# Check if model wants to call a function
if response["choices"][0]["message"].get("tool_calls"):
    tool_call = response["choices"][0]["message"]["tool_calls"][0]
    print(f"Function: {tool_call['function']['name']}")
    print(f"Args: {tool_call['function']['arguments']}")
```

## Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    llama.cpp Cheat Sheet                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Quick Start:                                               │
│    ./llama-cli -m model.gguf -p "prompt" -n 100            │
│                                                             │
│  Server Mode:                                               │
│    ./llama-server -m model.gguf --port 8080                │
│                                                             │
│  Python:                                                    │
│    pip install llama-cpp-python                            │
│    llm = Llama(model_path="model.gguf")                    │
│    llm("prompt", max_tokens=100)                           │
│                                                             │
│  Key Parameters:                                            │
│    -c: context size     --n-gpu-layers: GPU offload        │
│    -n: tokens to gen    --temp: temperature                │
│                                                             │
│  Convert Models:                                            │
│    python convert_hf_to_gguf.py model/ -o model.gguf       │
│    ./llama-quantize model.gguf model-q4.gguf Q4_K_M        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Next Steps

1. **vLLM** - For high-throughput production
2. **LM Studio** - GUI alternative
3. **Model Selection** - Choose the right model
