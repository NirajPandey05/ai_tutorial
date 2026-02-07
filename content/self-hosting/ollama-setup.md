# Ollama: The Easiest Way to Run Local LLMs

Ollama is the most user-friendly tool for running LLMs locally. It handles model management, provides an OpenAI-compatible API, and works on Mac, Windows, and Linux.

## Why Ollama?

```yaml
ollama_benefits:
  ease_of_use:
    - One-line model downloads
    - No Python/CUDA setup required
    - Automatic GPU detection
    - Simple CLI and API
    
  model_management:
    - Curated model library
    - Automatic updates
    - Easy model switching
    - Custom model creation
    
  api_compatibility:
    - OpenAI-compatible REST API
    - Drop-in replacement for many apps
    - Streaming support
    - Function calling support
    
  performance:
    - Optimized inference (llama.cpp)
    - Automatic GPU/CPU selection
    - Memory-efficient loading
    - Fast cold starts
```

## Installation

### Windows

```powershell
# Download and install from ollama.com
# Or use winget:
winget install Ollama.Ollama

# Verify installation
ollama --version
```

### macOS

```bash
# Download from ollama.com or use brew:
brew install ollama

# Start the service
ollama serve
```

### Linux

```bash
# One-line install
curl -fsSL https://ollama.com/install.sh | sh

# Or manual installation
wget https://ollama.com/download/ollama-linux-amd64
chmod +x ollama-linux-amd64
sudo mv ollama-linux-amd64 /usr/local/bin/ollama

# Start the service
ollama serve
```

### Docker

```bash
# CPU only
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# With NVIDIA GPU
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# Run a model
docker exec -it ollama ollama run llama3.2
```

## Basic Usage

### Running Models

```bash
# Download and run a model
ollama run llama3.2

# Run specific model size
ollama run llama3.2:3b
ollama run llama3.1:70b

# Run with custom system prompt
ollama run llama3.2 "You are a helpful coding assistant"

# List available models
ollama list

# Pull model without running
ollama pull mistral

# Remove a model
ollama rm mistral

# Show model info
ollama show llama3.2
```

### Popular Models

```yaml
recommended_models:
  general_purpose:
    - name: "llama3.2"
      sizes: ["1b", "3b"]
      notes: "Latest Meta model, excellent quality"
      
    - name: "llama3.1"
      sizes: ["8b", "70b", "405b"]
      notes: "Larger Llama models for advanced tasks"
      
    - name: "mistral"
      sizes: ["7b"]
      notes: "Great 7B model, fast and capable"
      
    - name: "qwen2.5"
      sizes: ["0.5b", "1.5b", "3b", "7b", "14b", "32b", "72b"]
      notes: "Strong multilingual, good at coding"
      
  coding:
    - name: "codellama"
      sizes: ["7b", "13b", "34b"]
      notes: "Code-specialized Llama"
      
    - name: "deepseek-coder-v2"
      sizes: ["16b", "236b"]
      notes: "Excellent code model, MoE architecture"
      
    - name: "qwen2.5-coder"
      sizes: ["1.5b", "7b", "32b"]
      notes: "State-of-the-art open code model"
      
  reasoning:
    - name: "phi4"
      sizes: ["14b"]
      notes: "Microsoft's reasoning model"
      
    - name: "gemma2"
      sizes: ["2b", "9b", "27b"]
      notes: "Google's open model"
      
  vision:
    - name: "llava"
      sizes: ["7b", "13b", "34b"]
      notes: "Vision-language model"
      
    - name: "llama3.2-vision"
      sizes: ["11b", "90b"]
      notes: "Meta's multimodal Llama"
```

## API Usage

### REST API

```python
import requests
import json

OLLAMA_BASE_URL = "http://localhost:11434"

# Generate completion
def generate(prompt: str, model: str = "llama3.2") -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

# Chat completion
def chat(messages: list, model: str = "llama3.2") -> str:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False
        }
    )
    return response.json()["message"]["content"]

# Streaming chat
def chat_stream(messages: list, model: str = "llama3.2"):
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": True
        },
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            if "message" in chunk:
                yield chunk["message"]["content"]

# Example usage
result = generate("What is the capital of France?")
print(result)

messages = [
    {"role": "user", "content": "Write a haiku about programming"}
]
result = chat(messages)
print(result)

# Streaming
for chunk in chat_stream(messages):
    print(chunk, end="", flush=True)
```

### OpenAI-Compatible API

```python
from openai import OpenAI

# Point to Ollama's OpenAI-compatible endpoint
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Required but not validated
)

# Chat completion (same as OpenAI!)
response = client.chat.completions.create(
    model="llama3.2",
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
    model="llama3.2",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Embeddings

```python
import requests

def get_embeddings(text: str, model: str = "nomic-embed-text") -> list:
    """Generate embeddings using Ollama."""
    
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": model,
            "prompt": text
        }
    )
    return response.json()["embedding"]

# Example
embedding = get_embeddings("Hello, world!")
print(f"Embedding dimension: {len(embedding)}")

# Using OpenAI client
embeddings_response = client.embeddings.create(
    model="nomic-embed-text",
    input="The quick brown fox"
)
print(embeddings_response.data[0].embedding[:5])
```

## Model Configuration

### Modelfile

Create custom models with a Modelfile:

```dockerfile
# Modelfile for a custom assistant
FROM llama3.2

# Set the temperature
PARAMETER temperature 0.7

# Set the system prompt
SYSTEM """
You are a senior software engineer with expertise in Python, JavaScript, and cloud architecture.
When asked about code:
1. First explain the concept
2. Then provide clean, well-commented code
3. Suggest best practices and potential improvements
Always be concise but thorough.
"""

# Set context window
PARAMETER num_ctx 8192

# Set stop tokens
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
```

Create and use the custom model:

```bash
# Create the model
ollama create code-assistant -f Modelfile

# Run it
ollama run code-assistant

# List your models
ollama list
```

### Advanced Modelfile Options

```dockerfile
# Full Modelfile example
FROM llama3.1:8b

# System prompt
SYSTEM "You are a helpful AI assistant specialized in data science."

# Parameters
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER num_predict 512
PARAMETER repeat_penalty 1.1
PARAMETER seed 42

# Template (for custom prompt formats)
TEMPLATE """
{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>
"""

# License
LICENSE "MIT"
```

### Importing Custom Models

```bash
# Import a GGUF file
ollama create my-model -f- <<EOF
FROM ./path/to/model.gguf
SYSTEM "Custom system prompt here"
EOF

# Import from Hugging Face (safetensors)
ollama create my-hf-model -f- <<EOF
FROM hf.co/username/model-name
EOF
```

## Performance Tuning

### GPU Configuration

```bash
# Force CPU only
OLLAMA_GPU_DRIVER=cpu ollama serve

# Set specific GPU layers (partial offload)
OLLAMA_NUM_GPU=35 ollama run llama3.1:70b

# Multiple GPUs
# Ollama automatically uses all available GPUs

# Check GPU usage
nvidia-smi -l 1  # Updates every second
```

### Memory Management

```bash
# Set maximum memory for models
OLLAMA_MAX_LOADED_MODELS=2 ollama serve

# Keep model in memory longer (default 5m)
OLLAMA_KEEP_ALIVE=30m ollama serve

# Unload all models
curl http://localhost:11434/api/generate -d '{"model": "llama3.2", "keep_alive": 0}'
```

### Context Window

```python
# Request with custom context
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2",
        "prompt": "Long document here...",
        "options": {
            "num_ctx": 8192,  # Increase context window
            "num_predict": 1024,  # Max tokens to generate
        }
    }
)
```

## Practical Examples

### Simple Chatbot

```python
import requests
import json

class OllamaChat:
    def __init__(self, model: str = "llama3.2", system_prompt: str = None):
        self.model = model
        self.base_url = "http://localhost:11434"
        self.messages = []
        
        if system_prompt:
            self.messages.append({
                "role": "system",
                "content": system_prompt
            })
    
    def chat(self, user_message: str) -> str:
        self.messages.append({
            "role": "user",
            "content": user_message
        })
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": self.messages,
                "stream": False
            }
        )
        
        assistant_message = response.json()["message"]["content"]
        self.messages.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def clear_history(self):
        system = [m for m in self.messages if m["role"] == "system"]
        self.messages = system

# Usage
bot = OllamaChat(
    model="llama3.2",
    system_prompt="You are a helpful assistant. Be concise."
)

print(bot.chat("What is Python?"))
print(bot.chat("How do I install it?"))  # Has context from previous message
```

### Code Assistant

```python
def code_review(code: str, language: str = "python") -> str:
    """Review code using local LLM."""
    
    prompt = f"""Review the following {language} code and provide:
1. A brief summary of what it does
2. Any bugs or issues
3. Suggestions for improvement
4. A rating from 1-10

Code:
```{language}
{code}
```"""
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5-coder:7b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Lower for more focused analysis
                "num_predict": 1000
            }
        }
    )
    
    return response.json()["response"]

# Example
code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

review = code_review(code)
print(review)
```

### RAG with Ollama

```python
import requests
import numpy as np

class SimpleRAG:
    def __init__(self, model: str = "llama3.2", embed_model: str = "nomic-embed-text"):
        self.model = model
        self.embed_model = embed_model
        self.documents = []
        self.embeddings = []
    
    def add_document(self, text: str):
        """Add a document to the knowledge base."""
        embedding = self._get_embedding(text)
        self.documents.append(text)
        self.embeddings.append(embedding)
    
    def _get_embedding(self, text: str) -> list:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": self.embed_model, "prompt": text}
        )
        return response.json()["embedding"]
    
    def _cosine_similarity(self, a: list, b: list) -> float:
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def query(self, question: str, top_k: int = 3) -> str:
        """Query the RAG system."""
        
        # Get question embedding
        q_embedding = self._get_embedding(question)
        
        # Find most similar documents
        similarities = [
            self._cosine_similarity(q_embedding, doc_emb)
            for doc_emb in self.embeddings
        ]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        context = "\n\n".join([self.documents[i] for i in top_indices])
        
        # Generate answer
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False}
        )
        
        return response.json()["response"]

# Usage
rag = SimpleRAG()
rag.add_document("Python was created by Guido van Rossum in 1991.")
rag.add_document("Python is known for its simple, readable syntax.")
rag.add_document("Python is widely used in data science and AI.")

answer = rag.query("Who created Python?")
print(answer)
```

## Troubleshooting

### Common Issues

```yaml
troubleshooting:
  model_not_loading:
    symptoms: "Model fails to load or crashes"
    solutions:
      - "Check available RAM/VRAM: nvidia-smi or free -h"
      - "Try a smaller model or higher quantization"
      - "Reduce num_ctx parameter"
      
  slow_inference:
    symptoms: "Generation is very slow"
    solutions:
      - "Check GPU is being used: watch nvidia-smi"
      - "Try a quantized model (e.g., llama3.2:3b-q4_0)"
      - "Reduce context length"
      
  connection_refused:
    symptoms: "Cannot connect to Ollama API"
    solutions:
      - "Ensure ollama serve is running"
      - "Check port 11434 is not blocked"
      - "For Docker, ensure port mapping is correct"
      
  out_of_memory:
    symptoms: "CUDA out of memory error"
    solutions:
      - "Use smaller model"
      - "Reduce num_ctx"
      - "Set OLLAMA_NUM_GPU to offload some layers to CPU"
```

### Useful Commands

```bash
# Check Ollama status
curl http://localhost:11434/api/version

# List loaded models
curl http://localhost:11434/api/ps

# Get model info
ollama show llama3.2 --modelfile

# Check logs (Linux/Mac)
journalctl -u ollama -f

# Check GPU memory
watch -n 1 nvidia-smi
```

## Next Steps

Now that you have Ollama running:
1. **llama.cpp** - For more control over inference
2. **vLLM** - For production deployments
3. **Model Selection** - Choose the best model for your use case
