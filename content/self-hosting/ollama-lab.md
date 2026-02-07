# Lab: Set Up Ollama and Run Local Models

In this hands-on lab, you'll install Ollama, run your first local LLM, and build a simple application that uses it.

## Prerequisites

- **Windows**: Windows 10/11 with WSL2 or native install
- **macOS**: macOS 11 Big Sur or later
- **Linux**: Any modern distribution
- **Hardware**: 8GB+ RAM, 10GB+ free disk space
- **Optional**: NVIDIA GPU with 6GB+ VRAM for faster inference

## Part 1: Installation

### Windows Installation

```powershell
# Option 1: Download installer from ollama.com
# Visit https://ollama.com/download and run the installer

# Option 2: Using winget
winget install Ollama.Ollama

# Verify installation
ollama --version
```

### macOS Installation

```bash
# Option 1: Download from ollama.com
# Visit https://ollama.com/download

# Option 2: Using Homebrew
brew install ollama

# Start Ollama service
ollama serve &

# Verify
ollama --version
```

### Linux Installation

```bash
# One-line install
curl -fsSL https://ollama.com/install.sh | sh

# Start the service
sudo systemctl start ollama

# Enable on boot
sudo systemctl enable ollama

# Verify
ollama --version
```

### ✅ Checkpoint 1

Run this command to verify Ollama is working:

```bash
curl http://localhost:11434/api/version
```

Expected output:
```json
{"version":"0.x.x"}
```

## Part 2: Running Your First Model

### Download and Run Llama 3.2

```bash
# Pull and run in one command
ollama run llama3.2:3b

# You'll see the model download progress, then a prompt
>>> 
```

### Interactive Chat

Try these prompts:

```
>>> What is Python?

>>> Write a haiku about programming

>>> Explain recursion like I'm five years old

>>> /bye  # Exit the chat
```

### Exploring Other Models

```bash
# List downloaded models
ollama list

# Pull a different model
ollama pull mistral:7b

# Pull a coding model
ollama pull qwen2.5-coder:7b

# Run with a different model
ollama run qwen2.5-coder:7b

# Try a code prompt
>>> Write a Python function to calculate Fibonacci numbers

>>> /bye
```

### ✅ Checkpoint 2

Verify you have multiple models:

```bash
ollama list
```

Expected output (models you've downloaded):
```
NAME                    ID              SIZE      MODIFIED
llama3.2:3b            abc123...       2.0 GB    2 minutes ago
qwen2.5-coder:7b       def456...       4.7 GB    1 minute ago
```

## Part 3: Using the API

### Basic API Call

Create a file `test_api.py`:

```python
import requests

# Simple generation
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2:3b",
        "prompt": "What is the capital of France?",
        "stream": False
    }
)

print(response.json()["response"])
```

Run it:
```bash
python test_api.py
```

### Chat API

Create `test_chat.py`:

```python
import requests

def chat(messages, model="llama3.2:3b"):
    """Send a chat request to Ollama."""
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False
        }
    )
    return response.json()["message"]["content"]

# Example conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What are three benefits of exercise?"}
]

response = chat(messages)
print("Assistant:", response)

# Continue the conversation
messages.append({"role": "assistant", "content": response})
messages.append({"role": "user", "content": "Which one is most important for mental health?"})

response = chat(messages)
print("\nAssistant:", response)
```

### Streaming Responses

Create `test_streaming.py`:

```python
import requests
import json

def stream_chat(messages, model="llama3.2:3b"):
    """Stream a chat response from Ollama."""
    response = requests.post(
        "http://localhost:11434/api/chat",
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
                print(chunk["message"]["content"], end="", flush=True)
    print()  # New line at end

# Test streaming
messages = [
    {"role": "user", "content": "Write a short poem about coding"}
]

print("Streaming response:")
stream_chat(messages)
```

### ✅ Checkpoint 3

Run the streaming test and verify you see text appearing gradually:

```bash
python test_streaming.py
```

## Part 4: Using the OpenAI-Compatible API

Ollama provides an OpenAI-compatible API, so you can use the OpenAI Python library:

```bash
pip install openai
```

Create `test_openai_compat.py`:

```python
from openai import OpenAI

# Point to Ollama's OpenAI-compatible endpoint
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Required but not validated
)

# Chat completion (same API as OpenAI!)
response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function to reverse a string"}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)

# List available models
models = client.models.list()
print("\nAvailable models:")
for model in models.data:
    print(f"  - {model.id}")
```

## Part 5: Building a Simple Chatbot

Create a complete chatbot application `chatbot.py`:

```python
import requests
import json
from datetime import datetime

class LocalChatbot:
    def __init__(self, model: str = "llama3.2:3b", system_prompt: str = None):
        self.model = model
        self.base_url = "http://localhost:11434"
        self.messages = []
        
        if system_prompt:
            self.messages.append({
                "role": "system",
                "content": system_prompt
            })
    
    def chat(self, user_input: str, stream: bool = True) -> str:
        """Send a message and get a response."""
        self.messages.append({
            "role": "user",
            "content": user_input
        })
        
        if stream:
            return self._stream_response()
        else:
            return self._get_response()
    
    def _get_response(self) -> str:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": self.messages,
                "stream": False
            }
        )
        
        content = response.json()["message"]["content"]
        self.messages.append({"role": "assistant", "content": content})
        return content
    
    def _stream_response(self) -> str:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": self.messages,
                "stream": True
            },
            stream=True
        )
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if "message" in chunk:
                    content = chunk["message"]["content"]
                    print(content, end="", flush=True)
                    full_response += content
        
        print()  # New line
        self.messages.append({"role": "assistant", "content": full_response})
        return full_response
    
    def clear_history(self):
        """Clear conversation history (keep system prompt)."""
        system_msgs = [m for m in self.messages if m["role"] == "system"]
        self.messages = system_msgs
    
    def save_conversation(self, filename: str = None):
        """Save conversation to a JSON file."""
        if not filename:
            filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                "model": self.model,
                "messages": self.messages,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"Conversation saved to {filename}")


def main():
    print("=" * 50)
    print("  Local LLM Chatbot (powered by Ollama)")
    print("=" * 50)
    print("\nCommands:")
    print("  /clear  - Clear conversation history")
    print("  /save   - Save conversation to file")
    print("  /model  - Change model")
    print("  /quit   - Exit\n")
    
    # Initialize chatbot
    bot = LocalChatbot(
        model="llama3.2:3b",
        system_prompt="You are a helpful, friendly AI assistant. Be concise but thorough."
    )
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == "/quit":
                print("Goodbye!")
                break
            elif user_input.lower() == "/clear":
                bot.clear_history()
                print("Conversation cleared.")
                continue
            elif user_input.lower() == "/save":
                bot.save_conversation()
                continue
            elif user_input.lower().startswith("/model"):
                parts = user_input.split()
                if len(parts) > 1:
                    bot.model = parts[1]
                    print(f"Model changed to: {bot.model}")
                else:
                    print(f"Current model: {bot.model}")
                continue
            
            # Get response
            print("\nAssistant: ", end="")
            bot.chat(user_input, stream=True)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    main()
```

### Run Your Chatbot

```bash
python chatbot.py
```

Try having a conversation:
```
You: What's the difference between Python lists and tuples?

You: Give me a code example

You: /save

You: /quit
```

## Part 6: Performance Testing

Create `benchmark.py` to measure your setup's performance:

```python
import requests
import time
import statistics

def benchmark_model(
    model: str = "llama3.2:3b",
    prompt: str = "Write a paragraph about artificial intelligence.",
    num_runs: int = 5
) -> dict:
    """Benchmark model performance."""
    
    times = []
    tokens = []
    
    print(f"Benchmarking {model}...")
    
    for i in range(num_runs):
        start = time.time()
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 100  # Generate ~100 tokens
                }
            }
        )
        
        elapsed = time.time() - start
        result = response.json()
        
        times.append(elapsed)
        
        # Extract token count from response
        if "eval_count" in result:
            tokens.append(result["eval_count"])
            tps = result["eval_count"] / elapsed
            print(f"  Run {i+1}: {elapsed:.2f}s, {result['eval_count']} tokens, {tps:.1f} t/s")
        else:
            print(f"  Run {i+1}: {elapsed:.2f}s")
    
    results = {
        "model": model,
        "avg_time": statistics.mean(times),
        "std_time": statistics.stdev(times) if len(times) > 1 else 0,
        "min_time": min(times),
        "max_time": max(times),
    }
    
    if tokens:
        total_tokens = sum(tokens)
        total_time = sum(times)
        results["avg_tokens_per_second"] = total_tokens / total_time
    
    return results


def main():
    print("=" * 50)
    print("  Ollama Performance Benchmark")
    print("=" * 50)
    
    # Get list of models
    response = requests.get("http://localhost:11434/api/tags")
    models = [m["name"] for m in response.json()["models"]]
    
    print(f"\nFound {len(models)} models: {', '.join(models)}")
    
    results = []
    for model in models[:3]:  # Benchmark first 3 models
        result = benchmark_model(model)
        results.append(result)
        print()
    
    # Summary
    print("\n" + "=" * 50)
    print("  Results Summary")
    print("=" * 50)
    
    for r in results:
        print(f"\n{r['model']}:")
        print(f"  Average time: {r['avg_time']:.2f}s (±{r['std_time']:.2f})")
        if "avg_tokens_per_second" in r:
            print(f"  Tokens/second: {r['avg_tokens_per_second']:.1f}")


if __name__ == "__main__":
    main()
```

### ✅ Checkpoint 4

Run the benchmark:
```bash
python benchmark.py
```

Record your results:
- Model: ____________
- Tokens/second: ____________
- Hardware: ____________

## Part 7: Creating a Custom Model

Create a custom model with a specific personality:

```bash
# Create a Modelfile
cat > Modelfile << 'EOF'
FROM llama3.2:3b

SYSTEM """
You are CodeBuddy, a friendly programming tutor. You:
- Explain concepts clearly with examples
- Encourage learners and celebrate their progress
- Break down complex topics into simple steps
- Use analogies to explain difficult concepts
- Always provide runnable code examples

When asked about code:
1. Explain what the code does
2. Show the code with comments
3. Explain any tricky parts
4. Suggest improvements or alternatives
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

# Create the model
ollama create codebuddy -f Modelfile

# Test it
ollama run codebuddy
```

Test prompts:
```
>>> Explain what a for loop is

>>> Show me how to read a file in Python

>>> What's the difference between == and is in Python?

>>> /bye
```

## Exercises

### Exercise 1: Multi-Model Comparison

Modify the benchmark script to compare the same prompt across different models and output a comparison table.

### Exercise 2: Build a Code Review Bot

Create a script that:
1. Accepts a Python file as input
2. Sends the code to the coding model
3. Gets a code review with suggestions

### Exercise 3: Local RAG System

Build a simple RAG system:
1. Pull `nomic-embed-text` for embeddings
2. Create embeddings for some text documents
3. Search for relevant context
4. Generate answers using context

```python
# Hint: Get embeddings like this
response = requests.post(
    "http://localhost:11434/api/embeddings",
    json={
        "model": "nomic-embed-text",
        "prompt": "Your text here"
    }
)
embedding = response.json()["embedding"]
```

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| "connection refused" | Run `ollama serve` or restart Ollama service |
| Model downloads slowly | Check internet connection, try different model |
| Out of memory | Try smaller model (3b vs 7b) or quantized version |
| Slow generation | Check GPU usage with `nvidia-smi`, use GPU model |
| API not responding | Check `curl http://localhost:11434/api/version` |

### Checking GPU Usage

```bash
# Linux/Windows
nvidia-smi

# Watch continuously
watch -n 1 nvidia-smi
```

## Summary

In this lab, you learned to:
- ✅ Install and configure Ollama
- ✅ Download and run local LLMs
- ✅ Use the REST API for chat and generation
- ✅ Use the OpenAI-compatible API
- ✅ Build a functional chatbot
- ✅ Benchmark model performance
- ✅ Create custom models with Modelfiles

## Next Steps

- Try larger models if you have the hardware
- Explore the [Ollama model library](https://ollama.com/library)
- Build more complex applications with local models
- Connect Ollama to existing tools using the OpenAI-compatible API
