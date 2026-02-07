# 🧪 Lab: Token Counter & Context Analyzer

## Objective

Build practical skills in counting tokens, estimating costs, and managing context effectively. You'll work with real examples to understand tokenization in practice.

## Prerequisites

- Completed the Tokenization and Context Windows lessons
- Basic Python knowledge
- Optional: `tiktoken` library installed (`pip install tiktoken`)

## Lab Overview

In this lab, you'll:
1. Count tokens manually and verify with tools
2. Analyze how different content affects token counts
3. Practice context budgeting
4. Implement a simple context manager

---

## Exercise 1: Token Counting Basics

### Task
Estimate token counts, then verify with actual tokenizer.

### Step 1: Make Your Estimates

Before running any code, estimate the token count for each:

| Text | Your Estimate | Actual |
|------|---------------|--------|
| "Hello" | | |
| "Hello, world!" | | |
| "The quick brown fox jumps over the lazy dog." | | |
| "def hello(): return 'world'" | | |
| "Hello" (with leading space: " Hello") | | |

### Step 2: Verify with Tokenizer

Use this code to check your estimates:

```python
# Token counting function
import tiktoken

def count_tokens(text, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    return len(tokens)

def show_tokens(text, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    print(f"Text: '{text}'")
    print(f"Total tokens: {len(tokens)}")
    print("Token breakdown:")
    for token in tokens:
        print(f"  {token} → '{enc.decode([token])}'")
```

### Step 3: Test with Your Own Text

Try these interesting cases:

```python
# Numbers
show_tokens("12345")
show_tokens("1000000000")
show_tokens("1e9")  # Scientific notation - fewer tokens!

# Code vs natural language
show_tokens("calculate the sum")
show_tokens("calculate_the_sum")

# Emojis
show_tokens("🚀")
show_tokens("🧑‍💻")  # Compound emoji - more tokens!
```

---

## Exercise 2: Content Type Analysis

### Task
Compare token efficiency across different content types.

### Test Each Content Type

```python
content_samples = {
    "english_prose": """
        The weather was beautiful that morning. Birds were singing 
        in the trees, and a gentle breeze carried the scent of 
        blooming flowers across the garden.
    """,
    
    "technical_writing": """
        The API endpoint accepts POST requests with JSON payloads.
        Authentication is handled via Bearer tokens in the header.
        Rate limiting is set to 100 requests per minute.
    """,
    
    "python_code": '''
def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
    ''',
    
    "json_data": '''
{
    "user": {
        "id": 12345,
        "name": "John Doe",
        "email": "john@example.com",
        "active": true
    }
}
    ''',
    
    "markdown": """
# Introduction

This is a **bold** statement with *italic* emphasis.

## Features
- Feature one
- Feature two
- Feature three
    """
}

# Analyze each
for name, content in content_samples.items():
    tokens = count_tokens(content)
    words = len(content.split())
    chars = len(content)
    
    print(f"{name}:")
    print(f"  Characters: {chars}")
    print(f"  Words: {words}")
    print(f"  Tokens: {tokens}")
    print(f"  Chars/Token: {chars/tokens:.1f}")
    print(f"  Words/Token: {words/tokens:.2f}")
    print()
```

### Questions to Answer

1. Which content type uses the most tokens per word?
2. Which is most token-efficient?
3. How does code compare to prose?

---

## Exercise 3: Cost Estimation

### Task
Build a cost calculator for API calls.

### Implement the Calculator

```python
# 2025 Pricing (approximate - check current rates)
PRICING = {
    "gpt-5.2": {"input": 5.00, "output": 15.00},
    "gpt-4-turbo": {"input": 2.50, "output": 7.50},
    "claude-4.5-sonnet": {"input": 3.00, "output": 15.00},
    "gemini-3-pro": {"input": 1.25, "output": 5.00},
}  # Per million tokens

def estimate_cost(
    input_text: str,
    expected_output_tokens: int,
    model: str = "gpt-4-turbo"
) -> dict:
    """Estimate the cost of an API call."""
    
    input_tokens = count_tokens(input_text)
    
    pricing = PRICING[model]
    
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (expected_output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": expected_output_tokens,
        "total_tokens": input_tokens + expected_output_tokens,
        "input_cost": f"${input_cost:.4f}",
        "output_cost": f"${output_cost:.4f}",
        "total_cost": f"${total_cost:.4f}",
    }
```

### Test Scenarios

```python
# Scenario 1: Simple Q&A
simple_prompt = "What is the capital of France?"
print("Simple Q&A:", estimate_cost(simple_prompt, 50, "gpt-4-turbo"))

# Scenario 2: Document summarization
long_document = "..." * 10000  # Simulate 10K word document
print("Document Summary:", estimate_cost(long_document, 500, "gpt-4-turbo"))

# Scenario 3: Code generation
code_prompt = "Write a Python class for a binary search tree with insert, delete, and search methods."
print("Code Generation:", estimate_cost(code_prompt, 800, "gpt-4-turbo"))
```

### Budget Exercise

You have a $100 monthly budget. Calculate:
1. How many simple Q&A calls can you make?
2. How many document summarizations?
3. What mix optimizes your use case?

---

## Exercise 4: Context Window Planning

### Task
Plan a context budget for a RAG application.

### The Scenario

You're building a customer support chatbot that:
- Has a system prompt defining its role
- Retrieves relevant FAQ documents
- Maintains conversation history
- Needs room for detailed responses

### Create Your Budget

```python
def plan_context_budget(
    context_window: int,
    system_prompt: str,
    max_response: int = 2000
) -> dict:
    """Plan how to allocate the context window."""
    
    # Calculate system prompt usage
    system_tokens = count_tokens(system_prompt)
    
    # Reserve for response
    reserved = system_tokens + max_response
    
    # Available for dynamic content
    available = context_window - reserved
    
    # Suggested allocation
    return {
        "total_window": context_window,
        "system_prompt": system_tokens,
        "max_response": max_response,
        "available": available,
        "suggested_allocation": {
            "documents": int(available * 0.6),
            "history": int(available * 0.3),
            "current_query": int(available * 0.1),
        }
    }

# Your system prompt
system_prompt = """
You are a helpful customer support assistant for TechCorp.
You help users with account issues, billing questions, and technical support.
Always be polite, professional, and concise.
If you don't know the answer, say so and offer to escalate.
"""

# Plan for GPT-4 Turbo (128K context)
budget = plan_context_budget(128000, system_prompt)
print("Context Budget:")
for key, value in budget.items():
    print(f"  {key}: {value}")
```

### Questions

1. How many FAQ documents (avg 500 tokens each) can you include?
2. How many conversation turns (avg 200 tokens each) can you keep?
3. What happens if a single document is 50,000 tokens?

---

## Exercise 5: Build a Simple Context Manager

### Task
Implement a basic context manager that respects token limits.

### Your Implementation

```python
class SimpleContextManager:
    """A basic context manager that respects token limits."""
    
    def __init__(self, max_tokens: int = 4000, model: str = "gpt-4"):
        self.max_tokens = max_tokens
        self.model = model
        self.enc = tiktoken.encoding_for_model(model)
        
        self.system_prompt = ""
        self.messages = []
    
    def _count(self, text: str) -> int:
        return len(self.enc.encode(text))
    
    def _total_tokens(self) -> int:
        total = self._count(self.system_prompt)
        for msg in self.messages:
            total += self._count(msg["content"])
        return total
    
    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt
        self._trim_if_needed()
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._trim_if_needed()
    
    def _trim_if_needed(self):
        """Remove oldest messages until within limit."""
        while self._total_tokens() > self.max_tokens and len(self.messages) > 1:
            removed = self.messages.pop(0)
            print(f"  [Trimmed: {removed['role']} message ({self._count(removed['content'])} tokens)]")
    
    def get_messages(self) -> list:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.messages)
        return messages
    
    def status(self):
        total = self._total_tokens()
        print(f"Context: {total}/{self.max_tokens} tokens ({total/self.max_tokens*100:.1f}%)")
        print(f"Messages: {len(self.messages)}")
```

### Test Your Implementation

```python
# Create manager with small limit for testing
manager = SimpleContextManager(max_tokens=500)

# Set system prompt
manager.set_system_prompt("You are a helpful assistant.")
manager.status()

# Add messages and watch trimming
messages = [
    ("user", "Hello, can you help me understand Python?"),
    ("assistant", "Of course! Python is a versatile programming language known for its readability and extensive libraries. What specific aspect would you like to learn about?"),
    ("user", "Tell me about list comprehensions."),
    ("assistant", "List comprehensions provide a concise way to create lists. The syntax is [expression for item in iterable if condition]. For example, squares = [x**2 for x in range(10)] creates a list of squares."),
    ("user", "Can you show me more complex examples?"),
    ("assistant", "Sure! Here's a nested list comprehension: matrix = [[i*j for j in range(3)] for i in range(3)]. This creates a 3x3 multiplication table."),
]

for role, content in messages:
    print(f"\nAdding {role} message...")
    manager.add_message(role, content)
    manager.status()

# Check final context
print("\n\nFinal messages in context:")
for msg in manager.get_messages():
    preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
    print(f"  [{msg['role']}]: {preview}")
```

---

## Challenge: Advanced Context Manager

### Extend your implementation to support:

1. **Priority levels** - Keep important messages longer
2. **Summarization** - Compress old messages instead of dropping
3. **Document injection** - Add/remove documents dynamically

```python
# Starter code for the challenge
class AdvancedContextManager:
    """Your advanced implementation here."""
    
    def __init__(self, max_tokens: int = 8000):
        # Your code here
        pass
    
    def add_message(self, role: str, content: str, priority: int = 1):
        # priority: 1 (low) to 5 (high, never drop)
        pass
    
    def add_document(self, doc: str, relevance: float = 0.5):
        # relevance: 0.0 to 1.0
        pass
    
    def compress_old_messages(self):
        # Summarize instead of dropping
        pass
```

---

## Reflection Questions

1. How close were your token estimates to actual counts?
2. What surprised you about token counts for different content?
3. How would you optimize a system with a $50/month budget?
4. What's the trade-off between keeping more history vs more documents?
5. When would you choose summarization over sliding window?

---

## Key Learnings

After completing this lab, you should be able to:

- [ ] Estimate token counts reasonably accurately
- [ ] Calculate API costs for different scenarios
- [ ] Plan context budgets for applications
- [ ] Implement basic context management
- [ ] Understand trade-offs in context strategies

---

## What's Next?

Congratulations on completing the LLM Fundamentals module! 🎉

You now have a solid foundation in:
- How LLMs work
- Prompt engineering techniques
- LLM parameters
- Token management

Next, explore the **RAG (Retrieval-Augmented Generation)** module to learn how to build LLM applications that use external knowledge.
