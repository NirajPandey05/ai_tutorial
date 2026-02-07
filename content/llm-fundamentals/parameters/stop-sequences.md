# Stop Sequences and Output Control

## What Are Stop Sequences?

**Stop sequences** are specific strings that tell the model to stop generating when encountered. They give you precise control over where output ends.

```python
response = client.chat(
    messages=[{"role": "user", "content": "List 3 colors:"}],
    stop=["\n4.", "Four"]  # Stop before a 4th item
)
```

Without stop sequences, the model might continue:
```
1. Red
2. Blue  
3. Green
4. Yellow    ← We didn't want this
5. Purple    ← Or this
```

With `stop=["\n4."]`:
```
1. Red
2. Blue
3. Green     ← Stops here!
```

## How Stop Sequences Work

```
┌─────────────────────────────────────────────────────────────────┐
│                  Stop Sequence Processing                       │
│                                                                 │
│   Model generating: "The answer is 42.\n\nNext question..."    │
│                                         ▲                       │
│                                         │                       │
│                              Stop sequence: "\n\n"              │
│                                         │                       │
│                                         ▼                       │
│   Final output: "The answer is 42."                            │
│                                                                 │
│   Note: The stop sequence itself is NOT included in output     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Behavior

1. **Stop sequence is excluded** - It's not in the final output
2. **First match wins** - Generation stops at the first occurrence
3. **Multiple sequences allowed** - Usually up to 4
4. **Case sensitive** - "END" ≠ "end" ≠ "End"

## Common Use Cases

### 1. Structured Output Control

```python
# Stop at the end of a JSON object
response = client.chat(
    messages=[{
        "role": "user",
        "content": "Return user data as JSON: {name, email, age}"
    }],
    stop=["\n\n", "```"]  # Stop after JSON block
)
```

### 2. Limiting List Length

```python
# Get exactly 5 items
response = client.chat(
    messages=[{
        "role": "user",
        "content": "List 5 programming languages:\n1."
    }],
    stop=["\n6.", "6."]  # Stop before 6th item
)
```

### 3. Dialogue Systems

```python
# Stop when AI's turn ends
response = client.chat(
    messages=[{
        "role": "user",
        "content": """
        Format: 
        Human: message
        AI: response
        
        Human: Hello!
        AI:"""
    }],
    stop=["Human:", "\nHuman"]  # Stop at next human turn
)
```

### 4. Code Generation

```python
# Stop at end of function
response = client.chat(
    messages=[{
        "role": "user",
        "content": "Write a Python function to reverse a string."
    }],
    stop=["\n\ndef ", "\nclass ", "```"]  # Stop at next function/class
)
```

### 5. Structured Extraction

```python
# Extract just the answer
prompt = """
Question: What is the capital of France?

Let me think step by step...
[reasoning here]

ANSWER: [your answer here]
END

Question: What is 2 + 2?

Let me think step by step..."""

response = client.chat(
    messages=[{"role": "user", "content": prompt}],
    stop=["END", "\n\nQuestion:"]
)
```

## Provider-Specific Behavior

### OpenAI

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-5.2",
    messages=[{"role": "user", "content": "..."}],
    stop=["END", "\n\n", "---"]  # Up to 4 sequences
)
```

### Anthropic

```python
import anthropic
client = anthropic.Client()

response = client.messages.create(
    model="claude-4-5-sonnet-20250514",
    messages=[{"role": "user", "content": "..."}],
    stop_sequences=["END", "\n\n"]  # Note: different parameter name
)
```

### Google (Gemini)

```python
import google.generativeai as genai

model = genai.GenerativeModel('gemini-3-pro')
response = model.generate_content(
    "...",
    generation_config={"stop_sequences": ["END"]}
)
```

## Advanced Patterns

### Pattern 1: Template Completion

```python
template = """
Product Review Analysis:

Product: {product_name}
Rating: {rating}/5

Summary:
[SUMMARY]

Pros:
[PROS]

Cons:
[CONS]

ANALYSIS_END
"""

response = client.chat(
    messages=[{
        "role": "user",
        "content": f"Complete this template for iPhone 16:\n{template}"
    }],
    stop=["ANALYSIS_END"]  # Clean ending
)
```

### Pattern 2: Multi-Turn Simulation

```python
def simulate_conversation(topic: str, turns: int = 3):
    prompt = f"""Simulate a conversation about {topic}.
    
Expert: Let me explain {topic}...
Student: """
    
    conversation = []
    
    for i in range(turns):
        # Student asks
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            stop=["Expert:", "\nExpert"]
        )
        student_message = response.content.strip()
        prompt += student_message + "\nExpert: "
        
        # Expert responds
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            stop=["Student:", "\nStudent"]
        )
        expert_message = response.content.strip()
        prompt += expert_message + "\nStudent: "
        
        conversation.append({
            "student": student_message,
            "expert": expert_message
        })
    
    return conversation
```

### Pattern 3: Controlled JSON Generation

```python
def generate_json_field(prompt: str, field_name: str) -> str:
    """Generate a single JSON field value."""
    response = client.chat(
        messages=[{
            "role": "user",
            "content": f'{prompt}\n\nRespond with just the value:\n"{field_name}": "'
        }],
        stop=['"', '\n']  # Stop at end of string value
    )
    return response.content.strip()

# Build JSON field by field
product = {
    "name": generate_json_field("Create a sci-fi book title", "name"),
    "author": generate_json_field("Create a pen name", "author"),
    "genre": generate_json_field("What genre? sci-fi, fantasy, etc", "genre"),
}
```

### Pattern 4: Structured Reasoning

```python
prompt = """
Analyze this problem using the following format:

PROBLEM: {problem}

ANALYSIS:
- Key factors:
- Constraints:
- Assumptions:

SOLUTION:
[Your solution here]

CONFIDENCE: [low/medium/high]

END_ANALYSIS
"""

response = client.chat(
    messages=[{
        "role": "user",
        "content": prompt.format(problem="How to reduce API costs?")
    }],
    stop=["END_ANALYSIS"]
)
```

## Common Pitfalls

### ❌ Stop Sequence Inside Desired Output

```python
# Bad: "END" might appear in regular text
stop = ["END"]

# The model might stop mid-sentence:
# "In the end..."  ← Stops here!
```

### ✅ Use Distinctive Stop Sequences

```python
# Good: Unlikely to appear naturally
stop = ["###END###", "[STOP]", "---END---"]
```

---

### ❌ Case Sensitivity Issues

```python
# This WON'T stop at "end" or "End"
stop = ["END"]
```

### ✅ Include Variations

```python
# Cover common variations
stop = ["END", "End", "end"]

# Or use post-processing for case-insensitive
response = get_response(prompt, stop=["END"])
# Then check: if "end" in response.lower(): truncate()
```

---

### ❌ Forgetting Newline Variations

```python
# May miss: "Human:\n" vs "Human: " vs "\nHuman:"
stop = ["Human:"]
```

### ✅ Include Whitespace Variations

```python
stop = ["Human:", "\nHuman:", "Human:\n", "\nHuman"]
```

## Combining with Other Parameters

### Stop Sequences + max_tokens

```python
# Double safety: stop at delimiter OR max length
response = client.chat(
    messages=messages,
    max_tokens=500,           # Length limit
    stop=["---END---"]        # Content-based stop
)
# Whichever comes first wins
```

### Stop Sequences + Temperature

```python
# Creative generation with clean endings
response = client.chat(
    messages=[{"role": "user", "content": "Write a creative story opening."}],
    temperature=1.2,          # Creative
    max_tokens=200,           # Not too long
    stop=["\n\nChapter 2"]    # Stop at next chapter
)
```

## Best Practices

### 1. Design Distinctive Markers

```python
# Good markers:
MARKERS = [
    "###END###",
    "[RESPONSE_COMPLETE]",
    "---DONE---",
    "%%STOP%%"
]
```

### 2. Include in Prompt

```python
prompt = """
Answer the question, then write "###END###" when done.

Question: What is machine learning?

Answer:
"""

response = client.chat(
    messages=[{"role": "user", "content": prompt}],
    stop=["###END###"]
)
```

### 3. Validate Output

```python
def get_clean_response(prompt: str, stop: list) -> str:
    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        stop=stop
    )
    
    result = response.content.strip()
    
    # Double-check: remove any stop sequences that slipped through
    for seq in stop:
        if result.endswith(seq):
            result = result[:-len(seq)].strip()
    
    return result
```

### 4. Test Edge Cases

```python
def test_stop_sequences():
    test_cases = [
        "Short response",
        "Response with END in middle of sentence",
        "Response\n\nwith double newlines",
        "Response that might not hit stop at all"
    ]
    
    for case in test_cases:
        result = generate_with_stop(case, stop=["END"])
        assert "END" not in result, f"Stop sequence leaked: {result}"
```

## Key Takeaways

1. **Stop sequences give precise control** - End generation exactly where you want
2. **They're excluded from output** - You get clean results
3. **Be specific** - Use distinctive, unlikely-to-occur sequences
4. **Handle variations** - Consider case and whitespace
5. **Combine strategically** - Use with max_tokens for double safety

## Next Steps

Now let's explore additional parameters like **frequency_penalty**, **presence_penalty**, and **logprobs** in the **Parameter Explorer Lab**.
