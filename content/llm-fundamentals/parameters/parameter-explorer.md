# 🧪 Lab: Parameter Explorer

## Objective

Experiment with LLM parameters to understand how they affect model outputs. You'll explore temperature, max_tokens, top_p, and stop sequences through hands-on exercises.

## Prerequisites

- Completed the Temperature and Tokens lessons
- API key for at least one LLM provider
- Basic understanding of token concepts

## Lab Overview

In this lab, you'll:
1. Observe temperature's effect on creativity
2. Control output length with max_tokens
3. Experiment with top_p (nucleus sampling)
4. Use stop sequences for precise control

---

## Exercise 1: Temperature Comparison

### Task
Generate the same prompt at different temperatures and observe the differences.

### The Experiment

```python
prompt = "Write a creative name for a coffee shop."
temperatures = [0.0, 0.5, 0.7, 1.0, 1.5]

# Run this prompt at each temperature and compare results
```

### What to Observe
- How consistent are outputs at temp=0?
- At what temperature do you start seeing creative names?
- When does it become too random?

### Record Your Results

| Temperature | Response | Creativity Level |
|-------------|----------|------------------|
| 0.0 | | |
| 0.5 | | |
| 0.7 | | |
| 1.0 | | |
| 1.5 | | |

---

## Exercise 2: Temperature for Different Tasks

### Task
Find the optimal temperature for each task type.

### Task A: Code Generation

```python
prompt = "Write a Python function to check if a string is a palindrome."
# Test with temperatures: 0.0, 0.3, 0.7, 1.0
# Which produces the most reliable code?
```

### Task B: Creative Writing

```python
prompt = "Write the opening line of a mystery novel."
# Test with temperatures: 0.3, 0.7, 1.0, 1.3
# Which produces the most engaging opening?
```

### Task C: Factual Q&A

```python
prompt = "What is the chemical formula for water?"
# Test with temperatures: 0.0, 0.5, 1.0
# Does temperature affect factual accuracy?
```

### Your Recommendations

| Task Type | Recommended Temperature | Why? |
|-----------|------------------------|------|
| Code generation | | |
| Creative writing | | |
| Factual Q&A | | |

---

## Exercise 3: Max Tokens Control

### Task
Control response length precisely.

### Step 1: Observe Default Behavior

```python
prompt = "Explain quantum computing."
# Call without max_tokens - how long is the response?
```

### Step 2: Enforce Short Response

```python
prompt = "Explain quantum computing in one sentence."
max_tokens = 50
# Does the model respect the limit?
# Does it cut off mid-sentence?
```

### Step 3: Detailed Response

```python
prompt = "Explain quantum computing with examples and applications."
max_tokens = 1000
# Is the response more comprehensive?
```

### Findings

| max_tokens | Actual Output Length | Quality |
|------------|---------------------|---------|
| 50 | | |
| 200 | | |
| 500 | | |
| 1000 | | |

---

## Exercise 4: Top-P Sampling

### Task
Compare temperature vs top_p for controlling randomness.

### Experiment Setup

```python
prompt = "Suggest a unique pizza topping combination."

# Configuration A: High temperature, default top_p
temperature = 1.2
top_p = 1.0

# Configuration B: Default temperature, low top_p
temperature = 1.0
top_p = 0.5

# Configuration C: Balanced
temperature = 0.8
top_p = 0.9
```

### Compare Results

Run each configuration 3 times and note the variety:

| Config | Run 1 | Run 2 | Run 3 | Variety Level |
|--------|-------|-------|-------|---------------|
| A (high temp) | | | | |
| B (low top_p) | | | | |
| C (balanced) | | | | |

---

## Exercise 5: Stop Sequences

### Task
Use stop sequences to control output structure.

### Scenario 1: Numbered List Control

```python
prompt = "List programming languages:\n1."
stop = ["\n4."]  # Stop at 3 items

# Does it stop cleanly after 3 items?
```

### Scenario 2: JSON Extraction

```python
prompt = """Extract the data as JSON:
"John Smith, 30 years old, lives in Seattle"

{"""

stop = ["\n\n", "```"]
# Is the JSON complete and valid?
```

### Scenario 3: Conversation Control

```python
prompt = """Simulate a conversation:
Customer: Hi, I need help with my order.
Support:"""

stop = ["Customer:", "\nCustomer"]
# Does it stop at the right place?
```

### Stop Sequence Results

| Scenario | Stop Sequence | Worked? | Clean Output? |
|----------|---------------|---------|---------------|
| List | ["\n4."] | | |
| JSON | ["\n\n"] | | |
| Chat | ["Customer:"] | | |

---

## Exercise 6: Combined Parameters

### Task
Find the optimal combination for a complex task.

### The Challenge

Generate a product description that is:
- Creative but coherent
- Exactly 3 bullet points
- Professional tone

```python
prompt = """Write a product description for wireless earbuds.

Format:
• Feature 1
• Feature 2
• Feature 3

Description:
• """

# Experiment with different combinations:
configs = [
    {"temperature": 0.7, "max_tokens": 200, "top_p": 1.0, "stop": ["\n\n"]},
    {"temperature": 1.0, "max_tokens": 150, "top_p": 0.9, "stop": ["• ", "\n\n"]},
    {"temperature": 0.8, "max_tokens": 200, "top_p": 0.95, "stop": ["\n• Feature 4"]},
]
```

### Which Configuration Won?

| Metric | Config 1 | Config 2 | Config 3 |
|--------|----------|----------|----------|
| Creativity | /5 | /5 | /5 |
| Format adherence | /5 | /5 | /5 |
| Professionalism | /5 | /5 | /5 |
| **Total** | /15 | /15 | /15 |

---

## Exercise 7: Consistency Test

### Task
Test output consistency across multiple runs.

### The Test

```python
prompt = "What are the three primary colors?"

# Run this prompt 5 times at temperature=0.0
# Are all responses identical?

# Now run 5 times at temperature=0.7
# How much variation is there?
```

### Consistency Results

| Temperature | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Consistent? |
|-------------|-------|-------|-------|-------|-------|-------------|
| 0.0 | | | | | | |
| 0.7 | | | | | | |

---

## Exercise 8: Token Estimation

### Task
Practice estimating token counts.

### Estimate Before Running

For each text, estimate the token count, then verify:

| Text | Your Estimate | Actual | Difference |
|------|--------------|--------|------------|
| "Hello, world!" | | | |
| "The quick brown fox jumps over the lazy dog." | | | |
| `def hello(): return "hi"` | | | |
| A 100-word paragraph | | | |

### Tips for Estimation
- English: ~4 characters ≈ 1 token
- Code: More tokens (special characters)
- Common words often = 1 token
- Rare/long words = multiple tokens

---

## Challenge: Build Your Own Configuration

### The Task

Create an optimal parameter configuration for:

**A customer service chatbot that:**
- Provides helpful, consistent answers
- Stays on topic
- Responds concisely (not too long)
- Sounds professional but friendly

### Your Configuration

```python
config = {
    "temperature": ___,
    "max_tokens": ___,
    "top_p": ___,
    "stop": [___],
}
```

### Justify Your Choices

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| temperature | | |
| max_tokens | | |
| top_p | | |
| stop | | |

---

## Reflection Questions

1. **Temperature insight**: What's the highest temperature you'd use for a production application? Why?

2. **max_tokens strategy**: How would you decide max_tokens for an unknown task?

3. **Stop sequences**: When are stop sequences essential vs. optional?

4. **Trade-offs**: How do you balance creativity vs. reliability?

5. **Provider differences**: Did you notice different behaviors across providers?

---

## Key Learnings

By completing this lab, you should understand:

- [ ] Temperature's effect on randomness and creativity
- [ ] How max_tokens controls response length
- [ ] When to use top_p vs temperature
- [ ] How stop sequences provide precise control
- [ ] How to combine parameters for specific tasks
- [ ] Estimating token counts

---

## What's Next?

You've mastered LLM parameters! Now let's dive deeper into **Tokens and Context Windows** to understand the fundamental unit of LLM processing.
