# Prompt Fundamentals

## What is a Prompt?

A **prompt** is the text input you provide to a Large Language Model (LLM) to guide its response. Think of it as giving instructions to a highly capable assistant who will interpret your words literally and contextually.

```
Prompt → LLM → Response
```

The quality of your prompt directly determines the quality of the output. This relationship is so crucial that "prompt engineering" has become a core skill in AI development.

## Anatomy of a Prompt

Effective prompts typically contain several components:

### 1. Context
Background information the model needs to understand the task.

```text
You are a senior Python developer reviewing code for a startup.
```

### 2. Instructions
The specific task you want the model to perform.

```text
Review the following code and identify potential security vulnerabilities.
```

### 3. Input Data
The actual content to process.

```python
def login(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    return db.execute(query)
```

### 4. Output Format (Optional)
How you want the response structured.

```text
Format your response as:
1. Vulnerability name
2. Risk level (High/Medium/Low)
3. Recommended fix
```

## Complete Example

Here's a well-structured prompt combining all elements:

```text
Context: You are a senior Python developer reviewing code for a startup.

Task: Review the following code and identify potential security vulnerabilities.

Code:
def login(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    return db.execute(query)

Format your response as:
1. Vulnerability name
2. Risk level (High/Medium/Low)  
3. Recommended fix with code example
```

## Best Practices for Prompt Writing

### 1. Be Specific and Clear
```text
❌ Bad:  "Write a poem"
✅ Good: "Write a haiku about autumn leaves using vivid imagery"
```

### 2. Provide Examples (Few-Shot Learning)
```text
Task: Classify customer feedback as positive or negative

Examples:
- "Love this product!" → Positive
- "Worst experience ever" → Negative
- "It works okay I guess" → Neutral

Now classify: "Exceeded my expectations!"
```

### 3. Use Step-by-Step Instructions
```text
Generate a product description following these steps:
1. Start with the product name and category
2. List 3 key benefits
3. Include one unique selling point
4. End with a call-to-action
```

### 4. Specify Output Format
```text
Analyze this code and respond in JSON format:
{
  "vulnerabilities": [...],
  "risk_level": "High|Medium|Low",
  "recommendations": [...]
}
```

## The CRISPE Framework

A popular framework for structuring prompts:

| Letter | Component | Description | Example |
|--------|-----------|-------------|---------|
| **C** | Capacity | What role should the AI assume? | "You are a senior data scientist" |
| **R** | Request | What specific task do you need? | "Analyze this dataset and find patterns" |
| **I** | Instructions | How should it approach the task? | "Use statistical methods, show your work" |
| **S** | Style | What tone/format for the output? | "Respond in technical but accessible language" |
| **P** | Persona | Who is the target audience? | "For C-suite executives, avoid jargon" |

### CRISPE in Action

```text
Capacity: You are a senior software engineer
Request: Review this code for performance issues
Instructions: 
  1. Analyze algorithmic complexity
  2. Identify memory inefficiencies  
  3. Suggest concrete optimizations
Style: Be direct and technical
Persona: Respond for a Python developer with 5 years experience

Code:
```python
def find_duplicates(items):
    for i in range(len(items)):
        for j in range(len(items)):
            if i != j and items[i] == items[j]:
                print(f"Duplicate: {items[i]}")
```

Response format: List each issue, severity (High/Medium/Low), and fix
```

## Template-Based Prompting

Use templates to ensure consistency across similar tasks:

```text
TEMPLATE: API Documentation Review

System: You are a technical writer reviewing API documentation.

Structure:
1. [ENDPOINT] - HTTP method and path
2. [DESCRIPTION] - What does it do?
3. [PARAMETERS] - Input requirements
4. [RESPONSE] - Expected output format
5. [EXAMPLES] - Real usage examples

For each section, assess:
- ✓ Clarity (Is it understandable?)
- ✓ Completeness (Is anything missing?)
- ✓ Accuracy (Is it correct?)
- ✓ Examples (Are they helpful?)

Content to review:
[INSERT API DOCS HERE]

Provide recommendations in the same structure.
```
| **E** | Examples | Sample inputs/outputs to guide |

### CRISPE Example

```text
Capacity: You are an expert technical writer.

Request: Create API documentation for a user registration endpoint.

Instructions: 
- Include endpoint URL, method, parameters
- Add request/response examples
- Note any rate limits or authentication

Style: Formal technical documentation

Persona: Writing for developers who will integrate with our API

Example format:
## POST /api/register
**Parameters:**
| Name | Type | Required | Description |
...
```

## Prompt Types

### 1. Direct Instructions
Simple, straightforward requests:

```text
Translate "Hello, how are you?" to French.
```

### 2. Role-Based Prompts
Assigning a persona:

```text
As an experienced data scientist, explain the difference between 
supervised and unsupervised learning to a business executive.
```

### 3. Task Decomposition
Breaking complex tasks into steps:

```text
Help me write a Python function that:
1. Takes a list of numbers as input
2. Filters out negative numbers
3. Calculates the average of remaining numbers
4. Returns the result rounded to 2 decimal places
```

### 4. Chain of Thought
Encouraging step-by-step reasoning:

```text
Solve this step by step:
A store sells apples at $2 each. If I buy 5 apples and pay with a $20 bill, 
how much change do I get? Show your reasoning.
```

## Common Prompt Pitfalls

### ❌ Vague Instructions
```text
Make this better.
```
**Problem:** "Better" is subjective and undefined.

### ✅ Clear Instructions
```text
Improve this email by:
- Making the tone more professional
- Adding a clear call-to-action
- Keeping it under 150 words
```

---

### ❌ Missing Context
```text
Fix the bug.
```
**Problem:** Model doesn't know what bug or what code.

### ✅ Complete Context
```text
I have this Python function that should return the sum of a list,
but it returns None for empty lists instead of 0. Fix it:

def sum_list(numbers):
    total = None
    for n in numbers:
        total = (total or 0) + n
    return total
```

---

### ❌ Ambiguous Output Format
```text
List some programming languages.
```

### ✅ Specific Output Format
```text
List 5 programming languages suitable for web development.
Format as a numbered list with one sentence explaining each.
```

## Iterative Refinement

Prompt engineering is rarely one-shot. Use this cycle:

```
┌─────────────────────────────────────────┐
│  1. Write initial prompt                │
│            ↓                            │
│  2. Test with the model                 │
│            ↓                            │
│  3. Analyze the output                  │
│            ↓                            │
│  4. Identify gaps or issues             │
│            ↓                            │
│  5. Refine the prompt                   │
│            ↓                            │
│  6. Repeat until satisfied              │
└─────────────────────────────────────────┘
```

## Prompt Length Considerations

| Prompt Type | Typical Length | Use Case |
|-------------|---------------|----------|
| Short | 1-2 sentences | Simple Q&A, translations |
| Medium | 1 paragraph | Most tasks with context |
| Long | Multiple paragraphs | Complex analysis, specific formats |
| Very Long | Full documents | Document processing, detailed instructions |

> **Note:** Longer isn't always better. Include relevant information but avoid unnecessary padding that might confuse the model.

## Key Takeaways

1. **Be specific** - Vague prompts get vague responses
2. **Provide context** - Models don't have access to your mental context
3. **Specify format** - Tell the model exactly how to structure output
4. **Iterate** - Refine prompts based on actual outputs
5. **Use roles** - Personas help set the right tone and expertise level

## Next Steps

Now that you understand the fundamentals, let's explore specific **prompt patterns** that solve common problems.
