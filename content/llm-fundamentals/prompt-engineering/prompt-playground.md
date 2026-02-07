# 🧪 Lab: Prompt Playground

## Objective

Practice prompt engineering techniques by experimenting with different approaches and observing how they affect LLM outputs.

## Prerequisites

- Completed the Prompt Fundamentals lesson
- API key for at least one LLM provider
- Understanding of zero-shot and few-shot learning

## Lab Overview

In this lab, you'll:
1. Compare zero-shot vs few-shot prompts
2. Practice the CRISPE framework
3. Experiment with Chain of Thought reasoning
4. Test different prompt patterns

---

## Exercise 1: Zero-Shot vs Few-Shot

### Task
You need to classify customer support tickets into categories.

### Step 1: Try Zero-Shot First

Try this prompt and observe the output:

```python
prompt = "Classify this support ticket into one of: billing, technical, account, shipping. Ticket: 'My package shows delivered but I never received it.' Category:"
```

### Step 2: Now Try Few-Shot

Add examples and compare:

```python
prompt = """Classify support tickets into categories: billing, technical, account, shipping.

Examples:
Ticket: "I was charged twice for my subscription"
Category: billing

Ticket: "The app keeps crashing when I open settings"
Category: technical

Ticket: "I need to update my email address"
Category: account

Ticket: "My order hasn't arrived yet"
Category: shipping

Now classify:
Ticket: "My package shows delivered but I never received it."
Category:"""
```

### Questions to Consider
- Did the outputs differ?
- Which approach gave more consistent results?
- When would you choose each approach?

---

## Exercise 2: CRISPE Framework

### Task
Use the CRISPE framework to write a prompt that generates product descriptions.

### Your Template

```python
prompt = """
Capacity: You are an expert e-commerce copywriter with 10 years of experience.

Request: Write a product description for a smart water bottle.

Instructions:
- Highlight the key features (temperature display, hydration reminders)
- Keep the description under 100 words
- Use persuasive but honest language

Style: Conversational and energetic, suitable for health-conscious millennials

Persona: Writing for busy professionals who want to improve their hydration habits

Example format:
[Catchy headline]
[2-3 sentence description]
[Key features as bullet points]
[Call to action]
"""
```

### Your Turn
Modify the CRISPE prompt for a different product of your choice.

---

## Exercise 3: Chain of Thought

### Task
Compare direct answers vs Chain of Thought for a logic problem.

### Step 1: Direct Prompt

```python
prompt = "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?"
```

### Step 2: Chain of Thought Prompt

```python
prompt = """A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?

Let's think through this step by step:
1. What does "all but 9" mean?
2. How does this relate to the total?
3. What's the final answer?"""
```

### Step 3: Test with a Harder Problem

```python
prompt = """In a room, there are 5 machines. Each machine produces 5 widgets per minute.
Each widget requires 5 minutes to be fully assembled.
How many widgets are fully assembled after 5 minutes?

Let's reason through this step by step."""
```

---

## Exercise 4: Prompt Patterns

### Pattern A: Persona + Constraint

```python
prompt = """As a senior software architect, explain microservices to a junior developer.

Constraints:
- Use a restaurant kitchen analogy
- Keep it under 150 words
- Avoid jargon
- End with one key takeaway"""
```

### Pattern B: Template Pattern

```python
prompt = """Analyze this business idea: "A subscription service for personalized vitamin packs"

Use this exact format:

## One-Sentence Summary
[Your summary]

## Target Market
[Who would buy this]

## Competitive Advantage
[What makes it unique]

## Main Challenge
[Biggest obstacle]

## Verdict
[👍 or 👎 with brief explanation]"""
```

### Pattern C: Critique Pattern

```python
prompt = """Write a Python function that checks if a string is a palindrome.

After writing the function:
1. Review it for edge cases
2. Identify at least 2 potential issues
3. Provide an improved version"""
```

---

## Exercise 5: Iterative Refinement

### Starting Prompt
```python
prompt = "Write an email."
```

### Iteration 1: Add Context
```python
prompt = "Write a professional email asking for a meeting."
```

### Iteration 2: Add Details
```python
prompt = "Write a professional email to a potential client asking for a 30-minute meeting to discuss our software services."
```

### Iteration 3: Add Format
```python
prompt = """Write a professional email to a potential client asking for a 30-minute meeting.

Include:
- A compelling subject line
- Brief introduction of our company
- Value proposition (why they should meet)
- 2-3 proposed time slots
- Professional sign-off

Keep it under 150 words."""
```

### Your Turn
Start with a simple prompt and iterate through 3-4 refinements.

---

## Exercise 6: Compare Providers

### Same Prompt, Different Providers

Run this prompt on different LLM providers and compare:

```python
prompt = """Explain quantum computing to a 10-year-old using only simple words 
and a fun analogy. Keep it under 100 words."""
```

### What to Compare
- Explanation clarity
- Analogy creativity  
- Age-appropriateness
- Length adherence

---

## Challenges

### Challenge 1: Format Enforcement
Write a prompt that ALWAYS returns valid JSON:
```json
{"sentiment": "positive|negative|neutral", "confidence": 0.0-1.0, "keywords": []}
```

### Challenge 2: Consistent Length
Write a prompt that consistently generates exactly 5 bullet points, each 10-15 words.

### Challenge 3: Error Handling
Write a prompt that gracefully handles ambiguous or incomplete input rather than making assumptions.

---

## Reflection Questions

After completing the exercises:

1. Which prompting technique was most effective for your use cases?
2. How much did the output quality improve with better prompts?
3. What patterns do you think you'll use most in real projects?
4. How did different providers handle the same prompts?

---

## What's Next?

You've practiced the fundamentals of prompt engineering! Next, we'll explore **LLM Parameters** to understand how temperature, top_p, and other settings affect model outputs.
